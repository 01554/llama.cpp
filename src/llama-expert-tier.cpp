#include "llama-expert-tier.h"

#include <algorithm>

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {
    struct tier_entry {
        ggml_tensor * dst_hot;    // [ne0, ne1, hot_s + 8]
        ggml_tensor * dst_cold;   // pinned-host cold bank [ne0, ne1, cold_s + 8], or null
        ggml_tensor * hot_lut;    // i32 [n_experts]
        ggml_tensor * cold_mask;  // f32 [n_experts] (1 = cold)
        ggml_tensor * cold_lut;   // i32 [n_experts] (bank slot / sentinel), or null
        ggml_tensor * hot_mask;   // f32 [n_experts] (1 = hot), or null
        ggml_tensor * cold_lut_cpu; // i32 [n_experts] CPU copy for the B2 cold path, or null
    };

    std::mutex g_mtx;
    std::unordered_map<ggml_tensor *, tier_entry> g_table;
}

void llama_expert_tier_register(ggml_tensor * src,
                                ggml_tensor * dst_hot,
                                ggml_tensor * dst_cold,
                                ggml_tensor * hot_lut,
                                ggml_tensor * cold_mask,
                                ggml_tensor * cold_lut,
                                ggml_tensor * hot_mask,
                                ggml_tensor * cold_lut_cpu) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table[src] = {dst_hot, dst_cold, hot_lut, cold_mask, cold_lut, hot_mask, cold_lut_cpu};
}

void llama_expert_tier_clear() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_table.clear();
}

bool llama_expert_tier_has(ggml_tensor * w) {
    std::lock_guard<std::mutex> lk(g_mtx);
    return g_table.find(w) != g_table.end();
}

// Remap real expert ids through a LUT and produce a 2d [n_expert_used,
// n_tokens] i32 tensor usable as `ids` for ggml_mul_mat_id.
//
// Flatten the per-token ids to 1D, run a stock 1D ggml_get_rows against the
// per-expert LUT (reshape to [1, n_experts] so each row picked is 1 scalar),
// then reshape the [1, n_eu*n_tok] result back to 2D. Avoids ggml_repeat_4d
// + view_2d strides that the CUDA mul_mat_id kernel mishandles on multi-token
// ubatches. `ggml_cont` defends against argsort views that may not be
// contiguous across the n_expert_used * n_tokens layout.
static ggml_tensor * remap_ids(ggml_context * ctx,
                              ggml_tensor * lut,
                              ggml_tensor * cold_mask,
                              ggml_tensor * selected,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    ggml_tensor * lut_rows = ggml_reshape_2d(ctx, lut, 1, n_experts);    // [1, n_experts]
    // selected (argsort_top_k view) may be non-contiguous; cont first, then reshape
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
        ggml_cont(ctx, selected), n_expert_used * n_tokens);             // [n_eu*n_tok] i32
    ggml_tensor * r = ggml_get_rows(ctx, lut_rows, flat_ids);            // [1, n_eu*n_tok, 1, 1] i32
    // lane-distinct sentinels: cold lanes map to hot_s + lane instead of a single
    // shared sentinel, so ids within one token never repeat (required by the
    // batched CUDA mul_mat_id compaction; verified empirically via draft
    // acceptance, which collapses with duplicated sentinel ids).
    ggml_tensor * mask_rows = ggml_reshape_2d(ctx, cold_mask, 1, n_experts);
    ggml_tensor * cold_flat = ggml_get_rows(ctx, mask_rows, flat_ids);   // [1, n_eu*n_tok] f32 (1=cold)
    cold_flat = ggml_reshape_1d(ctx, cold_flat, n_expert_used * n_tokens);
    ggml_tensor * lane = ggml_arange(ctx, 0.0f, (float) n_expert_used, 1.0f); // [n_eu] f32
    ggml_tensor * lane_rep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert_used * n_tokens);
    lane_rep = ggml_repeat(ctx, lane, lane_rep);                          // 0..n_eu-1 repeated
    ggml_tensor * off = ggml_mul(ctx, cold_flat, lane_rep);               // lane where cold, 0 where hot
    ggml_tensor * r_f = ggml_cast(ctx, ggml_reshape_1d(ctx, r, n_expert_used * n_tokens), GGML_TYPE_F32);
    ggml_tensor * sum = ggml_add(ctx, r_f, off);
    ggml_tensor * out = ggml_cast(ctx, sum, GGML_TYPE_I32);
    return ggml_reshape_2d(ctx, out, n_expert_used, n_tokens);            // [n_eu, n_tok]
}

// Build a per-(expert_used, token) mask f32 [1, n_expert_used, n_tokens, 1]
// (broadcastable against a mul_mat_id output of shape [out, n_eu, n_tok, 1]).
// Same flatten pattern as remap_ids.
static ggml_tensor * remap_mask(ggml_context * ctx,
                              ggml_tensor * mask,
                              ggml_tensor * selected,
                              int n_experts,
                              int n_expert_used,
                              int n_tokens) {
    ggml_tensor * mask_rows = ggml_reshape_2d(ctx, mask, 1, n_experts);  // [1, n_experts] f32
    ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
        ggml_cont(ctx, selected), n_expert_used * n_tokens);             // [n_eu*n_tok] i32
    ggml_tensor * r = ggml_get_rows(ctx, mask_rows, flat_ids);           // [1, n_eu*n_tok, 1, 1] f32
    return ggml_reshape_3d(ctx, r, 1, n_expert_used, n_tokens);          // [1, n_eu, n_tok]
}

ggml_tensor * llama_expert_tier_build(ggml_context * ctx,
                                      ggml_tensor * w,
                                      ggml_tensor * cur,
                                      ggml_tensor * ids,
                                      ggml_tensor * w_s) {
    // BUG-3E3: CUDA MMQ mul_mat_id compaction assumes distinct expert ids per token;
    // our sentinel duplicates OOB there. Decode (n_tokens==1) is safe via mmvq.
    // lane-distinct sentinels make ids unique per token, so small verification
    // batches are safe; larger (prefill-sized) batches still fall back for now.
    if (cur->ne[2] > 8) return nullptr;

    tier_entry ent;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_table.find(w);
        if (it == g_table.end()) {
            return nullptr;
        }
        ent = it->second;
    }

    const int n_experts     = (int) w->ne[2];
    const int n_expert_used = (int) ids->ne[0];
    const int n_tokens      = (int) cur->ne[2];

    // hot path: GPU tier tensor. Remap real expert ids through hot_lut
    // -> hot slot indices (sentinel S for cold experts = zero contribution).
    ggml_tensor * ids_hot = remap_ids(ctx, ent.hot_lut, ent.cold_mask, ids, n_experts, n_expert_used, n_tokens);
    ggml_tensor * hot = ggml_mul_mat_id(ctx, ent.dst_hot, cur, ids_hot);

    ggml_tensor * cold = nullptr;
    static const bool b2 = getenv("LLAMA_EHS_B2") != nullptr;
    if (b2 && ent.dst_cold != nullptr && ent.cold_lut_cpu != nullptr) {
        // B2: exclusive cold bank, computed on the CPU (latency-free host reads).
        // Real ids + mask semantics as the legacy op; bank rows resolved via the
        // CPU-resident lut.
        cold = ggml_mul_mat_id_cold_bank(ctx, ent.dst_cold, cur, ids, ent.cold_mask, ent.cold_lut_cpu);
    } else if (ent.dst_cold != nullptr && ent.cold_lut != nullptr && ent.hot_mask != nullptr) {
        // Stage C: cold experts live in a pinned-host bank the GPU reads via
        // UVA; both paths are plain GPU mul_mat_id ops and both banks carry
        // zero sentinel slots, so no output masking is needed.
        ggml_tensor * ids_cold = remap_ids(ctx, ent.cold_lut, ent.hot_mask, ids, n_experts, n_expert_used, n_tokens);
        cold = ggml_mul_mat_id(ctx, ent.dst_cold, cur, ids_cold);
        // tag the node so the CUDA mmvq path skips sentinel (>= cold_s) reads
        cold->op_params[14] = (int32_t) 0x51D3C01D;
        cold->op_params[15] = (int32_t) (ent.dst_cold->ne[2] - std::max(8, n_expert_used));
    } else {
        // legacy: dedicated CPU op that computes ONLY cold-selected experts.
        cold = ggml_mul_mat_id_cold(ctx, w, cur, ids, ent.cold_mask);
    }

    // per-expert quant scale on both paths
    if (w_s) {
        ggml_tensor * s_rows = ggml_reshape_2d(ctx, w_s, 1, n_experts);
        ggml_tensor * flat_ids = ggml_reshape_1d(ctx,
            ggml_cont(ctx, ids), n_expert_used * n_tokens);
        ggml_tensor * s = ggml_get_rows(ctx, s_rows, flat_ids);
        s = ggml_reshape_3d(ctx, s, 1, n_expert_used, n_tokens);
        hot  = ggml_mul(ctx, hot,  s);
        cold = ggml_mul(ctx, cold, s);
    }

    return ggml_add(ctx, hot, cold);
}