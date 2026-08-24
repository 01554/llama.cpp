// routing-stats: per-layer expert routing histogram collector.
//
// Feeds a corpus (-f file) through the model in context-sized chunks and
// accumulates, for every MoE layer, how often each expert id appears in the
// router's top-k ("ffn_moe_topk-<il>" tensors observed via cb_eval).
// Output: CSV "layer,expert,count" at $ROUTING_CSV (default routing_stats.csv).
//
// Derived from examples/eval-callback (same plumbing, different callback).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct routing_stats {
    // counts[layer][expert]
    std::vector<std::vector<int64_t>> counts;
    int64_t n_selections = 0;
    std::vector<int32_t> buf;
    FILE * raw = nullptr;

    void add(int layer, const int32_t * ids, int64_t n) {
        if (layer >= (int) counts.size()) {
            counts.resize(layer + 1);
        }
        auto & c = counts[layer];
        for (int64_t i = 0; i < n; i++) {
            const int32_t e = ids[i];
            if (e < 0) {
                continue;
            }
            if (e >= (int32_t) c.size()) {
                c.resize(e + 1, 0);
            }
            c[e]++;
            n_selections++;
        }
    }
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * stats = (routing_stats *) user_data;

    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) {
        return is_topk; // only fetch the router top-k id tensors
    }
    if (!is_topk || t->type != GGML_TYPE_I32) {
        return true;
    }

    const int layer = atoi(t->name + 13);
    // "ffn_moe_topk" is a NON-CONTIGUOUS view: ne[0]=k rows selected out of the
    // full argsort, nb[1] strides over all n_expert entries. A linear read would
    // capture the whole argsort permutation (uniform by construction!), so copy
    // row by row honoring nb[1].
    const int64_t k  = t->ne[0];
    const int64_t nt = t->ne[1] * t->ne[2] * t->ne[3];
    const int64_t n  = k * nt;
    stats->buf.resize(n);
    for (int64_t j = 0; j < nt; j++) {
        ggml_backend_tensor_get(t, stats->buf.data() + j * k, j * t->nb[1], k * sizeof(int32_t));
    }
    stats->add(layer, stats->buf.data(), n);
    if (stats->raw != nullptr) {
        const int32_t hdr[2] = { layer, (int32_t) n };
        fwrite(hdr, sizeof(int32_t), 2, stats->raw);
        fwrite(stats->buf.data(), sizeof(int32_t), n, stats->raw);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    routing_stats stats;
    if (const char * rp = getenv("ROUTING_RAW")) {
        stats.raw = fopen(rp, "wb");
    }
    common_params params;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &stats;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    LOG_INF("corpus tokens: %zu\n", tokens.size());
    if (tokens.empty()) {
        LOG_ERR("no input tokens (use -f corpus.txt)\n");
        return 1;
    }

    const int32_t n_ctx_use = (int32_t) llama_n_ctx(ctx) - 8;
    const int32_t n_batch   = (int32_t) params.n_batch;

    for (size_t off = 0; off < tokens.size(); off += n_ctx_use) {
        const size_t n_chunk = std::min((size_t) n_ctx_use, tokens.size() - off);
        for (size_t b = 0; b < n_chunk; b += n_batch) {
            const size_t n = std::min((size_t) n_batch, n_chunk - b);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + off + b, (int32_t) n))) {
                LOG_ERR("llama_decode failed at offset %zu\n", off + b);
                return 1;
            }
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        LOG_INF("processed %zu / %zu tokens\n", std::min(off + n_chunk, tokens.size()), tokens.size());
    }

    if (const char * gn = getenv("ROUTING_GEN_N")) {
        const int n_gen = atoi(gn);
        llama_memory_clear(llama_get_memory(ctx), true);
        const size_t tail = std::min((size_t) 2048, tokens.size());
        const size_t t0 = tokens.size() - tail;
        for (size_t b = 0; b < tail; b += n_batch) {
            const size_t n = std::min((size_t) n_batch, tail - b);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + t0 + b, (int32_t) n))) {
                LOG_ERR("gen prefill failed\n");
                return 1;
            }
        }
        llama_sampler * smpl = llama_sampler_init_greedy();
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        for (int i = 0; i < n_gen; i++) {
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
                LOG_ERR("gen decode failed at %d\n", i);
                break;
            }
            tok = llama_sampler_sample(smpl, ctx, -1);
            if ((i + 1) % 256 == 0) {
                LOG_INF("generated %d / %d\n", i + 1, n_gen);
            }
        }
        llama_sampler_free(smpl);
        LOG_INF("GEN_CAPTURE_DONE n=%d\n", n_gen);
    }

    const char * out = getenv("ROUTING_CSV");
    if (out == nullptr) {
        out = "routing_stats.csv";
    }
    FILE * f = fopen(out, "w");
    if (f == nullptr) {
        LOG_ERR("cannot open %s\n", out);
        return 1;
    }
    fprintf(f, "layer,expert,count\n");
    for (size_t il = 0; il < stats.counts.size(); il++) {
        for (size_t e = 0; e < stats.counts[il].size(); e++) {
            if (stats.counts[il][e] > 0) {
                fprintf(f, "%zu,%zu,%lld\n", il, e, (long long) stats.counts[il][e]);
            }
        }
    }
    fclose(f);
    if (stats.raw != nullptr) {
        fclose(stats.raw);
    }
    LOG_INF("ROUTING_STATS_OK selections=%lld layers=%zu -> %s\n",
            (long long) stats.n_selections, stats.counts.size(), out);

    llama_backend_free();
    return 0;
}
