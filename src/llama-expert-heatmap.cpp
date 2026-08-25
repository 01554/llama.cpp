#include "llama-expert-heatmap.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

llama_expert_heatmap::llama_expert_heatmap(
        int n_layers, int n_experts,
        float decay_rate, int log_period, int hot_s) :
    n_layers(n_layers),
    n_experts(n_experts),
    hot_s(hot_s),
    decay_rate(decay_rate),
    log_period(log_period),
    tokens_total(0),
    heat(n_layers * n_experts, 0.0f) {
    // Optional hotness prior: LLAMA_EHS_HEAT_PRIOR=<csv with header layer,expert,count>
    // (the routing-stats tool emits this format). Loaded as ~LLAMA_EHS_PRIOR_TOKENS
    // (default 500) tokens worth of evidence per layer, so live routing can still
    // overrule a stale prior within a few hundred tokens.
    const char * prior_path = getenv("LLAMA_EHS_HEAT_PRIOR");
    if (prior_path != nullptr) {
        FILE * f = fopen(prior_path, "r");
        if (f == nullptr) {
            LLAMA_LOG("EHS_PRIOR: cannot open %s\n", prior_path);
        } else {
            const char * w = getenv("LLAMA_EHS_PRIOR_TOKENS");
            const double prior_tokens = w ? atof(w) : 500.0;
            std::vector<double> raw(n_layers * n_experts, 0.0);
            std::vector<double> layer_sum(n_layers, 0.0);
            char line[256];
            if (fgets(line, sizeof(line), f)) {} // header
            int il, e; long long c;
            while (fgets(line, sizeof(line), f)) {
                if (sscanf(line, "%d,%d,%lld", &il, &e, &c) == 3 &&
                    il >= 0 && il < n_layers && e >= 0 && e < n_experts) {
                    raw[il * n_experts + e] = (double) c;
                    layer_sum[il] += (double) c;
                }
            }
            fclose(f);
            int loaded_layers = 0;
            for (int l = 0; l < n_layers; l++) {
                if (layer_sum[l] <= 0.0) {
                    continue;
                }
                loaded_layers++;
                // scale layer to prior_tokens * (selections per token in that layer)
                const double target = prior_tokens * (layer_sum[l] > 0 ? 1.0 : 0.0);
                for (int x = 0; x < n_experts; x++) {
                    heat[l * n_experts + x] = (float) (raw[l * n_experts + x] / layer_sum[l] * target * 6.0);
                }
            }
            LLAMA_LOG("EHS_PRIOR: loaded %s (%d layers, ~%.0f tokens of evidence)\n",
                      prior_path, loaded_layers, prior_tokens);
        }
    }
}

void llama_expert_heatmap::update(int layer_idx, const int32_t * expert_ids, int n_expert_used, int n_tokens) {
    float * layer_heat = heat.data() + layer_idx * n_experts;

    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert_used; e++) {
            int32_t id = expert_ids[t * n_expert_used + e];
            if (id >= 0 && id < n_experts) {
                layer_heat[id] += 1.0f;
            }
        }
    }
}
void llama_expert_heatmap::update_from_graph(const std::vector<std::pair<int, ggml_tensor *>> & moe_sel_experts) {
    if (moe_sel_experts.empty()) {
        return;
    }

    decay_all();

    int64_t n_tokens = 0;
    for (const auto & [il, tensor] : moe_sel_experts) {
        n_tokens = tensor->ne[1];

        if (!tensor->data) {
            continue;
        }

        std::vector<int32_t> expert_ids(tensor->ne[0] * n_tokens);
        ggml_backend_tensor_get(tensor, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));

        update(il, expert_ids.data(), tensor->ne[0], n_tokens);
    }

    tokens_total += n_tokens;
    if (log_period > 0 && tokens_total / log_period > (tokens_total - n_tokens) / log_period) {
        log();
    }
}

void llama_expert_heatmap::decay_all() {
    for (int i = 0; i < n_layers * n_experts; i++) {
        heat[i] *= decay_rate;
    }
}

void llama_expert_heatmap::log() const {
    LLAMA_LOG("=== Expert heatmap (tokens %" PRId64 ") ===\n", tokens_total);

    for (int l = 0; l < n_layers; l++) {
        const float * layer_heat = heat.data() + l * n_experts;
        int active_count = 0;
        float max_heat = 0.0f;
        int max_id = -1;

        for (int e = 0; e < n_experts; e++) {
            if (layer_heat[e] > 0.01f) {
                active_count++;
            }
            if (layer_heat[e] > max_heat) {
                max_heat = layer_heat[e];
                max_id = e;
            }
        }

        if (active_count > 0) {
            LLAMA_LOG("  layer %3d: %d warm experts, max heat=%.2f (expert %d)",
                l, active_count, max_heat, max_id);

            auto top = get_top_s(l, 8);
            LLAMA_LOG("  top-8=");
            for (size_t i = 0; i < top.size(); i++) {
                LLAMA_LOG("%s%d", i > 0 ? "," : "{", top[i]);
            }
            LLAMA_LOG("}\n");
        }
    }
}

float llama_expert_heatmap::get_score(int layer_idx, int expert_id) const {
    if (layer_idx < 0 || layer_idx >= n_layers || expert_id < 0 || expert_id >= n_experts) {
        return 0.0f;
    }
    return heat[layer_idx * n_experts + expert_id];
}

std::vector<int> llama_expert_heatmap::get_top_s(int layer_idx, int s) const {
    std::vector<int> result;
    if (layer_idx < 0 || layer_idx >= n_layers || s <= 0) {
        return result;
    }

    const float * layer_heat = heat.data() + layer_idx * n_experts;

    std::vector<int> indices(n_experts);
    for (int i = 0; i < n_experts; i++) {
        indices[i] = i;
    }

    int k = std::min(s, n_experts);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [layer_heat](int a, int b) {
            return layer_heat[a] > layer_heat[b];
        });

    result.assign(indices.begin(), indices.begin() + k);
    return result;
}
