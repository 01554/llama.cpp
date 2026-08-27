# expert-tier — a llama.cpp fork for running huge MoE models on small VRAM

**What this is**: upstream llama.cpp offloads MoE weights to CPU *per layer*. This fork
adds **per-expert placement**: the few hundred experts a model actually fires most live in
a VRAM hot cache (heat-tracked, migrating at a capped rate); everything else stays in
ordinary host RAM. On models with skewed routing this beats layer-split offload by
30-100% at the same VRAM budget. The branch also carries
[PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) (Qwen3.8-Flash-Next).

**Headline result** — Qwen3.8-Flash-Next (176B total, unsloth UD-Q4_K_XL) with VRAM
**capped to 32GB** (RTX 5090-sized) + 96GB host RAM:

| 32GB-cap config | decode |
|---|---:|
| plain layer-split (`--n-cpu-moe 36`) | 24.2 tok/s |
| **this fork, expert hot cache (`--expert-hot-s 140`)** | **32.9 tok/s (+36%)** |

Uncapped on the full 96GB card: 74.9 tok/s decode / 2922 t/s prefill (pp4096).
Hardware for all numbers: 1x RTX PRO 6000 Blackwell Max-Q (300W) + 128GB DDR4 dual-channel.

## Qwen3.8-Flash-Next (125B-A6B + 51B n-gram PLE, unsloth UD-Q4_K_XL 111GB)

| config | VRAM | decode | prefill (7k prompt) |
|---|---:|---:|---:|
| `-ngl 99 --n-cpu-moe 10` | 65 GB | 43.8 tok/s | |
| `-ngl 99 --n-cpu-moe 3` | 76 GB | 60.5 tok/s | |
| **`-ngl 99` (all experts on GPU)** | **80 GB** | **74.9 tok/s** | **2210 tok/s** |
| VRAM capped to 32GB (64GB balloon) `--n-cpu-moe 36` | ~28 GB | 24.2 tok/s |
| VRAM capped to 32GB, **`--cpu-moe --expert-hot-s 140`** (this fork) | ~31 GB | **32.9 tok/s** | |

- The 51B n-gram (PLE) table stays host-resident (~27GB) — its per-token cost is a few KB of
  row gathers, so "table in RAM" is effectively free. This is what makes the model fit.
- The 32GB-cap row simulates an RTX 5090: a physical 64GiB `cudaMalloc` balloon holds the
  memory, so the allocator genuinely has only ~31GB to work with.
- Sampling: official thinking-mode settings (temp 1.0, top-p 0.95). Server flags:
  `-c 8192 -fa on --jinja` (+`-b 2048 -ub 2048` for the prefill run).

## Reproduce (Flash-Next, 32GB-cap result)

```bash
git clone -b expert-tier https://github.com/01554/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -t llama-server -j $(nproc)

# model (111GB)
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" --local-dir models
```

Optional - simulate a 32GB card on a bigger one (hold 64GiB in a dummy process):

```c
// balloon.cu - build: nvcc -o balloon balloon.cu ; run: ./balloon 64 &
#include <cuda_runtime.h>
#include <unistd.h>
int main(int c, char **v) { void *p; cudaMalloc(&p, (c>1?atoll(v[1]):64ULL)<<30); for(;;) sleep(3600); }
```

Baseline (plain layer-split offload, ~24 tok/s under the 32GB cap):

```bash
./build/bin/llama-server -m models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 99 --n-cpu-moe 36 -c 8192 -fa on --jinja
```

This fork's expert hot cache (~33 tok/s under the same cap):

```bash
LLAMA_EHS_SWAPS_PER_TOK=1 \
./build/bin/llama-server -m models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 99 --cpu-moe --expert-hot-s 140 -c 8192 -fa on --jinja
```

Full-VRAM (no cap, 96GB card): drop the balloon and the offload flags, just `-ngl 99` (~75 tok/s).
Host RAM needed for the capped runs: ~96GB (60GB cold experts + 27GB n-gram table).
Sampling for quality checks: temp 1.0, top-p 0.95 (official thinking-mode settings).

## Expert-tier machinery (measured on DeepSeek V4-Flash 0731, UD-Q4_K_XL 155GB)

Single GPU cannot hold the 145GB expert pool; the branch adds:

- `--cpu-moe --expert-hot-s N` (from PR #26563, merged here + fixed): per-expert GPU hot
  store with heat-based migration. **19.8 → 40.3 tok/s** vs plain layer-split offload.
- `LLAMA_EHS_COLD_BANK=1 LLAMA_EHS_B2=1`: **exclusive banks** — every expert lives in exactly
  one place (VRAM hot store or a compact pinned-host cold bank), demotion is a D2H copy,
  no disk in the steady state, host RAM holds only the bank (not a full mmap copy).
  36-37.5 tok/s with full exclusivity; wikitext PPL identical to baseline (3.319 vs 3.324).
- `LLAMA_EHS_SWAPS_PER_TOK=K`: migration-rate cap (churn control).
- `LLAMA_EHS_HEAT_PRIOR=routing.csv`: warm-start expert placement from a measured routing
  profile (see `examples/routing-stats`).
- `examples/routing-stats`: per-layer expert routing histogram / raw-sequence capture tool
  (mind the `ffn_moe_topk` non-contiguous view — this tool reads it correctly).
- mmvq sentinel-skip: cold-bank `mul_mat_id` skips zero-sentinel slots before touching
  weight bytes (tagged via op_params; no effect on stock paths).
- Fixes made along the way: speculative-context hot-store clobber (draft acceptance
  93%→38% bug), `ffn_moe_topk` stride handling, `GGML_CUDA_ALLOW_HOST_BUFT=1` to let
  discrete GPUs compute over pinned host memory (UVA), and more — see commit log.

## Context

Written up during day-0 bring-up of Qwen3.8-Flash-Next; related reports:
transformers issues #48349 / #48350 / #48351, and
https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/discussions/2.
Status: experimental. Upstream-friendly pieces will be offered as PRs.
