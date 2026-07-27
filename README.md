# MiniMax Sparse Attention (MSA)

> **Fireworks integration branch.** The KV-outer sparse prefill backend on this
> branch is vendored from
> [fw-ai/minimax-kernels](https://github.com/fw-ai/minimax-kernels/tree/main),
> developed by [Fireworks AI](https://fireworks.ai). For upstream docs, the full
> test matrix, and benchmark results, see that repository. Background on the
> KV-outer work:
> [Kernel optimization for MiniMax M3 on NVIDIA Blackwell](https://fireworks.ai/blog/kernel-optimization-for-minimax-m3-on-nvidia-blackwell).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](pyproject.toml)
[![GPU](https://img.shields.io/badge/NVIDIA-SM100-76b900.svg)](#requirements)
[![Stack: CuTe-DSL + CUDA](https://img.shields.io/badge/stack-CuTe--DSL%20%2B%20CUDA-purple.svg)](#stacks)

**MSA** (`fmha_sm100`) ships dense FlashAttention and sparse top-k attention
kernels for **NVIDIA SM100**. Three kernel stacks plus a routing bridge share one
Python package; the public `fmha_sm100` / `fmha_sm100_plan` API is unchanged.

![MSA architecture](docs/architecture.png)

> **Note:** `docs/architecture.png` predates the KV-outer integration and shows the
> original two-stack layout. On this branch, sparse prefill is routed through
> KV-outer (GQA ≥ 8) or legacy CuTe (GQA &lt; 8) via `sparse_fmha_adapter.py`.

> Algorithm reference: [MiniMax Sparse Attention paper](docs/MiniMaxSparseAttention.pdf).

| Stack | Path | What it gives you |
|---|---|---|
| **csrc JIT** | `python/fmha_sm100/csrc/` | Dense FMHA (`fmha_sm100`, `fmha_sm100_plan`) + `sparse_topk_select` indexer, compiled from Jinja templates by `jit.py` at runtime. |
| **KV-outer** | `python/fmha_sm100/kvouter/` + `csrc/kvouter/` | Fireworks KV-outer sparse **prefill** (GQA ≥ 8): Python CuTe-DSL + optional C++ AOT extension (`fmha_sm100._C`). Default path when `qhead_per_kv ≥ 8`. |
| **CuTe-DSL (legacy sparse)** | `python/fmha_sm100/cute/` | CSR sparse prefill fallback (GQA &lt; 8), paged FP8 decode, BF16 / FP8 / NVFP4 / FP4 paths, compiled at runtime via `cute.compile`. |
| **Bridge** | `python/fmha_sm100/sparse_fmha_adapter.py` | Routes sparse prefill through KV-outer or legacy CuTe based on GQA ratio; adapts the `fmha_sm100` API to both backends. |

> **License: MIT.** Self-authored files carry `SPDX-License-Identifier: MIT`.
> See [LICENSE](LICENSE) and [NOTICE](NOTICE). Bundled / derived third-party
> code retains its own license — see [Third-party licenses](#third-party-licenses).

## Requirements

- **GPU**: NVIDIA SM100 (Blackwell).
- **Toolchain**: CUDA Toolkit **13.x** with `nvcc` on `PATH` (or `CUDA_HOME` / `CUDA_PATH` set). Required for all stacks (csrc JIT, KV-outer, legacy CuTe).
- **Python**: ≥ 3.10.
- **PyTorch**: ≥ 2.9 (see `pyproject.toml`).
- **OS**: Linux x86_64 (aarch64 untested; JIT builds may need small Makefile edits on WSL).

Quick sanity check before installing:

```bash
nvcc --version                # expect ≥ 13.x
nvidia-smi --query-gpu=compute_cap --format=csv | grep "10.0"  # confirm SM100
python -c "import sys; print(sys.version_info[:2])"              # ≥ (3, 10)
python -c "import torch; print(torch.__version__)"                # ≥ 2.9
```

## Using with the `kernels` library

> **Note:** The Hugging Face Hub build tracks upstream `main` and does not include
> the Fireworks KV-outer backend on this branch. For KV-outer sparse prefill,
> install from the `fireworks-msa` branch (see [Install](#install)).

To quickly get started using upstream MSA kernels, you can use the [`kernels` library](https://github.com/huggingface/kernels):

```py
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("MiniMaxAI/msa", version=0)
sparse_atten_func = kernel_module.sparse_atten_func

sparse_atten_func(...)
```

Check out the kernel on the Hugging Face Hub [here](https://huggingface.co/kernels/kernels-staging/msa).

## Install

```bash
# --recursive pulls the NVIDIA CUTLASS submodule (python/fmha_sm100/cutlass/),
# whose headers are required for dense FMHA JIT / AOT compilation.
git clone --recursive https://github.com/MiniMax-AI/MSA.git msa
cd msa
git checkout fireworks-msa   # this integration branch
# If you cloned without --recursive:
#   git submodule update --init --recursive
pip install -e . --no-build-isolation   # editable install; builds fmha_sm100._C (KV-outer)
# or
pip install . --no-build-isolation      # standard install
```

This pulls in `nvidia-cutlass-dsl`, `quack-kernels`, and `flash-attn-4` (see
`pyproject.toml`). Dense csrc kernels are JIT-compiled on first use and cached
under `~/.cache/minfer/fmha_sm100/` (delete that directory to force recompile).
The KV-outer C++ extension is built at install time. On the first KV-outer call
per config, CuTe-DSL kernels are AOT-exported into a per-process temp directory;
set `MINIMAX_KERNELS_CUTE_AOT_CACHE` to a fixed path to persist that cache across
restarts.

**KV-outer backend selection** (optional env vars):

| Variable | Effect |
|---|---|
| unset (default) | Use C++ AOT extension when `fmha_sm100._C` is available; else Python CuTe-DSL |
| `FMHA_SM100_KVOUTER_CPP=1` | Force C++ AOT |
| `FMHA_SM100_KVOUTER_CPP=0` | Force Python CuTe-DSL |
| `MINIMAX_KERNELS_KVOUTER_CPP` | Legacy alias for the same flags |
| `MINIMAX_KERNELS_CUTE_AOT_CACHE` | Persistent directory for KV-outer CuTe-DSL AOT exports (default: per-process temp dir) |

Sparse prefill uses KV-outer when `num_qo_heads // num_kv_heads ≥ 8` (e.g. TP1
config 64/4); lower GQA ratios still use the legacy CuTe CSR path under `cute/`.

## Verify

Run small CUDA smoke tests after install:

```bash
# Dense indexer JIT (first run compiles sparse_topk_select; 30 s – a few minutes
# on a cold nvcc cache is normal, not a hang). Cached under ~/.cache/minfer/fmha_sm100/.
python tests/smoke/test_sparse_topk_forced.py

# End-to-end sparse prefill via the fmha_sm100 API (exercises KV-outer when
# qhead_per_kv >= 8; default h_r_real=8). First call may AOT-export CuTe-DSL
# kernels into a per-process temp dir (or MINIMAX_KERNELS_CUTE_AOT_CACHE).
python tests/smoke/test_proxy_kv_smoke.py
```

## Usage

```python
import torch
from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select

# Page size and top-k for the sparse prefill path.
page_size, topk = 128, 16

# Dense proxy pass: compute per-block max score from a cheap Q slice.
proxy_plan = fmha_sm100_plan(
    qo_lens, kv_lens, proxy_q.shape[1],
    num_kv_heads=1,
    page_size=page_size,
    output_maxscore=True,
)
_, max_score = fmha_sm100(
    proxy_q, proxy_k_pages, proxy_v_pages, proxy_plan,
    kv_indices=kv_indices,
    output_o=False,
    output_maxscore=True,
)

# max_score -> sparse KV block indexes.
kv_block_indexes = sparse_topk_select(
    max_score.contiguous(), topk, num_valid_pages=num_pages,
)

# Sparse attention with the selected blocks.
sparse_plan = fmha_sm100_plan(
    qo_lens, kv_lens, q.shape[1],
    num_kv_heads=k_pages.shape[1],
    page_size=page_size,
    kv_block_num=topk,
)
out, _ = fmha_sm100(
    q, k_pages, v_pages, sparse_plan,
    kv_indices=kv_indices,
    kv_block_indexes=kv_block_indexes,
)
```

For block-sparse prefill with CSR metadata, NVFP4 K/V, and the paged FP8 decode
wrapper, see the **legacy CuTe-DSL deep dive** (still used for decode, NVFP4, and
GQA &lt; 8 prefill):

- [`python/fmha_sm100/cute/README.md`](python/fmha_sm100/cute/README.md)

KV-outer internals live under [`python/fmha_sm100/kvouter/`](python/fmha_sm100/kvouter/).
Upstream docs and benchmark tables:
[fw-ai/minimax-kernels](https://github.com/fw-ai/minimax-kernels/tree/main).
Design write-up:
[Kernel optimization for MiniMax M3 on NVIDIA Blackwell](https://fireworks.ai/blog/kernel-optimization-for-minimax-m3-on-nvidia-blackwell).

## Test

```bash
# Fast smoke tests.
python -m pytest tests/smoke -q

# API and end-to-end integration tests.
python -m pytest tests/integration -q
python tests/integration/test_proxy_kv_e2e.py

# Large regression suites.
python tests/regression/test_correctness.py
python tests/regression/test_sparse_attn.py

# CuTe-DSL forward-only sparse attention.
cd python/fmha_sm100/cute
python -m pytest test_sparse_atten.py -q
```

## Benchmark

`benchmarks/bench_sparse_attention_ops.py` covers dense prefill, paged
prefill, sparse prefill, dense decode, paged decode, and sparse decode, in
`fp8` and `bf16` (`nvfp4` is sparse-prefill only). Sparse prefill on this
branch uses the KV-outer backend for GQA ≥ 8 (default TP1: `h_q=64`, `h_k=4`).

```bash
FMHA_SM100_KVOUTER_CPP=1 python benchmarks/bench_sparse_attention_ops.py --help
```

For three-way KV-outer vs FlashInfer vs MSA comparisons and published latency
tables, see
[minimax-kernels benchmarks and perf results](https://github.com/fw-ai/minimax-kernels/tree/main).

Common invocations (output is TSV):

| Goal | Command |
|---|---|
| FP8 full sweep | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections all --output_mode o -o /tmp/msa_fp8.tsv` |
| BF16 full sweep | `python benchmarks/bench_sparse_attention_ops.py --dtype bf16 --sections all --output_mode o -o /tmp/msa_bf16.tsv` |
| NVFP4 sparse prefill | `python benchmarks/bench_sparse_attention_ops.py --dtype nvfp4 --sections sparse_prefill --output_mode o -o /tmp/msa_nvfp4.tsv` |
| Quick CI smoke | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections prefill,decode,sparse_decode --seqs 8192,16384 --tp 1,4 --decode-k 8192,131072 --decode-b 32 --dry-run-ms 50 --repeat-ms 200 -o /tmp/msa_smoke.tsv` |
| Output-mode checks (dense/paged) | `--output_mode maxscore` or `--output_mode full` |

## Layout

```
python/fmha_sm100/                  Python package
  __init__.py                       Public re-exports (lazy for the CuTe-DSL stack)
  api.py                            fmha_sm100 / fmha_sm100_plan / sparse_topk_select
  jit.py                            Runtime JIT (nvcc + ninja) for the csrc stack
  sparse.py                         Lazy shim that loads the cute/ stack
  sparse_fmha_adapter.py            Bridge: fmha_sm100 API → KV-outer or legacy CuTe
  kvouter/                          Vendored Fireworks KV-outer (Python + AOT export)
  csrc/kvouter/                     KV-outer C++ op (fmha_sm100._C)
  csrc/                             Dense CUDA kernels + Jinja templates (JIT-compiled)
    include/                        Vendored FlashInfer / CUTLASS-derived / TRT-LLM headers
  cutlass/                          NVIDIA CUTLASS git submodule (include/ + tools/util/include/)
  cute/                             Legacy CuTe-DSL sparse attention (loaded via sys.path)
setup.py                            Builds fmha_sm100._C CUDA extension
tests/                              Correctness tests
  smoke/  integration/  regression/
scripts/                            Warmup + cache-management helpers
benchmarks/                         bench_sparse_attention_ops.py
```

## Stacks

- **csrc JIT** — dense FlashAttention, page KV, and `sparse_topk_select`
  indexer. Compiled at runtime from `csrc/*.cu.jinja` plus
  `csrc/include/`. Public entry: `fmha_sm100_plan` → `fmha_sm100`.
- **KV-outer** — Fireworks block-sparse prefill for GQA ≥ 8. Vendored from
  [fw-ai/minimax-kernels](https://github.com/fw-ai/minimax-kernels/tree/main).
  See the
  [Blackwell kernel optimization blog post](https://fireworks.ai/blog/kernel-optimization-for-minimax-m3-on-nvidia-blackwell)
  for background. Index build + forward + combine; C++ AOT path preferred
  (`fmha_sm100._C`). Public entry: `sparse_fmha` → `kvouter_attention`
  (via `sparse_fmha_adapter`).
- **CuTe-DSL (legacy sparse)** — CSR sparse prefill fallback (GQA &lt; 8),
  FP8 / NVFP4 / FP4 quantization, paged FP8 decode
  (`SparseDecodePagedAttentionWrapper`), FP4 block-score indexer.
  Public entry: `fmha_sm100.sparse_atten_func`,
  `fmha_sm100.sparse_decode_atten_func`, `fmha_sm100.fp4_indexer_block_scores`.
- **Bridge** — `sparse_fmha_plan` / `sparse_fmha` adapt the dense-API call
  site to KV-outer or legacy CuTe for prefill; decode and indexer paths are
  unchanged.

## Third-party licenses

`fmha_sm100` bundles, derives from, or depends on the third-party components
below. Each retains its original license; this section summarizes them.
Authoritative text is shipped with each component.

### Vendored / derived source (shipped in this repo)

| Component | License | Where |
|---|---|---|
| **NVIDIA CUTLASS** | BSD-3-Clause | Git submodule at `python/fmha_sm100/cutlass/` (provides `include/` + `tools/util/include/`), plus BSD-3-tagged headers under `python/fmha_sm100/csrc/include/`. The SM100 MMA descriptor encodings in `python/fmha_sm100/cute/src/common/mma_sm100_desc.py` mirror CUTLASS hardware descriptors. Copyright (c) 2017–2025 NVIDIA CORPORATION & AFFILIATES. |
| **FlashInfer** | Apache-2.0 | Headers and sources under `python/fmha_sm100/csrc/` and `python/fmha_sm100/csrc/include/` that carry a `Copyright (c) <year> by FlashInfer team` line (e.g. `allocator.h`, `exception.h`, `utils.cuh`, `cutlass_utils.cuh`, `fmha_cutlass_sm100.cuh`, `sparse_topk_select.cuh`, `plan.cuh`, `sm100_fmha_reduction.hpp`, `tvm_ffi_utils.h`). Project: <https://github.com/flashinfer-ai/flashinfer>. |
| **NVIDIA TensorRT-LLM + NAVER Corp (CLOVA)** | Apache-2.0 | Portions of `python/fmha_sm100/csrc/include/sparse_topk_select.cuh` — `indexerTopK` histogram-step + insertion-sort derived from `tensorrt_llm/cpp/tensorrt_llm/kernels/indexerTopK.cu`. Copyright (c) 2019–2026 NVIDIA CORPORATION; Copyright (c) 2021 NAVER Corp. The per-file header in `sparse_topk_select.cuh` includes a function-level provenance map. |
| **Fireworks AI (minimax-kernels KV-outer)** | Apache-2.0 | Vendored under `python/fmha_sm100/kvouter/` and `python/fmha_sm100/csrc/kvouter/`. Upstream: <https://github.com/fw-ai/minimax-kernels>. Copyright (c) 2026 Fireworks AI. See [NOTICE](NOTICE). |

### Runtime dependencies (installed via pip)

| Package | Upstream | License |
|---|---|---|
| `quack-kernels` | <https://github.com/Dao-AILab/quack> | Apache-2.0 |
| `flash-attn-4` | FlashAttention CuTe SM100 kernels | BSD-3-Clause (see package) |
| `nvidia-cutlass-dsl` | NVIDIA CUTLASS Python DSL | NVIDIA / BSD-3-Clause (see package) |
| `apache-tvm-ffi` | Apache TVM FFI | Apache-2.0 |
| `cuda-python` | NVIDIA | NVIDIA / see package |
| `torch` | <https://github.com/pytorch/pytorch> | BSD-3-Clause |
| `jinja2` | <https://github.com/pallets/jinja> | BSD-3-Clause |
| `ninja` | <https://github.com/ninja-build/ninja> | Apache-2.0 |
| `pybind11` | <https://github.com/pybind/pybind11> | BSD-3-Clause |

The exact license of each installed package is distributed with that package;
consult its metadata (`pip show <pkg>`) for the authoritative text.

## Citation

If MSA helps your research, please cite it. (BibTeX entry coming once the
companion paper / technical report has a stable identifier — placeholder.)
The algorithmic reference is shipped at
[`docs/MiniMaxSparseAttention.pdf`](docs/MiniMaxSparseAttention.pdf).

```bibtex
@software{msa2026,
  title  = {MiniMax Sparse Attention (MSA): FlashAttention and block-sparse
            attention kernels for NVIDIA SM100},
  author = {{MiniMax}},
  year   = {2026},
  url    = {https://github.com/MiniMax-AI/MSA}
}
```

## Contributing

Issues and PRs welcome on the
[issue tracker](https://github.com/MiniMax-AI/MSA/issues). For kernel or
runtime-contract changes, open an issue first to align on the public
surface — `fmha_sm100.api`, `fmha_sm100.sparse` and
`cute.interface` are the stable entry points; everything else
is internal and may change without notice.
