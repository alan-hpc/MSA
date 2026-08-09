#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
#
# =============================================================================
# MSA sparse-attention configuration benchmark — full run
# =============================================================================
#
# Runs, in order:
#   0. environment report                (GPU, driver, torch, CUDA)
#   1. correctness gate                  (top-k config + block-size regressions)
#   2. baseline evaluation               (shipped MSA config: block 128 / top-16)
#   3. the 12-point configuration sweep  (block 32|64 x top-k 16|32 x head keep|sum|max,
#                                         force_init 128 / force_end 128)
#
# Steps 2 and 3 sweep several context lengths and emit CSV + JSON per stage
# (indexer / top-k select / CSR build / sparse attention), each with three
# views: eager latency, CUDA-graph GPU time, and host dispatch cost.
#
# The correctness gate runs first on purpose: a configuration that does not
# compute the right thing is not worth timing.  Set SKIP_TESTS=1 to bypass it
# when iterating on numbers only.
#
# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
#   ./benchmark.sh                       # full run, results under results/<stamp>/
#   SEQLENS=32768,131072 ./benchmark.sh  # custom context lengths
#   MODEL=n16 ./benchmark.sh          # benchmark a different geometry
#   CONFIGS=matrix ./benchmark.sh        # sweep only the 12 points, no baseline
#   SKIP_TESTS=1 ./benchmark.sh          # skip the correctness gate
#   QUICK=1 ./benchmark.sh               # short shapes + short timing windows
#
# Environment:
#   MODEL       target geometry (model-n32 | n16) (default model-n32)
#   GPU         CUDA device index                       (default 0)
#   SEQLENS     comma-separated KV lengths              (default 8192..1048576)
#   BATCH       requests per measurement                (default 1)
#   CONFIGS     'all' | 'matrix' | comma-separated list
#               (default baseline,cfg10,cfg12 = baseline + b32/k32/max + b64/k32/max)
#   OUT_DIR     results directory                       (default results/<UTC stamp>)
#   DRY_MS      warmup ms per stage                     (default 50)
#   REP_MS      measurement ms per stage                (default 200)
#   INDEXER     'measure' | 'skip'                      (default measure)
#   TIMING      'simple' | 'full'                       (default simple: single-op events)
#   COS         set empty to skip the cosine table      (default 1)
#   FA4_PATH    flash-attention checkout (FA4 reference)  (default ../flash-attention)
#   NO_FA4      set to 1 to skip the FA4 reference row
#   PREFILL_CHUNK    prefill query chunk in tokens; -1 (default) picks the
#                    largest the indexer's Int32 score tensor allows, 0 disables.
#                    Needed above 256K, where one unchunked score tensor exceeds
#                    both the Int32 limit and the card.
#   KVOUTER_PATH     fireworks-msa checkout; empty skips the KV-outer A/B
#   KVOUTER_PYTHON   interpreter with the branch's pinned deps (default PYTHON)
#   KVOUTER_FA4_PATH FA4 checkout for the KV-outer role, if not already importable
#   KVOUTER_SEQLENS / KVOUTER_TOPKS   A/B grid (block is fixed at 128 by the branch)
#   COS_MAX_SEQLEN  cosine only at or below this kv_len  (default 32768)
#   QK_SOURCE   'model' | 'random'                       (default model: real layer Q/K/V)
#   QK_MODEL    checkpoint for QK_SOURCE=model           (unset: falls back to random)
#   QK_LAYER    which layer to read                      (default 0)
#   DECODE      set empty to skip the decode sweep       (default 1)
#   DECODE_SEQLENS / DECODE_BATCH                        (default 8192..1048576 / 32)
#   PYTHON      interpreter                             (default python3)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
BATCH="${BATCH:-1}"
# baseline (block 128 / topk 16 / keep, no forced windows) plus the two the
# sweep is currently focused on: block 32 and 64, both at topk 32, head=max.
# CONFIGS=all restores the full 33-point grid.
CONFIGS="${CONFIGS:-baseline,cfg10,cfg12}"
INDEXER="${INDEXER:-measure}"
MODEL="${MODEL:-model-n32}"
TIMING="${TIMING:-simple}"
COS="${COS-1}"          # ${VAR-} not ${VAR:-}: empty must mean "off", not "default"
COS_MAX_SEQLEN="${COS_MAX_SEQLEN:-32768}"
QK_SOURCE="${QK_SOURCE:-model}"
QK_MODEL="${QK_MODEL:-}"
QK_LAYER="${QK_LAYER:-0}"
DECODE="${DECODE-1}"    # likewise -- see COS above
DECODE_SEQLENS="${DECODE_SEQLENS:-8192,16384,32768,65536,131072,262144,524288,1048576}"
DECODE_BATCH="${DECODE_BATCH:-32}"
FA4_PATH="${FA4_PATH:-../flash-attention}"
PREFILL_CHUNK="${PREFILL_CHUNK:--1}"
KVOUTER_PATH="${KVOUTER_PATH:-}"
KVOUTER_PYTHON="${KVOUTER_PYTHON:-}"
KVOUTER_FA4_PATH="${KVOUTER_FA4_PATH:-}"
KVOUTER_SEQLENS="${KVOUTER_SEQLENS:-32768,65536,131072,262144}"
KVOUTER_TOPKS="${KVOUTER_TOPKS:-4,8,16,32}"
NO_FA4="${NO_FA4:-}"

if [[ "${QUICK:-0}" == "1" ]]; then
    SEQLENS="${SEQLENS:-8192,32768}"
    DRY_MS="${DRY_MS:-20}"
    REP_MS="${REP_MS:-60}"
else
    # Sparse attention loses to dense below ~16K, so the grid starts at 8K --
    # far enough down to show the crossover, without spending time on lengths
    # where the answer is known and uninteresting.  The top two need query
    # chunking (see PREFILL_CHUNK).
    SEQLENS="${SEQLENS:-8192,16384,32768,65536,131072,262144,524288,1048576}"
    DRY_MS="${DRY_MS:-50}"
    REP_MS="${REP_MS:-200}"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/$STAMP}"
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/benchmark.log"
export CUDA_VISIBLE_DEVICES="$GPU"
# Deterministic JIT cache location so a cold first run is visible in the log
# rather than hidden in a stray home directory.
export MINFER_FMHA_CACHE_DIR="${MINFER_FMHA_CACHE_DIR:-$HOME/.cache/minfer/fmha_sm100}"

# Everything below is tee'd: the console shows progress, the log is the artifact.
exec > >(tee -a "$LOG") 2>&1

section() {
    echo
    echo "==============================================================================="
    echo "== $*"
    echo "==============================================================================="
}

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

# -----------------------------------------------------------------------------
section "0. Environment"
# -----------------------------------------------------------------------------
echo "timestamp     : $STAMP (UTC)"
echo "repo          : $REPO_ROOT"
echo "git           : $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
echo "out dir       : $OUT_DIR"
echo "model         : $MODEL"
echo "device index  : $GPU"
echo "seqlens       : $SEQLENS"
echo "batch         : $BATCH"
echo "configs       : $CONFIGS"
echo "timing window : dry=${DRY_MS}ms rep=${REP_MS}ms"
echo "jit cache     : $MINFER_FMHA_CACHE_DIR"
echo
command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
               --format=csv,noheader || echo "nvidia-smi unavailable"
echo
"$PYTHON" - <<'PYEOF' || fail "python/torch environment is not usable"
import sys, torch
print(f"python        : {sys.version.split()[0]}")
print(f"torch         : {torch.__version__}  (cuda {torch.version.cuda})")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
cap = torch.cuda.get_device_capability(0)
print(f"device        : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
if cap[0] < 10:
    raise SystemExit(f"MSA kernels require SM100 or newer, found sm_{cap[0]}{cap[1]}")
PYEOF

echo
MODEL="$MODEL" "$PYTHON" - <<'PYEOF'
import os, sys
sys.path.insert(0, "python")
from fmha_sm100.msa_config import BASELINE_CONFIG, CONFIG_MATRIX, model_by_name
m = model_by_name(os.environ["MODEL"])
print(f"target model  : {m.name}  Hq={m.num_qo_heads} Hkv={m.num_kv_heads} "
      f"(GQA {m.qhead_per_kv}x) head_dim={m.head_dim} ctx={m.max_position_embeddings}")
print(f"                {m.num_hidden_layers} layers, {m.full_attention_layers} full-attention"
      f" + {m.mtp_layers} MTP = {m.attention_layers_per_forward} attention layers/forward")
if m.notes:
    print(f"                {m.notes}")
for gap in m.compatibility():
    print(f"                NOTE {gap}")
print()
print("configuration matrix:")
print(f"  {'name':<9} {'label':<64} {'sink':>5} {'local':>6} {'free':>5} {'budget':>7}")
for c in (BASELINE_CONFIG, *CONFIG_MATRIX):
    print(f"  {c.resolved_name():<9} {c.label:<64} {c.force_init_blocks:>5} "
          f"{c.force_end_blocks:>6} {c.free_blocks:>5} {c.selected_tokens:>7}")
PYEOF

# FA4 is the strongest dense attention available here, so it is what the
# speedup tables divide by.  It ships as CuTe-DSL inside the flash-attention
# repo and needs no build, only a checkout.
if [[ -z "$NO_FA4" ]]; then
    if [[ -d "$FA4_PATH/flash_attn/cute" ]]; then
        echo "FA4 reference : $FA4_PATH"
    else
        echo "FA4 reference : not found at $FA4_PATH, cloning..."
        git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$FA4_PATH" \
            >/dev/null 2>&1 \
            && echo "                cloned" \
            || echo "                clone failed; tables will fall back to the repo's dense kernel"
    fi
else
    echo "FA4 reference : disabled (NO_FA4=$NO_FA4)"
fi

# Real Q/K only changes one table (the cosine one), but the failure to find a
# checkpoint used to surface as a FileNotFoundError three sections later, after
# the correctness gate had already run.  Decide it here, before anything is
# spent, and keep going on random rather than refusing to benchmark at all.
if [[ "$QK_SOURCE" == "model" ]]; then
    if [[ -z "$QK_MODEL" ]]; then
        echo "Q/K source    : random (QK_MODEL not set)"
        echo "                Set QK_MODEL=/path/to/checkpoint for real post-RoPE Q/K."
        echo "                Only the cosine table is affected: on random Q/K the"
        echo "                attention is near-uniform, so those numbers read as a"
        echo "                pessimistic floor.  All latency tables are unaffected."
        QK_SOURCE=random
    elif [[ ! -f "$QK_MODEL/config.json" ]]; then
        echo "Q/K source    : random ($QK_MODEL has no config.json)"
        echo "                QK_MODEL must be a checkpoint directory containing"
        echo "                config.json, the safetensors shards, and a tokenizer."
        QK_SOURCE=random
    else
        echo "Q/K source    : layer $QK_LAYER of $QK_MODEL"
    fi
else
    echo "Q/K source    : random (QK_SOURCE=$QK_SOURCE)"
fi

# -----------------------------------------------------------------------------
section "1. Correctness gate"
# -----------------------------------------------------------------------------
if [[ "${SKIP_TESTS:-0}" == "1" ]]; then
    echo "SKIPPED (SKIP_TESTS=1)"
else
    echo "--- legacy forced-block smoke (backwards compatibility) ---"
    "$PYTHON" tests/smoke/test_sparse_topk_forced.py \
        || fail "legacy sparse_topk_select smoke test failed"
    echo
    echo "--- configurable top-k selection (topk / head_mode / force windows / layout) ---"
    "$PYTHON" tests/regression/test_msa_topk_config.py \
        || fail "MSA top-k configuration regression failed"
    echo
    echo "--- block-sparse attention at blk_kv 32 / 64 / 128 ---"
    "$PYTHON" tests/regression/test_msa_block_size.py \
        || fail "MSA block-size regression failed"
fi

# -----------------------------------------------------------------------------
section "2. Baseline evaluation — shipped MSA configuration"
# -----------------------------------------------------------------------------
echo "block=128 topk=16 no forced windows head=keep, versus dense causal FMHA."
echo
"$PYTHON" benchmarks/bench_msa_configs.py \
    --model "$MODEL" \
    --configs baseline \
    --seqlens "$SEQLENS" \
    --batch "$BATCH" \
    --indexer-mode "$INDEXER" \
    --timing "$TIMING" \
    --qk-source "$QK_SOURCE" --qk-model "$QK_MODEL" --qk-layer "$QK_LAYER" \
    --fa4-path "$FA4_PATH" ${NO_FA4:+--no-fa4} \
    --dry-ms "$DRY_MS" --rep-ms "$REP_MS" \
    --gpu 0 \
    --csv "$OUT_DIR/baseline.csv" \
    --json "$OUT_DIR/baseline.json" \
    || fail "baseline benchmark failed"

# -----------------------------------------------------------------------------
section "3. Configuration sweep"
# -----------------------------------------------------------------------------
"$PYTHON" benchmarks/bench_msa_configs.py \
    --model "$MODEL" \
    --configs "$CONFIGS" \
    --seqlens "$SEQLENS" \
    --batch "$BATCH" \
    --indexer-mode "$INDEXER" \
    --timing "$TIMING" \
    --qk-source "$QK_SOURCE" --qk-model "$QK_MODEL" --qk-layer "$QK_LAYER" \
    --fa4-path "$FA4_PATH" ${NO_FA4:+--no-fa4} \
    ${COS:+--cos} --cos-max-seqlen "$COS_MAX_SEQLEN" \
    --chunk-q "$PREFILL_CHUNK" \
    --dry-ms "$DRY_MS" --rep-ms "$REP_MS" \
    --gpu 0 \
    --csv "$OUT_DIR/sweep.csv" \
    --json "$OUT_DIR/sweep.json" \
    || fail "configuration sweep failed"

# -----------------------------------------------------------------------------
section "3b. Decode sweep"
# -----------------------------------------------------------------------------
if [[ -z "$DECODE" ]]; then
    echo "SKIPPED (DECODE= )"
else
    echo "q_len=1 per request, batch=$DECODE_BATCH, kv_len=$DECODE_SEQLENS."
    echo "Long-context serving spends its time here, and the indexer's Int32 score-tensor"
    echo "cap stops binding because total_q collapses from the context length to the batch."
    echo
    "$PYTHON" benchmarks/bench_msa_configs.py \
        --model "$MODEL" \
        --mode decode \
        --batch "$DECODE_BATCH" \
        --configs "$CONFIGS" \
        --seqlens "$DECODE_SEQLENS" \
        --indexer-mode "$INDEXER" \
        --timing "$TIMING" \
        --qk-source "$QK_SOURCE" --qk-model "$QK_MODEL" --qk-layer "$QK_LAYER" \
        --fa4-path "$FA4_PATH" ${NO_FA4:+--no-fa4} \
        --dry-ms "$DRY_MS" --rep-ms "$REP_MS" \
        --gpu 0 \
        --csv "$OUT_DIR/decode.csv" \
        --json "$OUT_DIR/decode.json" \
        || echo "decode sweep reported failures (see the table for which shapes)"
fi

# -----------------------------------------------------------------------------
section "3c. KV-outer backend A/B (fireworks-msa branch)"
# -----------------------------------------------------------------------------
if [[ -z "$KVOUTER_PATH" ]]; then
    echo "SKIPPED (KVOUTER_PATH= ) — set it to a fireworks-msa checkout to run this."
    echo "The branch replaces the selection-consuming half of the pipeline (CSR build +"
    echo "Q-outer forward) with a KV-stationary loop plus an LSE merge, which is the"
    echo "structural answer to the partial-O traffic reported in the optimisation report."
else
    echo "Same selection, same Q/K/V, same shapes; only stages 3-4 differ."
    echo "checkout : $KVOUTER_PATH"
    echo "python   : ${KVOUTER_PYTHON:-$PYTHON} (the branch pins cutlass-dsl 4.5.x / flash-attn-4 b15)"
    echo "NOTE     : KV-outer asserts block_size==128 and head_dim==128, so the A/B is"
    echo "           only defined at block=128 — which is where the sweep's recommended"
    echo "           configuration already sits."
    echo
    rm -rf "$OUT_DIR/kvab" && mkdir -p "$OUT_DIR/kvab"
    "$PYTHON" benchmarks/bench_kvouter.py --role qouter \
        --exchange "$OUT_DIR/kvab" --model "$MODEL" \
        --seqlens "$KVOUTER_SEQLENS" --topks "$KVOUTER_TOPKS" \
        --dry-ms "$DRY_MS" --rep-ms "$REP_MS" \
        || echo "q-outer role reported failures"
    "${KVOUTER_PYTHON:-$PYTHON}" benchmarks/bench_kvouter.py --role kvouter \
        --exchange "$OUT_DIR/kvab" --model "$MODEL" \
        --seqlens "$KVOUTER_SEQLENS" --topks "$KVOUTER_TOPKS" \
        --kvouter-path "$KVOUTER_PATH" ${KVOUTER_FA4_PATH:+--fa4-path "$KVOUTER_FA4_PATH"} \
        --dry-ms "$DRY_MS" --rep-ms "$REP_MS" \
        || echo "kv-outer role reported failures"
    "$PYTHON" benchmarks/summarise_kvouter.py "$OUT_DIR/kvab" || true
fi

# -----------------------------------------------------------------------------
section "4. Summary — comparison tables"
# -----------------------------------------------------------------------------
QK_SOURCE="$QK_SOURCE" "$PYTHON" - "$OUT_DIR/sweep.csv" <<'PYSUM'
import csv, os, sys
from collections import OrderedDict

QK_SOURCE = os.environ.get("QK_SOURCE", "random")

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("no rows")
    raise SystemExit(0)


def f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


seqlens = sorted({int(r["seqlen_k"]) for r in rows})
names = list(OrderedDict.fromkeys(r["name"] for r in rows))
if "dense" in names:
    names.remove("dense")
    names.insert(0, "dense")

by_key = {(r["name"], int(r["seqlen_k"])): r for r in rows}
first = {}
for r in rows:
    first.setdefault(r["name"], r)

NAMEW = max(12, max(len(n) for n in names) + 1)


def label(name):
    if name == "dense":
        return ("dense (msa)", "-", "-", "-", "-")
    if name == "dense-fa4":
        return ("dense (FA4)", "-", "-", "-", "-")
    m = first[name]
    return (name, m.get("block_size", "-"), m.get("topk", "-"),
            m.get("head_mode", "-"), m.get("selected_tokens", "-"))


def emit(title, note, value_fn):
    print()
    print(title)
    head = (f"{'config':<{NAMEW}}{'blk':>5}{'topk':>5}{'head':>6}{'budget':>8}  "
            + "".join(f"{s // 1024:>8}K" for s in seqlens))
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for name in names:
        n, blk, k, hm, bud = label(name)
        cells = [value_fn(by_key.get((name, s)), s) for s in seqlens]
        print(f"{n:<{NAMEW}}{blk:>5}{k:>5}{hm:>6}{bud:>8}  " + "".join(cells))
    if note:
        print(note)


def attn_ms(r):
    if r is None or r.get("errors"):
        return None
    return f(r, "attn_ms_only") or f(r, "attn_ms")


emit("Table 1a  single-operator latency (ms) — sparse attention op vs dense gqa op",
     None,
     lambda r, s: (f"{attn_ms(r):>9.3f}" if attn_ms(r) else f"{'n/a':>9}"))


has_fa4 = any(r["name"] == "dense-fa4" for r in rows)
REF = "dense-fa4" if has_fa4 else "dense"
REF_LABEL = ("FlashAttention-4" if has_fa4 else "the repo's dense FMHA")


def speed(r, s):
    a, b = attn_ms(r), attn_ms(by_key.get((REF, s)))
    return f"{b / a:>8.2f}x" if (a and b) else f"{'n/a':>9}"


emit(f"Table 1b  single-operator speedup vs {REF_LABEL} (>1 = sparse faster)",
     "note: attention operator only; the indexer / top-k / CSR selection stages are excluded.\n"
     + ("      The denominator is FA4, the fastest dense attention available here -- comparing\n"
        "      only against our own dense kernel would leave open whether the speedup comes\n"
        "      from sparsity or from a weak baseline." if has_fa4 else
        "      FlashAttention-4 was unavailable, so the denominator is the repo's own dense\n"
        "      kernel. Treat the ratios as an upper bound."),
     speed)

# The dense rows carry cos_mean = 1.0 by construction (they are the
# reference), so asking whether *any* row has one is always true and the
# table prints empty when cosine was not measured.  Ask the sparse rows.
if any(f(r, "cos_mean") is not None for r in rows if not r["name"].startswith("dense")):
    emit("Table 2  output cosine similarity vs dense gqa (per (token, head) vector, averaged)",
         "note: selections come from exact block scores over these very Q/K; 1.0000 == identical to dense.\n"
         + ("      Q/K are a real layer's post-RoPE activations, so these are the cosines a\n"
            "      deployed model would see." if QK_SOURCE == "model" else
            "      Q/K here are random, so the attention is near-uniform and top-k can only capture\n"
            "      roughly its token-budget share of the mass -- read these as a pessimistic floor.")
         + "\n      For selection quality on a trained checkpoint see benchmarks/eval_msa_selection_quality.py.",
         lambda r, s: (f"{f(r, 'cos_mean'):>9.4f}"
                       if (r is not None and not r.get("errors")
                           and f(r, "cos_mean") is not None)
                       else f"{'n/a':>9}"))
else:
    print()
    print("Table 2  output cosine similarity: not measured (enable with COS=1 / --cos)")
PYSUM

if [[ -n "$DECODE" && -f "$OUT_DIR/decode.csv" ]]; then
"$PYTHON" - "$OUT_DIR/decode.csv" <<'PYDEC'
import csv, json, sys
from collections import OrderedDict

rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    raise SystemExit(0)


def f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


seqlens = sorted({int(r["seqlen_k"]) for r in rows})
names = list(OrderedDict.fromkeys(r["name"] for r in rows))
for special in ("dense-fa4", "dense"):
    if special in names:
        names.remove(special)
        names.insert(0, special)
by_key = {(r["name"], int(r["seqlen_k"])): r for r in rows}
first = {}
for r in rows:
    first.setdefault(r["name"], r)

NAMEW = max(12, max(len(n) for n in names) + 1)
head = (f"{'config':<{NAMEW}}{'blk':>5}{'topk':>5}{'head':>6}{'budget':>8}  "
        + "".join(f"{s // 1024:>10}K" for s in seqlens))
print()
print("Table 3  decode (q_len=1) pipeline latency, ms — value / speedup vs dense gqa")
print("=" * len(head))
print(head)
print("-" * len(head))
for name in names:
    m = first[name]
    lab = ("dense gqa", "-", "-", "-", "-") if name == "dense" else (
        name, m.get("block_size", "-"), m.get("topk", "-"),
        m.get("head_mode", "-"), m.get("selected_tokens", "-"))
    cells = []
    for s in seqlens:
        r = by_key.get((name, s))
        if r is None:
            cells.append(f"{'--':>11}")
            continue
        if (r.get("errors") or "").strip():
            stage = next(iter(json.loads(r["errors"])), "?")
            cells.append(f"{'X:' + stage:>11}")
            continue
        v, d = f(r, "pipeline_gpu_ms"), f(by_key.get(("dense", s)), "pipeline_gpu_ms")
        cells.append(f"{v:>7.3f}/{d / v:>3.1f}x" if (v and d) else f"{v:>11.3f}")
    print(f"{lab[0]:<{NAMEW}}{lab[1]:>5}{lab[2]:>5}{lab[3]:>6}{lab[4]:>8}  " + "".join(cells))
print("note: whole pipeline (indexer + top-k + CSR + attention); X:<stage> means that shape")
print("      cannot run -- see the optimisation report for which limit each one hits.")
PYDEC
fi

# -----------------------------------------------------------------------------
section "5. Render log.html"
# -----------------------------------------------------------------------------
if [[ -f "$REPO_ROOT/scripts/render_log_results.py" ]]; then
    "$PYTHON" "$REPO_ROOT/scripts/render_log_results.py" "$OUT_DIR" \
        --grid "$OUT_DIR/sweep.csv" \
        ${DECODE:+--decode "$OUT_DIR/decode.csv"} \
        || echo "log.html rendering failed (results are still in $OUT_DIR)"
else
    echo "renderer not found; skipping"
fi

echo
echo "==============================================================================="
echo "Done. Artifacts:"
ls -la "$OUT_DIR"
echo "==============================================================================="
