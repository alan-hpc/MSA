#!/usr/bin/env bash
# =============================================================================
# MSA three-stage pipeline benchmark + profiler driver
# =============================================================================
#
# Times, or profiles, the sparse pipeline:  indexer -> topk select -> sparse attn
# against the dense baseline, at a chosen attention / index geometry.
#
# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
#   ./init.sh                                       # one-time env bootstrap
#   ./benchmark.sh                                  # timing sweep, prefill + decode
#   MODE=prefill ./benchmark.sh                     # prefill only
#   MODE=decode  ./benchmark.sh                     # decode only
#   MODEL=m3 ./benchmark.sh                         # MiniMax-M3 geometry instead
#
# benchmark.sh runs ./init.sh itself when the environment is not ready, so on a
# fresh checkout ./benchmark.sh alone is enough.
#
#   PROFILE=nsys SEQLENS=131072 ./benchmark.sh      # timeline + per-stage kernel summary
#   PROFILE=ncu  SEQLENS=131072 ./benchmark.sh      # per-kernel counters
#   PROFILE=ncu  NCU_STAGE=idx ./benchmark.sh       # counters for one stage only
#
# Profiling implies short timing windows (DRY_MS/REP_MS) and a single sequence
# point -- a full sweep under a profiler takes hours and tells you nothing extra.
# Stages are wrapped in NVTX ranges (`msa_idx`, `msa_topk`, `msa_attn`,
# `msa_dense`) so both profilers can attribute kernels to a stage.
#
# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
#   MODEL       compassv4 (default) | m3        geometry preset
#   GPU         CUDA device index               (default 0)
#   MODE        all (default) | prefill | decode | fa4
#               fa4 runs the FA4 dense baseline instead of the MSA pipeline.
#               ONLY valid on node .3 -- FA4 mis-dispatches on .5 (19.1ms vs
#               5.8ms for the same 32K shape), so never publish .5 FA4 numbers.
#   FA4_PYTHON  interpreter with the FA4 cute checkout (auto-detected)
#   FA4_WARMUP / FA4_REP   FA4 iteration counts (default 10 / 30 prefill).
#                          1M costs ~6s per call, so 40 calls is ~4 min a point.
#   SEQLENS     prefill seq lengths, comma/space separated
#                                               (default 8192..1048576)
#   DECODE_KVS  decode kv lengths               (default 8192..1048576)
#   BATCH       decode batch size               (default 32)
#   QLEN        decode query length             (default 1)
#   IDX_DTYPE   indexer precision: fp8 (default) | bf16
#               fp8 falls back to bf16 automatically if maxscore is unsupported
#   PROFILE     '' (default, just time it) | nsys | ncu
#   NCU_STAGE   idx | topk | attn | dense | all  (default all)
#   NCU_SET     ncu metric set: basic (default) | detailed | full
#   NCU_LAUNCHES  kernels to capture            (default 20)
#   DRY_MS/REP_MS  timing windows in ms; 0 keeps per-stage defaults
#                  (profiling defaults to 5 / 20)
#   FORCE_BEGIN / FORCE_END   blocks always kept in the top-k budget: sink blocks
#               at the head, local window nearest the query. Both default 1.
#               They take slots *inside* topk, so they change which blocks are
#               attended, not how many -- measured latency-neutral at 128K.
#               Set 0/0 to reproduce the pure top-k-by-score published table.
#   DENSE_CHUNK auto (default) | off | N   query chunking for the dense baseline.
#               auto chunks only at h_q*seq == 2^24, where one-shot dense crashes.
#   SKIP_DENSE_FROM  manual guard: skip dense at/above this seq. Normally unset.
#   OUT_DIR     results directory               (default results/<UTC stamp>)
#   PYTHON      interpreter (default: whatever ./init.sh set up)
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- geometry preset --------------------------------------------------------
MODEL="${MODEL:-compassv4}"
case "$MODEL" in
  compassv4) H_Q=32; H_K=4; D=128; IDX_HQ=4; IDX_HKV=1 ;;
  m3)        H_Q=64; H_K=4; D=128; IDX_HQ=4; IDX_HKV=1 ;;
  *) echo "unknown MODEL: $MODEL (expected compassv4 | m3)" >&2; exit 1 ;;
esac
# explicit overrides still win
H_Q="${H_Q_OVERRIDE:-$H_Q}"; H_K="${H_K_OVERRIDE:-$H_K}"; D="${D_OVERRIDE:-$D}"
IDX_HQ="${IDX_HQ_OVERRIDE:-$IDX_HQ}"; IDX_HKV="${IDX_HKV_OVERRIDE:-$IDX_HKV}"

GPU="${GPU:-0}"
MODE="${MODE:-all}"
BATCH="${BATCH:-32}"
QLEN="${QLEN:-1}"
IDX_DTYPE="${IDX_DTYPE:-fp8}"
PROFILE="${PROFILE:-}"
NCU_STAGE="${NCU_STAGE:-all}"
NCU_SET="${NCU_SET:-basic}"
NCU_LAUNCHES="${NCU_LAUNCHES:-20}"
SKIP_DENSE_FROM="${SKIP_DENSE_FROM:-}"     # extra escape hatch; normally unset
SEQLENS="${SEQLENS:-8192 16384 32768 65536 131072 262144 524288 1048576}"
DECODE_KVS="${DECODE_KVS:-8192,16384,32768,65536,131072,262144,524288,1048576}"
SEQLENS="${SEQLENS//,/ }"

# ---- environment ------------------------------------------------------------
# init.sh owns all of it: scratch dir, CUTLASS submodule, the pinned venv, and
# the AOT precompile. Run it when the environment is not ready so that a bare
# ./benchmark.sh works on a fresh checkout. AOT is deferred here -- only decode
# needs it, and it costs ~5 min -- so aot_warmup() below asks for it explicitly.
# MODE=fa4 does not touch the MSA stack at all -- it runs in the FA4 venv on a
# different box, where fmha_sm100 is neither present nor wanted. Bootstrapping
# the MSA env there would build a venv nobody asked for.
[ -f "$REPO_ROOT/.msa_env" ] && . "$REPO_ROOT/.msa_env"
if [ "$MODE" != "fa4" ]; then
  if [ -z "${PYTHON:-}" ] || ! "$PYTHON" -c "import fmha_sm100" >/dev/null 2>&1; then
    echo "[env] not ready -- running ./init.sh (SKIP_AOT=1)"
    SKIP_AOT=1 GPU="$GPU" bash "$REPO_ROOT/init.sh" || {
      echo "[env] init.sh failed; fix the environment and retry" >&2; exit 1; }
    [ -f "$REPO_ROOT/.msa_env" ] && . "$REPO_ROOT/.msa_env"
  fi
fi
export TMPDIR="${TMPDIR:-/tmp}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/$STAMP}"
mkdir -p "$OUT_DIR"

# Profiling: short windows, and one sequence point unless told otherwise.
if [ -n "$PROFILE" ]; then
  DRY_MS="${DRY_MS:-5}"; REP_MS="${REP_MS:-20}"
  if [ "$(echo $SEQLENS | wc -w)" -gt 1 ]; then
    SEQLENS="$(echo $SEQLENS | awk '{print $(NF>4?5:NF)}')"   # default to 131072
    echo "[profile] narrowing to a single prefill point: $SEQLENS"
  fi
  if [[ "$DECODE_KVS" == *,* ]]; then
    DECODE_KVS="${DECODE_KVS##*,}"
    echo "[profile] narrowing to a single decode point: $DECODE_KVS"
  fi
else
  DRY_MS="${DRY_MS:-0}"; REP_MS="${REP_MS:-0}"
fi

DENSE_CHUNK="${DENSE_CHUNK:-auto}"
FORCE_BEGIN="${FORCE_BEGIN:-1}"; FORCE_END="${FORCE_END:-1}"
export H_Q H_K D IDX_HQ IDX_HKV IDX_DTYPE DRY_MS REP_MS DENSE_CHUNK
export FORCE_BEGIN FORCE_END

echo "=============================================================="
echo " MSA pipeline benchmark"
echo "   model      : $MODEL   attn h_q=$H_Q/h_kv=$H_K d=$D"
echo "   index      : h_q=$IDX_HQ/h_kv=$IDX_HKV dtype=$IDX_DTYPE"
echo "   topk       : 16  force_begin=$FORCE_BEGIN force_end=$FORCE_END"
echo "   mode       : $MODE    gpu=$GPU"
echo "   profile    : ${PROFILE:-off}"
echo "   out        : $OUT_DIR"
echo "=============================================================="
{ nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  [ -n "${PYTHON:-}" ] && "$PYTHON" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
} > "$OUT_DIR/env.txt" 2>&1
sed 's/^/  /' "$OUT_DIR/env.txt"

# -----------------------------------------------------------------------------
# ncu needs GPU performance counters, which the driver gates on a capability the
# container usually does not have. Say so up front rather than let the user read
# a wall of ERR_NVGPUCTRPERM and an empty report.
# -----------------------------------------------------------------------------
ncu_preflight() {
  local capeff cap
  capeff="$(awk '/^CapEff:/{print $2}' /proc/self/status 2>/dev/null)"
  [ -z "$capeff" ] && return 0
  cap=$((16#$capeff))
  if (( (cap >> 21 & 1) == 0 && (cap >> 38 & 1) == 0 )); then   # SYS_ADMIN / PERFMON
    cat >&2 <<'MSG'
[ncu] WARNING: this container has neither CAP_SYS_ADMIN nor CAP_PERFMON, so the
      driver will refuse counter collection (ERR_NVGPUCTRPERM) and ncu will
      profile zero kernels. In order of preference:
        * PROFILE=nsys -- tracing needs no counter permission, and already gives
          per-stage kernel time and launch counts
        * recreate the container with --cap-add SYS_ADMIN
        * host-side: NVreg_RestrictProfilingToAdminUsers=0
MSG
  fi
}

# -----------------------------------------------------------------------------
# profiler wrapper: echoes the argv prefix for a given run tag
# -----------------------------------------------------------------------------
profile_prefix() {
  local tag="$1"
  case "$PROFILE" in
    "")   echo "" ;;
    nsys) echo "nsys profile --trace=cuda,nvtx --cuda-memory-usage=false \
--force-overwrite=true -o $OUT_DIR/$tag" ;;
    ncu)  local nvtx_filter=""
          # ncu reads '/' in an NVTX name as range *nesting*, so the stage
          # ranges are named msa_idx / msa_topk / msa_attn / msa_dense. The
          # TRAILING slash is what marks it a push/pop range -- without it ncu
          # looks for a start/end range and silently profiles nothing.
          [ "$NCU_STAGE" != "all" ] && nvtx_filter="--nvtx --nvtx-include msa_$NCU_STAGE/"
          echo "ncu --target-processes all $nvtx_filter --set $NCU_SET \
--launch-count $NCU_LAUNCHES --force-overwrite -o $OUT_DIR/$tag" ;;
    *) echo "unknown PROFILE: $PROFILE (expected nsys | ncu)" >&2; exit 1 ;;
  esac
}

# nsys writes a .nsys-rep; turn it into a per-stage kernel table next to it.
# `:nvtx-name` prefixes each kernel with the NVTX range it ran under, so rows
# read `msa_idx/...`, `msa_topk/...`; `:base` keeps CUTLASS names readable.
# Note the sparse-attn kernels carry their own NVTX ranges from inside MSA
# (`Fwd_SparseAttn_Sm100_*`, `K2_Combine`), which take precedence over ours.
# The second table (nvtx_pushpop_sum) is WALL time of each stage block --
# allocation and warmup included -- so it is not a latency breakdown. Read
# kernel cost from the first table; read latency from the benchmark's own row.
nsys_report() {
  local tag="$1"
  local rep="$OUT_DIR/$tag.nsys-rep"
  [ -f "$rep" ] || return 0
  echo "[nsys] summarising $tag ..."
  nsys stats --report cuda_gpu_kern_sum:nvtx-name:base --format csv \
       --force-export=true --output "$OUT_DIR/$tag.kern" "$rep" > /dev/null 2>&1
  nsys stats --report nvtx_pushpop_sum --format csv \
       --output "$OUT_DIR/$tag.nvtx" "$rep" > /dev/null 2>&1
  for f in "$OUT_DIR/$tag".kern*.csv "$OUT_DIR/$tag".nvtx*.csv; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f") (top 15) ---"
    cut -c1-160 "$f" | head -16
  done
}

# -----------------------------------------------------------------------------
# The shipped dense FMHA raises cudaErrorIllegalAddress at exactly one point per
# geometry, and poisons the CUDA context on the way out (the error surfaces in
# the *next* call, and the handler's own empty_cache() re-raises, killing the
# process before it prints a row).
#
# It is not a size threshold. Measured on B300:
#     h_q=64: 256K crashes, 512K fine        64 * 262144 = 2^24
#     h_q=32: 256K fine, 512K crashes, 1M fine   32 * 524288 = 2^24
# Both failures sit on h_q * seq == 2^24 exactly, and both geometries are fine
# on either side of it -- so skipping "everything past N" would throw away the
# 1M row, which measures fine. Skip the one bad point instead.
# -----------------------------------------------------------------------------
# No longer skipped: bench_msa_full_pipeline.py chunks the dense query dimension
# at exactly this point (DENSE_CHUNK=auto), which keeps h_q*chunk at 2^23 and
# measures right through the crash -- 512K @ h_q=32 comes back as 1780.951 ms,
# 3.995x the 256K value, i.e. the N^2 a causal dense should follow. Chunked and
# one-shot agree to 0.4% where both run. SKIP_DENSE_FROM stays as a manual out.
dense_is_broken() {
  local s="$1"
  [ -n "$SKIP_DENSE_FROM" ] && [ "$s" -ge "$SKIP_DENSE_FROM" ] && return 0
  return 1
}

# -----------------------------------------------------------------------------
# prefill: one process per sequence point.
# Each point is isolated so that the dense crash described above cannot take the
# rest of the sweep with it; dense_is_broken() skips the one bad point.
# -----------------------------------------------------------------------------
run_prefill() {
  local log="$OUT_DIR/prefill.log"; : > "$log"
  for S in $SEQLENS; do
    local skip=0
    dense_is_broken "$S" && skip=1
    echo "===== seq=$S SKIP_DENSE=$skip =====" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$GPU SKIP_DENSE=$skip \
      $(profile_prefix "prefill_$S") \
      "$PYTHON" -u bench_msa_full_pipeline.py "$S" 2>&1 \
      | grep -vE '^\[aot_cache\]|^  File "|^ +\^+$|^ +[a-z_]+\(|^ +return |^Traceback|^Search for|^CUDA kernel|^Compile with|^During handling' \
      | tee -a "$log"
    [ "$PROFILE" = "nsys" ] && nsys_report "prefill_$S" | tee -a "$log"
  done
  echo "PREFILL_DONE" | tee -a "$log"
}

# -----------------------------------------------------------------------------
# decode
# -----------------------------------------------------------------------------
run_decode() {
  local log="$OUT_DIR/decode_bs${BATCH}.log"
  echo "===== decode kv=$DECODE_KVS batch=$BATCH =====" | tee "$log"
  CUDA_VISIBLE_DEVICES=$GPU DTYPE=bf16 IDX_DTYPE="$IDX_DTYPE" QLEN=$QLEN BATCHES=$BATCH \
    $(profile_prefix "decode_bs${BATCH}") \
    "$PYTHON" -u bench_msa_decode_pipeline.py "$DECODE_KVS" 2>&1 \
    | grep -vE '^\[aot_cache\]' | tee -a "$log"
  [ "$PROFILE" = "nsys" ] && nsys_report "decode_bs${BATCH}" | tee -a "$log"
}

# -----------------------------------------------------------------------------
# FA4 dense baseline. Lives in its own interpreter: FA4's cute sources need
# cutlass-dsl 4.6.x, which is exactly the version the MSA venv must NOT have, so
# the two cannot share an environment. FA2 owns the `flash_attn` namespace, so
# FA4 is loaded by path -- bench_fa4_prefill.py handles that and the fmax
# monkeypatch. Note H_KV/HD, not H_K/D: the FA4 script uses the other spelling.
# -----------------------------------------------------------------------------
find_fa4_python() {
  [ -n "${FA4_PYTHON:-}" ] && { echo "$FA4_PYTHON"; return; }
  for c in /workspace/sparse_atten_meng/fa4_venv/bin/python \
           "$REPO_ROOT/.fa4_venv/bin/python"; do
    [ -x "$c" ] && { echo "$c"; return; }
  done
  echo ""
}

run_fa4() {
  local py; py="$(find_fa4_python)"
  if [ -z "$py" ]; then
    cat >&2 <<'MSG'
[fa4] no FA4 interpreter found. FA4 needs cutlass-dsl 4.6.x and a flash-attention
      cute checkout, which cannot share the MSA venv (MSA is pinned to 4.4.1).
      Point FA4_PYTHON at one, e.g. on node .3:
        FA4_PYTHON=/workspace/sparse_atten_meng/fa4_venv/bin/python MODE=fa4 ./benchmark.sh
MSG
    return 1
  fi
  local log="$OUT_DIR/fa4_prefill.log"
  echo "[fa4] $py" | tee "$log"
  local csv; csv="$(echo $SEQLENS | tr ' ' ',')"
  CUDA_VISIBLE_DEVICES=$GPU H_Q=$H_Q H_KV=$H_K HD=$D PYTHONWARNINGS=ignore \
    $(profile_prefix "fa4_prefill") \
    "$py" -u bench_fa4_prefill.py "$csv" 2>&1 | tee -a "$log"
  [ "$PROFILE" = "nsys" ] && nsys_report "fa4_prefill" | tee -a "$log"

  if [ "${FA4_DECODE:-1}" = "1" ]; then
    local dlog="$OUT_DIR/fa4_decode_bs${BATCH}.log"
    # the decode script spells them B / SQ, not BATCH / QLEN
    CUDA_VISIBLE_DEVICES=$GPU H_Q=$H_Q H_KV=$H_K HD=$D B=$BATCH SQ=$QLEN \
      PYTHONWARNINGS=ignore "$py" -u bench_fa4_decode.py "$DECODE_KVS" 2>&1 | tee "$dlog"
  fi
}

# ---- decode needs the AOT kernels, or its first JIT compile deadlocks --------
aot_warmup() {
  [ -f "$REPO_ROOT/.aot_done" ] && { echo "[aot] cached, skip"; return 0; }
  GPU="$GPU" bash "$REPO_ROOT/init.sh"      # init.sh owns the precompile
}

[ "$PROFILE" = "ncu" ] && ncu_preflight

case "$MODE" in
  prefill) run_prefill ;;
  decode)  aot_warmup && run_decode ;;
  fa4)     run_fa4 ;;
  all)     run_prefill; aot_warmup && run_decode ;;
  *) echo "MODE must be all | prefill | decode | fa4" >&2; exit 1 ;;
esac

echo "ALL_DONE  results -> $OUT_DIR"
