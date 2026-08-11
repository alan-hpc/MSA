#!/usr/bin/env bash
# =============================================================================
# MSA environment bootstrap -- idempotent, safe to re-run
# =============================================================================
#
# Everything the benchmarks need before they can run:
#   1. scratch dir on a big volume   (AOT warmup fills /tmp otherwise)
#   2. CUTLASS submodule             (~154 MB, the dense JIT needs it)
#   3. a venv pinned to the versions MSA actually works with
#   4. `import fmha_sm100` proven to work
#   5. AOT precompile                (decode JIT-deadlocks without it)
#
# `benchmark.sh` calls this automatically when the environment is not ready,
# so normally you never run it by hand. Run it directly to set a box up, or
# after changing Python packages.
#
# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
#   ./init.sh                 # full bootstrap
#   SKIP_AOT=1 ./init.sh      # skip the ~5 min AOT precompile (prefill-only work)
#   FORCE_VENV=1 ./init.sh    # build the pinned venv even if the system env imports
#   VENV=/path/to/venv ./init.sh
#
# -----------------------------------------------------------------------------
# Why the versions are pinned
# -----------------------------------------------------------------------------
# MSA's pyproject only writes `>=`, and the newest releases do not work:
#   * cutlass-dsl 4.6.0 dropped `cute.core.ThrMma`, which MSA's cute code uses
#     -> AttributeError at import
#   * dropping back to 4.5.2 then breaks quack, which wants `cutlass._mlir_helpers`
#     (4.6-only) -> ModuleNotFoundError
# The combination that works is the two *minimum* versions MSA declares.
# The sparse-attn kernel is also version-sensitive: measured 6-14% slower under
# 4.6.0 than under 4.4.1, so numbers from an unpinned env are not comparable.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CUTLASS_DSL_PIN="${CUTLASS_DSL_PIN:-4.4.1}"
QUACK_PIN="${QUACK_PIN:-0.2.10}"
SKIP_AOT="${SKIP_AOT:-0}"
FORCE_VENV="${FORCE_VENV:-0}"
GPU="${GPU:-0}"

say() { printf '[init] %s\n' "$*"; }
die() { printf '[init] ERROR: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. scratch dir. The AOT warmup runs ~300 parallel nvcc; on these containers /
#    has a few GB free and ptxas dies with "Output file ... could not be opened",
#    failing every kernel. Put TMPDIR on whatever big volume exists.
# -----------------------------------------------------------------------------
pick_tmpdir() {
  if [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR}" ]; then echo "$TMPDIR"; return; fi
  for cand in /sparse/tmpdir /sparse/tmp /data1/tmp; do
    base="$(dirname "$cand")"
    if [ -d "$base" ] && [ -w "$base" ]; then echo "$cand"; return; fi
  done
  echo "$REPO_ROOT/.tmp"
}
TMPDIR="$(pick_tmpdir)"
mkdir -p "$TMPDIR" || die "cannot create TMPDIR=$TMPDIR"
export TMPDIR
say "TMPDIR = $TMPDIR ($(df -h "$TMPDIR" 2>/dev/null | awk 'NR==2{print $4}') free)"

# -----------------------------------------------------------------------------
# 2. CUTLASS submodule
# -----------------------------------------------------------------------------
# `git submodule status` prefixes an uninitialised module with '-'.
if [ -f "$REPO_ROOT/.gitmodules" ]; then
  if git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | grep -q '^-'; then
    say "fetching submodules (CUTLASS ~154 MB) ..."
    git -C "$REPO_ROOT" submodule update --init --recursive \
      || die "submodule fetch failed (no network?)"
  else
    say "submodules present"
  fi
fi

# -----------------------------------------------------------------------------
# 3. interpreter. Prefer one that already imports fmha_sm100 at the right pins;
#    otherwise build a venv that inherits the system torch and pins the rest.
# -----------------------------------------------------------------------------
if [ -z "${VENV:-}" ]; then
  if [ -x /sparse/msa_venv/bin/python ] && [ "$REPO_ROOT" = "/sparse/msa" ]; then
    VENV=/sparse/msa_venv                 # the box this harness was built on
  else
    VENV="$REPO_ROOT/.msa_venv"
  fi
fi

# Does a given interpreter import MSA at the pinned versions?
env_ok() {
  local py="$1"
  [ -x "$py" ] || return 1
  "$py" - "$CUTLASS_DSL_PIN" "$QUACK_PIN" <<'PY' >/dev/null 2>&1
import sys, importlib.metadata as md
want_dsl, want_quack = sys.argv[1], sys.argv[2]
import fmha_sm100  # noqa: F401
assert md.version("nvidia-cutlass-dsl") == want_dsl, "cutlass-dsl"
assert md.version("quack-kernels") == want_quack, "quack"
PY
}

PYTHON=""
for cand in "${PYTHON:-}" "$VENV/bin/python" "$(command -v python3)"; do
  [ -n "$cand" ] || continue
  if [ "$FORCE_VENV" = "1" ] && [ "$cand" != "$VENV/bin/python" ]; then continue; fi
  if env_ok "$cand"; then PYTHON="$cand"; break; fi
done

if [ -n "$PYTHON" ]; then
  say "environment already good: $PYTHON"
else
  say "building pinned venv at $VENV ..."
  BASE_PY="$(command -v python3)" || die "no python3 on PATH"
  if [ ! -x "$VENV/bin/python" ]; then
    # --system-site-packages so the container's torch is inherited, not rebuilt
    "$BASE_PY" -m venv --system-site-packages "$VENV" || die "venv creation failed"
  fi
  say "installing MSA (editable) + pins: cutlass-dsl==$CUTLASS_DSL_PIN quack-kernels==$QUACK_PIN"
  # pip warns about flashinfer / cutlass-libs resolver conflicts here. Harmless:
  # the venv path wins at import time, which is the whole point.
  "$VENV/bin/pip" install -q -e "$REPO_ROOT" || die "pip install -e failed (no network?)"
  "$VENV/bin/pip" install -q "nvidia-cutlass-dsl==$CUTLASS_DSL_PIN" \
                             "quack-kernels==$QUACK_PIN" || die "pin install failed"
  PYTHON="$VENV/bin/python"
fi

# -----------------------------------------------------------------------------
# 4. prove it imports, and report what we ended up with
# -----------------------------------------------------------------------------
"$PYTHON" - <<'PY' || die "fmha_sm100 does not import -- see the traceback above"
import importlib.metadata as md
import torch, fmha_sm100
print(f"[init] torch {torch.__version__} / cuda {torch.version.cuda}")
for p in ("nvidia-cutlass-dsl", "quack-kernels", "fmha_sm100"):
    try:
        print(f"[init] {p} {md.version(p)}")
    except Exception:
        print(f"[init] {p} MISSING")
print("[init] IMPORT_OK", fmha_sm100.__file__)
PY

if ! env_ok "$PYTHON"; then
  cat >&2 <<MSG
[init] WARNING: fmha_sm100 imports, but not at the pinned versions
       (want cutlass-dsl $CUTLASS_DSL_PIN / quack-kernels $QUACK_PIN).
       It will probably run, but the sparse-attn kernel is version-sensitive --
       measured 6-14% slower under 4.6.0 -- so treat the numbers as
       non-comparable with the published table. FORCE_VENV=1 ./init.sh builds
       a pinned venv alongside.
MSG
fi

# -----------------------------------------------------------------------------
# 5. AOT precompile. Without it the first decode run JIT-deadlocks: GPU at 0%,
#    the process parked in sigsuspend, no error, and CUDA_LAUNCH_BLOCKING=1
#    reports nothing either.
# -----------------------------------------------------------------------------
AOT_MARKER="$REPO_ROOT/.aot_done"
if [ "$SKIP_AOT" = "1" ]; then
  say "SKIP_AOT=1, not precompiling (decode will deadlock until you do)"
elif [ -f "$AOT_MARKER" ]; then
  say "AOT already done ($AOT_MARKER)"
else
  say "AOT precompile on GPU $GPU (~5 min first time, cached after) ..."
  mkdir -p "$REPO_ROOT/results"
  if CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/warmup_fmha_sm100.py \
        --preset m3-infer --include-sparse-aot -j 0 > "$REPO_ROOT/results/aot_warmup.log" 2>&1 \
     && grep -q "Sparse-attn AOT done" "$REPO_ROOT/results/aot_warmup.log"; then
    touch "$AOT_MARKER"; say "AOT OK"
  else
    say "AOT FAILED -- see results/aot_warmup.log"
    grep -iE "error|failed" "$REPO_ROOT/results/aot_warmup.log" | head -5 >&2
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 6. leave a sourceable record so benchmark.sh picks the same interpreter
# -----------------------------------------------------------------------------
cat > "$REPO_ROOT/.msa_env" <<EOF
# written by init.sh -- sourced by benchmark.sh
PYTHON="$PYTHON"
TMPDIR="$TMPDIR"
EOF
say "wrote $REPO_ROOT/.msa_env"
say "READY -- run ./benchmark.sh"
