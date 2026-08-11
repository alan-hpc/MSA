#!/usr/bin/env bash
# MSA 稀疏三段流水复测（indexer + topk + sparse attn）—— B300 .5 容器 dsa_stage2_meng2 内执行。
#
#   GPU=2 bash run_node5_msa.sh            # prefill + decode 全跑
#   GPU=2 STAGE=prefill bash run_node5_msa.sh
#   GPU=2 STAGE=decode  bash run_node5_msa.sh
#
# 前置：setup_node5.sh 已跑过，且两个 bench_msa_*.py 已拷进 $MSA_DIR。
# PAM adhoc ~10min 会超时 → 这个脚本本身请用 nohup 后台跑，轮询 $LOGDIR。
set -uo pipefail

MSA_DIR=${MSA_DIR:-/sparse/msa}
VENV=${VENV:-/sparse/msa_venv}
PY="$VENV/bin/python"
GPU=${GPU:-2}
STAGE=${STAGE:-all}
LOGDIR=${LOGDIR:-$MSA_DIR/logs}
PREFILL_SEQS=${PREFILL_SEQS:-"32768 131072 262144 524288 1048576"}
DECODE_KVS=${DECODE_KVS:-32768,131072,262144,524288,1048576}

# ⚠️ 必须：容器 / 只剩 ~12G，AOT warmup 起 ~300 个并行 nvcc 会把 /tmp 撑爆，
# ptxas 报 "Output file '/tmp/tmpxft_*.cubin' could not be opened" 全挂。
export TMPDIR=${TMPDIR:-/sparse/tmpdir}
mkdir -p "$TMPDIR" "$LOGDIR"
cd "$MSA_DIR"

# ---- decode 前置：AOT 预编译（不做的话 decode 首跑 JIT 死锁：GPU 0%、sigsuspend、无报错）
aot_warmup() {
  if [ -f "$LOGDIR/.aot_done" ]; then echo "[aot] cached, skip"; return; fi
  echo "[aot] warmup ~5min ..."
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u scripts/warmup_fmha_sm100.py \
      --preset m3-infer --include-sparse-aot -j 0 > "$LOGDIR/aot_warmup.log" 2>&1
  if grep -q "Sparse-attn AOT done" "$LOGDIR/aot_warmup.log"; then
    touch "$LOGDIR/.aot_done"; echo "[aot] OK"
  else
    echo "[aot] FAILED，看 $LOGDIR/aot_warmup.log"; return 1
  fi
}

# ---- prefill：⚠️ 每个 seq 单独起进程。
# .5 上 MSA 自带的 dense FMHA 在 seq=262144 必崩 cudaErrorIllegalAddress，且毒化 CUDA ctx
# （同进程后面所有 seq 一起废）。256k / 1M 用 SKIP_DENSE=1 只测稀疏三段。
# dense 列最终以 .3 的 FA4 为准（run_node3_fa4_dense.sh），MSA 自带 dense 只作参考。
run_prefill() {
  local log="$LOGDIR/prefill.log"; : > "$log"
  for S in $PREFILL_SEQS; do
    local skip=0
    [ "$S" -ge 262144 ] && [ "$S" -ne 524288 ] && skip=1     # 256k 崩、1M dense 本来就跑不了
    echo "===== seq=$S SKIP_DENSE=$skip =====" | tee -a "$log"
    # 只滤 aot_cache 噪声和 traceback 正文；注意表头是 5 个空格开头，别用 '^ +' 一起滤掉
    CUDA_VISIBLE_DEVICES=$GPU SKIP_DENSE=$skip "$PY" -u bench_msa_full_pipeline.py "$S" 2>&1 \
      | grep -vE '^\[aot_cache\]|^  File "|^ +\^+$|^ +[a-z_]+\(|^ +return |^Traceback|^Search for|^CUDA kernel|^Compile with|^During handling' \
      | tee -a "$log"
  done
  echo "PREFILL_DONE" | tee -a "$log"
}

# ---- decode：最终表口径 = bf16 / Sq=1 / BS=32（batch 内每个 request 独立 KV，不共享）
run_decode() {
  local log="$LOGDIR/decode_bs32.log"
  CUDA_VISIBLE_DEVICES=$GPU DTYPE=bf16 IDX_DTYPE=bf16 QLEN=1 BATCHES=32 \
    "$PY" -u bench_msa_decode_pipeline.py "$DECODE_KVS" 2>&1 \
    | grep -vE '^\[aot_cache\]' | tee "$log"
}

case "$STAGE" in
  prefill) run_prefill ;;
  decode)  aot_warmup && run_decode ;;
  all)     run_prefill; aot_warmup && run_decode ;;
  *) echo "STAGE 只能是 prefill / decode / all"; exit 1 ;;
esac
echo "ALL_DONE  logs -> $LOGDIR"
