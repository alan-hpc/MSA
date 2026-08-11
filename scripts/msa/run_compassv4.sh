#!/usr/bin/env bash
# Compass-V4 full sweep on B300 .5 (container dsa_stage2_meng2)
#   attention: num_query_heads=32, num_kv_heads=4, head_dim=128
#   index    : num_query_heads=4,  num_kv_heads=1
set -uo pipefail
export H_Q=32 H_K=4 D=128
export IDX_HQ=4 IDX_HKV=1
export IDX_DTYPE=${IDX_DTYPE:-fp8}          # falls back to bf16 if maxscore unsupported
export MSA_DIR=/sparse/msa
export LOGDIR=/sparse/msa/logs_compassv4
export TMPDIR=/sparse/tmpdir
export PREFILL_SEQS="8192 16384 32768 65536 131072 262144 524288 1048576"
export DECODE_KVS=8192,16384,32768,65536,131072,262144,524288,1048576
cd /sparse/msa
bash run_node5_msa.sh
