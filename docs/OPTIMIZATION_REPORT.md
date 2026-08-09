<!-- SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax -->
<!-- SPDX-License-Identifier: MIT -->

# MSA 当前版本可优化点报告

**基线**：MSA `main` @ `80434d7`
**硬件**：NVIDIA B300 SXM6 (sm_103, 275 GB)，torch 2.11.0+cu130 / CUDA 13.0.88
**目标模型**：model-n32 — Hq=32, Hkv=4, GQA 8×, head_dim=128；45 层全 full attention + 1 层 MTP = **每前向 46 层**；训练上下文 4096
**数据来源**：`benchmarks/bench_msa_configs.py` 三视角计时（eager 延迟 / CUDA-graph 纯 GPU / host 派发），
每阶段用真实内核在真实形状下测量。原始数据 `results/model-n32-20260808/sweep.csv`，工作记录 `log.html`。

> `head_dim=128`（来自 `--kv-channels 128`）与 `qhead_per_kv=8` 都是 MSA 原生支持的，
> 所以本轮用的是模型的**真实**注意力几何，没有任何近似。
> 先前一版按 model-n16（Hq=16/Hkv=2/**D=256**）扫过一遍，那个模型必须降到 D=128 才能测——见 §11。

---

## 摘要

| # | 问题 | 量化影响 | 状态 |
|---|---|---|---|
| 1 | indexer GPU 时间超二次增长，长上下文下反客为主 | 128K 时占稀疏流水线 **72%**；S 每翻倍 ×6.1–6.9（dense 是 ×4.0） | 🔴 **最高优先级** |
| 2 | indexer 分数张量按 Int32 寻址，超 2³¹ 元素越界 | `head=sum/max` 在 **S > 92,672** 直接非法访存；原先是静默内存踩踏 | ✅ 已加守卫 |
| 3 | indexer 每次调用跑 O(q_tiles²) 的 Python 循环 | 32K 下 **15.66 ms/次**纯 CPU × 46 层 = **720 ms/前向** | ✅ 已修复 61.6× |
| 4 | 稀疏注意力的 partial-O 流量 ∝ topk，与实际访问 token 数无关 | 等预算下 topk=32 比 topk=16 慢 **1.86×** | 📋 建议（结构性） |
| 5 | JIT 缓存不随源码失效，改内核静默用旧 kernel | 正确性风险 | ✅ 已修复 |
| 6 | top-k 输出布局与 CSR builder 不匹配，需全量 permute | 128K/topk32 下 67 MB 额外读写 | ✅ 已修复 |
| 7 | head=sum/max 需 8× 的 indexer 计算与写出，换来 0.2–1.1 点质量 | 64K 下选择级 40.4 ms vs keep 的 5.1 ms；128K 跑不了（见 #2）。`max` 被 `sum` 严格支配 | 📋 建议 |
| 8 | indexer 只能产出 128 token 粒度的块分数 | block=32/64 的打分级无法原生支持 | 📋 建议 |
| 9 | top-k 的 transpose workspace 是分数张量的额外一读一写 | 128K/block32/keep 下 8.6 GB 中转 | 📋 建议 |
| 10 | wrapper 常量 host 开销约 0.21 ms/次（四级合计） | × 46 层 = **9.5 ms/前向**纯 CPU | 📋 建议 |
| 11 | D=256 不支持 | **对 model-n32 不适用**；是 D=256 那类模型的阻塞项 | ⚪ 不适用 |
| 12 | decode 被三条独立限制卡住（内核 `qhead_per_kv=16` / CSR SMEM / top-k 12288 块） | 实测最好 **0.41×**，从未赢过 dense；B=32 时 512K 与 1M 全部不可达 | 🔴 **推理侧缺口** |
| 13 | 仓库自带 dense FMHA 只有 **55–64% MFU**，比 FA4 慢 13–28% | 用它当分母会把稀疏加速比虚高约 **1.15×**；已改用 FA4 作主基准 | ✅ 已换分母 |
| 14 | `block=128` + 小 topk 是被漏掉的最优区间 | cfg24（blk128/k8）128K 下 **18.26×** @ cos 0.9899，优于 cfg07 的 10.94× @ cos 0.9897 | 📋 **首选配置** |
| 15 | `fireworks-msa` 的 KV-outer 后端在推荐工作点只有 1.03–1.12× | 优势随 topk 增长，恰与"往小 topk 走"的结论错位；数值等价（cos 0.99999） | 📋 值得合，非首要 |
| 16 | decode 的 dense 已跑在 ~88% HBM 带宽上，稀疏路径距其带宽下界 **380×** | 说明 decode 的全部空间都被 §9(a) 的内核不匹配吃掉 | 🔴 印证 #12 |
| 18 | **FP4 scale-reorder 内核在 2³⁰ 元素处非法访存** | 同类问题第三处；**说明是系统性 32 位寻址问题而非孤立 bug** | ✅ 已加守卫 |
| 19 | `block=64 + topk=32` 在 512K prefill 内存放不下 | 确定性 OOM，同 topk 的 block=128 跑得通；从内存维度独立佐证 #14 | 📋 已记录 |
| 17 | **dense FMHA 在 512K prefill 非法访存**（Q/O 元素数正好 2³¹），且不总是崩 | 污染 CUDA context 带走整场扫描；`dense (msa)` 的 512K 格作废。**FA4 在 1.5×2³¹ 处仍正确**，证明这是本仓库的 32 位寻址 bug 而非固有限制 | ✅ 已加守卫 |

### 关键实测（batch=1, causal, bf16, GPU-only 单层时间）

| kv_len | dense causal | ×上一档 | indexer(keep) | ×上一档 | indexer 占流水线 | 最佳配置 | vs dense |
|---|---|---|---|---|---|---|---|
| 4K（训练长度） | 0.105 ms | — | 0.022 ms | — | 4% | cfg01 0.448 ms | **0.23×**（亏） |
| 8K | 0.385 ms | 3.67× | 0.049 ms | 2.20× | 5% | cfg03 0.789 ms | **0.49×**（亏） |
| 16K | 1.513 ms | 3.93× | 0.167 ms | 3.41× | 9% | cfg03 1.699 ms | **0.89×**（亏） |
| **~18.3K** | | | | | | | **← 交叉点** |
| 20K | 2.456 ms | 1.62× | 0.271 ms | 1.62× | 12% | cfg03 2.183 ms | **1.12×** |
| 24K | 3.550 ms | 1.45× | 0.413 ms | 1.52× | 15% | cfg03 2.636 ms | **1.35×** |
| 32K | 6.467 ms | 1.82× | 0.828 ms | 2.01× | 21% | **cfg03 3.876 ms** | **1.67×** |
| 64K | 27.270 ms | 4.22× | 5.074 ms | 6.13× | 45% | baseline 11.360 / **cfg03 11.388** | **2.40×** |
| 128K | 108.954 ms | 4.00× | 35.072 ms | **6.91×** | **72%** | baseline 48.457 / cfg03 49.283 | 2.25× / 2.21× |

**换算到每次前向（×46 层）**：

| kv_len | dense | baseline | 最佳配置 | 相对 dense |
|---|---|---|---|---|
| 4K | 4.8 ms | 23.7 ms | 20.6 ms | **+15.8 ms（亏）** |
| 8K | 17.7 ms | 43.3 ms | 36.3 ms | **+18.6 ms（亏）** |
| 16K | 69.6 ms | 81.6 ms | 78.1 ms | **+8.5 ms（亏）** |
| 20K | 113.0 ms | 104.9 ms | 100.4 ms | −12.6 ms |
| 24K | 163.3 ms | 130.9 ms | 121.2 ms | −42.1 ms |
| 32K | 297.5 ms | 179.5 ms | 178.3 ms | −119.2 ms |
| 64K | 1254.4 ms | 522.6 ms | 522.6 ms | −731.8 ms |
| 128K | 5011.9 ms | 2229.0 ms | 2229.0 ms | **−2782.8 ms** |

> **这两张表的分母是仓库自带的 dense FMHA，不是 FA4。** 后续接入 FlashAttention-4 作
> 独立对照后测得：FA4 比仓库 dense 快 **13–28%**（FA4 1389–1781 TFLOPS，仓库 dense 只有
> 55–64% MFU）。因此上表所有 "vs dense" 倍数按 FA4 分母应 **×0.87 左右**，交叉点相应右移。
> 详见 §13。表内数值保留原样，因为它们是"稀疏 vs 本仓库 dense"这一问题的正确答案；
> 只是不该被当成"稀疏 vs 最强 dense"来读。

---

## 1. 🔴 indexer GPU 时间超二次增长 —— 最高优先级

causal attention 的计算量是 O(S²)，dense 参照组精确地按 ×4.0 走。indexer 每翻倍却是
**×6.1–6.9**，明显超二次。后果是它在稀疏流水线里的占比从 4K 的 4% 一路涨到 128K 的 **72%**——
**稀疏注意力省下的时间正在被打分器吃掉**。稀疏相对 dense 的优势在 **64K 见顶
（cfg03 2.39×）**，到 128K 已回落到 2.21×——曲线已经掉头了。

另一个角度：128K 下 indexer 做 `4 head × (131072²/2) × 128 × 2 = 8.8 TFLOP`，用时 35.07 ms
→ **251 TFLOPS**；同一块卡上 dense BF16 attention 跑到 1291 TFLOPS。一个 **FP4** 的、
只做 QK 的 pass 却只有 BF16 全量 attention 吞吐的 19%，说明至少有一个量级的余量。

**建议**：先用 nsys 定位，不要凭猜改。怀疑点：
(a) causal compact 调度的负载不均随 S 增长；
(b) K 的分页 TMA 与分数写出之间的 DRAM page 冲突。

---

## 2. ✅ indexer 分数张量的 Int32 寻址上限（已加守卫）

本轮 benchmark 撞出来的一个真 bug。128K + `head=sum` 直接
`CUDA error: an illegal memory access`，并且污染整个 CUDA context，导致后续所有配置连带失败。

根因：`fp4_indexer_interface.py` 把分数张量的 CuTe layout 用 `Int32(...)` 构造，kernel 内部
因此以 32 位有符号算术计算线性偏移，可寻址的最后一个元素是 2³¹−1。分数张量元素数是
`scorer_heads × ceil(S/128) × total_q`：

| head_mode | scorer heads | S=32K | S=64K | S=128K | S=256K | 可跑的最大 S |
|---|---|---|---|---|---|---|
| `keep` | 4 (=Hkv) | 0.034e9 | 0.134e9 | 0.537e9 | 2.147e9（正好 =2³¹） | **262,144** |
| `sum`/`max` | 32 (=Hq) | 0.268e9 | 1.074e9 | **4.295e9 溢出** | **17.18e9 溢出** | **92,672** |

`keep` 恰好能撑到 256K（一个巧合般的边界），`sum`/`max` 在 **92,672 token** 就断了。

**已做**：在 `fp4_indexer_block_scores` 入口加显式校验，把「静默内存踩踏」换成带诊断的 `ValueError`；
`MsaSparseAttention.max_scorer_seqlen()` 把同一上界暴露给调用方做容量规划。

**根治**：要么把 layout 换成 Int64 extent，要么按 §5 把 GQA 归约下沉到 indexer——
后者顺带把 `sum`/`max` 的分数张量降 8×，上限直接推到和 `keep` 一样。

> 这条也改变了配置对比的读法：128K 下 cfg05–cfg12 的 indexer 跑不了，但 top-k / CSR /
> attention 三级仍能测。**把跑得通的几级加起来当总耗时是错的**——那等于把跑不通的部分
> 当成免费的，会得出「sum/max 在 128K 最快」这种完全反的结论。benchmark 与 `log.html`
> 已把这些行排除在排名之外并单独标注。

---

## 3. ✅ indexer 的 O(q_tiles²) host 循环（已修复）

`fp4_indexer_block_scores` 的 eager 延迟远高于 GPU 时间，且**与 head 数无关**——
2 head 和 16 head 都是同一个数，而 GPU 计算量差 8×，说明瓶颈完全在 host。

根因在 `_causal_compact_task_bound`：枚举 `q_tile_count` 个候选 q 长度，每个再调
`_causal_compact_task_count` 循环 `q_tile_count` 次——整体 **O(q_tiles²) 的纯 Python**，
且在**每一次** indexer 调用时重算。它只是给 kernel 算一个 grid 上界，与张量内容无关。

| 上下文 | q_tiles | 内层迭代 | 修复前 host | 修复后 host | 加速 | ×46 层（修复前） |
|---|---|---|---|---|---|---|
| 8K | 64 | ~2.1K | 1.321 ms | 0.269 ms | 4.9× | 61 ms |
| 32K | 256 | ~33K | 15.656 ms | 0.254 ms | **61.6×** | **720 ms** |
| 128K | 1024 | ~525K | ~250 ms（推算） | 0.193 ms | ~1300× | 11.5 s |

**已做**：给两个纯整数函数加 `functools.lru_cache`。
**后续**：`_causal_compact_task_count` 对 `q_len` 是分段线性的，可用等差数列求和写成闭式，
把首次调用的 O(n²) 也去掉、并避免 varlen 场景下 key 空间增长。

---

## 4. 📋 partial-O 流量 ∝ topk —— 最大的结构性优化空间

把 token 预算固定住，只改 block/topk 的分配（attn 阶段 GPU 时间，ms）：

| kv_len | cfg03 (b64/k16, 1024 tok) | cfg02 (b32/k32, 1024 tok) | 比值 | cfg01 (b32/k16, 512 tok) |
|---|---|---|---|---|
| 4K | 0.278 | 0.466 | 1.68× | 0.264 |
| 8K | 0.491 | 0.918 | 1.87× | 0.517 |
| 16K | 1.075 | 1.798 | 1.67× | 1.043 |
| 20K | 1.339 | 2.299 | 1.72× | 1.298 |
| 24K | 1.538 | 2.845 | 1.85× | 1.587 |
| 32K | 2.124 | 3.863 | 1.82× | 2.116 |
| 64K | 4.240 | 7.916 | 1.87× | 4.373 |
| 128K | **8.732** | **16.230** | **1.86×** | 9.066 |

cfg02 与 cfg03 访问的 KV token 数完全相同（1024），耗时却差 1.8×。更直接的证据在最后一列：
cfg01 只访问 512 token、cfg03 访问 1024 token，两者 topk 都是 16，**耗时几乎一样**。
**耗时跟 topk 走，不跟 token 数走。**

根因在 `cute/interface.py::_sparse_atten_csr_varlen_forward`：

```python
O_partial   = torch.empty(topK, total_q, head_q, dim, dtype=partial_dtype, ...)
LSE_partial = torch.empty(topK, total_q, head_q, dtype=torch.float32, ...)
```

一个 CTA 处理 `m_block_size=128` 个 packed Q-head（qhead_per_kv=8 时即 16 个 query token）
× **1 个** KV block，每个 tile 写出一条 partial O，最后由 `combine` 归约。partial 写出量
因此是 `topk × total_q × Hq × D`，**与 `block_size` 无关**：

| 上下文 | topk | O_partial (bf16) | 写+读往返 |
|---|---|---|---|
| 32K | 16 | 4.29 GB | 8.6 GB |
| 32K | 32 | 8.59 GB | 17.2 GB |
| 128K | 32 | **34.36 GB** | 68.7 GB |

副作用：**block 越小，epilogue 与计算的比值越差**。CTA 的 MMA 工作量是
`128 × block_size × 128`，partial 写出恒为 `128 × 128`，所以 block=32 的单位计算 epilogue
开销是 block=128 的 4 倍。这正是 12 组配置里小 block 没能换来收益的原因。

**建议**（按代价从低到高）：

1. **每个 split 吃 G 个 KV block（G>1）**。选择结果本来就按块索引升序，CTA 可以连续处理
   G 个块、在 TMEM 里用 online-softmax 累加，只写一条 partial，流量降为 `topk/G`。
   block=32 取 G=4 即可让 epilogue 比恢复到 block=128 的水平。收益最大且与现有 CSR/schedule 兼容。
2. **partial dtype 降到 fp8**。`partial_dtype` 已是参数（支持 fp8_e4m3），流量直接减半。
3. **partial 槽位数跟随 schedule 实际 split 数**而非 `topK` 上界；`split_counts` 已经算出来了，
   目前只用于 combine，没用来收缩分配。

---

## 5. 📋 三件事落在同一处 indexer epilogue，建议合并做

- **GQA 归约下沉**（收益最大）：`sum`/`max` 现在要写出 Hq=32 行分数、top-k 再读回来归约。
  在 indexer 的 CTA 内先把同组 8 个 head 归约掉，分数张量、写带宽、top-k 读入、workspace
  全部降 8×，**并且 §2 的 Int32 上限从 92,672 推到和 keep 一样的 262,144**。
  无法避免的只有 QK 的 MMA 量（那是 sum/max 语义本身要求的）。
  64K 下 `sum` 的选择级现在是 40.4 ms（keep 只要 5.1 ms），这一项做完应大幅回落。
- **子块粒度**：现在 epilogue 对整个 128 列 N tile 做一次 row-max。改成 `128/block_size`
  个子块累加器即可原生支持 block=32/64。B300 走的是
  `tcgen05.LdRed32x32bOp(Repetition.x128, ..., MAX)` 的硬件整行归约，改成按列区间发
  `Repetition.x32` 四次就能拿到 32 粒度，比在通用循环里加谓词更快也更干净。
  > **测量偏差声明**：本轮 benchmark 中 block=32/64 的 top-k / CSR / attention 三级都是
  > 真实精确的，唯独打分级仍产出 128 粒度分数（其 MMA 量与 block 无关，只有写出量会变）。
- **直接写转置布局**：`sparse_topk_select` 的第一级要把 `[H, K, qo]` 转成 `[H, qo, K]`
  存进 workspace——128K/block32/keep 时是 **8.6 GB** 的额外一读一写。indexer 的 epilogue
  本来就是逐 (q, ktile) 标量写，直接写成转置布局即可整个省掉 transpose kernel（需评估写合并度）。

---

## 6. ✅ JIT 缓存不随源码失效（已修复）

`jit.py::_do_compile_sparse_topk()` 原本只要 `.so` 存在就返回，而 `sparse_topk_select.cuh`
是通过 `-I` 从源码树读的、**不会**被拷进缓存目录。结果是改了 `.cuh` 里的 kernel，下次运行
仍加载旧的 `.so`，没有任何提示——本次改造中就是先踩到它才发现的。

修复：对 `.cu + .cuh + tvm_ffi_utils.h` 做 SHA-256 内容指纹并落 stamp 文件，指纹变化即重编。
编译前先删 stamp，避免中断后留下「新指纹 + 旧 .so」。用内容而非 mtime，因为 editable 安装、
`git checkout`、容器 bind-mount 都会无意义地改动 mtime。

**建议**：`jit.py` 里 variant / plan / reduction 三个模块用的是同一套
`if so_path.exists(): return`，应统一换成指纹机制。

---

## 7. ✅ top-k 输出布局与下游不匹配（已修复）

`sparse_topk_select` 原本固定写 `[total_q, Hkv, topk]`，而 `build_k2q_csr` 需要
`[Hkv, total_q, topk]`，生产路径必须做一次 `permute(1,0,2).contiguous()`。
128K/topk=32/Hkv=4 时这是 67 MB 的额外读写和一次 kernel 启动。

修复：给 kernel 加 `head_outermost` 开关。`bid = qo_head_idx * total_qo_len + t` 本来就是
head-outermost 的线性下标，所以只是把输出偏移从 `(t * H + h) * topk` 换成 `bid * topk`，
零额外成本。`MsaSparseAttention.select()` 默认走这条路径。

---

## 8. 📋 wrapper 常量 host 开销

修掉 §3 后，四级合计仍有约 **0.21 ms/次**（中位数）的 host 时间：每次调用重做张量校验、
compile-cache key 构造、输出分配、cute tensor 转换。

**× 46 层 = 9.5 ms/前向的纯 CPU**。在 4K/8K 这种短上下文下，这比 GPU 时间还多。

建议引入 plan/run 拆分——仓库里 `SparseDecodePagedAttentionWrapper` 已有这个模式：
把校验、key 构造、输出分配放进 `plan()`，`run()` 只更新指针并启动。

---

## 9. 🔴 decode 被三条独立限制卡住

decode（B=32, q_len=1）实测最好只有 **0.41×**，从来没赢过 dense。原因不是一条，是三条，
必须分开处理：

**(a) 走不到 decode 内核。** `sparse_decode_atten_func` /
`SparseDecodePagedAttentionWrapper` 硬性要求 `qhead_per_kv=16` 且 `page_size=128`，
model-n32 是 8。所以实测走的是 prefill 内核：`m_block_size=128` 打包 Q-head，
`qhead_per_kv=8` 时一个 tile 只有 16 个 query token，B=32 总共 2 个 tile，SM 大量闲置。
128K 下只选 1.6% 的 token，理论该快 60 倍，实测 attention 1.17 ms vs dense 1.21 ms。

**(b) CSR builder 的 SMEM 直方图溢出（已加守卫，未根治）。**
`build_k2q_csr.cu` 的降级循环有个下界缺陷：

```cpp
int per_warp_smem = ((total_rows + 1) >> 1) * sizeof(int);   // = total_rows × 2 字节
int kWarps_pick = 4;
while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > 228*1024) {
    kWarps_pick >>= 1;                    // 只降到 1 就停
}
```

降到 1 warp 后退出，**但从不检查单 warp 的直方图是否还放得下**，
`cudaFuncSetAttribute` 失败后向上只剩一句 `CUDA error: invalid argument`。
精确边界 `total_rows × 2 B ≤ 227 KB` → **`total_rows ≤ 116,224`**，
其中 `total_rows = batch × ⌈S/block⌉`。实测卡死在这条线上：512K/blk128
B=28（114,688 行）跑通、B=32（131,072 行）失败。prefill 因 batch=1 从未暴露。
现已在 `prepare_k2q_csr.py` 前置检查并报出行数/上限/绕行办法。
**根治方案**：global-memory 直方图回退，或按 batch 分块——后者最廉价，
且 512K/blk128 只差 4 个 batch。

**(c) top-k 的 12288 块上限**（见 §10），blk=32 在 512K、blk=64 在 1M 直接不可达。

三条叠加后 **B=32 decode 的可行域**：

| ctx | blk=32 | blk=64 | blk=128 |
|---|---|---|---|
| 32K | ✓ | ✓ | ✓ |
| 128K | ✗ csr（131,072 行） | ✓ | ✓ |
| 512K | ✗ topk（16,384 块） | ✗ csr（262,144 行） | ✗ csr（131,072 行，B≤28 可过） |
| 1M | ✗ topk | ✗ topk（16,384 块） | ✗ csr（262,144 行，B≤14 可过） |

**这是推理侧的实际缺口**：长上下文服务的主要成本在 decode，而 decode 恰恰是稀疏收益
最大的场景。优先级 (a) > (b) > (c)——(a) 决定有没有收益，(b)(c) 只决定跑不跑得起来。

---

## 10. 📋 其他

| 项 | 说明 |
|---|---|
| `max_k_tiles < 12288` | top-k 内核只实现了插入排序路径，radix-sort 路径在 fork 时被删掉了。`max_k_tiles = ⌈kv_len / block⌉`，与 batch 无关，所以按 block 换算可达上下文为 **blk32/64/128 → 393K / 786K / 1.5M token**。原断言把上限写成固定的 `12288 × 128` token，只在 block=128 时正确，已改为按 block 表述。 |
| 容器构建适配（已修） | `-arch` 按设备能力取（B300 是 sm_103，原本固定 sm_100 要走 PTX JIT）；CUDA 12.8 起 `-static-global-template-stub` 默认值变更导致匿名 namespace 模板 kernel 编译失败；pip CUDA stack 的 `cusparse.h` 只在 wheel 里，但整目录加 `-I` 会遮蔽 nvcc 自带的 `crt/host_runtime.h`，改为只软链缺失的头。 |
| benchmark 韧性（已修） | 长扫描中途遇到不可恢复的 CUDA 故障会污染 context 导致后续全挂；现在每行落盘一次、context 不可用时干净退出并保留已测数据。 |

---

## 11. ⚪ head_dim=256（对 model-n32 不适用）

**model-n32 的 `--kv-channels 128` 落在支持范围内，本条对它不构成阻塞。**
保留此节是因为它限制了 MSA 的适用面：model-n16 的 full-attention 层是
`head_dim=256`，而 MSA 全线断言 D=128（`cute/interface.py::_validate_csr_varlen_inputs`、
`src/sm100/fwd/atten_fwd.py::__init__`、FP4 indexer `_HEAD_DIM`）。
`MODEL=n16 ./benchmark.sh` 会在 D=128 下测量并显式打出这条偏差。

不能简单拆成两个 128 维各跑一遍再合并：QK 是可加的
（`q·k = q[:128]·k[:128] + q[128:]·k[128:]`），但 softmax 必须在完整 QK 之后做。
正解是让内核的 K-tile 循环跑两轮累加到同一个 S tile。主要阻力在 TMEM：当前布局 S 占
`2 × n_block_size` 列、O 占 `o_stage × head_dim = 2 × 128 = 256` 列，合计 512 列刚好用满
SM100 上限；D=256 时 O 单独就要 512 列，**必须把 `o_stage` 降到 1 或把 O 分段流水**。

---

## 12. 🟢 共享单 index（DSA 形态）—— 本轮找到的最大加速

`head_mode` 决定的是**怎么归约**；正交的另一维是**产出几份选择**。配置层新增 `shared_index`
表达后者：整层共享一份（DSA 形态）还是每个 KV head 一份。

**关键在于省的是打分器行数，不是归约。** `sum`/`max` 的打分器无论输出几份都要先算 `Hq=32` 行——
归约在昂贵那级的下游，所以 `sum@1` 相对 `sum@4` 几乎不省（32K：9.14 vs 9.80 ms）。
而 `keep` 的打分器行数**等于**它产出的选择数，`keep@1` 把打分器从 4 行压到 1 行，indexer 直接省 4×。

| 配置 | 打分器行数 | 选择份数 | 32K | 64K | 128K |
|---|---|---|---|---|---|
| dense causal | — | — | 6.57 | 26.72 | 108.41 |
| keep@4（cfg03） | 4 | 4 | 3.80 (1.80×) | 11.37 (2.40×) | 49.25 (2.21×) |
| **keep@1（DSA 形态）** | **1** | **1** | **2.56 (2.67×)** | **6.10 (4.46×)** | **19.02 (5.73×)** |
| **keep@1 + topk=32** | 1 | 1 | 4.17 | **9.55 (2.80×)** | **26.35 (4.11×)** |
| sum@4（cfg07） | 32 | 4 | 9.80 | 47.73 | n/a |
| sum@1 | 32 | 1 | 9.14 | 46.42 | n/a |

质量代价（8K，layer 0，相对 per-head oracle）：

| 模式 | 选择份数 | recall (topk=16) | recall (topk=32) | 组内最差 head (topk=16) |
|---|---|---|---|---|
| oracle | 32 | 1.000 | 1.000 | 0.633 |
| sum@4 | 4 | 0.935 | 0.956 | 0.452 |
| keep@4 | 4 | 0.924 | 0.949 | 0.443 |
| sum@1 | 1 | 0.906 | 0.938 | 0.359 |
| **keep@1** | 1 | **0.901** | **0.935** | 0.355 |
| max@1 | 1 | 0.858 | 0.906 | 0.358 |

**结论：把省下的打分成本换成更大的 topk。**
`keep@1 + topk=32` vs `keep@4 + topk=16`（cfg03）：128K 下 **26.35 vs 49.25 ms（快 1.87×）**，
同时 recall **0.935 vs 0.924（高 1.1 点）**——≥64K 严格占优；32K 慢 10% 但 recall 高 1.1 点，基本打平。

附带好处：`keep@1` 的分数张量只有 `1 × 1024 × 131072 = 1.34e8` 元素，远低于 §2 的 2³¹ 上限，
**可以一路扩到 256K 以上**。

**前提**：`keep@1` 需要一个**训练好的单 index 投影**——这正是 DSA lightning indexer 的形态。
MSA 本身不训练它，上表的质量数字用的是均值池化的未训练 proxy，是**下界**。

---

## 13. ✅ dense 分母换成 FlashAttention-4

只拿仓库自带的 dense FMHA 当分母，无法区分「加速来自稀疏」还是「分母太弱」。接入
FA4（`flash_attn/cute`，CuTe-DSL，无需 nvcc 构建）作独立对照后：

| | 仓库 dense FMHA | FA4 |
|---|---|---|
| MFU | 55–64% | — |
| TFLOPS | — | 1389–1781 |
| 相对快慢 | 基准 | **快 13–28%** |

**结论**：仓库的 dense 内核本身有 36–45% 的性能留在桌上，这是一个独立于稀疏的优化点。
所有对外报的加速比都应以 FA4 为分母；benchmark.sh 已默认跑 FA4 行（`NO_FA4=1` 关闭）。

**decode 下的一个坑（已修）**：FA4 默认 `num_splits=1`，而 q_len=1 时工作量只分解成
`batch × head_kv` 个 CTA（B=32/Hkv=4 → 128 个，不到一波），不切 KV 就跑不满机器，
一度测出 FA4 比仓库 dense 慢（0.88×）——那是基准不公平，不是 FA4 慢。现已按
`batch × head_kv` 与 SM 数实测挑选 `num_splits`，并把选中值记进结果行。

---

## 14. 📋 `block=128` + 小 topk 是被漏掉的最优区间

最初的 12 组只覆盖 `block ∈ {32,64}`、`topk ∈ {16,32}`。把网格扩到 33 组
（加 `block=128`、`topk ∈ {4,8}`）后，最优点整体移到了角落上：

| 配置 | block | topk | 128K vs FA4 | cos |
|---|---|---|---|---|
| cfg07 | 64 | 16 | 10.94× | 0.9897 |
| **cfg24**（=cfg17 同参） | **128** | **8** | **18.26×** | **0.9899** |

**同等甚至更好的质量下快 1.67×。** 这是 §4 的直接推论：耗时跟 `topk` 走而不跟 token
预算走，所以在固定预算（`block × topk`）下应当一路推向**大 block / 小 topk**。
cos 由 token 预算决定、与 block/topk 如何拆分几乎无关（实测 2048 预算 → 0.9945–0.9946，
1024 预算 → 0.9897–0.9903，跨配置差异在第 4 位小数），这条规律让这次外推是安全的。

**注意**：大 block 同时缓解 §9 的两条 decode 限制（`total_rows` 和 `max_k_tiles` 都
∝ 1/block），所以 `block=128` 在 decode 侧还有额外的可用性收益。

---

---

## 15. 📋 KV-outer 后端（`fireworks-msa` 分支）实测

`MiniMax-AI/MSA` 的 `fireworks-msa` 分支加了约 6000 行的 `kvouter` 模块，用
**KV-stationary 循环 + LSE merge** 替换 `build_k2q_csr` + Q-outer forward。
这正对着 §4：当前实现的 `O_partial = empty(topK, total_q, head_q, dim)` 让
partial-O 流量 ∝ topk 而非 ∝ 实际访问 token 数。

**口径**：两侧同一份 `selected`、同一份 Q/K/V（两个 checkout 都提供 `fmha_sm100`
包，无法同进程导入，故双方用相同种子逐位重建，只交换 selection 与输出）；
只比 stage 3–4，indexer 与 top-k 两侧相同、不计入。两侧输出 **cos = 0.99999**，
数值等价。KV-outer 有 `assert block_size == 128 and head_dim == 128`，故只在
block=128 上成立。

**重要保留**：跑的是 **Python CuTe-DSL 后端**（容器里 `cpp_backend_available()` 为
False），而分支把 AOT C++ 路径列为优先、Python 为 fallback。**下表低估分支实力，
小 shape 上尤甚**——0.67× 那格很可能主要是 Python dispatch 开销。

| kv_len | topk=4 | topk=8 | topk=16 | topk=32 |
|---|---|---|---|---|
| 32K | 0.67× | 0.82× | 0.93× | 1.03× |
| 64K | 0.83× | 0.94× | 1.03× | 1.09× |
| 128K | 0.93× | **1.03×** | 1.10× | 1.13× |
| 256K | 1.04× | **1.12×** | 1.15× | 1.15× |

（>1 = KV-outer 更快。加粗列为本报告 §14 推荐的 topk=8。）

**四条结论**：

1. **优势沿对角线出现**：上下文每翻一倍，追平所需的 topk 减半。256K 从 topk=4 就领先。
2. **短上下文 + 小 topk 反而慢**，最差 0.67×——index build 与 LSE merge 的固定开销摊不掉。
3. **和本报告的推荐配置错位。** §14 的结论是往「大 block、小 topk」走（cfg24 =
   blk128/topk8）；KV-outer 恰恰在**大 topk** 上收益最大，在 topk=8 这列只有
   1.03×（128K）～1.12×（256K）。**它没有移动最优配置的位置，只是让偏离最优点的
   那一侧惩罚变轻。**
4. **topk 伸缩性改善但非质变**：topk 4→32（8× 的 KV 读取量）下，q-outer 耗时涨
   4.5–5.5×，KV-outer 涨 3.0–5.0×。两者都远低于 8×，说明小 topk 侧都被固定开销主导；
   KV-outer 压低了 topk 相关的斜率，没有消掉 partial-O 的代价。

**优先级判断**：这条分支值得合，但**不是 §1 的替代品**。在推荐工作点上它给 1.03–1.12×，
而同一位置 indexer 占长上下文流水线的 **72%**——收益量级差一个数量级。
正确顺序仍是：先修 indexer 的超二次增长，再换 forward。

**复现**：`benchmark.sh` 的 `KVOUTER_PATH` / `KVOUTER_PYTHON` / `KVOUTER_FA4_PATH`，
详见 §16 的版本约束。

---

## 16. ⚠️ 跑通 `fireworks-msa` 的版本约束（都不是分支的 bug）

| 现象 | 原因 |
|---|---|
| `module 'cutlass.cute.core' has no attribute 'ThrMma'` | `cute.core.ThrMma` 与 `cute.make_fragment` 在 cutlass-dsl **4.6.0**（容器所装）里已移除；本仓库 main 的 `047c25b` 修的是同一件事。分支 pin 4.5.1。 |
| `FlashAttentionForwardSm100.__init__() got an unexpected keyword argument 'is_persistent'` | 手头 FA4 checkout 版本不符；分支要 `flash-attn-4==4.0.0b15`。 |
| 装了 b15 后仍报 ThrMma | b15 自身也用 `cute.core.ThrMma`，整条链锁死在 cutlass-dsl 4.5.x。 |

**处理**：没有去 patch 别人发布的 wheel（那会让"分支跑在它不支持的版本上"，
结果不可信），而是建 `--system-site-packages` venv 装上分支声明的版本
（cutlass-dsl 4.5.1 / quack 0.4.1 / flash-attn-4 b15），并撤回为 4.6.0 打的临时补丁。

---

## 17. 🔴 decode 的带宽账：稀疏路径差了 380 倍

把 decode 放到带宽尺度上量，§9 的结论会更锋利。128K decode / B=32 需读
`32 × 131072 × 4 heads × 128 dim × 2 B × 2 (K+V) = 8.59 GB`：

| | 耗时 | 等效带宽 | 相对 HBM 峰值 |
|---|---|---|---|
| dense（本仓库） | 1.215 ms | 7.07 TB/s | **~88%** |
| dense（FA4，已调 `num_splits`） | 1.275 ms | 6.74 TB/s | ~84% |
| cfg17 稀疏流水线 | 3.102 ms | — | — |
| cfg17 的**带宽下界**（只读 0.78% 的 KV = 67 MB） | ~0.008 ms | — | — |

两点直接推论：

1. **FA4 在 decode 下赢不了，是因为没得赢。** dense decode 已经跑在 ~88% 的 HBM
   带宽上，两个内核都顶在同一堵墙上。§13 里 FA4 快 13–28% 是**纯 prefill 现象**
   （那里是 compute-bound，仓库 dense 只有 55–64% MFU）；decode 下两者打平
   （1.05× @32K，0.95× @128K）是正确结果，不是测量问题。
2. **稀疏的空间全在那 380 倍里。** cfg17 只需读 0.78% 的 KV，带宽下界约 0.008 ms，
   实测 3.102 ms。这不是"稀疏收益小"，而是 §9(a) 的内核不匹配把收益全吃掉了——
   也再次说明放开 `qhead_per_kv` 的优先级高于一切 decode 侧的其它工作。

---
---

## 18. 🔴 dense FMHA 在 512K prefill 非法访存（已加守卫）——且 512K 那列数据不可信

重跑全网格时抓到的。34 个失败格全部只出现在 512K，其它 7 个上下文零失败，分三类：

| 类别 | 数量 | 原因 |
|---|---|---|
| `indexer: ValueError` | 34 | §2 的 Int32 分数张量守卫，全部配置。预期内。 |
| `topk: AssertionError` | 9 | §10 的 12288 块上限，恰好是 9 个 blk=32 配置。预期内。 |
| `attn: OutOfMemoryError` | 3 | 见下 |
| **`dense` 自身非法访存** | — | **本节，之前完全没被看见** |

**dense 参照组在 512K prefill 会 `CUDA error: an illegal memory access`。**
`total_qo_len × num_qo_heads × head_dim = 524288 × 32 × 128 = 2,147,483,648`，
**正好等于 2³¹**。实测边界：

| kv_len | Q/O 元素数 | 结果 |
|---|---|---|
| 262144 | 1.07e9 | 正常 |
| 458752 | 1.88e9 | 正常（1323 ms） |
| **524288** | **2.147e9 = 2³¹** | **每次都非法访存** |

和 §2 是同一类问题（32 位寻址），只是这次在 **dense/attention 路径**上，而且原本没有守卫。

**最坏的性质：它不总是崩。** 完整 run 里 dense@512K 报出了 1739.684 ms 并被写进表格，
而同一 shape 在干净进程里三次全崩（真实 Q/K 与随机 Q/K 都一样）。区别只在分配器状态——
这正是 2³¹ 边界问题的典型表现：是否触发取决于基址，不取决于数值。
**因此表中 `dense (msa)` 那一行的 512K 格是在已污染状态下侥幸跑出来的，作废。**

**FA4 没有这个问题——而且这正说明问题出在本仓库的内核上。** 用 `benchmarks/probe_fa4_int32.py`
在边界两侧实测，并且不只看崩不崩：32 位偏移溢出同样可能静默读错地址，所以每档都把输出
与 fp32 参考逐行比对，采样偏向高位行（含最后一行）：

| total_q | Q/O 元素数 | 结果 | cos vs fp32 参考 | rel_err |
|---|---|---|---|---|
| 458752 | 0.875 × 2³¹ | 正常 | 0.999995 | 2.375e-3 |
| **524288** | **1.000 × 2³¹** | **正常** | 0.999996 | 2.383e-3 |
| **786432** | **1.500 × 2³¹** | **正常** | 0.999996 | 2.420e-3 |

越过边界 1.5 倍仍然正确，且 `rel_err` 三档稳定（纯 bf16 累加噪声，没有随越界劣化）。
**所以这不是 shape 或硬件的固有限制，是本仓库内核的 32 位寻址 bug。**
实践含义有二：(a) 表 1b 以 FA4 为分母的 512K 加速比**仍然可信**，作废的只是
`dense (msa)` 行；(b) 根治方向明确——照 FA4 的做法把 offset 提升到 64 位即可，
不需要限制可用上下文。

**连带伤害**：非法访存会污染 CUDA context，导致该进程后续所有配置连带失败——
一个坏 shape 能带走整场扫描。

**已做**：在 `fmha_sm100()` 入口按 `q.shape` 加显式校验，把非法访存换成带诊断的
`ValueError`（给出元素数、上限，以及该 head 几何下的 token 天花板）。
验证后 512K 干净报错且 **context 存活**，同一进程里 cfg16 的稀疏路径继续跑完
（31.198 / 0.460 / 12.406 ms）。**根治**同 §2：把 layout 换成 Int64 extent。

> 稀疏 forward（CuTe 的 `sparse_atten_func`）在同一 shape 下没有崩。它是另一条入口，
> 不走这个守卫；但也不能据此断言它是 Int64 安全的，只能说这次没触发。

**另外 3 个 OOM**：cfg04/08/12（blk=64, topk=32）在 512K OOM，见 §20——
后续验证表明它是确定性的固有限制，不是扫描过程中的显存累积。

---
---

## 19. 🔴 FP4 scale-reorder 内核的 32 位寻址上限（已加守卫）

跑完整 `benchmark.sh` 时抓到的第三处同类问题。1M decode 在 **`setup` 阶段**
非法访存——早于所有已有守卫，且和预测中的 `csr` 失败无关。逐步同步二分后隔离到
`fp4_indexer_reorder_scales_for_mma_cute` 自身。卡边：

| K scale 张量 | 元素数 | 结果 |
|---|---|---|
| 262143 页 × 4 × 128 × 8 | 1,073,737,728 | 正常 |
| **262144 页 × 4 × 128 × 8** | **1,073,741,824 = 2³⁰** | **每次都非法访存** |

**上限是 2³⁰ 而不是 2³¹**，说明 kernel 内部某处按「输入 + 输出」两倍量形成偏移，
恰好在 2 × 2³⁰ 处溢出 Int32。

**这已经是同一类问题的第三处**（§2 indexer 分数张量 2³¹、§18 dense Q/O 2³¹、
本节 scale reorder 2³⁰）。**这不是三个孤立 bug，是这套 CuTe 栈系统性的 32 位寻址问题。**
三处的失败方式也一样：非法访存而非抛异常，污染 CUDA context，一个坏 shape 带走整场扫描。
根治办法对三者相同——把 layout 的 extent 提升到 Int64，FA4 已经证明这条路可行（§18）。

**已做**：在 `fp4_indexer_reorder_scales_for_mma_cute` 入口按 `k_scale.numel()` 加守卫。
注意这条限制**按 selection head 数缩放**，所以 `sum`/`max`（共享 1 份 selection）比
`keep`（Hkv 份）晚 4 倍才触发。

---

## 20. 📋 512K prefill 的内存墙：`block=64 + topk=32` 放不下

`block=64, topk=32` 在 512K prefill 下 attention 级 OOM（`Tried to allocate 4.00 GiB`，
268 GiB 卡上失败时只剩 1.98 GiB），而**同样 topk=32 的 `block=128` 跑得通**（attn 72.3 ms）。

**确定性的，不是偶然**：两次完整 run 命中同样 3 个格子（cfg04/08/12），
且单独在干净进程里跑同样 OOM。（先前一版报告把它归因为扫描过程中的显存累积，
那个解释是错的，已更正。）

selection→attention 路径上有某项随 **1/block** 增长，所以 block=64 需要 block=128 的两倍。
确切是哪一块分配尚未定位——这里记录的是**实测边界而非公式**，`verify_expectations.py`
里也照此标注。

**意义**：这是从内存维度独立佐证了 §14 的结论。§14 从延迟出发说「往大 block、小 topk 走」；
这里说的是**小 block + 大 topk 在长上下文下根本放不下**。两条独立理由指向同一个方向。

---
## 配置选择结论

> **最终推荐（33 组网格 + FA4 分母）：`block=128, topk=8`（cfg24）。**
> 128K 下 **18.26× vs FA4**，cos 0.9899——比原先的 cfg07 快 1.67× 且质量不降。
> 下面第 1–6 条是 12 组网格 + 仓库 dense 分母那一轮的结论，`head_mode` 与
> `sum`/`max` 的取舍仍然成立，但速度倍数请按 §13 换算、最优 block/topk 以 §14 为准。

1. **推荐生产配置：`force_init(128)-force_end(128)-block(64)-topk(16)-head(sum)`（cfg07）。**
   8 个上下文点全部最快或并列最快：16K 1.03×、32K 2.12×、64K 2.73×、128K 2.39× dense。
   交叉点 **~15.9K**，比 keep 系的 ~20.5K 早近 5K token。
2. **质量优先则选 cfg03（`head=keep`）**：recall 高 1.4–2.9 点，代价是慢 1.08–1.27×、
   交叉点晚到 ~20.5K。≥64K 时 sum 的速度优势只剩 1.08–1.14×，keep 的质量更划算。
3. **`max` 排除**：与 sum 速度差 <2%，质量差 3.2–4.2 点，严格被支配。
4. **`topk=16` 全面优于 `topk=32`**（cfg07 vs cfg08 在 128K 是 2.39× vs 2.05×）——
   耗时跟 topk 走而不跟 token 预算走（§4），同预算下永远选大 block / 小 topk。
5. **模型当前 4096 训练上下文远在交叉点之下**：4K 时最好的配置也只有 0.26×，
   ×46 层每前向多花 13.4 ms。**MSA 的价值在长上下文推理。**
6. **优势在 64K 见顶随后回落**（cfg07 2.73× → 2.39×），原因是 §1 的 indexer 超二次增长。

**测量口径**：机器上有其它容器负载，dense 参照组 run-to-run 波动约 ±8%（16K 下 1.38–1.48 ms）。
接近 1.0× 的格子应读作「约等于」；≥32K 的比值波动在 2% 以内，结论稳定。
