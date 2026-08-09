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
| 12 | decode 路径要求 `qhead_per_kv=16` | 本模型是 8，只覆盖 prefill | 📋 建议 |

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

## 9. 📋 decode 路径未覆盖

`sparse_decode_atten_func` / `SparseDecodePagedAttentionWrapper` 硬性要求
`qhead_per_kv=16` 且 `page_size=128`，model-n32 是 8，因此本轮只覆盖 prefill。

**这是推理侧的实际缺口**：长上下文服务的主要成本在 decode，而 decode 恰恰是稀疏收益
最大的场景（每个 token 只看 topk×block 个 KV）。要落地必须先放开 `qhead_per_kv` 与
`page_size` 的限制。

---

## 10. 📋 其他

| 项 | 说明 |
|---|---|
| `max_k_tiles < 12288` | top-k 内核只实现了插入排序路径。256K + block=32 = 8192 块尚可，512K 或 block=16 会超。radix-sort 路径在 fork 时被删掉了。 |
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

## 配置选择结论

> 按修正后的 `head_mode` 语义（`sum`/`max` 归约 keep 的 Hkv 行 → 1 份共享选择，
> 打分器工作量与 keep 相同）重测全部 12 组 × 8 个上下文，无 n/a。

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
