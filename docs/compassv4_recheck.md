# Compass-V4 MSA 复测（B300 `.5`）

> 2026-08-10 · 机器 `10.251.209.5` / 容器 `dsa_stage2_meng2` / MSA `80434d7`
> 协议与环境见内部交接文档 `MSA_复测交接.md`（不在本仓库）

## 形状

| | num_query_heads | num_kv_heads | head_dim |
|---|---:|---:|---:|
| attention | 32 | 4 | 128 |
| index | 4 | 1 | 128 |

`topk=16`、`page=128`、attention/dense 走 bf16。Prefill `B=1` causal；Decode `Sq=1, BS=32`
（batch 内各 request 独立 KV，不共享）。

**indexer 走 fp8。** 交接文档把它描述成 "bf16 MQA proxy"，但 `output_maxscore` 在
`float8_e4m3fn` 下是支持的，prefill / decode 都没触发回退。这是 decode 的主导项，
影响很大：1M decode indexer `0.6604 ms`(fp8) vs `1.2596 ms`(bf16)。

## Prefill（B=1, causal, bf16）

| seq | idx(ms) FP8 | topk(ms) | attn(ms) | MSA full(ms) | MSA dense(ms) | FA4(ms) | msa/FA4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.050 | 0.224 | 0.885 | 1.159 | 0.410 | 0.392 | 0.34× |
| 16,384 | 0.087 | 0.413 | 1.458 | 1.958 | 1.678 | 1.380 | 0.70× |
| 32,768 | 0.290 | 0.939 | 2.392 | 3.621 | 6.909 | 5.793 | 1.60× |
| 65,536 | 1.084 | 2.240 | 4.537 | 7.862 | 27.952 | 23.288 | 2.96× |
| 131,072 | 4.371 | 5.945 | 9.178 | 19.494 | 111.369 | 93.051 | 4.77× |
| 262,144 | 17.486 | 18.042 | 19.633 | 55.161 | 445.838 | 375.363 | 6.80× |
| 524,288 | 70.362 | 32.929 | 40.016 | 143.307 | 1780.951\* | 1503.135 | 10.49× |
| 1,048,576 | 314.615 | 112.001 | 79.949 | 506.565 | 6198.502 | 6011.293 | 11.87× |

## Decode（Sq=1, BS=32, bf16）

| KV | idx(ms) FP8 | topk(ms) | attn(ms) | MSA full(ms) | MSA dense(ms) | FA4(ms) | msa/FA4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.0236 | 0.0133 | 0.0402 | 0.0771 | 0.1035 | 0.1567 | 2.03× |
| 16,384 | 0.0298 | 0.0133 | 0.0418 | 0.0849 | 0.1834 | 0.2280 | 2.69× |
| 32,768 | 0.0419 | 0.0133 | 0.0400 | 0.0952 | 0.3290 | 0.3723 | 3.91× |
| 65,536 | 0.0658 | 0.0133 | 0.0412 | 0.1203 | 0.6236 | 0.6249 | 5.19× |
| 131,072 | 0.1074 | 0.0133 | 0.0402 | 0.1610 | 1.2155 | 1.2893 | 8.01× |
| 262,144 | 0.1834 | 0.0133 | 0.0402 | 0.2369 | 2.4065 | 2.5375 | 10.71× |
| 524,288 | 0.3412 | 0.0133 | 0.0400 | 0.3945 | 4.7839 | 5.1022 | 12.93× |
| 1,048,576 | 0.6604 | 0.0174 | 0.0419 | 0.7196 | 9.5878 | 10.2433 | 14.23× |

bf16 indexer 对照（decode）：1M `idx 1.2596 / full 1.3174 / 7.28×`，
32k `idx 0.0648 / full 0.1191 / 2.76×`。

## ⚠️ 跑在哪个容器里

上表全部测自 `.5` 的 **`dsa_stage2_meng2` + `/sparse/msa_venv`**（cutlass-dsl **4.4.1** /
quack **0.2.10**，MSA 硬 pin 的版本）。

另一个容器 `vllm-b300-dspark-min`（pip CUDA stack，cutlass-dsl 4.6.0 / quack 0.6.3）
现在也能跑通了，但 **数字不可与上表直接比较** —— sparse attn kernel 由 cutlass-dsl
版本决定，实测差异不小：

| seq | attn(ms) @4.4.1 | attn(ms) @4.6.0 |
|---:|---:|---:|
| 8,192 | 0.885 | 0.939 |
| 32,768 | 2.392 | 2.736 |

indexer / topk 两段基本一致（8k: 0.050/0.226 vs 0.050/0.224）。要出对外数字，用
`dsa_stage2_meng2`。

## 口径

- `MSA dense` 是 MSA 自带的 dense FMHA，**只作参考**；`msa/FA4` 的分母是 **FA4**。
- **FA4 这轮是我们自己在 `.3` 实测的**（`MODE=fa4 ./benchmark.sh`，容器
  `sparse_atten_meng` + `fa4_venv`），不再引用表里的旧值。MFU 稳定在 66-67%，
  与交接文档一致。和 `Sparse-Attention.xlsx` 的 GQA-FA4 列对比：≥32k 全部在
  **1.5%** 以内（32k −1.0% / 128k −0.8% / 256k −1.2% / 512k −1.5% / 1M −1.3%），
  8k `+5.9%`、16k `−8.0%` 偏差大一些——绝对值太小（0.4ms / 1.4ms），对噪声敏感。
  decode FA4 也是本轮实测（`B=32 Sq=1`，MBU 83-86%）：和表里旧值 128k−0.3% /
  256k +3.2% / 512k +4.5% / 1M +5.5%，32k、64k 差 −8.4% / −8.3%；8k、16k 旧表
  是空的，这轮补上了。
  ⚠️ FA4 **只能在 `.3` 测**：在 `.5` 会误 dispatch（同一个 32k 形状读到 19.1ms
  vs 5.8ms），别为了"统一机器"把 dense 挪到 `.5`。
- **1M 的 attn 是实测值，不是外推。** MiniMax-M3 形状（h_q=64）下 `bench_sparse` 要
  256 GiB 会 OOM，所以旧表 1M 的 `~155ms` 一直是外推；h_q=32 workspace 减半后跑通，
  1M 行现在全是实测。
- \* **dense FMHA 只在 `h_q × seq == 2^24` 这一个点上崩**（`cudaErrorIllegalAddress`，
  并且毒化 CUDA context——错误在下一次调用才浮出来，`_try` 里的
  `torch.cuda.empty_cache()` 又二次抛异常，直接杀掉进程，连 PARTIAL 行都打不出来）。
  这**不是**一个"超过多长就崩"的阈值，实测：

  | | 256k | 512k | 1M |
  |---|---|---|---|
  | h_q=64 | 崩（64×256k = 2^24） | 3107.925 | — |
  | h_q=32 | 445.838 | 崩（32×512k = 2^24） | 6198.502 |

  两个几何都在崩溃点两侧正常。

  \* **512k 的 dense 现在用 query 分块测出来了：1780.951 ms**（`DENSE_CHUNK=auto`，
  切成 2 块、每块 `h_q×chunk = 2^23`）。分块的正确性在两边都能跑的点上校验过：
  256k 一次性 445.838 vs 分成 4 块 447.536，差 **0.38%**；而且 1780.951 正好是
  256k 的 **3.995×**，符合 causal dense 的 N² 标度。分块能绕过崩溃这件事本身，
  也反过来支持"索引溢出"而不是"显存不够"的解释。

  meng.liu 原来的 `run_node5_msa.sh` 其实已经把这个非单调性写对了：

  ```bash
  [ "$S" -ge 262144 ] && [ "$S" -ne 524288 ] && skip=1   # 256k 崩、1M dense 本来就跑不了
  ```

  那个 `!= 524288` 的例外就是"跳过会崩的 256k、保留能测的 512k"。是我在写
  `benchmark.sh` 时把它压成了 `SKIP_DENSE_FROM=262144` 这样一个纯阈值，
  才把 h_q=32 下能测的 256k 丢掉。现在这个点不用跳了——`bench_msa_full_pipeline.py`
  在 `h_q × seq ≥ 2^24` 时自动对 dense 的 query 维分块（`DENSE_CHUNK=auto`），
  直接测过去；其余点仍走一次性路径，数字和之前逐位相同。
- prefill@1M 的 `topk=112.0ms`：这一项只依赖 kv 长度、与 h_q 无关，M3 复测同样是
  `111.68ms`。

## 复现

两个 bench 脚本已按 env 参数化（`H_Q/H_K/D`、`IDX_HQ/IDX_HKV`、`IDX_DTYPE`），
默认值保持 MiniMax-M3 原行为不变。把 `bench_msa_*.py`、`benchmark.sh` 与
`scripts/msa/*.sh` 拷进容器 `/sparse/msa/` 后，用 `benchmark.sh`（零配置默认就是
Compass-V4 形状）：

```bash
# .5（容器 dsa_stage2_meng2）—— MSA 稀疏三段
cd /sparse/msa && GPU=5 ./benchmark.sh              # prefill + decode 全跑
cd /sparse/msa && MODE=decode GPU=5 ./benchmark.sh  # 只跑 decode

# .3（容器 sparse_atten_meng）—— FA4 dense 基线
cd /workspace/sparse_atten_meng/msa_bench && MODE=fa4 GPU=5 ./benchmark.sh
```

`MODE=fa4` 自动找 FA4 解释器（`/workspace/sparse_atten_meng/fa4_venv`，或
`FA4_PYTHON` 指定），跑 prefill + decode 两条；`FA4_DECODE=0` 只跑 prefill。
1M 单次约 6 秒、默认 40 次迭代 ≈ 4 分钟一个点，急的话压 `FA4_REP`。
FA4 和 MSA **不能共用环境**：FA4 的 cute 要 cutlass-dsl 4.6.x，正是 MSA 必须避开
的版本，所以走各自的 venv。

**force init / force end 现在默认开（1/1）**——`FORCE_BEGIN` / `FORCE_END` 分别把
序列头部的 sink 块和最靠近 query 的 local 块钉进 top-k。它们占的是 topk=16 里面的
名额、不是额外增加，所以只改"attend 哪些块"、不改"多少块"，实测延迟中性（128k：
idx 4.3398→4.3499、topk 5.9441→5.8091、attn 9.1383→9.1730，都在噪声内）。
**上面两张表是 0/0 测的**，因为延迟中性所以数字照用；要复现纯 top-k 口径设
`FORCE_BEGIN=0 FORCE_END=0`。

也可以用旧的 `run_compassv4.sh`：

```bash
cd /sparse/msa && nohup env GPU=2 bash run_compassv4.sh > run_compassv4.log 2>&1 &
```

### Profile

`benchmark.sh` 的三段都包了 NVTX range（`msa_idx` / `msa_topk` / `msa_attn` /
`msa_dense`），profile 时自动收窄到单个序列点并压短计时窗口：

```bash
cd /sparse/msa && PROFILE=nsys SEQLENS=131072 GPU=5 ./benchmark.sh
```

跑完直接打印两张表：按 NVTX 段归属的 kernel 汇总
（`cuda_gpu_kern_sum:nvtx-name:base`，行长这样 `msa_idx/device_kernel`）
和每段的 NVTX wall time。

⚠️ **`PROFILE=ncu` 在当前容器里不可用**：容器既没有 `CAP_SYS_ADMIN` 也没有
`CAP_PERFMON`，驱动会拒绝采计数器（`ERR_NVGPUCTRPERM`），ncu profile 到 0 个
kernel。`benchmark.sh` 有 preflight 会提前把这个讲清楚。要用 ncu 得用
`--cap-add SYS_ADMIN` 重建容器。nsys 只做 trace、不需要计数器权限，所以能正常跑。

⚠️ `run_node5_msa.sh` 的 `run_decode()` 里硬编码了 `DTYPE=bf16 IDX_DTYPE=bf16`，
会**覆盖**外部导出的 `IDX_DTYPE=fp8`（prefill 不受影响）。fp8 decode 需手工跑：

```bash
cd /sparse/msa && CUDA_VISIBLE_DEVICES=2 \
  DTYPE=bf16 IDX_DTYPE=fp8 QLEN=1 BATCHES=32 \
  H_Q=32 H_K=4 D=128 IDX_HQ=4 IDX_HKV=1 TMPDIR=/sparse/tmpdir \
  /sparse/msa_venv/bin/python -u bench_msa_decode_pipeline.py \
  8192,16384,32768,65536,131072,262144,524288,1048576
```

日志与 CSV 落在容器内 `/sparse/msa/logs_compassv4/`。

## 环境校验：M3 基准复现

同一轮先按交接文档跑了 MiniMax-M3 形状（h_q=64）做环境校验，每一段都在 ±1% 内：

| prefill | idx | topk | attn | full | 基准 full |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 0.357 | 0.939 | 4.623 | 5.920 | 5.940 |
| 131,072 | 6.166 | 5.945 | 18.404 | 30.515 | 30.732 |
| 262,144 | 24.853 | 18.044 | 38.093 | 80.990 | 81.017 |
| 524,288 | 99.626 | 32.914 | 78.119 | 210.658 | 209.875 |
| 1,048,576 | 425.007 | 111.681 | OOM | — | — |

decode（BS=32）：1M `full 1.3165` vs 基准 `1.3179`；32k `0.1180` vs `0.1181`。


---

# 总对比表

> 全部实测于 `.5` / B300 SXM6 (sm_103)。除非注明，形状为 Compass-V4
> `attn h_q=32/h_kv=4/d=128`、`index h_q=4/h_kv=1`、`B=1 causal`、indexer fp8。
> 分支 = `msa-compassv4-recheck`（下称"当前分支"）。

## 1. 主口径全流水 · Prefill（block=128 / topk=16, attn bf16）

| seq | idx(ms) | topk(ms) | attn(ms) | full(ms) | MSA dense(ms) | FA4(ms) | msa/FA4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.050 | 0.224 | 0.885 | 1.159 | 0.410 | 0.392 | 0.34× |
| 16,384 | 0.087 | 0.413 | 1.458 | 1.958 | 1.678 | 1.380 | 0.70× |
| 32,768 | 0.290 | 0.939 | 2.392 | 3.621 | 6.909 | 5.793 | 1.60× |
| 65,536 | 1.084 | 2.240 | 4.537 | 7.862 | 27.952 | 23.288 | 2.96× |
| 131,072 | 4.371 | 5.945 | 9.178 | 19.494 | 111.369 | 93.051 | 4.77× |
| 262,144 | 17.486 | 18.042 | 19.633 | 55.161 | 445.838 | 375.363 | 6.80× |
| 524,288 | 70.362 | 32.929 | 40.016 | 143.307 | 1780.951* | 1503.135 | 10.49× |
| 1,048,576 | 314.615 | 112.001 | 79.949 | 506.565 | 6198.502 | 6011.293 | 11.87× |

\* dense 分块测得，见「口径」。FA4 实测自 `.3`。

**分段占比**：attn 从 8k 的 76% 降到 1M 的 16%；1M 时 idx+topk 占 **84%**。

## 2. 主口径全流水 · Decode（Sq=1, BS=32, attn bf16）

| KV | idx(ms) | topk(ms) | attn(ms) | full(ms) | MSA dense(ms) | FA4(ms) | msa/FA4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.0236 | 0.0133 | 0.0402 | 0.0771 | 0.1035 | 0.1567 | 2.03× |
| 16,384 | 0.0298 | 0.0133 | 0.0418 | 0.0849 | 0.1834 | 0.2280 | 2.69× |
| 32,768 | 0.0419 | 0.0133 | 0.0400 | 0.0952 | 0.3290 | 0.3723 | 3.91× |
| 65,536 | 0.0658 | 0.0133 | 0.0412 | 0.1203 | 0.6236 | 0.6249 | 5.19× |
| 131,072 | 0.1074 | 0.0133 | 0.0402 | 0.1610 | 1.2155 | 1.2893 | 8.01× |
| 262,144 | 0.1834 | 0.0133 | 0.0402 | 0.2369 | 2.4065 | 2.5375 | 10.71× |
| 524,288 | 0.3412 | 0.0133 | 0.0400 | 0.3945 | 4.7839 | 5.1022 | 12.93× |
| 1,048,576 | 0.6604 | 0.0174 | 0.0419 | 0.7196 | 9.5878 | 10.2433 | 14.23× |

## 3. 配置扫描 · attn 段（h_q=32, 当前分支）

KV 预算 = block × topk。**只有 128×16 能跑通整条流水**，其余配置的 idx/topk 段受限（见「限制」）。

| seq | dtype | 128×16 | 128×8 | 64×16 | 8/16 比 |
|---:|---|---:|---:|---:|---:|
| 8,192 | bf16 | 0.7813 | 0.5919 | 0.8408 | 0.76× |
| 32,768 | bf16 | 2.4402 | 1.5135 | 2.3139 | 0.62× |
| 65,536 | bf16 | 4.5639 | 2.6706 | 4.4389 | 0.59× |
| 131,072 | bf16 | 9.1022 | 5.2060 | 9.3122 | 0.57× |
| 32,768 | fp8 | 2.3561 | 1.4652 | — | 0.62× |
| 65,536 | fp8 | 4.1722 | 2.5241 | — | 0.61× |

**结论**：成本跟「访问了多少个块」走，不跟「读了多少 token」走。
- topk 减半 → **0.57–0.76×**（有效）
- block 减半（同 topk，预算也减半）→ **0.94–1.03×**（无效）
- 同为 1024 token 预算：128×8 比 64×16 快 **1.42–1.79×**
- fp8 vs bf16 → **0.92–0.97×**（几乎无效）

## 4. 实现对比 · attn 段（block=128 / topk=16）

⚠️ fireworks-msa 必须**编译 C++ 扩展**才是它的真实性能。它的 `interface.py` 是
`backend = "cpp" if cpp_backend_available() else "python"`——**静默回退**。而
`setup.py` 把扩展构建包在 `try/except: ext_modules = []` 里，装的时候少任何一个
build 依赖就悄悄跳过。博客原话："AOT-exported per config and driven from a C++ op,
removing Python launch overhead that dominates end-to-end latency"。

| 实现 | dtype | 32k | 64k | vs 当前分支 |
|---|---|---:|---:|---:|
| **当前分支**（KV-outer, CuTe CSR） | bf16 | **2.4402** | **4.5639** | — |
| fireworks-msa（C++ AOT 后端） | bf16 | 2.5497 | 5.0186 | 1.045× / 1.100× 慢 |
| fireworks-msa（Python 回退） | bf16 | 3.2573 | 5.4922 | 1.33× / 1.20× 慢 |
| **当前分支** | fp8 | **2.3685** | **4.1841** | — |
| fireworks-msa（C++ AOT 后端） | fp8 | 2.4433 | 4.6275 | 1.032× / 1.106× 慢 |
| fireworks-msa（Python 回退） | fp8 | 2.7812 | 4.9665 | 1.17× / 1.19× 慢 |

编译 C++ 扩展后差距从 1.17–1.33× 收窄到 **1.03–1.11×**，与 nsys 测到的
**GPU kernel 时间只差 4.6%** 吻合（32k bf16，每次迭代）：

| | 主 attn kernel | combine | 索引构建 | GPU 合计 |
|---|---:|---:|---:|---:|
| 当前分支 | 1.338 | 0.739 | 0.051 | 2.13 |
| fireworks | 1.419 | 0.708 | 0.091 | 2.22 |

fireworks 的 combine 反而比我们**快 4%**；主 kernel 慢 6%。

**仍未复现博客宣称的 ~1.6×**，但剩下的 1.03–1.11× 已经落在"硬件与调优差异"的量级：
他们测 B200 (sm_100)，我们是 B300 (sm_103)；分支停在 `e688998` 也未必对应博客版本。

另一个未验证的口径差异：我们的 benchmark **每次迭代都重建索引**（两边都是）。
KV-outer 的元数据在真实推理里可跨层复用（M3 有 46 层），他们报的端到端
1.18–1.43× 很可能是摊销过的。要公平比这一项，得改成"建一次索引跑 N 次 attn"。

| Q-outer（CUTLASS `SparseAttnMode::Sparse`，2048-token 分块） | bf16 | 113.542 | — | 46.5× 慢 |

Q-outer 一行是上界不是判决：它走 decode 取向的 dispatch、每 2048 token 一次 launch，
正是"M 维塌缩成 GQA factor"的病症。两者输出等价（cos = 1.0000）。

## 4b. 按 Fireworks 的测法：module-latency 分段

他们博客不是给一个总数，而是把 index mapping / scheduling / combine / attention
拆成独立的条。按同样口径 nsys 拆开（bf16, Compass-V4, block=128/topk=16,
**每次迭代的 GPU 时间**, fireworks 走 C++ AOT 后端）:

| seq | 分段 | 当前分支(µs) | fireworks(µs) | fw/ours |
|---:|---|---:|---:|---:|
| 32k | attention kernel | 1338.1 | 1418.8 | 1.060× |
| 32k | combine | 738.3 | **709.2** | **0.961×** |
| 32k | index mapping + sched | 64.3 | 89.0 | 1.384× |
| 32k | **attn kernel 合计** | **2076.4** | **2128.0** | **1.025×** |
| 32k | module (GPU 合计) | 2140.7 | 2217.0 | 1.036× |
| 64k | attention kernel | 2469.4 | 2787.4 | 1.129× |
| 64k | combine | 1485.0 | **1424.0** | **0.959×** |
| 64k | index mapping + sched | 89.6 | 162.7 | 1.816× |
| 64k | **attn kernel 合计** | **3954.4** | **4211.4** | **1.065×** |
| 64k | module (GPU 合计) | 4044.0 | 4374.1 | 1.082× |

**他们的 combine 优化是真的有效**：两个长度上都比我们快 **4%**。这正对应博客里
"store partial-O in contiguous blocks instead of scattered writes; the combine
kernel does gathered loads" —— 把 scatter 推迟到带宽受限的 combine 阶段。

但在 B300 上他们的**主 attention kernel 更慢**（32k +6%、64k +13%），
**index mapping 更贵**（1.38× / 1.82×），净结果是 attn kernel 合计
**1.025× / 1.065×**，我们略优。

GPU 分段合计（2217 / 4374 µs）与实测 wall（2549.7 / 5018.6 µs）的比值一致，
说明编上 C++ 扩展之后 host 侧开销已经基本消失——这条也反过来印证了
「之前那 33% 是 Python 回退」的结论。

**他们宣称的 attention kernel ~1.6× vs 开源 MSA，在 B300 + Compass-V4 下没有复现**：
按他们自己的分段口径，我们在 attn kernel 上反而快 2.5–6.5%。

## 5. 限制（为什么只有 128×16 有全流水数）

| 段 | block=64 | topk=8 |
|---|---|---|
| idx | ✅ 可跑 | ✅ 与 topk 无关 |
| topk | ❌ 打分 tile 固定 128，`num_valid_pages=512 > max_k_tiles=256` | ❌ `sparse_topk_select` 断言 `topk == 16` |
| attn | ✅ 已移植 CSR builder 的 blk 32/64/128 实例化 | ✅ 本来就支持 |

要补齐需移植 `msa-configurable-sparse-attention` 的 `fp4_indexer_interface.py`(+235)
与 `sparse_topk_select.cu`(+51，函数签名有变、会影响 `api.py` 调用点)。

按「idx/topk 两段不变」外推的 topk=8 全流水（**推算，非实测**）：
32k `2.746ms`（1.34×）、64k `5.993ms`（1.30×）。

## 6. 环境

| 用途 | 位置 | 关键版本 |
|---|---|---|
| MSA 主口径 | `.5` `dsa_stage2_meng2` + `/sparse/msa_venv` | cutlass-dsl **4.4.1** / quack **0.2.10** |
| fireworks-msa | `.5` `/sparse/msa_fw` + `/sparse/fw_venv` | cutlass-dsl 4.5.2 / quack 0.4.1 / flash-attn-4 4.0.0b15；**必须 `pip install -e . --no-build-isolation` 在依赖装齐之后重装**，否则 C++ 扩展静默跳过 |
| FA4 基线 | `.3` `sparse_atten_meng` + `fa4_venv` | 只能在 `.3` 测 |
| FlashInfer `msa_ops` | `.5` `/sparse/fi_venv` | 0.6.17，**B300 跑不了**（要 SM120/121） |

⚠️ cutlass-dsl 版本影响 attn kernel：4.6.0 比 pin 的 4.4.1 慢 6–14%，跨环境数字不可直接比。
