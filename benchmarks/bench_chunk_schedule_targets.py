"""Sweep sparse-attention schedule granularity for chunked prefill.

The selected ``(query, page)`` pairs stay identical.  Only
``target_q_per_cta`` changes, isolating whether short prefill chunks lose time
because the automatic scheduler creates too many work items.
"""

import os
import statistics

import torch


HQ, HKV, D, PAGE, TOPK = 32, 4, 128, 128, 16
LENGTH = int(os.environ.get("KLEN", "32768"))
CHUNK = int(os.environ.get("CHUNK", "8192"))
POOL = int(os.environ.get("POOL", "1"))
ITERS = int(os.environ.get("ITERS", "30"))
SAMPLES = int(os.environ.get("SAMPLES", "5"))
TARGETS = [
    None if value == "auto" else int(value)
    for value in os.environ.get(
        "TARGETS", "auto,1024,1536,2048,3072,4096,6144,8192"
    ).split(",")
]


def timed(fn) -> float:
    for _ in range(5):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / ITERS


def main() -> None:
    from vllm.model_executor.layers.compass_sparse_attn_msa import (
        _msa,
        msa_paged_kv_view,
    )

    mods = _msa()
    from src.sm100.prepare_scheduler import SPARSE_SCHEDULE_MODEL

    automatic_target = SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta
    generator = torch.Generator(device="cuda").manual_seed(0)
    pages = -(-LENGTH // PAGE)
    kv = torch.randn(
        (pages * POOL, HKV, PAGE, 2 * D),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    view = kv.transpose(1, 2)
    key_cache = msa_paged_kv_view(view[..., :D])
    value_cache = msa_paged_kv_view(view[..., D:])
    page_table = (
        torch.arange(pages, device="cuda", dtype=torch.int32) * POOL
    ).view(1, -1)
    q = torch.randn(
        (LENGTH, HQ, D),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    own_page = torch.arange(LENGTH, device="cuda") // PAGE
    offsets = torch.arange(TOPK, device="cuda")
    random_values = torch.rand(
        (LENGTH, TOPK), generator=generator, device="cuda"
    )
    selected = (
        random_values * (own_page + 1).view(-1, 1).float()
    ).floor().to(torch.int32)
    selected, _ = torch.sort(selected, dim=-1)
    selected = torch.where(
        own_page.view(-1, 1) >= offsets.view(1, -1),
        selected,
        torch.full_like(selected, -1),
    )
    selected = selected.unsqueeze(0).expand(HKV, LENGTH, TOPK).contiguous()

    def build(target: int | None):
        if target is None:
            SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta = automatic_target
        else:
            SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta = lambda **_: target

        pieces = []
        targets_used = []
        work_counts = []
        for lo in range(0, LENGTH, CHUNK):
            hi = min(lo + CHUNK, LENGTH)
            query = q[lo:hi].contiguous()
            selection = selected[:, lo:hi].contiguous()
            cu_q = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
            cu_k = torch.tensor([0, hi], dtype=torch.int32, device="cuda")
            used_k = torch.tensor([hi], dtype=torch.int32, device="cuda")
            row_ptr, q_indices, schedule = mods["build_k2q_csr"](
                selection,
                cu_q,
                cu_k,
                PAGE,
                total_k=hi,
                max_seqlen_k=hi,
                max_seqlen_q=hi - lo,
                total_rows=-(-hi // PAGE),
                qhead_per_kv=HQ // HKV,
                return_schedule=True,
            )
            targets_used.append(schedule.target_q_per_cta)
            work_counts.append(int(schedule.work_count.item()))
            pieces.append(
                (
                    query,
                    row_ptr,
                    q_indices,
                    schedule,
                    cu_q,
                    cu_k,
                    used_k,
                    hi,
                    hi - lo,
                )
            )
        return pieces, targets_used, work_counts

    def run(pieces, *, concatenate: bool = True):
        outputs = []
        for query, row_ptr, q_indices, schedule, cu_q, cu_k, used_k, hi, qlen in pieces:
            outputs.append(
                mods["sparse_atten_func"](
                    query,
                    key_cache,
                    value_cache,
                    row_ptr,
                    q_indices,
                    TOPK,
                    blk_kv=PAGE,
                    causal=True,
                    softmax_scale=D**-0.5,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_k=cu_k,
                    max_seqlen_q=qlen,
                    max_seqlen_k=hi,
                    schedule=schedule,
                    page_table=page_table,
                    seqused_k=used_k,
                )
            )
        return torch.cat(outputs) if concatenate else outputs

    results = []
    reference = None
    for target in TARGETS:
        pieces, targets_used, work_counts = build(target)
        output = run(pieces)
        torch.cuda.synchronize()
        if reference is None:
            reference = output
            cosine = 1.0
            max_abs = 0.0
        else:
            cosine = torch.nn.functional.cosine_similarity(
                output.float().flatten(), reference.float().flatten(), dim=0
            ).item()
            max_abs = (output.float() - reference.float()).abs().max().item()
        samples = [
            timed(lambda: run(pieces, concatenate=False)) for _ in range(SAMPLES)
        ]
        results.append(
            (
                "auto" if target is None else str(target),
                targets_used,
                work_counts,
                statistics.median(samples),
                cosine,
                max_abs,
            )
        )

    SPARSE_SCHEDULE_MODEL.balanced_target_q_per_cta = automatic_target
    print(
        f"device={torch.cuda.get_device_name(0)} length={LENGTH} chunk={CHUNK} "
        f"pool={POOL} pairs={int((selected[0] >= 0).sum())}"
    )
    print("target | targets/chunk | work/chunk | median_ms | cosine | max_abs")
    for label, targets_used, work_counts, elapsed, cosine, max_abs in results:
        print(
            f"{label:>6} | {targets_used!s:<25} | {work_counts!s:<24} | "
            f"{elapsed:>9.4f} | {cosine:.8f} | {max_abs:.6f}"
        )


if __name__ == "__main__":
    main()
