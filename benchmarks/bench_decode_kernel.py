"""Kernel-to-kernel decode comparison: the same kernel, with and without sparsity.

Every other decode number in this repo compares things that differ in more than
one way -- a sparse attention kernel against a dense one built by other people
with other tradeoffs, or a single kernel against another row's end-to-end cost.
Neither isolates what sparsity is worth.

This does.  One kernel, one set of inputs, one difference: the page table lists
every page, or it lists only the selected ones.  Everything else -- dtype, page
size, layout, launch configuration, the query tensor itself -- is byte for byte
identical between the two measurements, so the ratio is the sparsity effect and
cannot be anything else.

Inputs and reference come from the repo's test suite rather than being rebuilt
here.  Five separate wrong-cosine incidents on this path all traced to inputs
written by hand for a benchmark, never to a kernel.

Run under the interpreter that has the decode kernel's cutlass (see
benchmark.sh's DECODE_PYTHON), from the repo root:

    python benchmarks/bench_decode_kernel.py --seqlens 32768,131072,524288
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_test_module():
    """Import the test suite for its input builder and reference.

    It imports pytest at module scope only for decorators, so a stub is enough
    and saves making pytest a benchmark dependency.
    """
    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(REPO_ROOT / "python" / "fmha_sm100" / "cute"))
    if "pytest" not in sys.modules:
        stub = types.ModuleType("pytest")
        stub.skip = lambda *a, **k: None

        class _Mark:
            def __getattr__(self, _name):
                return lambda *a, **k: (a[0] if a and callable(a[0])
                                        else (lambda f: f))

        stub.mark = _Mark()
        stub.approx = lambda x, **k: x
        sys.modules["pytest"] = stub
    import test_sparse_atten as T  # noqa: PLC0415 - needs the paths above
    return T


def _median_ms(fn, *, flush, warmup=5, iters=15):
    """Median wall time with L2 flushed between iterations.

    Decode reads far more than L2 holds, so a warm cache would only flatter the
    sparse side -- it is the one whose working set could fit.
    """
    for _ in range(warmup):
        flush.zero_()
        fn()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _plan_and_run(T, inputs, *, seqlen_q):
    """Bind a kernel to one page table.  plan() is hoisted out of the timed
    region deliberately: it depends on shapes and seqused_k, both fixed across a
    decode loop, so timing it here would measure setup a real loop pays once."""
    q = inputs["q"]
    fn = T._get_sparse_decode_atten_func_for_benchmark()
    fn.plan(
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        seqlen_q=seqlen_q,
        max_seqlen_k=inputs["max_seqlen_k"],
        num_qo_heads=q.shape[1],
        num_kv_heads=inputs["k_paged"].shape[1],
        head_dim=q.shape[2],
    )
    return lambda: fn.run(q, inputs["k_paged"], inputs["v_paged"],
                          softmax_scale=inputs["softmax_scale"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlens", default="32768,131072,524288,1048576")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head-kv", type=int, default=4)
    ap.add_argument("--qhead-per-kv", type=int, default=8,
                    help="model-n32 is 8; the packed-q tile fixes seqlen_q at 128//this")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=16, help="selected pages per request")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if 128 % args.qhead_per_kv:
        print(f"qhead_per_kv ({args.qhead_per_kv}) must divide the 128-row packed-q tile")
        return 2

    T = _load_test_module()
    page = T.BLK_KV
    seqlen_q = 128 // args.qhead_per_kv
    batch = args.batch
    flush = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    print(f"batch={batch} Hkv={args.head_kv} qhead_per_kv={args.qhead_per_kv} "
          f"D={args.dim} page={page} seqlen_q={seqlen_q} topk={args.topk}")
    print("same kernel, same inputs; the page table is the only difference\n")
    head = (f"{'KV':>7} {'all pages':>10} {'topk only':>10} {'speedup':>9} "
            f"{'KV read':>9} {'ideal':>7} {'worst cos':>10}")
    print(head)
    print("-" * len(head))

    for seqlen_k in [int(x) for x in args.seqlens.split(",") if x.strip()]:
        pages = seqlen_k // page
        if args.topk >= pages:
            print(f"{seqlen_k//1024:>6}K  topk covers every page; nothing to compare")
            continue

        inputs = T._build_decode_paged_dense_inputs(
            kv_tokens=seqlen_k, batch=batch, seqlen_q=seqlen_q,
            head_kv=args.head_kv, qhead_per_kv=args.qhead_per_kv, dim=args.dim)

        gen = torch.Generator().manual_seed(args.seed)
        selection = torch.stack([
            torch.sort(torch.randperm(pages, generator=gen)[:args.topk]).values
            for _ in range(batch)]).cuda()

        # The only change.  gather maps a selection index to the physical page
        # id; computing that id arithmetically is what broke this once already.
        sparse = dict(inputs)
        sparse["page_table"] = torch.gather(
            inputs["page_table"], 1, selection).contiguous().to(torch.int32)
        sparse["kv_tokens"] = sparse["max_seqlen_k"] = args.topk * page
        sparse["seqused_k"] = torch.full((batch,), args.topk * page,
                                         dtype=torch.int32, device="cuda")

        run_sparse = _plan_and_run(T, sparse, seqlen_q=seqlen_q)
        out = run_sparse()
        torch.cuda.synchronize()
        out = out[0] if isinstance(out, (tuple, list)) else out
        ref, _ = T._decode_paged_dense_reference(sparse)
        cos = float(torch.nn.functional.cosine_similarity(
            out.float(), ref.float(), dim=-1).min())

        ms_sparse = _median_ms(run_sparse, flush=flush)
        ms_all = _median_ms(_plan_and_run(T, inputs, seqlen_q=seqlen_q),
                            flush=flush, iters=7)

        # Bytes the two runs actually touch, so the measured speedup can be read
        # against the one sparsity alone would buy.
        read_ratio = pages / args.topk
        print(f"{seqlen_k//1024:>6}K {ms_all:>9.4f}m {ms_sparse:>9.4f}m "
              f"{ms_all/ms_sparse:>8.1f}x {read_ratio:>8.0f}x {read_ratio:>6.0f}x "
              f"{cos:>10.5f}")

        del inputs, sparse, out, ref
        torch.cuda.empty_cache()

    print("\n'KV read' is how many times more KV the all-pages run touches; a "
          "speedup below it\nis fixed overhead the sparse run cannot amortise, "
          "not lost sparsity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
