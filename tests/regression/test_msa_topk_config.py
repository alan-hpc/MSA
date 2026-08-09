#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness regression for the configurable MSA top-k selection kernel.

Covers the v3.0_msa_config surface added to ``sparse_topk_select``:

* runtime ``topk`` (the kernel used to be pinned to 16),
* fused GQA head aggregation (``head_mode`` ``keep`` / ``sum`` / ``max``),
* forced sink / local windows, including the per-query causal variant,
* interaction with the existing ``num_valid_pages`` out-of-range clamp,
* both transpose paths (the float4 XOR-swizzle fast path and the padded-tile
  fallback), selected by whether ``total_qo_len`` and ``max_k_tiles`` are
  multiples of 32.

Every case is checked against a straightforward PyTorch reference.  Scores are
drawn from a continuous distribution and de-duplicated so the only ties are the
``FLT_MAX`` forced blocks, whose selection is order-independent — that keeps the
comparison exact rather than approximate.

Run directly (``python tests/regression/test_msa_topk_config.py``) or under
pytest.  Requires an SM100/SM103 GPU.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from fmha_sm100 import sparse_topk_select  # noqa: E402
from fmha_sm100.msa_config import DEFAULT_MODEL, MsaSparseConfig  # noqa: E402

# Selection is checked at the target model's GQA geometry so head aggregation is
# exercised with the group size production will actually use.
NUM_KV_HEADS = DEFAULT_MODEL.num_kv_heads
GQA_GROUP = DEFAULT_MODEL.qhead_per_kv

FLT_MAX = 3.4028234663852886e38


def make_scores(num_in_heads, max_k_tiles, total_q, num_valid_pages, *, seed):
    """Random block scores with ``-inf`` padding past ``num_valid_pages``.

    Values are made pairwise distinct so top-k has a unique answer; the kernel
    and the reference are then required to agree exactly.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    numel = num_in_heads * max_k_tiles * total_q
    # A random permutation gives distinct values; scale into a range wide enough
    # that fp32 keeps every one of them distinct.
    scores = torch.randperm(numel, generator=gen, device="cuda", dtype=torch.float32)
    scores = (scores / numel - 0.5) * 20.0
    scores = scores.view(num_in_heads, max_k_tiles, total_q).contiguous()
    if num_valid_pages < max_k_tiles:
        scores[:, num_valid_pages:, :] = float("-inf")
    return scores


def reference_select(
    scores,
    *,
    topk,
    num_valid_pages,
    force_begin,
    force_end,
    head_mode,
    num_out_heads,
    causal_end_block=None,
):
    """PyTorch mirror of ``sparse_topk_select``; returns ``[Q, H, topk]`` int32."""
    num_in_heads, max_k_tiles, total_q = scores.shape
    group = num_in_heads // num_out_heads
    if head_mode == "keep":
        reduced = scores
    elif head_mode == "sum":
        reduced = scores.view(num_out_heads, group, max_k_tiles, total_q).sum(dim=1)
    elif head_mode == "max":
        reduced = scores.view(num_out_heads, group, max_k_tiles, total_q).amax(dim=1)
    else:
        raise ValueError(head_mode)

    # -> [H, Q, K] to match the kernel's transposed working layout.
    reduced = reduced.permute(0, 2, 1).contiguous()

    k_idx = torch.arange(max_k_tiles, device=scores.device, dtype=torch.int32)
    if causal_end_block is None:
        q_end = torch.full((total_q,), num_valid_pages, device=scores.device, dtype=torch.int32)
    else:
        q_end = causal_end_block.clamp(min=0, max=num_valid_pages)
    # [Q, K] forced mask: sink window, plus the local window below each query's
    # own causal bound.
    sink = k_idx.unsqueeze(0) < force_begin
    local = (k_idx.unsqueeze(0) < q_end.unsqueeze(1)) & (
        k_idx.unsqueeze(0).to(torch.int64) + force_end >= q_end.unsqueeze(1).to(torch.int64)
    )
    forced = sink | local
    reduced = torch.where(forced.unsqueeze(0), torch.tensor(FLT_MAX, device=scores.device), reduced)

    idx = reduced.topk(topk, dim=-1).indices.to(torch.int32)          # [H, Q, topk]
    idx = torch.where(idx < num_valid_pages, idx, torch.full_like(idx, -1))
    # Ascending by index with the -1 sentinels pushed to the tail, matching the
    # kernel's fused warp bitonic sort on unsigned keys.
    sort_key = torch.where(idx < 0, torch.full_like(idx, 2**31 - 1), idx)
    order = sort_key.argsort(dim=-1, stable=True)
    idx = idx.gather(-1, order)
    return idx.permute(1, 0, 2).contiguous()                          # [Q, H, topk]


def check_case(
    *,
    num_out_heads,
    group,
    max_k_tiles,
    total_q,
    num_valid_pages,
    topk,
    force_begin,
    force_end,
    head_mode,
    causal_end,
    seed,
    head_outermost=False,
):
    num_in_heads = num_out_heads * (1 if head_mode == "keep" else group)
    scores = make_scores(num_in_heads, max_k_tiles, total_q, num_valid_pages, seed=seed)

    causal_end_block = None
    if causal_end:
        gen = torch.Generator(device="cuda").manual_seed(seed + 977)
        causal_end_block = torch.randint(
            1, num_valid_pages + 1, (total_q,), generator=gen, device="cuda", dtype=torch.int32
        ).contiguous()

    got = sparse_topk_select(
        scores,
        topk,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin,
        force_end_blocks=force_end,
        head_mode=head_mode,
        num_out_heads=num_out_heads,
        causal_end_block=causal_end_block,
        head_outermost=head_outermost,
    )
    if head_outermost:
        assert got.shape == (num_out_heads, total_q, topk), tuple(got.shape)
        got = got.permute(1, 0, 2).contiguous()
    want = reference_select(
        scores,
        topk=topk,
        num_valid_pages=num_valid_pages,
        force_begin=force_begin,
        force_end=force_end,
        head_mode=head_mode,
        num_out_heads=num_out_heads,
        causal_end_block=causal_end_block,
    )

    assert got.shape == want.shape, f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    if not torch.equal(got, want):
        bad = (got != want).any(dim=-1).nonzero()
        q, h = (int(v) for v in bad[0])
        raise AssertionError(
            f"mismatch at (q={q}, h={h})\n  got  {got[q, h].tolist()}\n  want {want[q, h].tolist()}"
        )

    # Contract checks that do not depend on the reference.
    valid = got[got >= 0]
    assert valid.numel() == 0 or int(valid.max()) < num_valid_pages, "index past num_valid_pages"
    order_ok = ((got[..., 1:] > got[..., :-1]) | (got[..., 1:] < 0)).all()
    assert bool(order_ok), "output must be strictly ascending until the -1 tail"


def iter_cases():
    """(name, kwargs) pairs covering the new surface."""
    base = dict(
        num_out_heads=NUM_KV_HEADS,
        group=GQA_GROUP,
        max_k_tiles=256,
        total_q=64,
        num_valid_pages=200,
        topk=16,
        force_begin=0,
        force_end=0,
        head_mode="keep",
        causal_end=False,
        seed=0,
    )

    # 1. Legacy contract: topk=16, keep, no forcing.
    yield "legacy_topk16", dict(base)

    # 2. Runtime topk, including the value the kernel could not do before.
    for topk in (4, 8, 16, 32):
        yield f"topk{topk}", {**base, "topk": topk, "seed": topk}

    # 3. Head aggregation, both reduction modes and both transpose paths.
    for head_mode, (total_q, max_k_tiles) in itertools.product(
        ("keep", "sum", "max"), ((64, 256), (48, 200))
    ):
        path = "xorf4" if (total_q % 32 == 0 and max_k_tiles % 32 == 0) else "fallback"
        yield f"head_{head_mode}_{path}", {
            **base,
            "head_mode": head_mode,
            "total_q": total_q,
            "max_k_tiles": max_k_tiles,
            "num_valid_pages": max_k_tiles - 56,
            "seed": hash((head_mode, total_q)) % 10_000,
        }

    # 4. Forced windows against the global sequence end (decode semantics).
    for force_begin, force_end in ((4, 0), (0, 4), (4, 4), (8, 8)):
        yield f"force_{force_begin}_{force_end}", {
            **base,
            "force_begin": force_begin,
            "force_end": force_end,
            "seed": 100 + force_begin * 16 + force_end,
        }

    # 5. Forced windows against per-query causal ends (prefill semantics).
    for head_mode in ("keep", "sum", "max"):
        yield f"causal_force_{head_mode}", {
            **base,
            "head_mode": head_mode,
            "force_begin": 4,
            "force_end": 4,
            "causal_end": True,
            "seed": 200 + len(head_mode),
        }

    # 6. The requested 12-point matrix at a realistic block count, exercising
    #    force windows and head modes together with topk 16/32.
    for cfg in _matrix_cases():
        yield cfg

    # 7. head_outermost output layout must be an exact transpose of the legacy
    #    layout, on both the top-k path and the trivial identity-fill path.
    for head_mode in ("keep", "sum", "max"):
        yield f"head_outermost_{head_mode}", {
            **base,
            "head_mode": head_mode,
            "topk": 32,
            "force_begin": 4,
            "force_end": 4,
            "causal_end": True,
            "head_outermost": True,
            "seed": 300 + len(head_mode),
        }
    yield "head_outermost_trivial", {
        **base, "max_k_tiles": 16, "num_valid_pages": 12, "topk": 16, "head_outermost": True,
    }

    # 8. Degenerate / boundary shapes.
    yield "trivial_all_blocks", {**base, "max_k_tiles": 16, "num_valid_pages": 12, "topk": 16}
    yield "nvp_equals_ktiles", {**base, "num_valid_pages": 256, "force_begin": 2, "force_end": 2}
    yield "single_query", {**base, "total_q": 1, "max_k_tiles": 64, "num_valid_pages": 50}


def _matrix_cases():
    from fmha_sm100.msa_config import CONFIG_MATRIX

    # 8192-token context: 256 blocks at block=32, 128 at block=64.
    seqlen_k = 8192
    for cfg in CONFIG_MATRIX:
        num_blocks = cfg.num_blocks(seqlen_k)
        yield f"matrix_{cfg.name}_{cfg.slug}", dict(
            num_out_heads=NUM_KV_HEADS,
            group=GQA_GROUP,
            max_k_tiles=num_blocks,
            total_q=32,
            num_valid_pages=num_blocks - 3,
            topk=cfg.topk,
            force_begin=cfg.force_init_blocks,
            force_end=cfg.force_end_blocks,
            head_mode=cfg.head_mode,
            causal_end=True,
            seed=int(cfg.name[3:]) * 31,
        )


def test_msa_topk_config():
    """pytest entry point."""
    assert torch.cuda.is_available(), "CUDA device required"
    for name, kwargs in iter_cases():
        check_case(**kwargs)


def _validate_config_guardrails():
    """The config layer must reject windows that cannot fit in the budget."""
    try:
        MsaSparseConfig(
            block_size=32, topk=4, force_init_tokens=128, force_end_tokens=128
        ).validate()
    except ValueError:
        return
    raise AssertionError("MsaSparseConfig.validate() should reject an over-subscribed budget")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    torch.cuda.set_device(0)
    print(f"=== MSA configurable top-k selection regression ({torch.cuda.get_device_name(0)}) ===")
    print(f"    geometry: {DEFAULT_MODEL.name}  Hkv={NUM_KV_HEADS} GQA group={GQA_GROUP}")
    failures = []
    for name, kwargs in iter_cases():
        try:
            check_case(**kwargs)
        except Exception as exc:  # noqa: BLE001 - report every case, fail at the end
            failures.append((name, exc))
            print(f"  [FAIL] {name}: {exc}")
        else:
            print(f"  [PASS] {name}")
    _validate_config_guardrails()
    print("  [PASS] config guardrails")
    if failures:
        print(f"\n{len(failures)} case(s) FAILED")
        return 1
    print("\nAll MSA top-k configuration tests PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
