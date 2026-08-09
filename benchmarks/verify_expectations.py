#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Check a benchmark.sh run against what the optimisation report predicts.

A sweep this large produces failures by design — three separate hardware or
kernel limits are reachable inside the configured grid, and the report names
each one and where it should bind.  Eyeballing several hundred cells for
"looks right" does not distinguish a limit behaving as documented from a
regression that happens to fail in a similar place, so this asserts the
prediction instead:

  * every cell the report says should run, ran;
  * every cell it says should fail, failed **for the stated reason**;
  * nothing failed that was not predicted.

Anything in the third category is the interesting output — it is a finding the
report does not yet cover.

Usage:  verify_expectations.py <results-dir>
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

#: Q/O tensors are indexed with 32-bit arithmetic (report §18).
QO_INT32_LIMIT = 2**31
#: FP4 indexer score tensor, same class of limit (report §2).
SCORE_INT32_LIMIT = 2**31
#: top-k implements only the insertion-sort path (report §10).
MAX_K_TILES = 12288
#: build_k2q_csr keeps a 2-byte histogram entry per CSR row in shared memory (§9b).
CSR_MAX_ROWS = 116224
#: the FP4 scale-reorder kernel's own 32-bit indexing limit (report §19).
SCALE_REORDER_LIMIT = 2**30
#: nvfp4 scale groups per 128-element page row.
SCALE_GROUPS = 8

HEAD_Q, HEAD_KV, HEAD_DIM = 32, 4, 128


def predict(*, mode, block, topk, head_mode, seqlen, batch):
    """Which stage should fail first at this cell, or None if it should run."""
    total_q = batch * (1 if mode == "decode" else seqlen)
    num_blocks = math.ceil(seqlen / block)

    # Stage order matters: the sweep runs indexer -> topk -> csr -> attn and
    # records every stage, so predict the full set, not just the first.
    failures = {}

    scorer_heads = HEAD_KV  # every head_mode scores with one proxy query per KV head
    score_elems = scorer_heads * math.ceil(seqlen / 128) * total_q
    if score_elems > SCORE_INT32_LIMIT:
        failures["indexer"] = f"score tensor {score_elems} > {SCORE_INT32_LIMIT}"
    else:
        # The indexer stage also builds its MMA-ordered scales, which has a
        # separate and lower 32-bit ceiling; it binds first at large batch.
        # The K side is sized by the number of *selections*, so sum/max --
        # which collapse to one shared selection -- reach it Hkv times later
        # than keep does.
        sel_heads = HEAD_KV if head_mode == "keep" else 1
        pages = batch * math.ceil(seqlen / 128)
        scale_elems = pages * sel_heads * 128 * SCALE_GROUPS
        if scale_elems >= SCALE_REORDER_LIMIT:
            failures["indexer"] = f"scale reorder {scale_elems} >= {SCALE_REORDER_LIMIT}"

    if num_blocks >= MAX_K_TILES:
        failures["topk"] = f"max_k_tiles {num_blocks} >= {MAX_K_TILES}"

    # Stage dependency: the CSR build consumes top-k's output, so a failed
    # top-k means CSR never runs and cannot report an error of its own.  The
    # indexer is independent -- the sweep falls back to synthetic block scores
    # and carries on -- so its failure does not mask anything downstream.
    csr_rows = batch * num_blocks
    if csr_rows > CSR_MAX_ROWS and "topk" not in failures:
        failures["csr"] = f"csr rows {csr_rows} > {CSR_MAX_ROWS}"

    # Memory wall (report §20).  Measured, not derived: at 512K prefill,
    # block=64/topk=32 exhausts a 268 GiB card even alone in a fresh process,
    # while block=128 at the same topk completes.  Something on the
    # selection->attention path scales with 1/block; the exact allocation is
    # not yet pinned down, so this encodes the observed boundary and should be
    # re-derived if the grid changes.  Attention only runs if top-k and CSR
    # both produced their outputs, so it can only fail when they did not.
    reached_attn = "topk" not in failures and "csr" not in failures
    if reached_attn and mode == "prefill" and seqlen >= 524288 and block <= 64 and topk >= 32:
        failures["attn"] = "out of memory: block<=64 with topk>=32 at 512K prefill"

    return failures


def predict_dense(*, mode, seqlen, batch):
    total_q = batch * (1 if mode == "decode" else seqlen)
    elems = total_q * HEAD_Q * HEAD_DIM
    if elems >= QO_INT32_LIMIT:
        return f"Q/O tensor {elems} >= {QO_INT32_LIMIT}"
    return None


def check(csv_path: Path, mode: str, batch: int):
    if not csv_path.exists():
        return [f"{csv_path.name}: MISSING"], 0, 0
    rows = list(csv.DictReader(open(csv_path)))
    problems, checked, expected_fail = [], 0, 0

    for r in rows:
        name = r["name"]
        seqlen = int(r["seqlen_k"])
        errors = json.loads(r["errors"]) if r.get("errors") else {}
        errors = {k: v for k, v in errors.items() if k != "cos"}
        checked += 1

        if name.startswith("dense"):
            want = predict_dense(mode=mode, seqlen=seqlen, batch=batch)
            if name == "dense-fa4":
                # FA4 is verified correct past 1.5x the boundary (report §18).
                want = None
            if want and not errors:
                problems.append(f"{name} @{seqlen}: expected to fail ({want}) but ran")
            elif want and errors:
                expected_fail += 1
                if "ValueError" not in str(errors):
                    problems.append(f"{name} @{seqlen}: failed but not via the guard: {errors}")
            elif not want and errors:
                problems.append(f"{name} @{seqlen}: UNEXPECTED FAILURE {errors}")
            continue

        want = predict(mode=mode, block=int(r["block_size"]), topk=int(r["topk"]),
                       head_mode=r["head_mode"], seqlen=seqlen, batch=batch)
        got = set(errors)
        if want:
            expected_fail += 1
        missing = set(want) - got
        extra = got - set(want)
        if missing:
            problems.append(f"{name} @{seqlen}: expected {sorted(missing)} to fail, it ran "
                            f"({'; '.join(want[k] for k in missing)})")
        if extra:
            problems.append(f"{name} @{seqlen}: UNPREDICTED failure in {sorted(extra)}: "
                            + "; ".join(f"{k}={errors[k][:90]}" for k in sorted(extra)))
    return problems, checked, expected_fail


def check_kvouter(ex: Path):
    problems = []
    qf, kf = ex / "qouter.json", ex / "kvouter.json"
    if not qf.exists() or not kf.exists():
        return ["kvouter A/B: results missing"]
    q = {(r["seqlen"], r["topk"]): r for r in json.loads(qf.read_text())}
    kv = {(r["seqlen"], r["topk"]): r for r in json.loads(kf.read_text())}
    for key in sorted(set(q) | set(kv)):
        a, b = q.get(key), kv.get(key)
        if not (a and a.get("ok")):
            problems.append(f"kvouter A/B {key}: q-outer failed: {(a or {}).get('error')}")
            continue
        if not (b and b.get("ok")):
            problems.append(f"kvouter A/B {key}: kv-outer failed: {(b or {}).get('error')}")
            continue
        # The two backends compute the same attention over the same selection.
        if b.get("cos_mean", 0) < 0.999:
            problems.append(f"kvouter A/B {key}: outputs diverge, cos={b['cos_mean']:.5f}")
    return problems


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_expectations.py <results-dir>", file=sys.stderr)
        return 2
    out = Path(sys.argv[1])
    all_problems = []

    print("=" * 78)
    print("Checking the run against the optimisation report's predictions")
    print("=" * 78)

    for label, fname, mode, batch in (
        ("baseline", "baseline.csv", "prefill", 1),
        ("prefill sweep", "sweep.csv", "prefill", 1),
        ("decode sweep", "decode.csv", "decode", 32),
    ):
        problems, checked, expected_fail = check(out / fname, mode, batch)
        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"\n{label:<16} {checked:>4} cells, {expected_fail:>3} predicted to fail  -> {status}")
        for p in problems:
            print(f"   ! {p}")
        all_problems += problems

    kvp = check_kvouter(out / "kvab")
    print(f"\n{'kv-outer A/B':<16} {'':>4}                                 -> "
          + ("OK" if not kvp else f"{len(kvp)} PROBLEM(S)"))
    for p in kvp:
        print(f"   ! {p}")
    all_problems += kvp

    print("\n" + "=" * 78)
    if all_problems:
        print(f"VERDICT: {len(all_problems)} deviation(s) from the documented behaviour.")
        print("Each one is either a regression or a limit the report does not yet cover.")
        return 1
    print("VERDICT: every cell matched the prediction — the ones that ran, and the")
    print("         ones that failed, each for the reason the report states.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
