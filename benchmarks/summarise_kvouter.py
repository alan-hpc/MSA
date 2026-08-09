#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Render the q-outer vs KV-outer A/B produced by ``bench_kvouter.py``.

Reads ``qouter.json`` and ``kvouter.json`` from the exchange directory the two
roles shared and prints one table.  Also writes ``kvouter_ab.csv`` next to them
so the numbers can be diffed across runs.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarise_kvouter.py <exchange-dir>", file=sys.stderr)
        return 2
    ex = Path(sys.argv[1])
    qf, kf = ex / "qouter.json", ex / "kvouter.json"
    if not qf.exists() or not kf.exists():
        print("KV-outer A/B: no results to summarise "
              f"({'qouter.json' if not qf.exists() else 'kvouter.json'} missing)")
        return 0

    q = {(r["seqlen"], r["topk"]): r for r in json.loads(qf.read_text())}
    kv = {(r["seqlen"], r["topk"]): r for r in json.loads(kf.read_text())}
    keys = sorted(set(q) | set(kv))
    if not keys:
        print("KV-outer A/B: no shapes ran")
        return 0

    print()
    print("Table 4  q-outer (this repo) vs KV-outer (fireworks-msa), block=128")
    print("=" * 104)
    print(f"{'kv_len':>8} {'topk':>5} | {'csr':>8} {'attn':>9} {'q-outer':>9} | "
          f"{'kv-outer':>9} | {'speedup':>8} | {'cos':>8}  note")
    print("-" * 104)

    rows = []
    for seqlen, topk in keys:
        a, b = q.get((seqlen, topk)), kv.get((seqlen, topk))
        rec = {"seqlen": seqlen, "topk": topk, "block": 128}
        if not (a and a.get("ok")):
            why = (a or {}).get("error", "not run")
            print(f"{seqlen:>8} {topk:>5} | {'':>8} {'':>9} {'X':>9} | {'-':>9} | {'-':>8} | "
                  f"{'-':>8}  q-outer failed: {why[:40]}")
            rows.append({**rec, "note": f"q-outer failed: {why}"})
            continue
        if not (b and b.get("ok")):
            why = (b or {}).get("error", "not run")
            print(f"{seqlen:>8} {topk:>5} | {a['csr_ms']:>8.3f} {a['attn_ms']:>9.3f} "
                  f"{a['total_ms']:>9.3f} | {'X':>9} | {'-':>8} | {'-':>8}  "
                  f"kv-outer failed: {why[:40]}")
            rows.append({**rec, "qouter_ms": a["total_ms"], "note": f"kv-outer failed: {why}"})
            continue

        speed = a["total_ms"] / b["total_ms"]
        # The two backends compute the same attention over the same selection, so
        # anything below ~0.999 is a real numerical divergence, not sparsity loss.
        note = "kv-outer faster" if speed > 1.02 else ("q-outer faster" if speed < 0.98 else "tie")
        if b.get("cos_mean", 1.0) < 0.999:
            note += " -- OUTPUTS DIVERGE, ratio not comparable"
        print(f"{seqlen:>8} {topk:>5} | {a['csr_ms']:>8.3f} {a['attn_ms']:>9.3f} "
              f"{a['total_ms']:>9.3f} | {b['total_ms']:>9.3f} | {speed:>7.2f}x | "
              f"{b.get('cos_mean', float('nan')):>8.5f}  {note}")
        rows.append({**rec, "csr_ms": a["csr_ms"], "attn_ms": a["attn_ms"],
                     "qouter_ms": a["total_ms"], "kvouter_ms": b["total_ms"],
                     "speedup": speed, "cos_mean": b.get("cos_mean"), "note": note})

    print("note: stages 1-2 (FP4 indexer, top-k select) are identical and excluded from both\n"
          "      sides; this compares only the half of the pipeline the two designs differ in.\n"
          "      'cos' is KV-outer's output against q-outer's on the same selection, so it is a\n"
          "      correctness check (~1.0 expected), not a sparsity-quality number.")

    fields = ["seqlen", "topk", "block", "csr_ms", "attn_ms", "qouter_ms", "kvouter_ms",
              "speedup", "cos_mean", "note"]
    with open(ex / "kvouter_ab.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {ex / 'kvouter_ab.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
