#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Render benchmark CSVs into the result sections of ``log.html``.

Keeps the work log's numbers derived from the measured artifacts rather than
transcribed by hand: every table in sections 6 and 7 is generated from
``results/<run>/sweep.csv``, so re-running the benchmark and re-running this
script keeps the log honest.

    python scripts/render_log_results.py results/b300-20260808

Content is spliced between ``<!-- BEGIN:name -->`` / ``<!-- END:name -->``
markers, so the surrounding prose is never touched.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_msa_config():
    """Import ``msa_config`` by path.

    Going through the ``fmha_sm100`` package would pull in ``api.py`` and with it
    torch, but this script only needs the pure-Python model registry so it can
    run on a laptop with the CSVs.
    """
    import importlib.util

    path = REPO_ROOT / "python" / "fmha_sm100" / "msa_config.py"
    spec = importlib.util.spec_from_file_location("msa_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["msa_config"] = module
    spec.loader.exec_module(module)
    return module


DEFAULT_MODEL = _load_msa_config().DEFAULT_MODEL


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def load(run_dir: Path):
    rows = list(csv.DictReader(open(run_dir / "sweep.csv")))
    by_len = {}
    for row in rows:
        by_len.setdefault(int(row["seqlen_k"]), []).append(row)
    return by_len


def complete(row):
    """Rows with a failed stage are not comparable — their total omits real work."""
    return not (row.get("errors") or "").strip()


def unsupported_reason(row):
    import json
    try:
        errs = json.loads(row["errors"])
    except Exception:  # noqa: BLE001
        return row.get("errors", "")
    return next(iter(errs.values()), "").split("\n")[0]


def stage_cells(row):
    return [
        fmt(num(row, "indexer_gpu_ms")),
        fmt(num(row, "topk_gpu_ms")),
        fmt(num(row, "csr_gpu_ms")),
        fmt(num(row, "attn_gpu_ms")),
    ]


def render_baseline(by_len) -> str:
    out = [
        '<div class="scroll"><table>',
        "<thead><tr>"
        '<th class="num">kv_len</th><th>行</th>'
        '<th class="num">indexer</th><th class="num">top-k</th>'
        '<th class="num">CSR</th><th class="num">attn</th>'
        '<th class="num">GPU 合计</th><th class="num">eager</th>'
        '<th class="num">host</th><th class="num">vs dense</th>'
        "</tr></thead><tbody>",
    ]
    for seqlen in sorted(by_len):
        group = by_len[seqlen]
        dense = next((r for r in group if r["name"] == "dense"), None)
        base = next((r for r in group if r["name"] == "baseline"), None)
        dense_gpu = num(dense, "pipeline_gpu_ms") if dense else None
        if dense is not None:
            out.append(
                f'<tr class="dense"><td class="num">{seqlen}</td><td>dense causal FMHA</td>'
                f'<td class="num">—</td><td class="num">—</td><td class="num">—</td>'
                f'<td class="num">{fmt(num(dense, "attn_gpu_ms"))}</td>'
                f'<td class="num">{fmt(dense_gpu)}</td>'
                f'<td class="num">{fmt(num(dense, "pipeline_ms"))}</td>'
                f'<td class="num">{fmt(num(dense, "pipeline_host_ms"))}</td>'
                f'<td class="num">1.00x</td></tr>'
            )
        if base is not None:
            gpu = num(base, "pipeline_gpu_ms")
            speed = f"{dense_gpu / gpu:.2f}x" if (dense_gpu and gpu) else "—"
            cells = stage_cells(base)
            out.append(
                f'<tr class="hl"><td class="num">{seqlen}</td>'
                f"<td>baseline block=128 topk=16 keep</td>"
                + "".join(f'<td class="num">{c}</td>' for c in cells)
                + f'<td class="num">{fmt(gpu)}</td>'
                f'<td class="num">{fmt(num(base, "pipeline_ms"))}</td>'
                f'<td class="num">{fmt(num(base, "pipeline_host_ms"))}</td>'
                f'<td class="num">{speed}</td></tr>'
            )
    out.append("</tbody></table></div>")
    m = DEFAULT_MODEL
    out.append(
        f"<p>单位 ms，batch=1，causal，bf16，"
        f"Hq={m.num_qo_heads} / Hkv={m.num_kv_heads} / D={m.bench_head_dim()}"
        f"（{m.name} 的真实注意力几何）。</p>"
    )
    out.append(_per_forward_table(by_len))
    return "\n".join(out)


def _per_forward_table(by_len) -> str:
    """Scale one layer's cost to a whole forward pass — the number that matters."""
    m = DEFAULT_MODEL
    n = m.attention_layers_per_forward
    out = [
        f"<p>{m.name} 每次前向有 <b>{n}</b> 层 softmax attention"
        f"（{m.full_attention_layers} 层 + {m.mtp_layers} 层 MTP），把单层耗时乘上去："
        "</p>",
        '<div class="scroll"><table><thead><tr>'
        '<th class="num">kv_len</th><th class="num">dense × ' + str(n) + '</th>'
        '<th class="num">baseline × ' + str(n) + '</th>'
        '<th class="num">最佳配置</th><th class="num">最佳 × ' + str(n) + '</th>'
        '<th class="num">相对 dense 省下</th></tr></thead><tbody>',
    ]
    for seqlen in sorted(by_len):
        group = by_len[seqlen]
        dense = next((r for r in group if r["name"] == "dense"), None)
        base = next((r for r in group if r["name"] == "baseline"), None)
        cfgs = [r for r in group
                if r["name"] != "dense" and num(r, "pipeline_gpu_ms") and complete(r)]
        if not (dense and cfgs):
            continue
        cfgs.sort(key=lambda r: num(r, "pipeline_gpu_ms"))
        best = cfgs[0]
        d = num(dense, "pipeline_gpu_ms")
        b = num(base, "pipeline_gpu_ms") if base else None
        w = num(best, "pipeline_gpu_ms")
        delta = (d - w) * n
        saved = (
            f"{delta:.1f} ms"
            if delta > 0
            else f"<span style='color:var(--bad)'>+{-delta:.1f} ms</span>"
        )
        out.append(
            f"<tr><td class='num'>{seqlen}</td>"
            f"<td class='num'>{d * n:.1f}</td>"
            f"<td class='num'>{f'{b * n:.1f}' if b else '—'}</td>"
            f"<td class='num'>{best['name']}</td>"
            f"<td class='num'>{w * n:.1f}</td>"
            f"<td class='num'>{saved}</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


def render_matrix(by_len) -> str:
    """One config-by-context matrix — the view for comparing configurations."""
    seqlens = sorted(by_len)
    # Row order follows the CSV rather than a hardcoded cfg01..cfg12 range: the
    # matrix grows over time and a fixed range would silently drop new points.
    meta = {}
    for rows_ in by_len.values():
        for r in rows_:
            meta.setdefault(r["name"], r)
    order = [n for n in meta if n != "dense"]
    order.sort(key=lambda n: (n != "baseline", n))

    def cells(name, fmt):
        out = []
        for s in seqlens:
            row = next((r for r in by_len[s] if r["name"] == name), None)
            gpu = num(row, "pipeline_gpu_ms") if row else None
            dense = num(next(r for r in by_len[s] if r["name"] == "dense"), "pipeline_gpu_ms")
            if row is None or gpu is None or not complete(row):
                out.append("<td class='num' style='color:var(--muted)'>n/a</td>")
            elif fmt == "ms":
                out.append(f"<td class='num'>{gpu:.2f}</td>")
            else:
                ratio = dense / gpu
                style = " style='color:var(--ok);font-weight:600'" if ratio >= 1 else ""
                out.append(f"<td class='num'{style}>{ratio:.2f}x</td>")
        return "".join(out)

    head = ("<thead><tr><th>配置</th><th class='num'>block</th><th class='num'>topk</th>"
            "<th>head</th><th class='num'>预算</th>"
            + "".join(f"<th class='num'>{s // 1024}K</th>" for s in seqlens)
            + "</tr></thead>")

    # The "GPU time" column is a CUDA-graph replay only under --timing full;
    # with the default single-op timing it is the plain event-measured latency,
    # so the heading has to follow the data rather than assert one of them.
    graph_timed = any(
        num(r, "attn_gpu_ms") is not None for rows_ in by_len.values() for r in rows_
    )
    ms_title = ("GPU 时间（ms / 层，CUDA graph 计时）" if graph_timed
                else "流水线延迟（ms / 层，单算子逐级相加）")

    blocks = []
    for title, fmt in ((ms_title, "ms"),
                       ("相对 dense 的加速比（&gt;1 表示稀疏更快）", "ratio")):
        rows_html = []
        dense_cells = "".join(
            f"<td class='num'>{num(next(r for r in by_len[s] if r['name'] == 'dense'), 'pipeline_gpu_ms'):.2f}</td>"
            if fmt == "ms" else "<td class='num'>1.00x</td>"
            for s in seqlens
        )
        rows_html.append("<tr class='dense'><td>dense causal</td><td class='num'>—</td>"
                         "<td class='num'>—</td><td>—</td><td class='num'>—</td>"
                         + dense_cells + "</tr>")
        for name in order:
            m = meta.get(name)
            if m is None:
                continue
            hl = " class='hl'" if name == "cfg03" else ""
            rows_html.append(
                f"<tr{hl}><td>{name}</td><td class='num'>{m['block_size']}</td>"
                f"<td class='num'>{m['topk']}</td><td>{m['head_mode']}</td>"
                f"<td class='num'>{m['selected_tokens']}</td>" + cells(name, fmt) + "</tr>"
            )
        blocks.append(f"<h3>{title}</h3><div class='scroll'><table>{head}<tbody>"
                      + "".join(rows_html) + "</tbody></table></div>")

    blocks.append(
        "<p><code>n/a</code> = 该形状下打分级无法运行（128K 的 sum/max 触发 §8.2 的 Int32 上限），"
        "这些格子<b>没有</b>参与任何排名。加粗绿色为稀疏胜过 dense 的格子。"
        "cfg03 高亮为综合推荐配置。</p>"
    )
    return "\n".join(blocks)


def render_sweep(by_len) -> str:
    out = []
    for seqlen in sorted(by_len):
        group = by_len[seqlen]
        dense = next((r for r in group if r["name"] == "dense"), None)
        dense_gpu = num(dense, "pipeline_gpu_ms") if dense else None
        cfgs = [r for r in group
                if r["name"] != "dense" and num(r, "pipeline_gpu_ms") and complete(r)]
        cfgs.sort(key=lambda r: num(r, "pipeline_gpu_ms"))
        best = cfgs[0]["name"] if cfgs else None
        skipped = [r for r in group if r["name"] != "dense" and not complete(r)]

        out.append(f"<h3>kv_len = {seqlen}"
                   + (f"　<span class='pill'>dense causal GPU {dense_gpu:.3f} ms</span>"
                      if dense_gpu else "") + "</h3>")
        out.append('<div class="scroll"><table><thead><tr>'
                   '<th class="num">#</th><th>配置</th><th class="num">block</th>'
                   '<th class="num">topk</th><th>head</th><th class="num">预算</th>'
                   '<th class="num">块数</th><th class="num">indexer</th>'
                   '<th class="num">top-k</th><th class="num">CSR</th>'
                   '<th class="num">attn</th><th class="num">GPU 合计</th>'
                   '<th class="num">vs dense</th></tr></thead><tbody>')
        for rank, row in enumerate(cfgs, 1):
            gpu = num(row, "pipeline_gpu_ms")
            speed = f"{dense_gpu / gpu:.2f}x" if dense_gpu else "—"
            cls = ' class="hl"' if row["name"] == best else ""
            cells = stage_cells(row)
            out.append(
                f"<tr{cls}><td class='num'>{rank}</td><td>{row['name']}</td>"
                f"<td class='num'>{row['block_size']}</td>"
                f"<td class='num'>{row['topk']}</td><td>{row['head_mode']}</td>"
                f"<td class='num'>{row['selected_tokens']}</td>"
                f"<td class='num'>{row['num_blocks']}</td>"
                + "".join(f"<td class='num'>{c}</td>" for c in cells)
                + f"<td class='num'>{fmt(gpu)}</td><td class='num'>{speed}</td></tr>"
            )
        for row in skipped:
            out.append(
                f"<tr class='dense'><td class='num'>—</td><td>{row['name']}</td>"
                f"<td class='num'>{row['block_size']}</td><td class='num'>{row['topk']}</td>"
                f"<td>{row['head_mode']}</td><td class='num'>{row['selected_tokens']}</td>"
                f"<td class='num'>{row['num_blocks']}</td>"
                f"<td colspan='6'>此形状下不可运行 — {unsupported_reason(row)}</td></tr>"
            )
        out.append("</tbody></table></div>")
        if skipped:
            out.append(
                "<p><b>注意</b>：上表中标为「不可运行」的配置<b>没有</b>参与排名。"
                "它们的 top-k / CSR / attention 三级都能跑，但打分级跑不了——"
                "把跑得通的几级加起来当作总耗时，等于把跑不通的那部分当成免费的，"
                "会得出「它们最快」这种反的结论。原因见 §8.2。</p>"
            )
    return "\n".join(out)


def render_equal_budget(by_len) -> str:
    """The (block=64,topk=16) vs (block=32,topk=32) comparison at equal budget."""
    out = ['<div class="scroll"><table><thead><tr>'
           '<th class="num">kv_len</th>'
           '<th class="num">cfg03 attn<br>b64/k16, 1024 tok</th>'
           '<th class="num">cfg02 attn<br>b32/k32, 1024 tok</th>'
           '<th class="num">比值</th>'
           '<th class="num">cfg01 attn<br>b32/k16, 512 tok</th>'
           '<th class="num">cfg03 attn<br>b64/k16, 1024 tok</th>'
           '<th class="num">比值</th>'
           "</tr></thead><tbody>"]
    for seqlen in sorted(by_len):
        idx = {r["name"]: r for r in by_len[seqlen]}
        a = num(idx.get("cfg03", {}), "attn_gpu_ms")
        b = num(idx.get("cfg02", {}), "attn_gpu_ms")
        c = num(idx.get("cfg01", {}), "attn_gpu_ms")
        if None in (a, b, c):
            continue
        out.append(
            f"<tr><td class='num'>{seqlen}</td>"
            f"<td class='num'>{fmt(a)}</td><td class='num'>{fmt(b)}</td>"
            f"<td class='num'><b>{b / a:.2f}×</b></td>"
            f"<td class='num'>{fmt(c)}</td><td class='num'>{fmt(a)}</td>"
            f"<td class='num'>{a / c:.2f}×</td></tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)




# ---------------------------------------------------------------------------
# Grid / decode tables (single-operator timing, real Q/K)
# ---------------------------------------------------------------------------


def _load_flat(path: Path):
    rows = list(csv.DictReader(open(path)))
    seqlens = sorted({int(r["seqlen_k"]) for r in rows})
    meta = {}
    for r in rows:
        meta.setdefault(r["name"], r)
    order = [n for n in meta if n != "dense"]

    def sort_key(name):
        m = meta[name]
        if name == "baseline":
            return (0, 0, 0, "")
        return (1, int(m["selected_tokens"] or 0), int(m["block_size"] or 0), m["head_mode"] or "")

    order.sort(key=sort_key)
    return rows, seqlens, meta, order


def _attn_ms(row):
    if row is None or (row.get("errors") or "").strip():
        return None
    return num(row, "attn_ms_only") or num(row, "attn_ms")


def _grid_table(rows, seqlens, meta, order, *, title, subtitle, value, best="max"):
    """One config-by-context table with the best cell per column highlighted."""
    by_key = {(r["name"], int(r["seqlen_k"])): r for r in rows}

    # Best value per column, so a reader can find the winner without scanning.
    column_best = {}
    for s in seqlens:
        vals = [value(by_key.get((n, s))) for n in order]
        vals = [v for v in vals if v is not None]
        if vals:
            column_best[s] = max(vals) if best == "max" else min(vals)

    out = [f"<h4>{title}</h4>", f"<p>{subtitle}</p>",
           '<div class="scroll"><table class="sortable"><thead><tr>'
           '<th>配置</th><th class="num">block</th><th class="num">topk</th>'
           '<th>head</th><th class="num">预算</th>'
           + "".join(f'<th class="num">{s // 1024}K</th>' for s in seqlens)
           + "</tr></thead><tbody>"]

    dense = by_key.get(("dense", seqlens[0]))
    if dense is not None:
        cells = []
        for s in seqlens:
            v = value(by_key.get(("dense", s)))
            cells.append(f"<td class='num'>{'—' if v is None else f'{v:.4f}' if v <= 1.001 and best == 'max' and v > 0.5 else f'{v:.2f}'}</td>")
        out.append("<tr class='dense'><td>dense gqa</td><td class='num'>—</td>"
                   "<td class='num'>—</td><td>—</td><td class='num'>—</td>"
                   + "".join(cells) + "</tr>")

    for name in order:
        m = meta[name]
        cells = []
        for s in seqlens:
            v = value(by_key.get((name, s)))
            if v is None:
                cells.append("<td class='num' style='color:var(--muted)'>n/a</td>")
                continue
            is_best = column_best.get(s) is not None and abs(v - column_best[s]) < 1e-9
            style = " style='color:var(--ok);font-weight:600'" if is_best else ""
            txt = f"{v:.4f}" if v <= 1.001 else f"{v:.2f}x"
            cells.append(f"<td class='num'{style}>{txt}</td>")
        out.append(
            f"<tr><td>{name}</td><td class='num'>{m['block_size']}</td>"
            f"<td class='num'>{m['topk']}</td><td>{m['head_mode']}</td>"
            f"<td class='num'>{m['selected_tokens']}</td>" + "".join(cells) + "</tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


def render_grid(path: Path) -> str:
    rows, seqlens, meta, order = _load_flat(path)
    parts = [
        "<p>行按 <b>token 预算</b>排序（预算相同的配置相邻），方便看「同预算下不同 "
        "block/topk 拆分」的差异。点表头可按列排序。每列最优值以绿色加粗标出。</p>"
    ]
    parts.append(_grid_table(
        rows, seqlens, meta, order,
        title="单算子相对 dense gqa 加速比",
        subtitle="只计 attention 算子本身，不含 indexer / top-k / CSR。&gt;1 表示稀疏更快。",
        value=lambda r: (
            None if _attn_ms(r) is None or _attn_ms(rows_dense(rows, r)) is None
            else _attn_ms(rows_dense(rows, r)) / _attn_ms(r)
        ),
    ))
    parts.append(_grid_table(
        rows, seqlens, meta, order,
        title="输出余弦相似度 vs dense gqa",
        subtitle="Q/K/V 取自 model-n32 checkpoint <code><checkpoint></code> 的真实一层，"
                 "每 (token, head) 向量取 cos 后平均。",
        value=lambda r: (None if r is None or (r.get("errors") or "").strip()
                         else num(r, "cos_mean")),
    ))
    return "\n".join(parts)


def rows_dense(rows, row):
    if row is None:
        return None
    s = int(row["seqlen_k"])
    return next((r for r in rows if r["name"] == "dense" and int(r["seqlen_k"]) == s), None)


def render_decode(path: Path) -> str:
    import json as _json

    rows, seqlens, meta, order = _load_flat(path)
    by_key = {(r["name"], int(r["seqlen_k"])): r for r in rows}
    out = ['<div class="scroll"><table><thead><tr>'
           '<th>配置</th><th class="num">block</th><th class="num">topk</th>'
           '<th>head</th><th class="num">预算</th>'
           + "".join(f'<th class="num">{s // 1024}K</th>' for s in seqlens)
           + "</tr></thead><tbody>"]

    d_cells = []
    for s in seqlens:
        v = num(by_key.get(("dense", s)), "pipeline_gpu_ms")
        d_cells.append(f"<td class='num'>{'—' if v is None else f'{v:.3f}'}</td>")
    out.append("<tr class='dense'><td>dense gqa</td><td class='num'>—</td><td class='num'>—</td>"
               "<td>—</td><td class='num'>—</td>" + "".join(d_cells) + "</tr>")

    for name in order:
        m = meta[name]
        cells = []
        for s in seqlens:
            r = by_key.get((name, s))
            if r is None:
                cells.append("<td class='num' style='color:var(--muted)'>—</td>")
                continue
            if (r.get("errors") or "").strip():
                stage = next(iter(_json.loads(r["errors"])), "?")
                cells.append(f"<td class='num' style='color:var(--bad)'>✗ {stage}</td>")
                continue
            v = num(r, "pipeline_gpu_ms")
            d = num(by_key.get(("dense", s)), "pipeline_gpu_ms")
            cells.append(f"<td class='num'>{v:.3f}<br>"
                         f"<span style='color:var(--muted);font-size:11px'>{d / v:.2f}x</span></td>"
                         if (v and d) else "<td class='num'>—</td>")
        out.append(
            f"<tr><td>{name}</td><td class='num'>{m['block_size']}</td>"
            f"<td class='num'>{m['topk']}</td><td>{m['head_mode']}</td>"
            f"<td class='num'>{m['selected_tokens']}</td>" + "".join(cells) + "</tr>"
        )
    out.append("</tbody></table></div>")
    return "\n".join(out)


def splice(html: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", re.DOTALL
    )
    if not pattern.search(html):
        raise SystemExit(f"marker BEGIN:{name} not found in log.html")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", html)


def main(argv):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default="results/model-n32-v2-20260808",
                    help="directory holding sweep.csv for sections 6 and 7")
    ap.add_argument("--grid", default=None,
                    help="CSV from a single-op / real-Q_K grid run (section 7c)")
    ap.add_argument("--decode", default=None,
                    help="CSV from a decode run (section 7d)")
    args = ap.parse_args(argv[1:])

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    by_len = load(run_dir)

    log_path = REPO_ROOT / "log.html"
    html = log_path.read_text()
    html = splice(html, "baseline", render_baseline(by_len))
    html = splice(html, "sweep", render_matrix(by_len) + "\n"
                  + "<h3>逐上下文明细</h3>\n"
                  + render_sweep(by_len) + "\n"
                  + "<h3>等 token 预算对照</h3>\n"
                  + "<p>把 token 预算固定住，只改 block/topk 的分配方式：</p>\n"
                  + render_equal_budget(by_len))
    for name, path in (("grid", args.grid), ("decode", args.decode)):
        if not path:
            continue
        csv_path = Path(path)
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        body = render_grid(csv_path) if name == "grid" else render_decode(csv_path)
        html = splice(html, name, body)
        print(f"  section {name} <- {csv_path}")

    log_path.write_text(html)
    print(f"log.html updated from {run_dir}")
    for seqlen in sorted(by_len):
        print(f"  kv_len={seqlen}: {len(by_len[seqlen])} rows")


if __name__ == "__main__":
    main(sys.argv)
