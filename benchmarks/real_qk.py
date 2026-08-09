#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Real post-RoPE Q/K/V from a single attention layer of a trained checkpoint.

Benchmarks that only measure latency can run on random tensors — kernel time
does not depend on the values.  Anything that measures *quality* cannot: random
Q/K produce near-uniform attention, where any top-k selection can only recover
roughly its token-budget share of the mass, so every configuration looks
equally bad and the ranking carries no signal.

This module supplies the alternative: the projections of one real layer, applied
to real token embeddings.  Only six tensors are read straight out of the
safetensors shards, so it costs a few hundred MB and a couple of seconds rather
than materialising a 35B MoE.

Shared by ``bench_msa_configs.py`` (``--qk-source model``) and
``eval_msa_selection_quality.py``.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_prompt_text(min_chars: int, *, allow_short: bool = False) -> str:
    """A long, non-repeating natural prompt built from this repo's own sources.

    Repeating one passage to reach the target length would manufacture
    artificial long-range matches and flatter any selector that finds them, so
    the prompt is assembled from distinct files instead.  Source code is not
    prose, but it is real text with genuine long-range structure — far better
    than random token ids, which would strip out exactly the content-driven
    dependencies this evaluation is about.
    """
    parts, total = [], 0
    roots = [REPO_ROOT / "python" / "fmha_sm100", REPO_ROOT / "tests", REPO_ROOT / "benchmarks"]
    files = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")))
        files.extend(sorted(root.rglob("*.cuh")))
        files.extend(sorted(root.rglob("*.cu")))
    for path in files:
        if "cutlass" in str(path):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        parts.append(f"\n\n===== {path.relative_to(REPO_ROOT)} =====\n{text}")
        total += len(parts[-1])
        if total >= min_chars:
            break
    if total < min_chars and not allow_short:
        raise RuntimeError(
            f"only gathered {total} chars, need {min_chars}. Ask for a shorter capture, "
            f"or pass allow_short=True if the caller tiles the result (decode does)."
        )
    return "".join(parts)


def _rms_norm(x, weight, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()


def _rope(q, k, *, positions, head_dim, theta):
    """Standard rotate-half RoPE over the full head dim (rotary_percent=1.0)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv.unsqueeze(0)          # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)                              # [S, D]
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)            # [S, 1, D]

    def rot(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def capture_qk(model_path: str, *, seqlen: int, layer: int, device: torch.device,
               allow_short: bool = False):
    """Post-RoPE Q/K for **one** attention layer, without instantiating the model.

    Only six tensors are read straight out of the safetensors shards — the token
    embedding, the layer's input norm, and its ``q_proj`` / ``k_proj`` /
    ``q_layernorm`` / ``k_layernorm``.  Materialising a 35B MoE just to look at
    one layer's projections would be several minutes and ~70 GB for nothing.

    The head dimension is derived from ``q_proj``'s shape rather than the config
    (which omits ``head_dim`` and would wrongly imply ``hidden/heads``).

    **Approximation**: hidden states are ``input_layernorm(embed(ids))``, i.e.
    layer 0's input.  Correct mid-stack hidden states would require running the
    prefix.  At layer 0 the hyper-connection streams are still copies of the
    embedding, so this is close there; the trade is that early-layer attention
    is more local than mid-stack, which *understates* how much the query heads
    inside a GQA group disagree — so the head-mode gaps reported here are a
    lower bound.
    """
    import glob
    import json as _json
    import os

    from safetensors import safe_open

    cfg = _json.load(open(os.path.join(model_path, "config.json")))
    theta = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    head_q = int(cfg["num_attention_heads"])
    head_kv = int(cfg["num_key_value_heads"])

    prefix = f"model.layers.{layer}.self_attn."
    wanted = {
        "embed": "model.embed_tokens.weight",
        "in_norm": f"model.layers.{layer}.input_layernorm.weight",
        "q_proj": prefix + "q_proj.weight",
        "k_proj": prefix + "k_proj.weight",
        "v_proj": prefix + "v_proj.weight",
        "q_norm": prefix + "q_layernorm.weight",
        "k_norm": prefix + "k_layernorm.weight",
    }
    got: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as fh:
            keys = set(fh.keys())
            for name, key in wanted.items():
                if name not in got and key in keys:
                    got[name] = fh.get_tensor(key)
        if len(got) == len(wanted):
            break
    missing = [n for n in wanted if n not in got]
    if missing:
        raise RuntimeError(f"layer {layer}: missing tensors {[wanted[m] for m in missing]}")

    head_dim = got["q_proj"].shape[0] // head_q
    if got["k_proj"].shape[0] // head_kv != head_dim:
        raise RuntimeError("q_proj and k_proj disagree on head_dim")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = build_prompt_text(min_chars=seqlen * 4, allow_short=allow_short)
    ids = tok(text, return_tensors="pt").input_ids[0, :seqlen]
    if ids.numel() < seqlen:
        if not allow_short:
            raise RuntimeError(
                f"prompt tokenised to {ids.numel()} < {seqlen}; the repo does not contain enough "
                f"text for a real sequence this long"
            )
        # Decode tiles the capture across the batch anyway, so a shorter but
        # genuinely-real span is strictly better than padding it with noise.
        seqlen = int(ids.numel())

    embed = got["embed"].to(device=device, dtype=torch.float32)
    hidden = _rms_norm(embed[ids.to(device)], got["in_norm"].to(device), eps)   # [S, hidden]

    q = hidden @ got["q_proj"].to(device=device, dtype=torch.float32).T
    k = hidden @ got["k_proj"].to(device=device, dtype=torch.float32).T
    # V carries no norm and no RoPE.
    v = (hidden @ got["v_proj"].to(device=device, dtype=torch.float32).T).view(
        seqlen, head_kv, head_dim
    )
    q = _rms_norm(q.view(seqlen, head_q, head_dim), got["q_norm"].to(device), eps)
    k = _rms_norm(k.view(seqlen, head_kv, head_dim), got["k_norm"].to(device), eps)
    q, k = _rope(q, k, positions=torch.arange(seqlen, device=device),
                 head_dim=head_dim, theta=theta)

    del embed, got
    torch.cuda.empty_cache()
    return {
        "seqlen": seqlen,
        "q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous(),
        "scale": head_dim ** -0.5,
        "head_q": head_q, "head_kv": head_kv, "head_dim": head_dim,
    }
