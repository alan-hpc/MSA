# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Declarative configuration for the MSA block-sparse attention pipeline.

The MSA sparse path has four independent knobs that together decide both the
quality and the cost of a sparse attention layer:

``block_size``
    KV granularity, in tokens, shared by the block scorer, the top-k selection
    and the sparse attention kernel (``blk_kv``).  Smaller blocks select more
    precisely for the same token budget but multiply the number of scored and
    sorted candidates by ``128 / block_size``.

``topk``
    Number of KV blocks each query attends to.  The attended token budget is
    ``topk * block_size``, so ``(block=64, topk=32)`` and ``(block=32, topk=32)``
    differ in *precision at equal k*, while ``(block=64, topk=16)`` and
    ``(block=32, topk=32)`` differ in *precision at equal budget*.

``force_init_tokens`` / ``force_end_tokens``
    Attention-sink and local-window guarantees, expressed in **tokens** so a
    configuration keeps the same meaning as ``block_size`` changes.  They are
    converted to whole blocks by rounding up, and they consume part of the
    ``topk`` budget rather than adding to it.

``head_mode``
    How indexer scores map onto the per-KV-head selection the sparse attention
    kernel consumes.  See :data:`HEAD_MODES`.

Nothing here touches CUDA; this module is importable on any machine and is the
single source of truth for both the runtime pipeline and the benchmark sweep.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from typing import Iterator, Optional

__all__ = [
    "HEAD_MODES",
    "SUPPORTED_BLOCK_SIZES",
    "SUPPORTED_TOPK",
    "MsaSparseConfig",
    "MsaModelShape",
    "MODEL_N32",
    "MODEL_N16",
    "MODEL_SHAPES",
    "DEFAULT_MODEL",
    "model_by_name",
    "BASELINE_CONFIG",
    "CONFIG_MATRIX",
    "config_by_name",
    "iter_configs",
]


#: ``head_mode`` -> human-readable contract.
#:
#: Every mode starts from the same input: one score row per **KV head**, from
#: MSA's MQA "proxy-Q" indexer.  The mode decides what is done with those rows.
#:
#: ``keep``
#:     Rows are used as-is — each KV head gets its own selection.  This is the
#:     shipped behaviour.
#: ``sum``
#:     The ``Hkv`` rows are summed into **one** selection shared by the whole
#:     layer.  A block wins if it is broadly useful across the KV heads.
#: ``max``
#:     Same, reduced with max: a block wins if *any* KV head wants it.  Max is
#:     scale-free across heads and is the more conservative choice for recall.
#:
#: So ``sum``/``max`` are **cheaper** than ``keep``: the scorer does identical
#: work and the top-k emits one row instead of ``Hkv``.  The single selection is
#: broadcast back to ``Hkv`` for the CSR, whose metadata is keyed on KV heads.
HEAD_MODES = ("keep", "sum", "max")

#: Block sizes the whole pipeline (scorer, top-k, attention) is wired for.
SUPPORTED_BLOCK_SIZES = (32, 64, 128)

#: ``topk`` values the sparse attention kernel accepts (``_SUPPORTED_SPARSE_TOPK``
#: in ``cute/interface.py``); the top-k selection kernel additionally caps at 64.
SUPPORTED_TOPK = (4, 8, 16, 32)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class MsaSparseConfig:
    """One point in the MSA sparse-attention configuration space.

    Attributes
    ----------
    block_size : int
        KV block size in tokens.  Used for the block scorer, the top-k
        selection and ``sparse_atten_func(blk_kv=...)`` alike.
    topk : int
        Blocks selected per query, inclusive of the forced windows.
    force_init_tokens : int
        Leading tokens always attended (attention sinks).
    force_end_tokens : int
        Trailing tokens always attended, relative to each query's own causal
        position (local window).
    head_mode : str
        One of :data:`HEAD_MODES`.
    name : str, optional
        Stable identifier used by the benchmark harness and reports.  Derived
        from the other fields when omitted.
    """

    block_size: int = 128
    topk: int = 16
    force_init_tokens: int = 0
    force_end_tokens: int = 0
    head_mode: str = "keep"
    #: Collapse the whole layer onto **one** shared selection (DSA-style) instead
    #: of one per KV head.  Orthogonal to ``head_mode``: the mode says how rows
    #: are combined, this says how many rows come out.
    shared_index: bool = False
    name: Optional[str] = None

    # ---- derived quantities ------------------------------------------------

    @property
    def force_init_blocks(self) -> int:
        """Sink window rounded **up** to whole blocks.

        Rounding up rather than down keeps the token guarantee intact: with
        ``block_size=48`` a 128-token sink still covers all 128 tokens.
        """
        return _ceil_div(self.force_init_tokens, self.block_size)

    @property
    def force_end_blocks(self) -> int:
        """Local window rounded **up** to whole blocks (see :attr:`force_init_blocks`)."""
        return _ceil_div(self.force_end_tokens, self.block_size)

    @property
    def selected_tokens(self) -> int:
        """Upper bound on KV tokens attended per query."""
        return self.topk * self.block_size

    @property
    def free_blocks(self) -> int:
        """Blocks left for score-driven selection after the forced windows."""
        return self.topk - self.force_init_blocks - self.force_end_blocks

    def num_scorer_heads(self, num_qo_heads: int, num_kv_heads: int) -> int:
        """Score rows the block scorer must emit.

        MSA's indexer scores with a proxy query **per KV head**, so every mode
        starts from the same ``num_kv_heads`` rows — ``head_mode`` only decides
        what happens to them afterwards.  ``shared_index`` is the one knob that
        changes the scorer itself, collapsing it to a single row (the DSA-style
        "one index for the whole layer").

        ``num_qo_heads`` is accepted for signature stability and validation; the
        scorer cost does not depend on it.
        """
        if num_kv_heads <= 0 or num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads ({num_qo_heads}) must be a positive multiple of "
                f"num_kv_heads ({num_kv_heads})"
            )
        return 1 if self.shared_index else num_kv_heads

    def num_selection_heads(self, num_kv_heads: int) -> int:
        """Selections produced: one per KV head (``keep``) or one shared (``sum``/``max``).

        ``keep`` hands each KV head its own selection — the shipped behaviour.
        ``sum`` / ``max`` reduce those same per-KV-head score rows down to a
        single selection shared by the whole layer, so they are **cheaper** than
        ``keep``: the scorer does identical work and the top-k emits a quarter
        of the rows (at ``Hkv=4``).
        """
        if self.shared_index or self.head_mode != "keep":
            return 1
        return num_kv_heads

    def head_group_size(self, num_qo_heads: int, num_kv_heads: int) -> int:
        """Score rows consumed per output selection row."""
        return self.num_scorer_heads(num_qo_heads, num_kv_heads) // self.num_selection_heads(
            num_kv_heads
        )

    def num_blocks(self, seqlen_k: int) -> int:
        """KV blocks covering ``seqlen_k`` tokens."""
        return _ceil_div(seqlen_k, self.block_size)

    def sparsity(self, seqlen_k: int) -> float:
        """Fraction of KV tokens attended at ``seqlen_k`` (clamped to 1.0)."""
        if seqlen_k <= 0:
            return 0.0
        return min(1.0, self.selected_tokens / seqlen_k)

    # ---- naming ------------------------------------------------------------

    @property
    def slug(self) -> str:
        """Filesystem/CSV-safe identifier, round-trippable through :meth:`parse`."""
        return (
            f"fi{self.force_init_tokens}-fe{self.force_end_tokens}"
            f"-b{self.block_size}-k{self.topk}-h{self.head_mode}"
            + ("-shared" if self.shared_index else "")
        )

    @property
    def label(self) -> str:
        """Human-facing label matching the requirement wording."""
        return (
            f"force_init({self.force_init_tokens})-force_end({self.force_end_tokens})"
            f"-block({self.block_size})-topk({self.topk})-head({self.head_mode})"
        )

    def resolved_name(self) -> str:
        return self.name or self.slug

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.label

    # ---- validation --------------------------------------------------------

    def validate(self, *, strict: bool = True) -> "MsaSparseConfig":
        """Check internal consistency; returns ``self`` so it can be chained.

        ``strict`` additionally requires ``block_size`` / ``topk`` to be values
        the shipped kernels are compiled for.  Pass ``strict=False`` to explore
        configurations that only the reference implementation supports.
        """
        if self.head_mode not in HEAD_MODES:
            raise ValueError(f"head_mode must be one of {HEAD_MODES}, got {self.head_mode!r}")
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError(f"block_size must be a positive power of two, got {self.block_size}")
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if self.force_init_tokens < 0 or self.force_end_tokens < 0:
            raise ValueError("force_init_tokens and force_end_tokens must be non-negative")
        if strict:
            if self.block_size not in SUPPORTED_BLOCK_SIZES:
                raise ValueError(
                    f"block_size must be one of {SUPPORTED_BLOCK_SIZES}, got {self.block_size}"
                )
            if self.topk not in SUPPORTED_TOPK:
                raise ValueError(f"topk must be one of {SUPPORTED_TOPK}, got {self.topk}")
        forced = self.force_init_blocks + self.force_end_blocks
        if forced > self.topk:
            raise ValueError(
                f"{self.label}: forced windows need {forced} blocks "
                f"({self.force_init_blocks} sink + {self.force_end_blocks} local) "
                f"but topk is only {self.topk}. Raise topk, raise block_size, or "
                f"shrink the windows."
            )
        return self

    # ---- (de)serialisation -------------------------------------------------

    _SLUG_RE = re.compile(
        r"^fi(?P<fi>\d+)-fe(?P<fe>\d+)-b(?P<b>\d+)-k(?P<k>\d+)-h(?P<h>keep|sum|max)"
        r"(?P<shared>-shared)?$"
    )
    _LABEL_RE = re.compile(
        r"force_init\((?P<fi>\d+)\)-force_end\((?P<fe>\d+)\)"
        r"-block\((?P<b>\d+)\)-topk\((?P<k>\d+)\)-head\((?P<h>[a-z]+)\)$"
    )

    @classmethod
    def parse(cls, spec: str) -> "MsaSparseConfig":
        """Build a config from a slug, a label, or a ``key=value`` list.

        Accepted forms::

            fi128-fe128-b32-k16-hkeep
            force_init(128)-force_end(128)-block(32)-topk(16)-head(keep)
            block=32,topk=16,force_init=128,force_end=128,head=keep

        A bare name from :data:`CONFIG_MATRIX` also resolves.
        """
        spec = spec.strip()
        named = config_by_name(spec, default=None)
        if named is not None:
            return named
        for pattern in (cls._SLUG_RE, cls._LABEL_RE):
            m = pattern.match(spec)
            if m:
                return cls(
                    block_size=int(m["b"]),
                    topk=int(m["k"]),
                    force_init_tokens=int(m["fi"]),
                    force_end_tokens=int(m["fe"]),
                    head_mode=m["h"],
                    shared_index=bool(m.groupdict().get("shared")),
                ).validate()

        aliases = {
            "block": "block_size",
            "blk": "block_size",
            "block_size": "block_size",
            "topk": "topk",
            "k": "topk",
            "force_init": "force_init_tokens",
            "force_init_tokens": "force_init_tokens",
            "fi": "force_init_tokens",
            "force_end": "force_end_tokens",
            "force_end_tokens": "force_end_tokens",
            "fe": "force_end_tokens",
            "head": "head_mode",
            "head_mode": "head_mode",
            "shared": "shared_index",
            "shared_index": "shared_index",
        }
        kwargs: dict[str, object] = {}
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"cannot parse MSA config fragment {part!r} in {spec!r}")
            key, value = (t.strip() for t in part.split("=", 1))
            if key not in aliases:
                raise ValueError(f"unknown MSA config key {key!r} in {spec!r}")
            field = aliases[key]
            if field == "head_mode":
                kwargs[field] = value
            elif field == "shared_index":
                kwargs[field] = value.lower() in ("1", "true", "yes", "on")
            else:
                kwargs[field] = int(value)
        if not kwargs:
            raise ValueError(f"cannot parse MSA config {spec!r}")
        return cls(**kwargs).validate()  # type: ignore[arg-type]

    @classmethod
    def from_env(
        cls, prefix: str = "MSA_", default: Optional["MsaSparseConfig"] = None
    ) -> "MsaSparseConfig":
        """Read a config from environment variables.

        ``{prefix}CONFIG`` takes a whole spec string; otherwise the individual
        ``{prefix}BLOCK_SIZE`` / ``TOPK`` / ``FORCE_INIT`` / ``FORCE_END`` /
        ``HEAD_MODE`` variables override ``default`` field by field.
        """
        spec = os.environ.get(f"{prefix}CONFIG")
        if spec:
            return cls.parse(spec)
        base = default if default is not None else BASELINE_CONFIG
        overrides: dict[str, object] = {}
        for env_name, field, caster in (
            ("BLOCK_SIZE", "block_size", int),
            ("TOPK", "topk", int),
            ("FORCE_INIT", "force_init_tokens", int),
            ("FORCE_END", "force_end_tokens", int),
            ("HEAD_MODE", "head_mode", str),
        ):
            raw = os.environ.get(f"{prefix}{env_name}")
            if raw is not None:
                overrides[field] = caster(raw)
        if not overrides:
            return base
        return replace(base, name=None, **overrides).validate()  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        return {
            "name": self.resolved_name(),
            "label": self.label,
            "slug": self.slug,
            "block_size": self.block_size,
            "topk": self.topk,
            "force_init_tokens": self.force_init_tokens,
            "force_end_tokens": self.force_end_tokens,
            "force_init_blocks": self.force_init_blocks,
            "force_end_blocks": self.force_end_blocks,
            "head_mode": self.head_mode,
            "shared_index": self.shared_index,
            "selected_tokens": self.selected_tokens,
        }


@dataclass(frozen=True)
class MsaModelShape:
    """Attention geometry of a target model, for benchmark shape derivation."""

    name: str
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    num_hidden_layers: int
    #: Layers running softmax attention (i.e. layers MSA can replace).  Equal to
    #: ``num_hidden_layers`` unless the model interleaves another attention kind.
    full_attention_layers: int
    #: Extra multi-token-prediction layers, which also run attention.
    mtp_layers: int = 0
    #: Head dim the shipped MSA kernels actually support (``D=128`` today).
    kernel_head_dim: int = 128
    #: Training / configured context length, benchmarked alongside longer ones.
    train_seqlen: Optional[int] = None
    notes: str = ""

    @property
    def qhead_per_kv(self) -> int:
        return self.num_qo_heads // self.num_kv_heads

    @property
    def head_dim_supported(self) -> bool:
        return self.head_dim == self.kernel_head_dim

    @property
    def qhead_per_kv_supported(self) -> bool:
        # SparseAttentionForwardSm100 compiles for these GQA ratios only.
        return self.qhead_per_kv in (1, 2, 4, 8, 16)

    @property
    def attention_layers_per_forward(self) -> int:
        return self.full_attention_layers + self.mtp_layers

    def bench_head_dim(self) -> int:
        """Head dim to benchmark with, clamped to what the kernels compile for."""
        return self.head_dim if self.head_dim_supported else self.kernel_head_dim

    def compatibility(self) -> list[str]:
        """Human-readable list of MSA support gaps for this geometry."""
        gaps = []
        if not self.head_dim_supported:
            gaps.append(
                f"head_dim={self.head_dim} exceeds the MSA kernels' D={self.kernel_head_dim} "
                f"limit; benchmarks run at D={self.bench_head_dim()}"
            )
        if not self.qhead_per_kv_supported:
            gaps.append(
                f"qhead_per_kv={self.qhead_per_kv} is outside the supported set "
                "{1, 2, 4, 8, 16}"
            )
        # The paged FP8 decode wrapper is pinned to a 16x GQA ratio.
        if self.qhead_per_kv != 16:
            gaps.append(
                f"sparse paged decode requires qhead_per_kv=16, model has {self.qhead_per_kv}; "
                "only the prefill path is covered"
            )
        return gaps


#: ``model-n32`` — the default benchmark target.
#:
#: A GQA transformer with a Megatron-style pretraining config::
#:
#:     num_layers 45, hidden_size 2048, ffn_hidden_size 6144
#:     num_attention_heads 32, group_query_attention, num_query_groups 4
#:     kv_channels 128, seq_length 4096
#:     qk_layernorm, rotary_base 10000, mtp_num_layers 1
#:     MoE: 256 experts, moe_ffn_hidden_size 512, router topk 12
#:
#: ``kv_channels=128`` explicitly overrides the default head dim
#: (``hidden_size / num_attention_heads`` = 2048/32 = 64), so Q projects
#: 2048 -> 32*128 = 4096 while K/V project to 4*128 = 512 each.
#:
#: Its MoE layer frequency selects dense vs MoE **MLP**, not the attention kind,
#: so all 45 layers run softmax attention, plus the MTP layer — 46 per forward.
#:
#: MSA compatibility: ``D=128`` and ``qhead_per_kv=8`` are both natively
#: supported, so this geometry needs no kernel work to run.
MODEL_N32 = MsaModelShape(
    name="model-n32",
    num_qo_heads=32,
    num_kv_heads=4,
    head_dim=128,
    max_position_embeddings=4096,
    num_hidden_layers=45,
    full_attention_layers=45,
    mtp_layers=1,
    train_seqlen=4096,
    notes="all 45 layers are full attention (the MoE layer frequency selects the MLP kind); +1 MTP layer",
)


#: ``model-n16`` — a hybrid-attention MoE, kept as a contrast case.
#:
#: ``layer_types`` alternates three linear-attention layers with one
#: full-attention layer, so only 10 of its 40 layers run the softmax attention
#: that MSA replaces.
#:
#: NOTE ``head_dim=256`` exceeds what the shipped MSA kernels support (D=128
#: only, asserted in ``cute/interface.py`` and ``atten_fwd.py``).  Benchmarks
#: therefore run at D=128; see the optimisation report for the cost of lifting
#: this.
MODEL_N16 = MsaModelShape(
    name="model-n16",
    num_qo_heads=16,
    num_kv_heads=2,
    head_dim=256,
    max_position_embeddings=262144,
    num_hidden_layers=40,
    full_attention_layers=10,
    mtp_layers=1,
    notes="hybrid stack: 30 of 40 layers are linear attention and are not MSA's to replace",
)


#: Benchmarkable model geometries, keyed by lowercase short name.
MODEL_SHAPES = {
    "model-n32": MODEL_N32,
    "n32": MODEL_N32,
    "model-n16": MODEL_N16,
    "n16": MODEL_N16,
}

#: The geometry benchmarks target unless told otherwise.
DEFAULT_MODEL = MODEL_N32


def model_by_name(name: str) -> MsaModelShape:
    """Look a model geometry up by short name (case-insensitive)."""
    key = str(name).strip().lower()
    if key in MODEL_SHAPES:
        return MODEL_SHAPES[key]
    known = ", ".join(sorted(MODEL_SHAPES))
    raise KeyError(f"unknown model {name!r}; known: {known}")


#: The shipped MSA production point: 128-token blocks, top-16, no forced
#: windows, per-KV-head (proxy-Q) scoring.  Used as the reference row that every
#: swept configuration is compared against.
BASELINE_CONFIG = MsaSparseConfig(
    block_size=128,
    topk=16,
    force_init_tokens=0,
    force_end_tokens=0,
    head_mode="keep",
    name="baseline",
).validate()


def _build_matrix() -> tuple[MsaSparseConfig, ...]:
    """The swept configuration grid, with stable names.

    ``cfg01``..``cfg12`` are emitted first and in their original order — head
    mode outer, then ``block_size`` in ``(32, 64)``, then ``topk`` in
    ``(16, 32)``.  Every later addition is appended, so a name never changes
    meaning and results collected earlier stay comparable.

    The extension fills out the rest of the supported grid
    (``block_size`` x ``topk``).  Combinations whose forced windows do not fit
    in the budget are skipped rather than silently clamped: with
    ``force_init=force_end=128``, ``block_size=32`` needs 4+4 blocks, so
    ``topk=4`` is not expressible.
    """
    legacy = [
        (head, block, topk)
        for head in ("keep", "sum", "max")
        for block in (32, 64)
        for topk in (16, 32)
    ]
    full = [
        (head, block, topk)
        for head in ("keep", "sum", "max")
        for block in SUPPORTED_BLOCK_SIZES
        for topk in SUPPORTED_TOPK
    ]
    ordered = legacy + [combo for combo in full if combo not in legacy]

    configs, index = [], 1
    for head_mode, block_size, topk in ordered:
        candidate = MsaSparseConfig(
            block_size=block_size,
            topk=topk,
            force_init_tokens=128,
            force_end_tokens=128,
            head_mode=head_mode,
            name=f"cfg{index:02d}",
        )
        if candidate.force_init_blocks + candidate.force_end_blocks > topk:
            continue  # forced windows do not fit; not a usable configuration
        configs.append(candidate.validate())
        index += 1
    return tuple(configs)


#: The requested sweep, ordered so ``CONFIG_MATRIX[i]`` is requirement item i+1.
CONFIG_MATRIX = _build_matrix()


def config_by_name(name: str, default: Optional[MsaSparseConfig] = ...) -> Optional[MsaSparseConfig]:
    """Look a config up by ``name`` or ``slug`` (``baseline`` included)."""
    for cfg in (BASELINE_CONFIG, *CONFIG_MATRIX):
        if name in (cfg.name, cfg.slug):
            return cfg
    if default is ...:
        known = ", ".join(c.resolved_name() for c in (BASELINE_CONFIG, *CONFIG_MATRIX))
        raise KeyError(f"unknown MSA config {name!r}; known names: {known}")
    return default


def iter_configs(selector: Optional[str] = None) -> Iterator[MsaSparseConfig]:
    """Yield configs named by ``selector``.

    ``None`` / ``"all"`` yields baseline + the full matrix.  ``"matrix"`` yields
    the 12 sweep points only.  Anything else is a comma-separated list of names,
    slugs or full specs.
    """
    if selector in (None, "", "all"):
        yield BASELINE_CONFIG
        yield from CONFIG_MATRIX
        return
    if selector == "matrix":
        yield from CONFIG_MATRIX
        return
    for token in selector.split(","):
        token = token.strip()
        if token:
            yield MsaSparseConfig.parse(token)


# The original 12-point request must keep its names and its meaning.
assert [c.name for c in CONFIG_MATRIX[:12]] == [f"cfg{i:02d}" for i in range(1, 13)]
assert all(
    (c.head_mode, c.block_size, c.topk) == combo
    for c, combo in zip(
        CONFIG_MATRIX[:12],
        [(h, b, k) for h in ("keep", "sum", "max") for b in (32, 64) for k in (16, 32)],
    )
), "cfg01..cfg12 must keep their original (head, block, topk) assignment"
# block=32 with topk=4 cannot hold the 128/128 forced windows, so 3 of the 36
# grid points are legitimately absent.
assert len(CONFIG_MATRIX) == 33, f"expected 33 usable grid points, got {len(CONFIG_MATRIX)}"
