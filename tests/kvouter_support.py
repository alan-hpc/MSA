# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Notes for tests that exercise the Fireworks KV-outer sparse backend."""

# sparse_fmha() routes prefill through KV-outer when qhead_per_kv >= this value.
MIN_KVOUTER_QHEADS_PER_KV = 8

# Use this string in commented-out test cases so the reason stays consistent.
KVOUTER_DISABLED_REASON = (
    "Disabled: KV-outer sparse prefill only supports qhead_per_kv >= 8 "
    "(MHA / GQA2 / GQA4 not implemented yet)."
)
