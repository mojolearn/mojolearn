# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PARTS OF A NUMERIC PROFILE (lane fixed15-train, 2026-11-25; Andrew:
"configurable for EACH part").

A model's `numeric_profile=` is a PRESET (`"fixed15_v1"`, `"fp32_v1"`) or a
preset with per-part overrides, `{"preset": "fixed15_v1", "lm_head":
"fp32_v1"}`. It is resolved ONCE, at construction or load, into the PARTS
MASK below: bit `i` set means part `i` computes under
`mojolearn.identical.gemm.int15i64.v1`, clear means under
`mojolearn.identical.gemm.fp32.v1`. Every native entry (the device block,
the host oracle, the CausalLM head) reads the mask it was built with and
never a name, so no call looks anything up.

The parts are the products the profiles define for a Llama-shaped model
(`python/mojolearn/_numeric_profile.py` PARTS mirrors this table, same order):

    bit  part          product
    0    q_proj        X W_q^T, contracted over d_model
    1    k_proj        X W_k^T
    2    v_proj        X W_v^T
    3    o_proj        ctx W_o^T, contracted over n_heads * head_dim
    4    gate_proj     X W_gate^T (gated MLP only)
    5    up_proj       X W_up^T
    6    down_proj     H W_down^T, contracted over the intermediate width
    7    lm_head       h W_head^T (CausalLM / ParallelCausalLM only)
    8    attn_scores   Q K^T (S11), contracted over head_dim
    9    attn_context  P V (S19): fp32_v1 ONLY. Under the per-row rule its
                       scales would depend on the key span, so decode would
                       not equal prefill (int15_block.mojo); a fixed15
                       format for it is refused by name.

`fixed15_v1` is bits 0..8 (511); `fp32_v1` is 0.
"""

comptime NP_PART_Q = 0
comptime NP_PART_K = 1
comptime NP_PART_V = 2
comptime NP_PART_O = 3
comptime NP_PART_GATE = 4
comptime NP_PART_UP = 5
comptime NP_PART_DOWN = 6
comptime NP_PART_HEAD = 7
comptime NP_PART_SCORES = 8
comptime NP_PART_CONTEXT = 9
comptime NP_PART_COUNT = 10

#: The preset masks.
comptime NP_MASK_FP32_V1 = 0
comptime NP_MASK_FIXED15_V1 = 511
#: The bits a part may set (attn_context may not).
comptime NP_MASK_ALLOWED = 511
#: The seven projections of a block (bits 0..6).
comptime NP_MASK_PROJECTIONS = 127


@always_inline
def np_part_on(mask: Int, part: Int) -> Bool:
    """True when `part` computes under int15i64.v1 in `mask`."""
    return ((mask >> part) & 1) == 1


def np_part_name(part: Int) -> String:
    if part == NP_PART_Q:
        return "q_proj"
    if part == NP_PART_K:
        return "k_proj"
    if part == NP_PART_V:
        return "v_proj"
    if part == NP_PART_O:
        return "o_proj"
    if part == NP_PART_GATE:
        return "gate_proj"
    if part == NP_PART_UP:
        return "up_proj"
    if part == NP_PART_DOWN:
        return "down_proj"
    if part == NP_PART_HEAD:
        return "lm_head"
    if part == NP_PART_SCORES:
        return "attn_scores"
    return "attn_context"


def np_part_index(name: String) -> Int:
    """The bit of a part name, -1 for an unknown name."""
    for i in range(NP_PART_COUNT):  # small-loop(10: the part names): name lookup, no data
        if np_part_name(i) == name:
            return i
    return -1


def np_preset_mask(name: String) raises -> Int:
    """The mask of a registered preset; anything else is refused by name."""
    if name == "fixed15_v1":
        return NP_MASK_FIXED15_V1
    if name == "fp32_v1":
        return NP_MASK_FP32_V1
    raise Error("ValueError: mojolearn: numeric_profile preset " + name
                + " has no parts table; presets: 'fixed15_v1', 'fp32_v1'")


def np_mask_check(mask: Int) raises:
    """Refuse a mask no resolution can produce (a set attn_context bit or a
    bit beyond the table), by name."""
    if mask < 0 or (mask & ~NP_MASK_ALLOWED) != 0:
        raise Error("ValueError: mojolearn: numeric_profile parts mask " + String(mask)
                    + " sets a part that cannot compute under fixed15_v1 (attn_context is fp32_v1 only)")
