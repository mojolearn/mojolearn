# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The parts mask as a native entry receives it, and the sabotage arms of the
per-part profile (lane fixed15-parts-native, 2026-11-25).

`np_block_parts` is the part of a model's parts mask
(`core/numeric_profile_parts.mojo`) that a Transformer block computes: the
seven projections and attn_scores (bits 0..6 and 8). The head bit (7) is the
CausalLM head's, and attn_context is never set.

THE SABOTAGE ARMS, one pair per side, all default off. Each must change the
digest of an override case on its own side only; the verify sabotage registry
lists them (the verify sub-lane wires them):

  MOJOLEARN_NP_PARTS_SABOTAGE_WRONG_PART_HOST    the q_proj bit is applied to
  MOJOLEARN_NP_PARTS_SABOTAGE_WRONG_PART_DEVICE  k_proj and the k_proj bit to
                                                 q_proj
  MOJOLEARN_NP_PARTS_SABOTAGE_IGNORE_HOST        the overrides are ignored: any
  MOJOLEARN_NP_PARTS_SABOTAGE_IGNORE_DEVICE      mask with a bit set becomes
                                                 the fixed15_v1 preset's (511)

The HOST arms act where the host block receives its mask (the CPU column's
`_run_forward`, every host entry of the block). The DEVICE arms act where a
device model resolves its mask at construction or load
(`bindings/causal_lm_construct_door.mojo` `_model_parts_mask`, a `gpu`
route only), BEFORE the planes are made: a device part the arm turns on is
quantized from its float32 weight there, so the device column computes the
sabotaged mask with its own kernels.

Lane fixed15-decode (2026-11-25), one more pair: the GENERATION routes
ignore the mask.

  MOJOLEARN_NP_PARTS_SABOTAGE_PROGRAM_IGNORE_HOST    the CPU generation
                                                     engine runs every
                                                     fixed15 layer on the
                                                     float32 entry (op 33)
  MOJOLEARN_NP_PARTS_SABOTAGE_PROGRAM_IGNORE_DEVICE  the GPU decode program
                                                     drops a layer's
                                                     attn_scores bit (the
                                                     one bit its addresses
                                                     do not carry)

Each moves the default `generate` case of numeric-profile-parts on its own
side only (the override with attn_scores on fp32_v1 stays for the device
arm)."""

from std.sys import is_defined

from core.numeric_profile_parts import (
    NP_MASK_FIXED15_V1, NP_MASK_PROJECTIONS, NP_PART_K, NP_PART_Q, NP_PART_SCORES,
    np_mask_check,
)

comptime NP_PARTS_SABOTAGE_WRONG_PART_HOST = is_defined["MOJOLEARN_NP_PARTS_SABOTAGE_WRONG_PART_HOST"]()
comptime NP_PARTS_SABOTAGE_WRONG_PART_DEVICE = is_defined["MOJOLEARN_NP_PARTS_SABOTAGE_WRONG_PART_DEVICE"]()
comptime NP_PARTS_SABOTAGE_IGNORE_HOST = is_defined["MOJOLEARN_NP_PARTS_SABOTAGE_IGNORE_HOST"]()
comptime NP_PARTS_SABOTAGE_IGNORE_DEVICE = is_defined["MOJOLEARN_NP_PARTS_SABOTAGE_IGNORE_DEVICE"]()
comptime NP_PARTS_SABOTAGE_PROGRAM_IGNORE_HOST = is_defined["MOJOLEARN_NP_PARTS_SABOTAGE_PROGRAM_IGNORE_HOST"]()
comptime NP_PARTS_SABOTAGE_PROGRAM_IGNORE_DEVICE = is_defined["MOJOLEARN_NP_PARTS_SABOTAGE_PROGRAM_IGNORE_DEVICE"]()

#: The bits a Transformer block computes: the seven projections and attn_scores.
comptime NP_MASK_BLOCK = NP_MASK_PROJECTIONS | (1 << NP_PART_SCORES)


@always_inline
def np_block_parts(mask: Int) -> Int:
    """The block's bits of a model's parts mask (module docstring)."""
    return mask & NP_MASK_BLOCK


def np_entry_parts(mask: Int, what: String) raises -> Int:
    """A fixed15 block entry's mask word: refused by name when it sets a bit
    no resolution produces, or no part the block computes (such a block runs
    the fp32_v1 entry, never this one); returned as the block's bits."""
    np_mask_check(mask)
    var block = np_block_parts(mask)
    if block == 0:
        raise Error(what + ": the parts mask " + String(mask) + " sets no part of the block; "
                    + "a block with no fixed15_v1 part runs the fp32_v1 entry")
    return block


@always_inline
def _np_swap_qk(mask: Int) -> Int:
    var q = (mask >> NP_PART_Q) & 1
    var k = (mask >> NP_PART_K) & 1
    var cleared = mask & ~((1 << NP_PART_Q) | (1 << NP_PART_K))
    return cleared | (k << NP_PART_Q) | (q << NP_PART_K)


@always_inline
def _np_ignore(mask: Int) -> Int:
    return NP_MASK_FIXED15_V1 if mask != 0 else mask


def np_parts_sabotage_host(mask: Int) -> Int:
    """The host block's mask under the host arms (identity when none is on)."""
    var out = mask
    comptime if NP_PARTS_SABOTAGE_WRONG_PART_HOST:
        out = _np_swap_qk(out)
    comptime if NP_PARTS_SABOTAGE_IGNORE_HOST:
        out = _np_ignore(out)
    return out


def np_parts_sabotage_device(mask: Int) -> Int:
    """A device model's mask under the device arms (identity when none is on)."""
    var out = mask
    comptime if NP_PARTS_SABOTAGE_WRONG_PART_DEVICE:
        out = _np_swap_qk(out)
    comptime if NP_PARTS_SABOTAGE_IGNORE_DEVICE:
        out = _np_ignore(out)
    return out


def np_parts_sabotage_program(mask: Int) -> Int:
    """The decode program's layer mask under its device arm (identity when
    it is off): the attn_scores bit dropped while a projection keeps the
    block on its fixed15 entry."""
    var out = mask
    comptime if NP_PARTS_SABOTAGE_PROGRAM_IGNORE_DEVICE:
        if (out & NP_MASK_PROJECTIONS) != 0:
            out = out & ~(1 << NP_PART_SCORES)
    return out


def np_parts_sabotage_name() -> String:
    """The armed parts sabotage defines, comma separated, or "" (a report
    word for the check and the verify banner)."""
    var names = String("")
    comptime if NP_PARTS_SABOTAGE_WRONG_PART_HOST:
        names += "MOJOLEARN_NP_PARTS_SABOTAGE_WRONG_PART_HOST,"
    comptime if NP_PARTS_SABOTAGE_WRONG_PART_DEVICE:
        names += "MOJOLEARN_NP_PARTS_SABOTAGE_WRONG_PART_DEVICE,"
    comptime if NP_PARTS_SABOTAGE_IGNORE_HOST:
        names += "MOJOLEARN_NP_PARTS_SABOTAGE_IGNORE_HOST,"
    comptime if NP_PARTS_SABOTAGE_IGNORE_DEVICE:
        names += "MOJOLEARN_NP_PARTS_SABOTAGE_IGNORE_DEVICE,"
    comptime if NP_PARTS_SABOTAGE_PROGRAM_IGNORE_HOST:
        names += "MOJOLEARN_NP_PARTS_SABOTAGE_PROGRAM_IGNORE_HOST,"
    comptime if NP_PARTS_SABOTAGE_PROGRAM_IGNORE_DEVICE:
        names += "MOJOLEARN_NP_PARTS_SABOTAGE_PROGRAM_IGNORE_DEVICE,"
    return names
