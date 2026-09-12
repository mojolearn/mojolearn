# SPDX-License-Identifier: Apache-2.0
"""WHERE THE BYTE LM'S SHAPE GUARDS BIND AS `length` GROWS. Host only; no
GPU import, no context, seconds.

Every byte LM identity check in the tree runs the default profile, whose
`length` is 32. `ByteConfig.validate` carries two guards that a longer
sequence walks into, and NOBODY HAD MEASURED WHICH ONE BINDS FIRST:

  * the length cap, "length exceeds the absolute-position ceiling 8192"
    (DEVIATION 812: the Cody-Waite domain of `_cephes_sincosf_core`, NOT a
    table size -- nothing is tabulated at 8192 entries);
  * the int32 indexing guard on the shape products, whose largest term is
    `batch * length * n_heads * length` -- QUADRATIC in length, so it is
    the term that grows fastest and the one a long-context sweep meets.

This file prints, rather than asserts, the boundary: for each shape family
it sweeps `length` over the identity sweep's values, then holds `length` at
the cap and raises `batch` until a guard refuses, naming the guard that
fired. A guard that moves shows up here as a changed boundary line.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
        training/checks/byte_lm_length_sweep_check.mojo
"""
from training.byte_lm_config import ByteConfig


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def verdict(config: ByteConfig) raises -> String:
    """"OK", or the guard's own message."""
    try:
        config.validate()
    except e:
        return String(e)
    return String("OK")


def largest_product(config: ByteConfig) -> Int:
    """The largest term `validate` checks against int32 indexing. The
    quadratic one is `batch * length * n_heads * length`."""
    var m = config.batch * config.length
    var best = m * config.vocab_size
    var candidates: List[Int] = [
        m * config.d_model,
        m * config.intermediate,
        m * config.n_heads * config.length,
        m * config.n_kv * config.head_dim,
    ]
    for c in candidates:
        if c > best:
            best = c
    return best


def sweep(label: String, base: ByteConfig) raises:
    print("--- " + label)
    var lengths: List[Int] = [32, 128, 512, 2048, 8192]
    for length in lengths:
        var c = base.copy()
        c.length = length
        var v = verdict(c)
        print(
            "  length=" + String(length) + " largest_product="
            + String(largest_product(c)) + " verdict=" + v
        )
    # The cap held, the batch raised: the int32 guard's real boundary.
    var b = 1
    var first_refused = 0
    var message = String("")
    while b <= 64:
        var c = base.copy()
        c.length = 8192
        c.batch = b
        var v = verdict(c)
        if v != "OK":
            first_refused = b
            message = v
            break
        b += 1
    if first_refused == 0:
        print("  at length=8192 no batch in [1,64] is refused")
    else:
        var c = base.copy()
        c.length = 8192
        c.batch = first_refused
        print(
            "  at length=8192 the largest accepted batch is "
            + String(first_refused - 1) + "; batch=" + String(first_refused)
            + " has largest_product=" + String(largest_product(c))
            + " and is refused: " + message
        )


def main() raises:
    # The shipped identity profile: b2 L32 d32 h4 kv2 hd8 ff64 blocks2 v256.
    sweep("default identity profile (the shape every byte LM check runs)",
          ByteConfig())
    # The LM target shape (tools/lm_step_memory_probe.py TARGET_SHAPE), the
    # one the attention step and the lean training step actually run.
    sweep("LM target shape b1 d768 h12 kv12 hd64 ff2048 blocks12 v50257",
          ByteConfig(1, 2048, 768, 12, 12, 64, 2048, 12, 50257))
    # The cap itself, both sides, on the profile's own fields.
    require(verdict(ByteConfig(2, 8192)) == "OK", "length 8192 refused")
    require(verdict(ByteConfig(2, 8193)) != "OK", "length 8193 accepted")
    print("PASS byte LM length sweep: boundaries printed above")
