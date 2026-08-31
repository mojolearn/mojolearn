# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One bootstrap, the identity card, and the mode it ran in.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.resample.card \\
        tools/with_build_lock.sh pixi run mojo run -I . resample/resample_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.resample.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . resample/resample_main.mojo

    python3 tools/identity_trace_diff.py \\
        /tmp/mac.resample.identical.card /tmp/<other>.resample.identical.card

Environment knobs, all optional: `MOJOLEARN_RESAMPLE_STAT` (default `mean`;
any name `stat_from_name` takes), `MOJOLEARN_RESAMPLE_METHOD` (default
`percentile`), `MOJOLEARN_RESAMPLE_N` (default 10000 resamples),
`MOJOLEARN_RESAMPLE_Q` is not offered -- the quantile arm runs at q = 0.5,
because a card whose stage meaning depends on an unrecorded environment
variable is a card two runs cannot be compared through.

THE CARD, in the order it is written:

    resample.key         u32 x2   the derived key (DEVIATION 1691)
    resample.index_map   i32      4 replicates x 32 positions of the map
    resample.theta       f32      the bootstrap distribution, replicate order
    resample.sorted      f32      the same values, sorted (the interval's input)
    resample.point       f32 x1   theta_hat, the point estimate
    resample.order_pos   i32 x2   the two order-statistic positions
    resample.se          f32 x1   the standard error
    resample.interval    f32 x2   the two endpoints
    resample.bca.z0p     f32 x1   BCa's bias percentile        (diagnostics on)
    resample.jackknife   f32      the n leave-one-out statistics (diagnostics on)
    resample.bca.ahat    f32 x1   BCa's acceleration           (diagnostics on)

A cross-vendor run that diverges has an ADDRESS, and each address has a
cause. `resample.key` is pure integer arithmetic and CANNOT differ -- if it
does, the two runs were not given the same seed. `resample.index_map` is pure
integer arithmetic too, so a difference there is a Philox port defect and
`ensemble/bench/philox_oracle.txt` is where to take it. `resample.theta` is
the first float stage and is the pinned tree (IDENTITY_PATHS rows 9, 10, 20,
21). `resample.sorted` is a permutation of `resample.theta`, so a difference
there WITH `resample.theta` matching is a sort defect and nothing else.
`resample.point`, `resample.se` and `resample.interval` are host scalars over
the same tree.

The fixture performs no host floating-point operation on a hashed value; the
integer fixtures are exact by construction (`resample/mojo_only/
resample_fixture.mojo`). Prints the first values as decimal AND hex, because
`String(Float32)` does not round-trip in this toolchain.

NOTHING BELOW HAS BEEN RUN.
"""

from std.memory import bitcast
from std.os import getenv

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from resample.estimator import bootstrap_host
from resample.mojo_only.intervals import (
    ALT_TWO_SIDED,
    METHOD_BCA,
    method_from_name,
    method_name,
)
from resample.mojo_only.resample_fixture import (
    FIX_HASHED,
    build_sample,
    fixture_d,
    fixture_n,
)
from resample.mojo_only.statistics import (
    STAT_DIFF_MEANS,
    STAT_MEAN,
    STAT_STD,
    stat_from_name,
    stat_name,
)


comptime RESAMPLE_MAIN_SEED: UInt64 = 20260825
comptime RESAMPLE_MAIN_CONFIDENCE = Float32(0.95)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def main() raises:
    var stat_str = String(getenv("MOJOLEARN_RESAMPLE_STAT"))
    if stat_str == "":
        stat_str = String("mean")
    var method_str = String(getenv("MOJOLEARN_RESAMPLE_METHOD"))
    if method_str == "":
        method_str = String("percentile")
    var n_str = String(getenv("MOJOLEARN_RESAMPLE_N"))
    var n_resamples = 10000
    if n_str != "":
        n_resamples = Int(atol(n_str))

    var stat = stat_from_name(stat_str)
    var method = method_from_name(method_str)
    var fix = FIX_HASHED
    var n = fixture_n(fix)
    var d = fixture_d(fix)
    var x = build_sample(fix)

    print(
        "== resample/resample_main.mojo ["
        + _mode_name()
        + "] statistic="
        + stat_name(stat)
        + " method="
        + method_name(method)
        + " n="
        + String(n)
        + " d="
        + String(d)
        + " n_resamples="
        + String(n_resamples)
        + " seed="
        + String(RESAMPLE_MAIN_SEED)
        + " =="
    )

    if method == METHOD_BCA:
        # DEVIATION 1699. Print the refusal rather than letting the driver
        # die with a stack trace: a card run that asks for BCa should say
        # WHY it got no card, and the wording is `bca_refuse`'s.
        print(
            "  method=BCa is REFUSED (DEVIATION 1699). Run with"
            " MOJOLEARN_RESAMPLE_METHOD=percentile or basic; the BCa"
            " diagnostics (bias percentile, jackknife, acceleration) are"
            " recorded on the card of every run whose statistic has a"
            " jackknife arm."
        )
        return

    # The BCa diagnostics ride along for the three statistics that have a
    # jackknife arm, so the identical half of DEVIATION 1699 appears on the
    # card of an ordinary run rather than only inside a check.
    var with_bca = (
        stat == STAT_MEAN or stat == STAT_STD or stat == STAT_DIFF_MEANS
    )

    var res = bootstrap_host(
        x,
        n,
        d,
        stat,
        n_resamples,
        RESAMPLE_MAIN_SEED,
        method,
        RESAMPLE_MAIN_CONFIDENCE,
        ALT_TWO_SIDED,
        Float32(0.5),
        0,
        256,
        with_bca,
    )

    var trace_path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if trace_path == "":
        print(
            "no MOJOLEARN_IDENTITY_TRACE set: the bootstrap ran, no card"
            " written"
        )
    else:
        var stages = 11 if with_bca else 8
        print(
            "card written to "
            + trace_path
            + " ("
            + String(stages)
            + " stages"
            + (
                ", BCa diagnostics included"
                if with_bca
                else ", no jackknife arm for this statistic"
            )
            + ")"
        )

    print(
        "  point estimate  = "
        + String(res.point_estimate)
        + "  "
        + _hex32(res.point_estimate)
    )
    print(
        "  standard error  = "
        + String(res.standard_error)
        + "  "
        + _hex32(res.standard_error)
    )
    print(
        "  interval        = ["
        + String(res.interval.low)
        + ", "
        + String(res.interval.high)
        + "]  ["
        + _hex32(res.interval.low)
        + ", "
        + _hex32(res.interval.high)
        + "]"
    )
    print(
        "  order positions = ["
        + String(res.order_low)
        + ", "
        + String(res.order_high)
        + "] of "
        + String(n_resamples)
        + " sorted replicates"
    )
    for i in range(4):
        print(
            "  theta["
            + String(i)
            + "] = "
            + String(res.distribution[i])
            + "  "
            + _hex32(res.distribution[i])
        )
    print(
        "  sorted[0] = "
        + _hex32(res.sorted_distribution[0])
        + "  sorted[last] = "
        + _hex32(res.sorted_distribution[n_resamples - 1])
    )
