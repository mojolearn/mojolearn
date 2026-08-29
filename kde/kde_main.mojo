"""KernelDensity driver: one hashed fit, the identity card, the mode it ran in.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.kde.card \\
        tools/with_build_lock.sh pixi run mojo run -I . kde/kde_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.kde.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . kde/kde_main.mojo

    python3 tools/identity_trace_diff.py /tmp/mac.kde.identical.card /tmp/<other>.kde.identical.card

Environment knobs (all optional): `MOJOLEARN_KDE_KERNEL` (default
`gaussian`), `MOJOLEARN_KDE_METRIC` (default `euclidean`),
`MOJOLEARN_KDE_WEIGHTED` (`1` to attach hashed weights; default on).

THE CARD: seven stages, FNV-1a64 over raw bytes --
`kde.dists` (n_query x n_train), `kde.logk` (same), `kde.rowmax`,
`kde.logsumexp`, `kde.logsw` (scalar), `kde.lognorm` (scalar),
`kde.scores`. A cross-vendor run that diverges has an address: `dists` is
IDENTITY_PATHS rows 9/10 (the per-cell contraction and flush), `logk` is
row 12 (the device log/cos) plus row 10, `rowmax` is row 13 (the zero-sign
question, closed in `logsumexp_kernel`'s docstring), `logsumexp` is row 12
(exp/log) over a serial fold that is a pure function of `n_train`, `logsw`
and `lognorm` are HOST scalars (row 18's class; `lognorm` is DEVIATION
601's construction under IDENTICAL), and `scores` is two subtractions.

The fixture performs no host floating-point operation (see
`kde/mojo_only/kde_fixture.mojo`). Prints the first eight scores as decimal
AND hex, because `String(Float32)` does not round-trip.
"""

from std.memory import bitcast
from std.os import getenv

from kde.estimator import kde_score_samples_host
from kde.mojo_only.kde_fixture import query_fixture, train_fixture, weight_fixture
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


comptime KDE_MAIN_N_TRAIN = 1024
comptime KDE_MAIN_N_QUERY = 256
comptime KDE_MAIN_N_FEATURES = 8
comptime KDE_MAIN_BANDWIDTH = Float32(2.75)


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
    var kernel = String(getenv("MOJOLEARN_KDE_KERNEL"))
    if kernel == "":
        kernel = String("gaussian")
    var metric = String(getenv("MOJOLEARN_KDE_METRIC"))
    if metric == "":
        metric = String("euclidean")
    var weighted = String(getenv("MOJOLEARN_KDE_WEIGHTED")) != "0"
    print(
        "== kde/kde_main.mojo [" + _mode_name() + "] kernel=" + kernel + " metric=" + metric
        + " weighted=" + String(weighted) + " n_train=" + String(KDE_MAIN_N_TRAIN)
        + " n_query=" + String(KDE_MAIN_N_QUERY) + " d=" + String(KDE_MAIN_N_FEATURES)
        + " h=" + String(KDE_MAIN_BANDWIDTH) + " =="
    )
    var train = train_fixture(KDE_MAIN_N_TRAIN, KDE_MAIN_N_FEATURES, 1)
    var query = query_fixture(train, KDE_MAIN_N_TRAIN, KDE_MAIN_N_QUERY, KDE_MAIN_N_FEATURES, 1)
    var weights = weight_fixture(KDE_MAIN_N_TRAIN, 1)
    var scores = kde_score_samples_host(
        train, KDE_MAIN_N_TRAIN, query, KDE_MAIN_N_QUERY, KDE_MAIN_N_FEATURES,
        KDE_MAIN_BANDWIDTH, kernel, metric, weights, weighted,
    )
    var trace_path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if trace_path == "":
        print("no MOJOLEARN_IDENTITY_TRACE set: scores computed, no card written")
    else:
        print("card written to " + trace_path + " (7 stages)")
    for i in range(8):
        print("  score[" + String(i) + "] = " + String(scores[i]) + "  " + _hex32(scores[i]))
    var n_sentinel = 0
    for i in range(KDE_MAIN_N_QUERY):
        if scores[i] < Float32(-1e38):
            n_sentinel += 1
    print(
        "  " + String(KDE_MAIN_N_QUERY - n_sentinel) + " finite scores, " + String(n_sentinel)
        + " at the FLOAT_MIN sentinel (outside every training point's support)"
    )
