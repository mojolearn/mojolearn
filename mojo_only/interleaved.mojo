"""Interleaved arms in one process, because this box drifts.

NO CATBOOST COUNTERPART. It is the measurement discipline from the mojotrees
repository, brought over because every number this port has produced needed
the same caveat and it is cheaper to fix the harness than to keep repeating
it.

THE RULE IT ENFORCES
--------------------
**Arms run round-robin inside ONE process, and only arms from the same run
compare.** This machine has been measured drifting two- to threefold between
thermal windows, and this port has already produced a depth-8 tree at 59.7 ms
and 44.8 ms for IDENTICAL work an hour apart. Sequential arms in separate
runs compare nothing, whoever wrote them.

WHAT IT REPORTS
---------------
Median, min, max, and a verdict: RESOLVED when the ranges are disjoint,
INDISTINGUISHABLE when they overlap. A ratio without that verdict is an
invitation to read noise as a result, which is how the replication experiment
in this repository nearly got believed at 32 replicas before it was rerun.
"""

from std.time import perf_counter_ns


@fieldwise_init
struct ArmResult(Copyable, Movable):
    var name: String
    var median_ms: Float64
    var min_ms: Float64
    var max_ms: Float64


def summarize(name: String, mut samples: List[Float64]) raises -> ArmResult:
    """Median, min and max of one arm's samples."""
    if len(samples) == 0:
        raise Error("arm " + name + " has no samples")
    # insertion sort; the sample count is a handful by design
    for i in range(1, len(samples)):
        var v = samples[i]
        var j = i - 1
        while j >= 0 and samples[j] > v:
            samples[j + 1] = samples[j]
            j -= 1
        samples[j + 1] = v
    var n = len(samples)
    var med = samples[n // 2]
    if n % 2 == 0:
        med = (samples[n // 2 - 1] + samples[n // 2]) / 2.0
    return ArmResult(name, med, samples[0], samples[n - 1])


def report(arms: List[ArmResult]) raises:
    """Print every arm and the verdict against the first.

    The first arm is the baseline by convention, so a caller puts the shipped
    configuration first and the candidate second.
    """
    if len(arms) == 0:
        raise Error("nothing to report")
    print("    arm                         median      min      max")
    for i in range(len(arms)):
        print(
            "    ",
            arms[i].name,
            "  ",
            arms[i].median_ms,
            "  ",
            arms[i].min_ms,
            "  ",
            arms[i].max_ms,
        )
    ref base = arms[0]
    for i in range(1, len(arms)):
        ref a = arms[i]
        var disjoint = a.max_ms < base.min_ms or base.max_ms < a.min_ms
        var faster = a.median_ms < base.median_ms
        var ratio = base.median_ms / a.median_ms
        print(
            "    ",
            a.name,
            "is",
            ratio,
            "x the baseline,",
            "FASTER" if faster else "SLOWER",
            "--",
            "RESOLVED, ranges disjoint" if disjoint
            else "INDISTINGUISHABLE, ranges overlap",
        )
