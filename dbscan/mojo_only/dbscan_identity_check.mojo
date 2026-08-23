"""What `IDENTICAL` promises about DBSCAN, gated on ONE device.

DBSCAN is the easiest of the three to state and the hardest to be casual
about. Its arithmetic is ONE expression -- `sum (x - y)^2` compared against
`eps^2` -- and everything after it is integer: a degree, a boolean adjacency,
a CSR, and a minimum over labels. So there is no accumulation order to pin
downstream and no float atomic anywhere. What there IS:

1. **The eps compare is a cliff, not a tolerance.** One ULP in the squared
   distance moves a point across `<= eps` and the adjacency BIT flips. Two
   clusters become one, or a border point becomes noise. That is why the
   accumulator is pinned (DEVIATION 506) even though DBSCAN has no
   reduction: the hazard is not drift, it is a discrete decision on a float
   comparison.

2. **The propagation is order-independent ONLY AT ITS FIXED POINT.**
   `weak_cc_label_kernel` pushes and pulls through `atomicMin`, and a
   minimum does not care in what order it is taken -- but that is a
   statement about where the loop STOPS, not about a loop that was cut off
   at `max_iterations`. DEVIATION 507 refuses a truncated run under
   IDENTICAL instead of returning the snapshot.

3. **The batch count is a MEMORY number that reaches the algorithm.**
   `max_mbytes_per_batch = 0` -- the default, and cuML's -- derives it from
   the device's free memory, so two GPUs decompose the same problem
   differently. The labels must not notice. That is an argument (each
   batch's `weak_cc` canonicalizes to a component minimum, and
   `merge_labels` merges under the same order) and arguments of that shape
   have been wrong here before, so it is also a gate.

WHY THE FIXTURE IS BORDER-HEAVY
--------------------------------
Every existing DBSCAN check uses well-separated blobs, where a batch-count
change cannot show: the interior of a blob is core points all the way down
and their labels are decided by the component minimum whatever order they
were found in. The points that CAN move are the border points -- non-core
points adjacent to core points of more than one cluster -- because a border
point receives a label and never propagates one, so which label it receives
is the one place a decomposition could be visible. This file's fixture is
two dense blobs with a deliberate bridge of border points between them, and
the checks assert on those points specifically.
"""

from max.gpu.host import DeviceContext, HostBuffer

from dbscan.ported.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC
from dbscan.estimator import dbscan_fit
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime DB_N = 1020
comptime DB_D = 4
#: The bridge occupies rows `DB_BRIDGE_FROM ..< DB_BRIDGE_FROM + DB_BRIDGE`.
comptime DB_BRIDGE_FROM = 1000
#: TWENTY points across ten units, so consecutive bridge points sit 0.5
#: apart: inside `eps` of each other (2 neighbours each side) and well
#: below `min_pts`, which is what makes them NON-CORE. A first attempt used
#: 200 points at 0.05 spacing; every one of them had ~40 neighbours, all of
#: them were core, and the fixture degenerated into one long cluster with
#: no border point anywhere -- caught by this check's own fixture guard.
comptime DB_BRIDGE = 20
comptime DB_EPS = 1.05
comptime DB_MIN_PTS = 8


def _mode_name() -> String:
    comptime if IDENTICAL_BUILD:
        return String("IDENTICAL")
    return String("FAST")


def _jitter(i: Int, f: Int) -> Float32:
    var h = UInt64(i + 3) * UInt64(0x9E3779B97F4A7C15) + UInt64(
        f + 11
    ) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> UInt64(32))
    return Float32(Int(h & UInt64(0xFFFF))) / Float32(65536.0)


def _build(ctx: DeviceContext) raises -> HostBuffer[DType.float32]:
    """Two dense blobs plus a sparse BRIDGE of would-be border points.

    The blobs sit at 0 and at 10 with a jitter well inside `eps`, so each is
    one component of core points. The bridge points are spread along the
    line between them at a spacing that keeps their own neighbourhoods
    BELOW `min_pts` -- so they are not core -- while the ones at each end
    fall inside `eps` of a blob. Those are the points whose label is a
    choice, and they are the reason this fixture exists.
    """
    var host = ctx.enqueue_create_host_buffer[DType.float32](DB_N * DB_D)
    ctx.synchronize()
    for i in range(DB_N):
        if i < DB_BRIDGE_FROM:
            var center = Float32(0.0) if (i % 2) == 0 else Float32(10.0)
            for f in range(DB_D):
                host.unsafe_ptr().unsafe_store(
                    i * DB_D + f, center + _jitter(i, f) * Float32(0.6)
                )
        else:
            var t = Float32(i - DB_BRIDGE_FROM) / Float32(DB_BRIDGE)
            for f in range(DB_D):
                var v = t * Float32(10.0)
                if f > 0:
                    v = _jitter(i, f) * Float32(0.2)
                host.unsafe_ptr().unsafe_store(i * DB_D + f, v)
    return host^


def _fit(
    ctx: DeviceContext,
    mut x: HostBuffer[DType.float32],
    max_mbytes: Int,
    method: Int,
    max_iterations: Int,
    mut out: List[Int32],
) raises -> Int:
    var labels = ctx.enqueue_create_host_buffer[DType.int32](DB_N)
    ctx.synchronize()
    var batches = dbscan_fit(
        ctx,
        x.unsafe_ptr(),
        DB_N,
        DB_D,
        Float64(DB_EPS),
        DB_MIN_PTS,
        labels.unsafe_ptr(),
        max_mbytes,
        max_iterations,
        method,
    )
    for i in range(DB_N):
        out.append(labels.unsafe_ptr().unsafe_load(i))
    _ = labels^
    return batches


def check_dbscan_batch_count_invariance() raises:
    """The memory budget may not reach the labels, border points included.

    Four budgets that decompose the same problem into different numbers of
    batches, on a fixture built so border points exist. Every label of every
    point must be bit-equal, and the check refuses to report a pass unless
    the fixture actually produced border points -- otherwise it proved
    nothing about the case it was built for.

    This holds in BOTH modes: batching independence is not something
    `IDENTICAL` buys, it is a property of `weak_cc` canonicalizing each
    batch to a component minimum before `merge_labels` folds them. If it
    ever fails, the argument is wrong and the ledger row must change, not
    the check.
    """
    var ctx = DeviceContext()
    var x = _build(ctx)

    var budgets = List[Int]()
    budgets.append(0)
    budgets.append(1)
    budgets.append(2)
    budgets.append(16)

    var base = List[Int32]()
    var base_batches = 0
    var seen_batches = List[Int]()
    for b in range(len(budgets)):
        var got = List[Int32]()
        var batches = _fit(ctx, x, budgets[b], EPS_NN_BRUTE_FORCE, 200, got)
        seen_batches.append(batches)
        if b == 0:
            base = got.copy()
            base_batches = batches
        else:
            var moved = 0
            for i in range(DB_N):
                if got[i] != base[i]:
                    moved += 1
            if moved != 0:
                raise Error(
                    "check_dbscan_batch_count_invariance: budget "
                    + String(budgets[b])
                    + " MB ("
                    + String(batches)
                    + " batches) moved "
                    + String(moved)
                    + " of "
                    + String(DB_N)
                    + " labels against the default budget ("
                    + String(base_batches)
                    + " batches). A memory number is reaching the answer."
                )

    # THE FIXTURE GUARD. Border points are non-core points that got a label
    # anyway; noise is a non-core point that got none. If the bridge
    # produced neither, this fixture is two blobs and the check is vacuous.
    var bridge_labelled = 0
    var bridge_noise = 0
    for i in range(DB_BRIDGE_FROM, DB_N):
        if base[i] < Int32(0):
            bridge_noise += 1
        else:
            bridge_labelled += 1
    if bridge_labelled == 0 or bridge_noise == 0:
        raise Error(
            "check_dbscan_batch_count_invariance: the bridge produced "
            + String(bridge_labelled)
            + " labelled and "
            + String(bridge_noise)
            + " noise points, so it is not the mixed border/noise fixture"
            " this check needs. Adjust the spacing, not the assertion."
        )

    var spread = String("")
    for b in range(len(seen_batches)):
        spread += String(seen_batches[b])
        if b + 1 < len(seen_batches):
            spread += "/"
    print(
        "check_dbscan_batch_count_invariance OK (" + _mode_name() + "):",
        DB_N,
        "labels bit-identical at",
        spread,
        "batches, with",
        bridge_labelled,
        "border points and",
        bridge_noise,
        "noise points in the bridge",
    )


def check_dbscan_arms_agree_on_the_border() raises:
    """The two neighbourhood arms must agree, border points included.

    `EPS_NN_RBC` and `EPS_NN_BRUTE_FORCE` compute the same predicate by
    different routes -- the ball-cover index prunes with the triangle
    inequality, the brute-force kernel does not -- and both accumulate
    `sum (x - y)^2` in float32. Under IDENTICAL both accumulators are
    pinned (DEVIATION 506), so a point exactly on the eps boundary must
    fall the same way in both. This is the check that would catch a pin
    applied to one arm and forgotten on the other, which is how row 9's
    checklist has been missed before.
    """
    var ctx = DeviceContext()
    var x = _build(ctx)

    var brute = List[Int32]()
    var rbc = List[Int32]()
    _ = _fit(ctx, x, 0, EPS_NN_BRUTE_FORCE, 200, brute)
    _ = _fit(ctx, x, 0, EPS_NN_RBC, 200, rbc)

    var moved = 0
    for i in range(DB_N):
        if brute[i] != rbc[i]:
            moved += 1
    if moved != 0:
        raise Error(
            "check_dbscan_arms_agree_on_the_border: the rbc and"
            " brute-force arms disagree on "
            + String(moved)
            + " of "
            + String(DB_N)
            + " labels. They compute the same predicate; a disagreement is"
            " one arm's accumulator rounding differently from the other's."
        )
    print(
        "check_dbscan_arms_agree_on_the_border OK (" + _mode_name() + "):",
        DB_N,
        "labels identical through the ball-cover index and through brute",
        "force",
    )


def check_dbscan_refuses_truncated_propagation() raises:
    """DEVIATION 507. A cap that binds is not a fixed point.

    `max_iterations = 1` cannot settle this fixture: the bridge is a long
    chain and a label has to walk it. Under IDENTICAL the run must RAISE --
    the labels it would return are a snapshot of the atomic order on this
    machine. Under FAST it must return upstream's truncated answer, which
    this check requires to DIFFER from the converged one, because a
    refusal that costs nothing would not be worth having.
    """
    var ctx = DeviceContext()
    var x = _build(ctx)

    var converged = List[Int32]()
    _ = _fit(ctx, x, 0, EPS_NN_BRUTE_FORCE, 200, converged)

    var truncated = List[Int32]()
    var raised = False
    try:
        _ = _fit(ctx, x, 0, EPS_NN_BRUTE_FORCE, 1, truncated)
    except e:
        raised = True
        var msg = String(e)
        if msg.find("did not converge") < 0 and msg.find("did not settle") < 0:
            raise Error(
                "check_dbscan_refuses_truncated_propagation: the run"
                " raised, but not with the non-convergence refusal: " + msg
            )

    comptime if IDENTICAL_BUILD:
        if not raised:
            var moved = 0
            for i in range(DB_N):
                if truncated[i] != converged[i]:
                    moved += 1
            raise Error(
                "check_dbscan_refuses_truncated_propagation (IDENTICAL):"
                " max_iterations = 1 returned a labelling ("
                + String(moved)
                + " labels away from the converged one) instead of"
                " raising. DEVIATION 507 is not reached."
            )
        print(
            "check_dbscan_refuses_truncated_propagation OK (IDENTICAL):",
            "a propagation cut off at one pass is refused, not returned",
        )
    else:
        if raised:
            raise Error(
                "check_dbscan_refuses_truncated_propagation (FAST): the"
                " truncated run raised. Under FAST the upstream's silent"
                " truncation is the behaviour, and changing it would be an"
                " improvement on cuML rather than a port of it."
            )
        var moved = 0
        for i in range(DB_N):
            if truncated[i] != converged[i]:
                moved += 1
        if moved == 0:
            raise Error(
                "check_dbscan_refuses_truncated_propagation (FAST): one"
                " pass produced the SAME labelling as convergence, so this"
                " fixture cannot show what the refusal is worth. Lengthen"
                " the bridge."
            )
        print(
            "check_dbscan_refuses_truncated_propagation OK (FAST): one"
            " pass returns a labelling",
            moved,
            "of",
            DB_N,
            "labels away from the converged one -- silently, which is",
            "upstream's behaviour and what IDENTICAL refuses",
        )


def main() raises:
    check_dbscan_batch_count_invariance()
    check_dbscan_arms_agree_on_the_border()
    check_dbscan_refuses_truncated_propagation()
