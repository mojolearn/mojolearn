"""Does the per-node FEATURE SAMPLER, on the device, agree with the host
transcription slot for slot?

    pixi run mojo run -I . extratrees/mojo_only/sampler_kernel_check.mojo

NO CUML COUNTERPART -- this is a check. It covers the device half of
`extratrees/ported/decisiontree/batched_levelalgo/kernels/builder_kernels.mojo`:
`excess_sample_kernel`, `sample_all_features_kernel`,
`algo_l_sample_kernel` and the `sample_features_device` dispatch, which carry
DEVIATIONS 195-199.

THE CLAIM, AND WHY IT IS THE STRONGEST ONE AVAILABLE
----------------------------------------------------
Not "the device draws an equivalent sample". **Bit-identical `colids`, per
slot, per work item, against the host transcription** -- and the host
transcription is already checked against cuML's algorithm branch by branch
over 1,063,780 cells by `feature_sampler_check.mojo`, so matching it IS
matching `builder_kernels.cuh:152-248`.

That the comparison CAN be exact is a property of the algorithm and not a
hope. Every draw is keyed on `(seed, threadIdx, treeid, nodeid)` through
`excess_subsequence`, so no thread's stream depends on when it ran; the sort
is a fixed network over a fixed set of slots; the dedupe is a pure function of
two adjacent slots; and the two scan outputs are block collectives whose
aggregate is broadcast. Given `BLOCK_THREADS` and `MAX_SAMPLES_PER_THREAD`
there is exactly one right answer and the host has it. A tolerance here would
hide a defect, so there is none: every assertion below is `==` on `Int32`.

WHAT THE DEVICE SAYS ABOUT ITSELF, BECAUSE THE HOST MUST NOT INFER IT
---------------------------------------------------------------------
Rule 8: a check that cannot NAME the kernel it ran can pass about a different
one. `d_report` is `[arm, instantiation, iterations]` per work item, written
ONLY by the kernels -- no host code touches it -- and seeded with
`SAMPLER_UNVISITED`, which is not a legal outcome. Every case asserts, per
work item, that the seed is gone and that the arm and the instantiation are
the ones `plan_feature_sampling` said. Three cells past the end of `d_report`
are allocated and never written by anything: they are asserted to STILL hold
the seed, which is what proves the seed took and that no kernel writes out of
its range. Rule 15: reach is per branch, and a digest cannot tell a working
change from a no-op.

THE ARMS, WHICH ARE THREE AND NOT TWO
--------------------------------------
`excess_sample_with_replacement_kernel` is instantiated at
`MAX_SAMPLES_PER_THREAD = 1` and at `= 72` (`builder.cuh:434-455`) and the two
consume the RNG at different rates, so they are different kernels returning
different column sets. The `(n, k)` pairs below are the ones
`feature_sampler_check.mojo` already proved route to each arm, reused
deliberately: a pair whose routing is asserted in two files cannot drift in
one of them.

WHAT THIS CHECK CANNOT DO, STATED HERE RATHER THAN OMITTED
-----------------------------------------------------------
`algo_l_sample_kernel` NEVER RUNS ON THIS BOX. Metal has no `double` and
cuML's algorithm L is a `double` algorithm. The check asserts the REFUSAL
instead -- by name, from `sample_features_device` -- and that is an assertion
about the shipping path, not a skip. DEVIATION 199 has the measured backend
error and the reason a float32 substitute was not shipped.
"""

from std.sys import has_accelerator
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    FeatureSamplerPlan,
    InstanceRange,
    NodeWorkItem,
    SAMPLE_ALGO_L,
    SAMPLE_ALL_FEATURES,
    SAMPLE_EXCESS,
    SAMPLER_BLOCK_THREADS,
    SAMPLER_MAX_SAMPLES_PER_THREAD,
    SAMPLER_OVERRUN,
    SAMPLER_UNVISITED,
    SAMP_SAB_AGG_ONE_PER_THREAD,
    SAMP_SAB_ALL_REVERSED,
    SAMP_SAB_CUML_FILLER,
    SAMP_SAB_CUML_MASK0,
    SAMP_SAB_KEY_NO_THREAD,
    SAMP_SAB_LOOP_ANY_UNIQUE,
    SAMP_SAB_NONE,
    SAMP_SAB_NO_DEDUPE,
    SAMP_SAB_SORT_DESCENDING,
    device_has_float64,
    excess_scratch_stride,
    plan_feature_sampling,
    sample_features,
    sample_features_device,
    sampler_arm_name,
    sampler_report_len,
    sampler_scratch_len,
)


@fieldwise_init
struct SabCase(Copyable, Movable):
    """One sabotage arm, and the `(n, k)` whose instantiation actually has the
    mechanism it attacks."""

    var sab: Int32
    var name: String
    var n: Int
    var k: Int


def _vendor() -> String:
    if has_apple_gpu_accelerator():
        return String("apple")
    if has_nvidia_gpu_accelerator():
        return String("nvidia")
    if has_amd_gpu_accelerator():
        return String("amd")
    return String("unknown-vendor")


def work_items_for(node_ids: List[Int]) -> List[NodeWorkItem]:
    """The SAME batch shape `feature_sampler_check.work_items_for` builds, so
    the host oracle this file compares against is the one that file checked.
    The instance range is filler and deliberately not uniform: neither sampler
    reads it (`builder_kernels.cuh:165` and `:279` take only `.idx`), and if
    one ever started, that would show."""
    var items = List[NodeWorkItem]()
    for i in range(len(node_ids)):
        items.append(
            NodeWorkItem(
                Int32(node_ids[i]),
                Int32(i % 7),
                InstanceRange(Int32(i * 13), Int32(11 + i)),
            )
        )
    return items^


@fieldwise_init
struct DeviceRun(Copyable, Movable):
    """One launch's output, brought back."""

    var colids: List[Int32]
    var report: List[Int32]
    var tail: List[Int32]
    """The three cells past the end of `report`, which nothing writes."""
    var plan: FeatureSamplerPlan


def run_device(
    ctx: DeviceContext,
    node_ids: List[Int],
    tree_id: Int32,
    seed: UInt64,
    n: Int,
    k: Int,
    sabotage: Int32,
) raises -> DeviceRun:
    """One `sample_features_device` launch, with fresh buffers every time.

    Fresh buffers on purpose: a reused `colids` would let a kernel that writes
    nothing pass by leaving the previous arm's answer in place. Every cell is
    seeded with a value the samplers cannot produce.
    """
    var w = len(node_ids)
    var items = work_items_for(node_ids)
    var plan = plan_feature_sampling(n, k)

    var n_col = w * k
    var n_rep = sampler_report_len(w)
    var n_scratch = sampler_scratch_len(w, n, k)

    var d_colids = ctx.enqueue_create_buffer[DType.int32](n_col)
    var d_scratch = ctx.enqueue_create_buffer[DType.int32](n_scratch)
    # THREE CELLS OF OVERHANG. Nothing writes them; they are asserted to hold
    # the seed afterwards, which is what proves the seed took at all.
    var d_report = ctx.enqueue_create_buffer[DType.int32](n_rep + 3)
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        w * size_of[NodeWorkItem]()
    )
    var h_items = ctx.enqueue_create_host_buffer[DType.uint8](
        w * size_of[NodeWorkItem]()
    )
    ctx.synchronize()

    var ip = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    for i in range(w):
        ip[unsafe_offset=i] = items[i]
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())
    # `-9` is not a column id and not a legal report value.
    d_colids.enqueue_fill(Int32(-9))
    d_report.enqueue_fill(SAMPLER_UNVISITED)
    d_scratch.enqueue_fill(Int32(-9))
    ctx.synchronize()

    var got_plan = sample_features_device(
        ctx,
        d_colids.unsafe_ptr(),
        d_scratch.unsafe_ptr(),
        d_report.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        w,
        tree_id,
        seed,
        n,
        k,
        sabotage,
    )
    assert_equal(
        got_plan.arm, plan.arm, "the driver must report the planned arm"
    )

    var o_col = ctx.enqueue_create_host_buffer[DType.int32](n_col)
    var o_rep = ctx.enqueue_create_host_buffer[DType.int32](n_rep + 3)
    ctx.enqueue_copy(dst_buf=o_col, src_buf=d_colids)
    ctx.enqueue_copy(dst_buf=o_rep, src_buf=d_report)
    ctx.synchronize()

    var colids = List[Int32]()
    for i in range(n_col):
        colids.append(o_col.unsafe_ptr().unsafe_load(i))
    var report = List[Int32]()
    for i in range(n_rep):
        report.append(o_rep.unsafe_ptr().unsafe_load(i))
    var tail = List[Int32]()
    for i in range(3):
        tail.append(o_rep.unsafe_ptr().unsafe_load(n_rep + i))
    return DeviceRun(colids^, report^, tail^, plan)


def iter_span(run: DeviceRun, w: Int) -> String:
    """The min and max iteration count the DEVICE reported across the batch.
    Printed rather than summarised because the do-while's iteration count is
    what decides whether the `n_uniques >= k` mechanism exists in a case at
    all: a batch that is 1 everywhere cannot see a sabotage of it."""
    var lo = run.report[2]
    var hi = run.report[2]
    for b in range(w):
        var v = run.report[3 * b + 2]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return String(lo, "..", hi)


def host_reference(
    node_ids: List[Int], tree_id: Int32, seed: UInt64, n: Int, k: Int
) raises -> List[Int32]:
    var items = work_items_for(node_ids)
    var colids = List[Int32](length=len(node_ids) * k, fill=Int32(-1))
    _ = sample_features(colids, items, tree_id, seed, n, k)
    return colids^


def assert_report(
    run: DeviceRun, w: Int, expected_arm: Int, expected_m: Int, label: String
) raises -> Int:
    """The DEVICE's statement about which kernel ran, asserted per work item.
    Returns the number of assertions made."""
    var cells = 0
    for i in range(3):
        assert_equal(
            run.tail[i],
            SAMPLER_UNVISITED,
            String(
                label,
                ": a cell PAST the end of `report` was written -- either a"
                " kernel indexes out of its range, or the seed never took and"
                " every 'the device said so' assertion below is vacuous",
            ),
        )
        cells += 1
    for b in range(w):
        var arm = Int(run.report[3 * b + 0])
        var inst = Int(run.report[3 * b + 1])
        var iters = run.report[3 * b + 2]
        assert_true(
            Int32(arm) != SAMPLER_UNVISITED,
            String(
                label,
                ": work item ",
                b,
                " still holds SAMPLER_UNVISITED -- no block served it, so the"
                " grid is too small or the kernel never ran",
            ),
        )
        assert_equal(
            arm,
            expected_arm,
            String(
                label,
                ": work item ",
                b,
                " reports arm ",
                sampler_arm_name(arm),
                ", the dispatch planned ",
                sampler_arm_name(expected_arm),
            ),
        )
        assert_equal(
            inst,
            expected_m,
            String(
                label,
                ": work item ",
                b,
                " reports instantiation MAX_SAMPLES_PER_THREAD=",
                inst,
                ", expected ",
                expected_m,
            ),
        )
        assert_true(
            iters != SAMPLER_OVERRUN,
            String(
                label,
                ": work item ",
                b,
                " hit EXCESS_MAX_ITERATIONS and wrote no columns",
            ),
        )
        if expected_arm == SAMPLE_EXCESS:
            assert_true(
                iters >= Int32(1),
                String(label, ": work item ", b, " ran ", iters, " iterations"),
            )
        cells += 4
    return cells


def compare_slots(
    got: List[Int32], want: List[Int32], label: String
) raises -> Int:
    """Bit-identical, per slot. Returns the number of cells compared."""
    assert_equal(len(got), len(want), String(label, ": length"))
    for i in range(len(want)):
        assert_equal(
            got[i],
            want[i],
            String(
                label,
                ": slot ",
                i,
                " -- device ",
                got[i],
                ", host transcription ",
                want[i],
                ". This comparison is exact by construction; see the header.",
            ),
        )
    return len(want)


def differing(got: List[Int32], want: List[Int32]) -> Int:
    var d = 0
    for i in range(len(want)):
        if got[i] != want[i]:
            d += 1
    return d


def run_case(
    ctx: DeviceContext,
    n: Int,
    k: Int,
    node_ids: List[Int],
    tree_id: Int32,
    seed: UInt64,
    expected_arm: Int,
    expected_m: Int,
    label: String,
) raises -> Int:
    """One named `(n, k)`: dispatch, ASSERT WHICH KERNEL RAN from the device's
    own report, then compare every slot against the host transcription."""
    var run = run_device(
        ctx, node_ids, tree_id, seed, n, k, SAMP_SAB_NONE
    )
    var want = host_reference(node_ids, tree_id, seed, n, k)
    var cells = assert_report(run, len(node_ids), expected_arm, expected_m, label)
    cells += compare_slots(run.colids, want, label)
    print(
        "   ",
        label,
        " n=",
        n,
        " k=",
        k,
        " nodes=",
        len(node_ids),
        " -> device says ",
        sampler_arm_name(Int(run.report[0])),
        " M=",
        run.report[1],
        " iters=",
        iter_span(run, len(node_ids)),
        " : ",
        len(want),
        " slots bit-identical",
    )
    return cells


def main() raises:
    comptime assert has_accelerator(), "this check must run on a GPU"
    print("=== sampler_kernel_check ===")
    print(
        "[device] accelerator present;",
        _vendor(),
        "-- every arm below runs the SAME binary, selected by a kernel",
        "argument",
    )

    var ctx = DeviceContext()
    var cells = 0
    var seed = UInt64(0xC0FFEE)

    var nodes8 = List[Int]()
    for i in range(8):
        nodes8.append(i)
    var nodes16 = List[Int]()
    for i in range(16):
        nodes16.append(i * 5)
    var nodes64 = List[Int]()
    for i in range(64):
        nodes64.append(i * 3 + 1)

    # =====================================================================
    # 1. ARM: excess sampling, instantiation MAX_SAMPLES_PER_THREAD = 1
    # =====================================================================
    print("")
    print("[excess M=1] builder_kernels.cuh:152-248, one draw per thread")
    cells += run_case(
        ctx, 64, 8, nodes64, 2, seed, SAMPLE_EXCESS, 1, String("small")
    )
    cells += run_case(
        ctx, 1000, 20, nodes64, 7, seed, SAMPLE_EXCESS, 1, String("wide")
    )
    cells += run_case(
        ctx, 100, 1, nodes64, 1, seed, SAMPLE_EXCESS, 1, String("k==1")
    )

    # =====================================================================
    # 2. ARM: excess sampling, instantiation MAX_SAMPLES_PER_THREAD = 72
    # =====================================================================
    print("")
    print("[excess M=72] the same kernel at the other instantiation")
    comptime M72 = SAMPLER_MAX_SAMPLES_PER_THREAD
    cells += run_case(
        ctx, 100, 99, nodes8, 3, seed, SAMPLE_EXCESS, M72, String("k==n-1")
    )
    cells += run_case(
        ctx,
        20000,
        128,
        nodes8,
        3,
        seed,
        SAMPLE_EXCESS,
        M72,
        String("just over the M boundary"),
    )
    cells += run_case(
        ctx, 2000, 1500, nodes8, 5, seed, SAMPLE_EXCESS, M72, String("dense")
    )

    # THE TWO INSTANTIATIONS MUST DISAGREE ON DEVICE TOO. Same (n, k, seed,
    # tree, node), and the dispatch is what changes M -- so this is asserted
    # across the `n_parallel_samples <= block_threads` boundary rather than by
    # calling one kernel twice.
    var a1000 = run_device(ctx, nodes8, 7, seed, 1000, 20, SAMP_SAB_NONE)
    var a20000 = run_device(ctx, nodes8, 7, seed, 20000, 128, SAMP_SAB_NONE)
    assert_equal(Int(a1000.report[1]), 1, "n=1000 k=20 is the M=1 arm")
    assert_equal(Int(a20000.report[1]), M72, "n=20000 k=128 is the M=72 arm")
    cells += 2

    # =====================================================================
    # 3. ARM: the all-features fill, DEVIATION 156
    # =====================================================================
    print("")
    print("[all-features] k == n: builder.cuh:399 does not launch; we fill")
    for n in [1, 2, 17, 256]:
        cells += run_case(
            ctx,
            n,
            n,
            nodes16,
            9,
            seed,
            SAMPLE_ALL_FEATURES,
            0,
            String("k==n, n=", n),
        )

    # =====================================================================
    # 4. ARM: algo-L. IT DOES NOT RUN HERE, AND THE REFUSAL IS THE ASSERTION
    # =====================================================================
    print("")
    print("[algo-L] builder_kernels.cuh:268-316, one THREAD per work item")
    print(
        "    device_has_float64() =",
        device_has_float64(),
        " on",
        _vendor(),
    )
    for pair in [(20000, 7385), (2000, 1990)]:
        var n = pair[0]
        var k = pair[1]
        var plan = plan_feature_sampling(n, k)
        assert_equal(
            plan.arm,
            SAMPLE_ALGO_L,
            String("(", n, ", ", k, ") must route to algo-L"),
        )
        cells += 1
        var refused = False
        try:
            _ = run_device(ctx, nodes16, 4, seed, n, k, SAMP_SAB_NONE)
        except e:
            refused = True
            assert_true(
                String(e).find("DEVIATION 199") >= 0,
                String(
                    "the algo-L refusal must name DEVIATION 199; got: ", e
                ),
            )
            print("    n=", n, " k=", k, " -> REFUSED by name (correct here)")
        if device_has_float64():
            assert_true(
                not refused,
                "a target with double must RUN algo-L, not refuse it --"
                " see DEVIATION 199, this arm has never been exercised",
            )
        else:
            assert_true(
                refused,
                String(
                    "a target without double must refuse the algo-L arm by"
                    " name rather than substitute float32 for cuML's"
                    " double W; (n, k) = (",
                    n,
                    ", ",
                    k,
                    ")",
                ),
            )
        cells += 2

    # =====================================================================
    # 5. THE TWO FIXED cuML BUGS STAY FIXED ON THE DEVICE PATH
    # =====================================================================
    # DEVIATION 164: at k = 1 their column 0 was drawn 0 of 64, at every one
    # of n = 2, 3, 4, 8. Not under-drawn, NEVER drawn. DEVIATION 165: their
    # column n-1 came out 662 times against 512 expected. Both are asserted
    # here on the DEVICE's own output, and both are then re-measured with the
    # bug deliberately restored by a kernel argument, so the fix is shown to
    # be load-bearing rather than asserted to be.
    print("")
    print("[deviations 164/165] the two fixed cuML bugs, on device")
    for n in [2, 3, 4, 8]:
        var fixed = run_device(ctx, nodes64, 1, seed, n, 1, SAMP_SAB_NONE)
        var buggy = run_device(ctx, nodes64, 1, seed, n, 1, SAMP_SAB_CUML_MASK0)
        var zeros_fixed = 0
        var zeros_buggy = 0
        for i in range(len(nodes64)):
            if fixed.colids[i] == Int32(0):
                zeros_fixed += 1
            if buggy.colids[i] == Int32(0):
                zeros_buggy += 1
        var want = host_reference(nodes64, 1, seed, n, 1)
        _ = compare_slots(
            fixed.colids, want, String("k==1 fixed, n=", n)
        )
        print(
            "    n=",
            n,
            " k=1 over 64 nodes: column 0 drawn",
            zeros_fixed,
            "times (device, fixed) vs",
            zeros_buggy,
            "with cuML's mask[0] predecessor restored",
        )
        assert_true(
            zeros_fixed > 0,
            String(
                "DEVIATION 164 on device: column 0 must be reachable at n=",
                n,
                ", k=1. cuML drew it 0 of 64.",
            ),
        )
        assert_equal(
            zeros_buggy,
            0,
            String(
                "SAMP_SAB_CUML_MASK0 must REPRODUCE cuML's bug at n=",
                n,
                " -- if it does not, this arm is not restoring the bug and"
                " the 164 fix is unguarded on device",
            ),
        )
        cells += 2 + len(want)

    # DEVIATION 165, ON `feature_sampler_check`'S OWN SWEEP, so the two files
    # measure the same thing: n = 64, k = 8, 512 nodes x 8 trees = 32768
    # selections, 512 expected per column. Theirs measured column n-1 at 662
    # before the fix. Small sweeps cannot see a 1.29x bias, which is why this
    # one is not smaller.
    var sweep_nodes = List[Int]()
    for i in range(512):
        sweep_nodes.append(i)
    var last_fixed = 0
    var last_buggy = 0
    var sweep_total = 0
    for tree in range(8):
        var f = run_device(
            ctx, sweep_nodes, Int32(tree), 0xA5A5, 64, 8, SAMP_SAB_NONE
        )
        var g = run_device(
            ctx, sweep_nodes, Int32(tree), 0xA5A5, 64, 8, SAMP_SAB_CUML_FILLER
        )
        for i in range(len(f.colids)):
            sweep_total += 1
            if f.colids[i] == Int32(63):
                last_fixed += 1
            if g.colids[i] == Int32(63):
                last_buggy += 1
    var expected_each = Float64(sweep_total) / 64.0
    print(
        "    n=64 k=8 over 512 nodes x 8 trees: column n-1 drawn",
        last_fixed,
        "times against",
        expected_each,
        "expected (ratio",
        Float64(last_fixed) / expected_each,
        "); with cuML's n-1 filler restored:",
        last_buggy,
        "(ratio",
        Float64(last_buggy) / expected_each,
        ")",
    )
    assert_true(
        Float64(last_fixed) < 1.25 * expected_each,
        String(
            "DEVIATION 165 on device: column n-1 is over-represented at ",
            last_fixed,
            " against ",
            expected_each,
        ),
    )
    # THE ASSERTION IS RELATIVE, AND THE ABSOLUTE 662 IS NOT THE TARGET.
    # DEVIATION 165's 662 was measured with BOTH cuML bugs present, because
    # that is what cuML is; `SAMP_SAB_CUML_FILLER` restores ONE of them and
    # there is no selector that restores both at once. What the arm must show
    # is that the filler moves column n-1 UP a lot -- measured 194 -> 487,
    # 2.5x -- which is the mechanism DEVIATION 165 names.
    #
    # And the direction of the FIXED number is worth reading rather than
    # skipping: 194 against 512 is column n-1 UNDER-drawn, 0.38x. That is not
    # a residue of the bug, it is point 5 of the host docstring -- the gather
    # takes the k SMALLEST uniques (`:243-246`), so the top of the column
    # range is systematically less likely at a small `n_parallel_samples`.
    # `feature_sampler_check` bands columns 1..n/2 and bounds n-1 only from
    # ABOVE for exactly that reason, and the host oracle reports the same 194.
    assert_true(
        last_buggy > 2 * last_fixed,
        String(
            "SAMP_SAB_CUML_FILLER must REPRODUCE the n-1 over-representation"
            " DEVIATION 165 names; it produced ",
            last_buggy,
            " against the fixed path's ",
            last_fixed,
            ". If it does not, the 165 fix is unguarded on device.",
        ),
    )
    cells += 2

    # =====================================================================
    # 6. EDGES
    # =====================================================================
    print("")
    print("[edges] n==1, k==1, k==n-1, and the refusals")
    # n == 1 is the all-features arm and the only legal k is 1.
    var e1 = run_device(ctx, nodes8, 11, seed, 1, 1, SAMP_SAB_NONE)
    cells += compare_slots(
        e1.colids, host_reference(nodes8, 11, seed, 1, 1), String("n==1")
    )
    cells += assert_report(
        e1, len(nodes8), SAMPLE_ALL_FEATURES, 0, String("n==1")
    )
    for n in [2, 3, 5000]:
        cells += run_case(
            ctx,
            n,
            1,
            nodes64,
            1,
            seed,
            SAMPLE_EXCESS,
            1,
            String("k==1, n=", n),
        )
    for n in [3, 100]:
        var plan = plan_feature_sampling(n, n - 1)
        cells += run_case(
            ctx,
            n,
            n - 1,
            nodes8,
            3,
            seed,
            SAMPLE_EXCESS,
            plan.max_samples_per_thread,
            String("k==n-1, n=", n),
        )

    # The refusals are `plan_feature_sampling`'s, DEVIATION 159, and the
    # device driver must inherit them rather than launch a grid of zero.
    for bad in [(10, 0), (10, 11), (10, -3)]:
        var raised = False
        try:
            _ = run_device(
                ctx, nodes8, 1, seed, bad[0], bad[1], SAMP_SAB_NONE
            )
        except:
            raised = True
        assert_true(
            raised,
            String(
                "sample_features_device must refuse k=",
                bad[1],
                " n=",
                bad[0],
                " the same way the host driver does (DEVIATION 159)",
            ),
        )
        cells += 1
    print("    refused (n, k) = (10, 0), (10, 11), (10, -3)")

    # =====================================================================
    # 7. SABOTAGE, ONE PER MECHANISM
    # =====================================================================
    # Rule: each must turn a COMPARISON RED, and the comparison it turns red
    # is named. A sabotage that leaves the check green is a defect in the
    # CHECK, so the count of differing slots is PRINTED for every arm and
    # every arm is asserted non-zero on a case where its mechanism exists.
    print("")
    print("[sabotage] one per mechanism, each an argument to the SAME binary")

    # THE MATRIX, not one cell per sabotage. Each arm is run at FOUR (n, k),
    # two on each instantiation, and every count is printed -- including the
    # zeros. A zero is not a pass and not a failure: it means the fixture does
    # not reach that mechanism at that size, and the designated case is the
    # one the assertion is made on. Two of these cells started at zero and the
    # FIXTURE was what changed; see the note beside each designated case.
    # (3, 2) is in the list because it is the one M=1 case whose do-while
    # goes round twice (the device reports iters=1..2), which is what makes
    # the `n_uniques >= k` mechanism exist at that instantiation at all.
    var sab_cases = [(1000, 20), (64, 8), (3, 2), (100, 99), (2000, 1500)]
    var sabs = [
        SabCase(
            SAMP_SAB_SORT_DESCENDING, String("sort direction"), 1000, 20
        ),
        # DESIGNATED (100, 99): at (1000, 20) only 21 slots draw from 1000
        # columns, so a repeat is rare and "every slot is a head" is already
        # true. The mechanism needs COLLISIONS to be visible.
        SabCase(SAMP_SAB_NO_DEDUPE, String("dedupe"), 100, 99),
        # DESIGNATED (2000, 1500): structurally inert at M=1, where a thread
        # HAS one item -- see the selector's docstring.
        SabCase(
            SAMP_SAB_AGG_ONE_PER_THREAD,
            String("block-scan aggregate"),
            2000,
            1500,
        ),
        SabCase(SAMP_SAB_KEY_NO_THREAD, String("key chain"), 1000, 20),
        # DESIGNATED (100, 99): the do-while only matters when it goes round
        # more than once. At (1000, 20) the first pass already yields k
        # uniques (the device reports iters=1), so `n_uniques >= 1` and
        # `n_uniques >= k` stop at the same place and the arm is inert.
        SabCase(
            SAMP_SAB_LOOP_ANY_UNIQUE,
            String("enough-uniques condition"),
            100,
            99,
        ),
    ]
    for si in range(len(sabs)):
        var c = sabs[si].copy()
        var hit = 0
        for ci in range(len(sab_cases)):
            var n = sab_cases[ci][0]
            var k = sab_cases[ci][1]
            var want = host_reference(nodes8, 7, seed, n, k)
            var base = run_device(ctx, nodes8, 7, seed, n, k, SAMP_SAB_NONE)
            var hurt = run_device(ctx, nodes8, 7, seed, n, k, c.sab)
            var d_base = differing(base.colids, want)
            var d_hurt = differing(hurt.colids, want)
            assert_equal(
                d_base,
                0,
                String(
                    c.name,
                    ": the SHIPPING arm must be bit-identical at (n=",
                    n,
                    ", k=",
                    k,
                    "); it is the control",
                ),
            )
            var designated = n == c.n and k == c.k
            print(
                "    ",
                c.name,
                " @ (n=",
                n,
                ", k=",
                k,
                ", M=",
                base.report[1],
                ", iters=",
                iter_span(base, len(nodes8)),
                "): ",
                d_hurt,
                "of",
                len(want),
                "slots differ",
                " <-- DESIGNATED" if designated else "",
            )
            if designated:
                hit = d_hurt
            cells += 2
        assert_true(
            hit > 0,
            String(
                c.name,
                ": sabotage did NOT turn the comparison red at its DESIGNATED"
                " (n=",
                c.n,
                ", k=",
                c.k,
                "). That is a defect in THIS FILE, not evidence about the"
                " kernel.",
            ),
        )
        cells += 1

    # The all-features fill has its own mechanism -- the identity, IN ORDER.
    var af_want = host_reference(nodes16, 9, seed, 17, 17)
    var af_base = run_device(ctx, nodes16, 9, seed, 17, 17, SAMP_SAB_NONE)
    var af_hurt = run_device(
        ctx, nodes16, 9, seed, 17, 17, SAMP_SAB_ALL_REVERSED
    )
    var af_d = differing(af_hurt.colids, af_want)
    print(
        "     all-features order at (n=17, k=17): shipping differs in",
        differing(af_base.colids, af_want),
        ", reversed differs in",
        af_d,
        "of",
        len(af_want),
    )
    assert_equal(
        differing(af_base.colids, af_want), 0, "all-features control"
    )
    assert_true(
        af_d > 0,
        "SAMP_SAB_ALL_REVERSED must turn the identity fill red; a check that"
        " compared only the column SET would not see it",
    )
    cells += 2

    # And the two bug-restoring arms, already asserted in section 5, are
    # counted here as the sabotages they are.
    print(
        "     cuML mask[0] predecessor and cuML n-1 filler: asserted in"
        " section 5 above, both turn their own measurement red"
    )

    print("")
    print("sampler_kernel: ", cells, " cells")
    print("sampler_kernel_check: PASS")
