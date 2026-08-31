# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MultiClass oracle: gradient, blocked Hessian, and the gauge fixing.

    pixi run check-multiclass-oracle

NO CATBOOST COUNTERPART: a gate, so `checks/`.

WHAT IS UNDER TEST. `BinOptimizedOracle`'s `rowSize > 1` arm -- the branches
of `pointwise_oracle.cpp` that every pointwise loss skips:

    WriteValueAndFirstDerivatives   the multi-column reduce and the
                                    gradient's RECONSTRUCTED last
                                    component (`:93-101`)
    WriteSecondDerivatives          the blocked lower-triangular Hessian,
                                    mirrored, with lambda on the diagonal
                                    (`:125-184`)
    MakeEstimationResult            the gauge fixing (`:18-33`)
    MoveTo                          which projects BEFORE it subtracts
                                    (`:43-47`), so the cursor and the
                                    walker's point live in different gauges
    Regularize                      which zeroes every approx dimension of
                                    an under-weight leaf, not just the
                                    first (`oracle_interface.h:47-50`)

THREE OF THE FOUR GATES ARE ANALYTIC, and that is deliberate: the per-cell
reference here would otherwise be a transcription of the same formulas the
code under test uses, which rule 8 warns is a tally rather than a gate.

1. **THE GRADIENT SUMS TO ZERO ACROSS ALL numClasses COMPONENTS.** That is
   not an accident of the arithmetic, it is the identity their
   reconstruction is built on: they never compute the last component, they
   set it to `-total`. So the sum is zero BY CONSTRUCTION and testing it
   only proves the construction ran -- which is why gate 2 exists.

2. **THE FIRST cursorDim COMPONENTS MATCH AN INDEPENDENT PER-LEAF TALLY.**
   The host sums `weight * ((label == k) - p_k)` over each leaf's rows in
   float64, walking the leaves through the SAME offsets the device was
   given. This is the gate that would catch a wrong reduce, a wrong leaf
   segmentation, or a plane transposition, none of which gate 1 can see.

3. **EVERY HESSIAN ROW SUMS TO EXACTLY lambda.** Row `i` of
   `diag(p) - p p^T` sums to `p_i - p_i * sum_j p_j = 0` over all
   numClasses, so the leaf-summed Hessian has zero row sums and adding
   lambda to the diagonal makes every row sum exactly lambda. This is
   algebra about the multinomial Hessian and holds whatever the code does,
   which makes it the strongest structural gate in the file. It also pins
   the fact the port nearly got wrong: the Hessian is
   `numClasses x numClasses`, INCLUDING the pinned class's row, and a
   matrix one row short would not have this property.

4. **THE HESSIAN IS SYMMETRIC**, cell for cell, which is what the mirroring
   in `_write_blocked_second_derivatives` is for.

5. **THE GAUGE FIXING IS SHIFT-INVARIANT.** Adding the same constant to all
   `numClasses` components of a leaf's point must leave
   `make_estimation_result` unchanged, because the softmax cannot see a
   common shift. That is the property the projection exists to enforce and
   it is checkable without knowing what the projection is. It is exact in
   real arithmetic and ONE ULP in float32 -- `(v + c) - (u + c)` rounds
   differently from `v - u` -- so it is checked to 1e-5 relative and not
   bitwise.

THE SABOTAGES:

    C1  lambda changed                    reaches the DIAGONAL only, so the
                                          row sums move by exactly the
                                          difference and gate 3 sees it
    C2  one leaf's rows given to another   the segmentation reaches the
        leaf in the offsets the DEVICE     reduce
        gets, not in the host tally
    C3  a class plane of the cursor
        shifted                            the cursor reaches the softmax
"""

from max.gpu.host import DeviceContext
from std.ffi import external_call

from gbdt.methods.leaves_estimation.pointwise_oracle import (
    make_bin_optimized_oracle,
)
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_NEWTON,
)
from gbdt.targets.kernel.pointwise_targets import OBJECTIVE_MULTICLASS

comptime LAMBDA = Float64(3.0)


def c_exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hashed_unit(seed: UInt64, i: Int) -> Float64:
    var h = splitmix(seed ^ UInt64(i * 2654435761))
    return Float64(h >> 11) * (1.0 / Float64(1 << 53))


def host_probs(approx: List[Float64], eff: Int) -> List[Float64]:
    """The softmax with their max-subtraction, in float64. `eff + 1` out."""
    var mx = Float64(0.0)
    for k in range(eff):
        if approx[k] > mx:
            mx = approx[k]
    var se = Float64(0.0)
    for k in range(eff):
        se += c_exp(approx[k] - mx)
    se += c_exp(0.0 - mx)
    var out = List[Float64]()
    for k in range(eff):
        out.append(c_exp(approx[k] - mx) / se)
    out.append(c_exp(0.0 - mx) / se)
    return out^


def run_case(
    ctx: DeviceContext, num_classes: Int, sab: Int, verbose: Bool
) raises -> Int:
    var eff = num_classes - 1
    var sizes = List[Int]()
    for v in [7, 1, 40, 0, 129, 512, 3]:
        sizes.append(v)
    var n_leaves = len(sizes)
    var n = 0
    for i in range(n_leaves):
        n += sizes[i]

    var offsets = List[Int]()
    var acc = 0
    for i in range(n_leaves):
        offsets.append(acc)
        acc += sizes[i]

    var dev_offsets = offsets.copy()
    var dev_sizes = sizes.copy()
    # C2: hand the DEVICE a different segmentation than the host tally
    if sab == 2:
        var t = dev_sizes[0]
        dev_sizes[0] = dev_sizes[2]
        dev_sizes[2] = t

    var d_target = ctx.enqueue_create_buffer[DType.float32](n)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_cursor = ctx.enqueue_create_buffer[DType.float32](eff * n)
    var d_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var d_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)

    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_c = ctx.enqueue_create_host_buffer[DType.float32](eff * n)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)

    var labels = List[Int]()
    var wts = List[Float64]()
    var apx = List[Float64]()
    for i in range(n):
        var lab = Int(hashed_unit(UInt64(0x71), i) * Float64(num_classes))
        if lab >= num_classes:
            lab = num_classes - 1
        labels.append(lab)
        var w = Float64(
            Float32(0.5 + hashed_unit(UInt64(0x72), i) * 2.0)
        )
        wts.append(w)
        h_t.unsafe_ptr().unsafe_store(i, Float32(lab))
        h_w.unsafe_ptr().unsafe_store(i, Float32(w))
    for k in range(eff):
        for i in range(n):
            var a = hashed_unit(UInt64(0x80 + k), i) * 4.0 - 2.0
            # C3: shift one class plane on the DEVICE only
            var dev_a = a
            if sab == 3 and k == 0:
                dev_a = a + 0.5
            apx.append(Float64(Float32(a)))
            h_c.unsafe_ptr().unsafe_store(k * n + i, Float32(dev_a))
    for i in range(n_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(dev_offsets[i]))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(dev_sizes[i]))

    ctx.enqueue_copy(dst_buf=d_target, src_ptr=h_t.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_cursor, src_ptr=h_c.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.synchronize()

    var lam = LAMBDA
    if sab == 1:
        lam = LAMBDA + 1.0

    var oracle = make_bin_optimized_oracle(
        ctx, n, n_leaves, sizes,
        d_target^, d_w^, d_cursor^, d_off^, d_sz^,
        True, OBJECTIVE_MULTICLASS, Float32(0.0), Float32(0.5),
        Float32(0.5), lam, -1, LEAF_ESTIMATION_NEWTON, num_classes,
    )

    var value = Float64(0.0)
    var gradient = List[Float64]()
    oracle.write_value_and_first_derivatives(value, gradient)
    var hessian = List[Float64]()
    oracle.write_second_derivatives(hessian)

    var bad = 0
    var shown = 0

    # ---- GATE 1 + 2: the gradient ------------------------------------
    for leaf in range(n_leaves):
        var total = Float64(0.0)
        for dim in range(num_classes):
            total += gradient[leaf * num_classes + dim]
        if total > 1e-6 or total < -1e-6:
            bad += 1
            if verbose and shown < 3:
                print("    leaf", leaf, "gradient sum", total, "!= 0")
                shown += 1

        # the independent per-leaf tally, over the HOST's segmentation
        var want = List[Float64]()
        for _ in range(eff):
            want.append(Float64(0.0))
        for r in range(sizes[leaf]):
            var row = offsets[leaf] + r
            var approx = List[Float64]()
            for k in range(eff):
                approx.append(apx[k * n + row])
            var p = host_probs(approx, eff)
            for k in range(eff):
                var ind = 1.0 if labels[row] == k else 0.0
                want[k] += wts[row] * (ind - p[k])
        for k in range(eff):
            var got = gradient[leaf * num_classes + k]
            var d = got - want[k]
            if d < 0.0:
                d = -d
            var scale = want[k] if want[k] > 0.0 else -want[k]
            if scale < 1.0:
                scale = 1.0
            if d / scale > 1e-4:
                bad += 1
                if verbose and shown < 3:
                    print(
                        "    leaf", leaf, "dim", k, "got", got,
                        "want", want[k],
                    )
                    shown += 1

    # ---- GATE 3 + 4: the Hessian -------------------------------------
    var hbs = num_classes
    for leaf in range(n_leaves):
        var base = leaf * hbs * hbs
        for i in range(hbs):
            var row_sum = Float64(0.0)
            for j in range(hbs):
                row_sum += hessian[base + i * hbs + j]
                # GATE 4: symmetry, cell for cell
                var a = hessian[base + i * hbs + j]
                var b = hessian[base + j * hbs + i]
                var d = a - b
                if d < 0.0:
                    d = -d
                if d > 1e-9:
                    bad += 1
            # GATE 3: every row sums to exactly lambda
            # THE GATE KEEPS `LAMBDA`, not `lam`. C1 hands the ORACLE a
            # different lambda and leaves this expectation alone; a
            # sabotage that moved both sides would move nothing, which is
            # the mistake the first version of this check made and the
            # same one `check-exact-estimation` made before it.
            var e = row_sum - LAMBDA
            if e < 0.0:
                e = -e
            if e > 1e-4 * (1.0 + LAMBDA):
                bad += 1
                if verbose and shown < 3:
                    print(
                        "    leaf", leaf, "hessian row", i, "sums to",
                        row_sum, "want", lam,
                    )
                    shown += 1

    # ---- GATE 5: the gauge fixing is shift-invariant ------------------
    if sab == 0:
        var point = List[Float32]()
        var shifted = List[Float32]()
        for leaf in range(n_leaves):
            var c = Float32(0.37 + Float32(leaf) * 0.11)
            for dim in range(num_classes):
                var v = Float32(
                    hashed_unit(UInt64(0x90), leaf * num_classes + dim)
                    * 2.0
                    - 1.0
                )
                point.append(v)
                shifted.append(v + c)
        var pa = oracle.make_estimation_result(point)
        var pb = oracle.make_estimation_result(shifted)
        # THE INVARIANCE IS EXACT IN REAL ARITHMETIC AND ONE ULP IN
        # FLOAT32. `make_estimation_result` computes
        # `point[dim] - point[cursorDim]`; with a common shift `c` that
        # becomes `(v + c) - (u + c)`, which is `v - u` mathematically and
        # differs in the last bit once both sums are rounded. The first
        # version of this gate demanded bitwise equality and failed at
        # every class count on differences of about 1e-7 relative -- a
        # property of float addition, not of the projection.
        for i in range(len(pa)):
            var d = Float64(pa[i]) - Float64(pb[i])
            if d < 0.0:
                d = -d
            var m = Float64(pa[i]) if pa[i] > 0 else -Float64(pa[i])
            if m < 1.0:
                m = 1.0
            if d / m > 1e-5:
                bad += 1
                if verbose and shown < 3:
                    print(
                        "    gauge: a common shift moved component", i,
                        pa[i], "vs", pb[i],
                    )
                    shown += 1
    return bad


def check_multiclass_oracle(ctx: DeviceContext) raises:
    var failures = 0
    print("-- honest run: gradient, blocked Hessian, gauge --")
    for nc in [2, 3, 7]:
        var bad = run_case(ctx, nc, 0, True)
        if bad != 0:
            print("  FAIL numClasses", nc, "bad", bad)
            failures += 1
        else:
            print("  ok   numClasses", nc)

    print()
    print("-- sabotages --")
    var sabs = [
        (1, "lambda + 1 (reaches the Hessian diagonal)"),
        (2, "device segmentation swapped against the host tally"),
        (3, "class plane 0 of the cursor shifted"),
    ]
    for si in range(len(sabs)):
        var sid = sabs[si][0]
        var name = sabs[si][1]
        var moved = run_case(ctx, 7, sid, False)
        if moved == 0:
            print("  FAIL C" + String(sid), name, "moved nothing")
            failures += 1
        else:
            print("  ok   C" + String(sid), name, "->", moved)

    if failures != 0:
        raise Error(
            "multiclass oracle check: " + String(failures) + " failures"
        )
    print()
    print("multiclass oracle check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_multiclass_oracle(ctx)
