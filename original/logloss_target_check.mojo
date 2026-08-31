# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cross_entropy_kernel` against a float64 host oracle and libm anchors.

    pixi run check-logloss-target

WHAT GATES WHAT. The kernel is the port of `CrossEntropyImpl`
(`pointwise_targets.cu:327-390`), the one kernel Logloss and CrossEntropy
reach on their GPU. No CatBoost fit can gate its DERIVATIVES bitwise -- the
kernel's own deviation block explains why (`__expf` vs `std.math.exp`) -- so
this check gates the ARITHMETIC three ways:

1. ANCHORS: six cases computed OUTSIDE this repository by CPython's libm in
   float64 (2026-08-20, values pasted below as decimals and compared under
   tolerance, never bitwise). They pin the branches a bulk sweep undersamples:
   the `exp` overflow arm (v=200), the p clamp at the other end (v=-200,
   where the float32 der2 of ~1e-40 may legally flush to zero on Metal, so
   the compare treats |x| < 1e-38 as zero), a target EXACTLY at the border
   (their compare is STRICT `>`, so c = 0), and the borderless soft-target
   arm.
2. BULK: 4,133 rows (odd, so the in_range tail is real) of hashed scattered
   values -- per `uniform-test-data-hides-permutation`, no two rows expect
   the same cell -- compared per element against the same formula evaluated
   on the host in float64. Every 211th target is planted at exactly 0.5 so
   the strict-border mechanism is bulk-covered, not anchor-only.
3. MODES: has_border true/false, weighted/weightless (plane 0 must be
   bit-exactly the weight / 1.0), and the fv/magnitude partials folded in
   fixed host order against the float64 sums.

Proven to have teeth, measured 2026-08-20: flipping the host reference's
border compare from strict `>` to `>=` fails with 43 mismatches (the
planted rows in both weighted modes, the fv/magnitude sums they feed, and
the at-border anchor); scaling the reference der by (1 + 1e-5) fails with
8,895; scaling the reference der2 by the same amount in the estimation
section fails with 390 (the extremes sit in the zero band by design, the
mid-range rows catch it); all restored and the check returned green
before this landed.
"""
from max.gpu.host import DeviceContext
from std.math import exp, log
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    cross_entropy_kernel,
)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def frac(i: Int, salt: UInt64) -> Float64:
    return Float64(splitmix(UInt64(i) * UInt64(2654435761) + salt) >> 11) * (
        1.0 / 9007199254740992.0
    )


def close(a: Float64, b: Float64, tol: Float64) -> Bool:
    # |x| < 1e-38 is the zero band: float32 subnormals may flush on device.
    if abs(a) < 1e-38 and abs(b) < 1e-38:
        return True
    return abs(a - b) <= tol * (1.0 + max(abs(a), abs(b)))


def main() raises:
    comptime N = 4133
    var ctx = DeviceContext()

    var d_t = ctx.enqueue_create_buffer[DType.float32](N)
    var d_w = ctx.enqueue_create_buffer[DType.float32](N)
    var d_p = ctx.enqueue_create_buffer[DType.float32](N)
    var d_stats = ctx.enqueue_create_buffer[DType.float32](2 * N)
    comptime BLOCKS = (N + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var d_fv = ctx.enqueue_create_buffer[DType.float32](BLOCKS)
    var d_mag = ctx.enqueue_create_buffer[DType.float32](2 * BLOCKS)

    var h_t = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_p = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_stats = ctx.enqueue_create_host_buffer[DType.float32](2 * N)
    var h_fv = ctx.enqueue_create_host_buffer[DType.float32](BLOCKS)
    var h_mag = ctx.enqueue_create_host_buffer[DType.float32](2 * BLOCKS)
    ctx.synchronize()

    var failures = 0

    # both border modes; border-mode inputs are 0/1 classes, the other soft
    for mode in range(2):
        var has_border = mode == 0
        for hw in range(2):
            var has_weights = hw == 0
            for i in range(N):
                var v = Float32((frac(i, UInt64(11)) - 0.5) * 16.0)
                if i % 97 == 0:
                    v = Float32(150.0) if i % 194 == 0 else Float32(-150.0)
                var t: Float32
                if has_border:
                    t = Float32(1.0) if frac(i, UInt64(22)) > 0.5 else (
                        Float32(0.0)
                    )
                    if i % 211 == 0:
                        t = Float32(0.5)  # exactly AT the strict border
                else:
                    t = Float32(frac(i, UInt64(33)))
                var w = Float32(0.5 + 2.0 * frac(i, UInt64(44)))
                # the six libm anchors sit in the first slots of their mode
                if has_border and i < 5:
                    var av: InlineArray[Float32, 5] = [
                        0.3, -1.7, 200.0, -200.0, 1.0
                    ]
                    var at: InlineArray[Float32, 5] = [
                        1.0, 0.0, 0.0, 1.0, 0.5
                    ]
                    var aw: InlineArray[Float32, 5] = [
                        1.0, 2.5, 1.0, 1.0, 1.0
                    ]
                    v = av[i]
                    t = at[i]
                    w = aw[i] if has_weights else Float32(1.0)
                if (not has_border) and i == 0:
                    v = Float32(0.7)
                    t = Float32(0.3)
                    w = Float32(1.3) if has_weights else Float32(1.0)
                h_p.unsafe_ptr().unsafe_store(i, v)
                h_t.unsafe_ptr().unsafe_store(i, t)
                h_w.unsafe_ptr().unsafe_store(i, w)
            ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_p, src_ptr=h_p.unsafe_ptr())

            if has_border:
                ctx.enqueue_function[cross_entropy_kernel[True]](
                    d_t.unsafe_ptr(), d_w.unsafe_ptr(), Int32(N),
                    d_p.unsafe_ptr(),
                    Int32(1) if has_weights else Int32(0),
                    Float32(0.5),
                    d_stats.unsafe_ptr(),
                    d_fv.unsafe_ptr(), Int32(1),
                    d_mag.unsafe_ptr(), Int32(1),
                    grid_dim=(BLOCKS, 1, 1),
                    block_dim=(MSE_BLOCK_SIZE, 1, 1),
                )
            else:
                ctx.enqueue_function[cross_entropy_kernel[False]](
                    d_t.unsafe_ptr(), d_w.unsafe_ptr(), Int32(N),
                    d_p.unsafe_ptr(),
                    Int32(1) if has_weights else Int32(0),
                    Float32(0.0),
                    d_stats.unsafe_ptr(),
                    d_fv.unsafe_ptr(), Int32(1),
                    d_mag.unsafe_ptr(), Int32(1),
                    grid_dim=(BLOCKS, 1, 1),
                    block_dim=(MSE_BLOCK_SIZE, 1, 1),
                )
            ctx.enqueue_copy(dst_ptr=h_stats.unsafe_ptr(), src_buf=d_stats)
            ctx.enqueue_copy(dst_ptr=h_fv.unsafe_ptr(), src_buf=d_fv)
            ctx.enqueue_copy(dst_ptr=h_mag.unsafe_ptr(), src_buf=d_mag)
            ctx.synchronize()

            # float64 host oracle, same formula, straight through
            var fv_ref = Float64(0.0)
            var mag_w_ref = Float64(0.0)
            var mag_g_ref = Float64(0.0)
            for i in range(N):
                var v = Float64(h_p.unsafe_ptr().unsafe_load(i))
                var t = Float64(h_t.unsafe_ptr().unsafe_load(i))
                var w = Float64(
                    h_w.unsafe_ptr().unsafe_load(i)
                ) if has_weights else Float64(1.0)
                var ev = exp(v)
                var p = ev / (1.0 + ev)
                if p > 1.0 - 1e-40:
                    p = 1.0 - 1e-40
                if p < 1e-40:
                    p = 1e-40
                var c = (
                    (1.0 if t > 0.5 else 0.0) if has_border else t
                )
                var der = w * (c - p)
                fv_ref += w * (c * v - log(1.0 + ev))
                mag_w_ref += abs(w)
                mag_g_ref += abs(der)

                var got_w = Float64(h_stats.unsafe_ptr().unsafe_load(i))
                var got_g = Float64(h_stats.unsafe_ptr().unsafe_load(N + i))
                if Float32(got_w) != Float32(w):
                    if failures < 8:
                        print("plane0 row", i, "got", got_w, "want", w)
                    failures += 1
                if not close(got_g, der, 3e-6):
                    if failures < 8:
                        print(
                            "der row", i, "got", got_g, "want", der,
                            "(v", v, "t", t, "w", w, ")"
                        )
                    failures += 1

            # THE EXTERNAL ANCHORS: CPython-libm float64 decimals, pasted.
            # `stats[N+a]` holds weight*der, and the pasted values already
            # include each case's weight. This is the leg that pins
            # `std.math.exp`/`log` to an oracle OUTSIDE this repository.
            if has_border and has_weights:
                var want: InlineArray[Float64, 5] = [
                    0.42555748318834097,
                    -0.38616316270883677,
                    -1.0,
                    1.0,
                    -0.7310585786300049,
                ]
                for a in range(5):
                    var got = Float64(
                        h_stats.unsafe_ptr().unsafe_load(N + a)
                    )
                    if not close(got, want[a], 3e-6):
                        print("libm anchor", a, "got", got, "want", want[a])
                        failures += 1
            if (not has_border) and has_weights:
                var got0 = Float64(h_stats.unsafe_ptr().unsafe_load(N))
                if not close(got0, -0.47864410381861605, 3e-6):
                    print("libm soft anchor got", got0)
                    failures += 1

            var fv_got = Float64(0.0)
            var mw_got = Float64(0.0)
            var mg_got = Float64(0.0)
            for b in range(BLOCKS):
                fv_got += Float64(h_fv.unsafe_ptr().unsafe_load(b))
                mw_got += Float64(h_mag.unsafe_ptr().unsafe_load(2 * b))
                mg_got += Float64(h_mag.unsafe_ptr().unsafe_load(2 * b + 1))
            if not close(fv_got, fv_ref, 2e-5):
                print("fv mode", mode, hw, "got", fv_got, "want", fv_ref)
                failures += 1
            if not close(mw_got, mag_w_ref, 2e-5) or not close(
                mg_got, mag_g_ref, 2e-5
            ):
                print("mags mode", mode, hw, "got", mw_got, mg_got,
                      "want", mag_w_ref, mag_g_ref)
                failures += 1

    # ------- estimation mode: their ApproximateAt outputs -------------
    # plane 0 = der = w*(c-p), plane 1 = der2 = w*p*(1-p). Weighted only:
    # the mode exists for the leaves oracle, which always carries weights.
    for mode in range(2):
        var has_border = mode == 0
        for i in range(N):
            var v = Float32((frac(i, UInt64(11)) - 0.5) * 16.0)
            if i % 97 == 0:
                v = Float32(150.0) if i % 194 == 0 else Float32(-150.0)
            var t: Float32
            if has_border:
                t = Float32(1.0) if frac(i, UInt64(22)) > 0.5 else (
                    Float32(0.0)
                )
                if i % 211 == 0:
                    t = Float32(0.5)
            else:
                t = Float32(frac(i, UInt64(33)))
            var w = Float32(0.5 + 2.0 * frac(i, UInt64(44)))
            if has_border and i < 5:
                var av: InlineArray[Float32, 5] = [
                    0.3, -1.7, 200.0, -200.0, 1.0
                ]
                var at: InlineArray[Float32, 5] = [
                    1.0, 0.0, 0.0, 1.0, 0.5
                ]
                var aw: InlineArray[Float32, 5] = [
                    1.0, 2.5, 1.0, 1.0, 1.0
                ]
                v = av[i]
                t = at[i]
                w = aw[i]
            if (not has_border) and i == 0:
                v = Float32(0.7)
                t = Float32(0.3)
                w = Float32(1.3)
            h_p.unsafe_ptr().unsafe_store(i, v)
            h_t.unsafe_ptr().unsafe_store(i, t)
            h_w.unsafe_ptr().unsafe_store(i, w)
        ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_p, src_ptr=h_p.unsafe_ptr())
        if has_border:
            ctx.enqueue_function[cross_entropy_kernel[True, True]](
                d_t.unsafe_ptr(), d_w.unsafe_ptr(), Int32(N),
                d_p.unsafe_ptr(), Int32(1), Float32(0.5),
                d_stats.unsafe_ptr(),
                d_fv.unsafe_ptr(), Int32(1),
                d_mag.unsafe_ptr(), Int32(0),
                grid_dim=(BLOCKS, 1, 1),
                block_dim=(MSE_BLOCK_SIZE, 1, 1),
            )
        else:
            ctx.enqueue_function[cross_entropy_kernel[False, True]](
                d_t.unsafe_ptr(), d_w.unsafe_ptr(), Int32(N),
                d_p.unsafe_ptr(), Int32(1), Float32(0.0),
                d_stats.unsafe_ptr(),
                d_fv.unsafe_ptr(), Int32(1),
                d_mag.unsafe_ptr(), Int32(0),
                grid_dim=(BLOCKS, 1, 1),
                block_dim=(MSE_BLOCK_SIZE, 1, 1),
            )
        ctx.enqueue_copy(dst_ptr=h_stats.unsafe_ptr(), src_buf=d_stats)
        ctx.enqueue_copy(dst_ptr=h_fv.unsafe_ptr(), src_buf=d_fv)
        ctx.synchronize()

        var fv_ref = Float64(0.0)
        for i in range(N):
            var v = Float64(h_p.unsafe_ptr().unsafe_load(i))
            var t = Float64(h_t.unsafe_ptr().unsafe_load(i))
            var w = Float64(h_w.unsafe_ptr().unsafe_load(i))
            var ev = exp(v)
            var pp = ev / (1.0 + ev)
            if pp > 1.0 - 1e-40:
                pp = 1.0 - 1e-40
            if pp < 1e-40:
                pp = 1e-40
            var c = (1.0 if t > 0.5 else 0.0) if has_border else t
            var der = w * (c - pp)
            var der2 = w * pp * (1.0 - pp)
            fv_ref += w * (c * v - log(1.0 + ev))
            var got_d = Float64(h_stats.unsafe_ptr().unsafe_load(i))
            var got_d2 = Float64(h_stats.unsafe_ptr().unsafe_load(N + i))
            if not close(got_d, der, 3e-6):
                if failures < 8:
                    print("est der row", i, "got", got_d, "want", der)
                failures += 1
            if not close(got_d2, der2, 3e-6):
                if failures < 8:
                    print("est der2 row", i, "got", got_d2, "want", der2)
                failures += 1
        var fv_got = Float64(0.0)
        for b in range(BLOCKS):
            fv_got += Float64(h_fv.unsafe_ptr().unsafe_load(b))
        if not close(fv_got, fv_ref, 2e-5):
            print("est fv mode", mode, "got", fv_got, "want", fv_ref)
            failures += 1

        # der2 libm anchors (CPython float64, decimal-pasted). Slot 3's
        # 1e-40 sits in the zero band by design: float32 flushes it.
        if has_border:
            var want2: InlineArray[Float64, 5] = [
                0.24445831169074586,
                0.32651436741552015,
                0.0,
                1e-40,
                0.19661193324148185,
            ]
            for a in range(5):
                var got = Float64(h_stats.unsafe_ptr().unsafe_load(N + a))
                if not close(got, want2[a], 3e-6):
                    print("libm der2 anchor", a, "got", got,
                          "want", want2[a])
                    failures += 1
        else:
            var got0 = Float64(h_stats.unsafe_ptr().unsafe_load(N))
            if not close(got0, 0.28822673528104176, 3e-6):
                print("libm soft der2 anchor got", got0)
                failures += 1

    if failures != 0:
        raise Error(
            "logloss target check FAILED with "
            + String(failures)
            + " mismatches"
        )
    print(
        "logloss target check: 4 mode passes x", N,
        "rows, anchors, planted borders, fv/mags -- all match"
    )
