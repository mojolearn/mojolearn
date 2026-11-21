# SPDX-License-Identifier: Apache-2.0
"""AdditiveChi2Sampler / SkewedChi2Sampler RUN: typed requests in, typed
results out.

The boundary (bindings/neighbors_semi_native_door.mojo) admits the public
arguments into these requests, releases the GIL once for each function here
and publishes the results. Nothing in this module imports Python. Every
address is admitted storage the boundary keeps alive through the call. The
checks run at the point the public method made them, so the refusal order
and wording are unchanged.
"""
from core.neighbors_native_driver import XnDriver
from core.family_cast_impl import family_min_f32
from neighbors.neighbors_graph_run import xn_f32_word
from neighbors.neighbors_semi_run import (
    XN_SEMI_U_LOG, xn_achi2_negative_check, xn_achi2_fit_negative, xn_achi2_width, xn_achi2_csr_count,
    xn_achi2_csr_fill, xn_schi2_skew_check,
)

comptime _CHI2_F32P = MutPointer[Float32, MutUntrackedOrigin]


def chi2_minimum(x: Int, count: Int) -> Float64:
    """The minimum of `count` >= 1 float32 words at `x` (parallel row tasks)."""
    return Float64(family_min_f32(_CHI2_F32P(unsafe_from_address=x), count))


# ---------------------------------------------------------------- AdditiveChi2Sampler


@fieldwise_init
struct Achi2FitRequest(Copyable, Movable):
    # `device_scan`: the bound binary carries the FAST + Apple device scan
    # (XN_FAST_ACHI2_DEVSCAN); it answers -2 below its size gate.
    var x: Int
    var count: Int
    var device_scan: Bool


def achi2_fit_run(driver: XnDriver, request: Achi2FitRequest) raises:
    """fit's refusal of a negative entry: the device scan when the binary has
    it, else the host minimum of a non-empty X."""
    var first = -2
    if request.count != 0 and request.device_scan:
        var a: Array[Int, 1] = [request.x]
        var p: Array[Int, 1] = [request.count]
        var f: Array[Float64, 0] = []
        first = driver.run("achi2_first_negative", a, p, f)
    var lo = Float64(0.0)
    if first == -2 and request.count != 0:
        lo = chi2_minimum(request.x, request.count)
    xn_achi2_negative_check(xn_achi2_fit_negative(first, request.count, lo))


def achi2_interval_check(valid: Bool) raises:
    if not valid:
        raise Error("If sample_steps is not in [1, 2, 3], you need to provide sample_interval")


@fieldwise_init
struct Achi2Request(Copyable, Movable):
    # `interval_valid` False: no interval was given and sample_steps has no
    # default one; refused after the negative check, where the public method
    # read the interval. `checked`: fit has just refused a negative entry of
    # this same X. `sparse`: count the map's stored entries.
    var x: Int
    var dst: Int
    var rows: Int
    var features: Int
    var steps: Int
    var interval: Float64
    var interval_valid: Bool
    var checked: Bool
    var sparse: Bool


def achi2_transform_run(driver: XnDriver, request: Achi2Request) raises -> Int:
    """The dense map into `request.dst` (rows x features * (2 * steps - 1)
    float32). Returns the stored-entry count of its CSR form for a sparse
    input, else 0."""
    var count = request.rows * request.features
    if not request.checked and count != 0:
        xn_achi2_negative_check(chi2_minimum(request.x, count) < 0.0)
    achi2_interval_check(request.interval_valid)
    var a: Array[Int, 2] = [request.x, request.dst]
    var p: Array[Int, 3] = [request.rows, request.features, request.steps]
    var f: Array[Float64, 1] = [xn_f32_word(request.interval)]
    _ = driver.run("achi2", a, p, f)
    if request.sparse:
        return xn_achi2_csr_count(request.dst, request.rows * xn_achi2_width(request.features, request.steps))
    return 0


def achi2_csr_run(dense: Int, rows: Int, cols: Int, data: Int, indices: Int, indptr: Int):
    """The CSR arrays of the dense map (row-major, columns ascending) into
    the three buffers the boundary sized from `achi2_transform_run`'s count."""
    xn_achi2_csr_fill(dense, rows, cols, data, indices, indptr)


# ---------------------------------------------------------------- SkewedChi2Sampler


@fieldwise_init
struct Schi2FitRequest(Copyable, Movable):
    # `seed` is the seed's low 32 bits; `weights` is features x components
    # float32, `offset` is components float32.
    var seed: Int
    var features: Int
    var components: Int
    var weights: Int
    var offset: Int


def schi2_fit_run(driver: XnDriver, request: Schi2FitRequest) raises:
    """`random_weights_` and `random_offset_` from the counter-based draw."""
    var a: Array[Int, 2] = [request.weights, request.offset]
    var p: Array[Int, 3] = [request.seed, request.features, request.components]
    var f: Array[Float64, 0] = []
    _ = driver.run("schi2_fit", a, p, f)


# `schi2_weights_run` (the weights of a pending z) deleted with the door's
# pending-z paths (lane fx-leftovers item 10, 2026-11-21: no producer of a
# pending z since 5a2b5cecb9); `Schi2Request.pending` is always False there.


@fieldwise_init
struct Schi2Request(Copyable, Movable):
    # `weights` is `random_weights_`, or the pending z when `pending` (then
    # `weights_out` receives the weights; 0 otherwise; only with `lazy`). `lazy`: the one-call
    # FAST + Apple transform (XN_FAST_SCHI2_LAZYW); else `scratch` (rows x
    # features float32) receives log(x + skewedness) and the map follows.
    var x: Int
    var weights: Int
    var weights_out: Int
    var offset: Int
    var scratch: Int
    var dst: Int
    var rows: Int
    var features: Int
    var components: Int
    var skewedness: Float64
    var lazy: Bool
    var pending: Bool


def schi2_check_run(x: Int, count: Int, skewedness: Float64) raises:
    """transform's refusal of an entry at or below -skewedness (non-empty X)."""
    if count != 0:
        xn_schi2_skew_check(chi2_minimum(x, count), skewedness)


def schi2_transform_run(driver: XnDriver, request: Schi2Request) raises:
    """The map into `request.dst` (rows x components float32)."""
    if request.lazy:
        var za: Array[Int, 5] = [request.x, request.weights, request.weights_out, request.offset, request.dst]
        var zp: Array[Int, 4] = [request.rows, request.features, request.components, 1 if request.pending else 0]
        var zf: Array[Float64, 1] = [xn_f32_word(request.skewedness)]
        _ = driver.run("schi2_lazy_transform", za, zp, zf)
        return
    var la: Array[Int, 2] = [request.x, request.scratch]
    var lp: Array[Int, 2] = [request.rows * request.features, XN_SEMI_U_LOG]
    var lf: Array[Float64, 2] = [1.0, xn_f32_word(request.skewedness)]
    _ = driver.run("unary", la, lp, lf)
    var a: Array[Int, 4] = [request.scratch, request.weights, request.offset, request.dst]
    var p: Array[Int, 3] = [request.rows, request.features, request.components]
    var f: Array[Float64, 0] = []
    _ = driver.run("skew_transform", a, p, f)
