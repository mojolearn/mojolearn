# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the native PLS owner (PLSRegression, PLSCanonical, CCA;
bindings/pls_native.mojo is UNPACK and PACK): `_PLS._center_scale`, the one
XD_PLS_FIT component loop, the rotations through `_pinv`, the coefficients
and the projections, over the typed kit (bindings/decomp_kit_run.mojo).
Same x_decomp operations, same words, same order: the bits do not move. No
Python (lane entangled-decompkit, 2026-11-12)."""
from std.memory import unsafe_memcpy
from bindings.decomp_kit_run import DKit, DK_SUB, DK_ADD, DK_MUL, DK_DIV, DK_SCALE, DK_SQ, DK_SQRT, DK_SELECT, DK_RECIP
from x_decomp.descriptor_schema import XD_PLS_FIT

comptime PLS_F32_EPS = 1.1920928955078125e-07


def pls_center_scale(mut kit: DKit, a: Int, scale: Bool) raises -> Tuple[Int, Int, Int]:
    # `_PLS._center_scale`: (centered and scaled A, mean, std).
    var n = kit.nrows(a)
    var mean = kit.colmean(a)
    var centered = kit.ew(DK_SUB, a, mean)
    var std: Int
    if scale:
        var squares = kit.ew(DK_SQ, centered)
        var sums = kit.colsum(squares)
        var variance = kit.ew(DK_SCALE, sums, scalar=1.0 / Float64(n - 1))
        var root = kit.ew(DK_SQRT, variance)
        var one = kit.const(1.0)
        std = kit.ew(DK_SELECT, root, root, one, 0.0)
        centered = kit.ew(DK_DIV, centered, std)
    else:
        std = kit.const(1.0, 1, kit.ncols(a))
    return (centered, mean, std)


def pls_pinv(mut kit: DKit, a: Int) raises -> Int:
    # `_pinv` (rcond None): V diag(1/s) U^T through the thin SVD, values at or
    # below max(shape) * float32 eps * s_max dropped.
    var rows = kit.nrows(a)
    var cols = kit.ncols(a)
    var r = min(rows, cols)
    var usv = kit.thin_svd(a, r, True)
    var cond = Float64(max(rows, cols)) * PLS_F32_EPS
    var cut = Float64(0.0)
    if r:
        cut = Float64(Float32(kit.word(usv[1], 0) * cond))
    var zero = kit.zeros(1, 1)
    var kept = kit.ew(DK_SELECT, usv[1], usv[1], zero, cut)
    var inverse = kit.ew(DK_RECIP, kept)
    var scaled = kit.ew(DK_MUL, usv[0], inverse)
    return kit.mm(usv[2], scaled, ta=True, tb=True)


def pls_rotation(mut kit: DKit, weights: Int, loadings: Int) raises -> Int:
    var gram = kit.mm(loadings, weights, ta=True)
    var inverse = pls_pinv(kit, gram)
    return kit.mm(weights, inverse)


def pls_project(mut kit: DKit, a: Int, mean: Int, std: Int, rotations: Int) raises -> Int:
    # ((A - mean) / std) @ rotations.
    var centered = kit.ew(DK_SUB, a, mean)
    var scaled = kit.ew(DK_DIV, centered, std)
    return kit.mm(scaled, rotations)


def pls_rebuild(mut kit: DKit, z: Int, loadings: Int, std: Int, mean: Int) raises -> Int:
    # (Z @ loadings^T) * std + mean.
    var product = kit.mm(z, loadings, tb=True)
    var scaled = kit.ew(DK_MUL, product, std)
    return kit.ew(DK_ADD, scaled, mean)


def pls_predict_run(mut kit: DKit, x: Int, x_mean: Int, coef: Int, y_mean: Int) raises -> Int:
    var centered = kit.ew(DK_SUB, x, x_mean)
    var product = kit.mm(centered, coef, tb=True)
    return kit.ew(DK_ADD, product, y_mean)


struct PlsFitWords(Copyable, Movable):
    """The fit's options, read once by UNPACK."""
    var components: Int
    var max_iter: Int
    var mode_b: Bool
    var canonical: Bool
    var svd_algo: Bool
    var scale: Bool
    var tol: Float64

    def __init__(out self, components: Int, max_iter: Int, mode_b: Bool, canonical: Bool, svd_algo: Bool,
                 scale: Bool, tol: Float64):
        self.components = components
        self.max_iter = max_iter
        self.mode_b = mode_b
        self.canonical = canonical
        self.svd_algo = svd_algo
        self.scale = scale
        self.tol = tol


struct PlsFit(Movable):
    """The fit's matrices and words for PACK. `outs`: x_weights, y_weights,
    x_scores, y_scores, x_loadings, y_loadings (XD_PLS_FIT's six blocks, cut
    to `done` columns); `iterations` the completed components' counts;
    `tx` / `ty` fit_transform's projections (-1 when not asked)."""
    var outs: List[Int]
    var done: Int
    var iterations: List[Int32]
    var x_rotations: Int
    var y_rotations: Int
    var coef: Int
    var x_mean: Int
    var x_std: Int
    var y_mean: Int
    var y_std: Int
    var tx: Int
    var ty: Int

    def __init__(out self):
        self.outs = List[Int]()
        self.done = 0
        self.iterations = List[Int32]()
        self.x_rotations = -1
        self.y_rotations = -1
        self.coef = -1
        self.x_mean = -1
        self.x_std = -1
        self.y_mean = -1
        self.y_std = -1
        self.tx = -1
        self.ty = -1


def pls_fit_run(mut kit: DKit, x: Int, ym: Int, words: PlsFitWords, transform: Bool) raises -> PlsFit:
    """`_PLS.fit` after admission (and fit_transform's projections of the X
    and Y the fit holds: the fitted means, stds and rotations are these
    matrices, the words the public fields get)."""
    var n = kit.nrows(x)
    var p = kit.ncols(x)
    var q = kit.ncols(ym)
    var nc = words.components
    var result = PlsFit()
    var xs = pls_center_scale(kit, x, words.scale)
    var ys = pls_center_scale(kit, ym, words.scale)
    # The component loop (NIPALS or the SVD pair, the flips, the deflations)
    # is ONE call writing six host blocks: a = [x, y, the six blocks, the
    # int32 iteration counts, the int64 word nits comes back in].
    var sizes = [p, q, n, n, p, q]
    var outs = List[Int]()
    var addresses: List[Int] = [kit.at(xs[0]), kit.at(ys[0])]
    for k in range(6):  # driver-loop: over the six fitted blocks
        var block = kit.zeros(sizes[k], nc)
        outs.append(block)
        addresses.append(kit.at(block))
    var its = List[Int32](length=max(nc, 1), fill=0)
    var written = List[Int](length=1, fill=0)
    addresses.append(Int(its.unsafe_ptr()))
    addresses.append(Int(written.unsafe_ptr()))
    var params: List[Int] = [n, p, q, nc, words.max_iter, 1 if words.mode_b else 0, Int(words.canonical),
                             Int(words.canonical), Int(words.svd_algo)]
    var tol: List[Float64] = [words.tol]
    var done = kit.xd(XD_PLS_FIT, addresses, params, tol)
    var nits = written[0]
    _ = written^
    for k in range(6):  # driver-loop: over the six fitted blocks
        kit.wrote_host(outs[k])
    if done < nc:
        for k in range(6):  # driver-loop: over the six fitted blocks
            outs[k] = kit.cols_of(outs[k], 0, done)
    # The completed counts, bounded as the old list[:nits] for the driver's count.
    var stop = max(nc + nits, 0) if nits < 0 else min(nits, nc)
    result.iterations = List[Int32](length=stop, fill=0)
    if stop > 0:
        unsafe_memcpy(dest=result.iterations.unsafe_ptr(), src=its.unsafe_ptr(), count=stop)
    _ = its^
    result.x_rotations = pls_rotation(kit, outs[0], outs[4])
    result.y_rotations = pls_rotation(kit, outs[1], outs[5])
    var coef = kit.mm(result.x_rotations, outs[5], tb=True)
    var scaled = kit.ew(DK_MUL, coef, ys[2])
    var x_std_t = kit.transpose(xs[2])
    var divided = kit.ew(DK_DIV, scaled, x_std_t)
    result.coef = kit.transpose(divided)
    result.done = done
    result.x_mean = xs[1]
    result.x_std = xs[2]
    result.y_mean = ys[1]
    result.y_std = ys[2]
    if transform:
        result.tx = pls_project(kit, x, xs[1], xs[2], result.x_rotations)
        result.ty = pls_project(kit, ym, ys[1], ys[2], result.y_rotations)
    result.outs = outs^
    return result^
