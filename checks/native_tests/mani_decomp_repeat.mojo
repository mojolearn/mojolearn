# SPDX-License-Identifier: Apache-2.0
"""Repeatability of the manifold decomposition kit (plan
NATIVE_ENTRY_POINTS_AFTER_0866 rows N290, N291, N295, N301), natively.

Ported from python/mojolearn/tests/test_x_decomp_kernel_table.py
(test_repeated_transforms_return_the_same_bits for Isomap nn5/nc2 and LLE
nn6/nc2, test_isomap_transform_matches_a_fresh_fit_of_the_same_data,
test_isomap_radius_transform_is_repeatable,
test_concurrent_transforms_of_one_model_agree,
test_fit_is_repeatable_across_calls_with_the_shared_zero) and from
python/mojolearn/tests/test_native_lle_iteration_lifetime.py (300x16 LLE,
n_neighbors 10, nc 2, random_state 0, the iterative shift-invert solve: two
fits give the same embedding bits and reconstruction error, finite, X
unchanged).

Each public operation is reproduced in typed values exactly as the door
bindings/mani_decomp_native_door.mojo composes it: one `MRun` per operation,
the x_decomp image as an `MImage` (its dispatch an `EntryRef` built directly
from the host or device descriptor, its capability word read through
XD_CAPABILITIES as kit_connect reads it), X admitted by `from_snapshot`, the
RUN piece of bindings/mani_decomp_run.mojo, then PACK in the door's order (an
output's words copied out before its fitted owner is made, because copying a
device matrix out moves it to the host), the fitted owners as
`ManifoldMatrix` values (mk_owner), transform reading them as `MFitted`
(mk_fitted) and keeping the copies it made on them (mk_keep), and the image's
device zero cached across operations of one kit (mk_finish, row N290).

Data: the Python tests drew NumPy standard_normal; every assertion here is a
relation (repeat bits, agreement of two fits or of concurrent reads), so the
same shapes are drawn from a fixed hash through Box-Muller instead.

Host column: the six fixtures and the twelve concurrent transforms (six of
one fitted Isomap, six of one fitted LLE; each in its own MRun reading the
same fitted owners) are one batch of independent units across host cores.
The two models those twelve read are fitted, and their expected transform
taken, before the batch: the units under test must all read one model.
Device column: the fixtures run one after another on the one device context
(the shared-DeviceContext concurrent-submission failure recorded with
spectral_prediction stays open), and the concurrency fixture becomes six
ordered reads of one fitted model; kernels keep their own parallelism.
"""
from std.math import cos, isfinite, log, sqrt
from std.memory import bitcast, unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved, entry_ref_word
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.descriptor_schema import XD_OPERATION_COUNT, XD_CAPABILITIES, XD_DEV_FREE
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.descriptor_device import x_decomp_device_v1
from bindings.mani_kit_run import MRun, MImage, MFitted, MK_F32P
from bindings.mani_decomp_run import (
    MANI_METRIC_EUCLIDEAN, IsoPlan, IsoState, LlePlan,
    cmds_core, iso_fit_run, iso_transform_run, lle_fit_run, lle_transform_run,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require

# The fitted owner, as bindings/manifold_storage_native_unpack.mojo
# `ManifoldMatrix` holds it (that module's file name marks it a boundary
# module, which a check may not import): host words and/or a pool id, the
# image key and dispatch of that id, the side the fit left it on; the pool id
# goes back through its own image's dispatch when the owner dies.
struct ManifoldMatrix(Movable):
    var rows: Int
    var cols: Int
    var words: List[Float32]
    var has_words: Bool
    var device: Int
    var device_key: Int
    var device_entry: EntryRef
    var on_device: Bool

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.words = List[Float32]()
        self.has_words = False
        self.device = -1
        self.device_key = 0
        self.device_entry = entry_ref_unresolved()
        self.on_device = False

    def words_address(self) -> Int:
        if not self.has_words:
            return 0
        return Int(self.words.unsafe_ptr())

    def __deinit__(deinit self):
        if self.device >= 0:
            try:
                _ = entry_ref_call(self.device_entry, XD_DEV_FREE, List[Int](), [self.device], List[Float64]())
            except:
                pass


comptime MANI_REPEAT_FIXTURES = 6
comptime MANI_CONCURRENT = 6
# the door's iterative-LLE gate (bindings/mani_decomp_native_door.mojo)
comptime _LLE_ITER_MIN_N = 200
comptime _LLE_ITER_MAX_K = 10
# the public LocallyLinearEmbedding defaults (python/mojolearn/_expansion_decomp.py)
comptime _LLE_REG = 1e-3
comptime _LLE_MAX_ITER = 100


# ---------------------------------------------------------------- data
def _mix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _unit(seed: Int, i: Int) -> Float64:
    """A fixed value in [0, 1) for (seed, i): 53 hashed bits."""
    var h = _mix(UInt64(seed) * UInt64(0x100000001B3) + UInt64(i))
    return Float64(Int(h >> 11)) / 9007199254740992.0


def normal_matrix(rows: Int, cols: Int, seed: Int) -> List[Float32]:
    """`default_rng(seed).standard_normal((rows, cols)).astype(float32)`
    stand-in: Box-Muller over the hash, C order."""
    var count = rows * cols
    var out = List[Float32](length=count, fill=0.0)
    for i in range(count):  # native data loop: one small fixture matrix
        var u1 = 1.0 - _unit(seed, 2 * i)
        var u2 = _unit(seed, 2 * i + 1)
        out[i] = Float32(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2))
    return out^


def same_bits(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(one output's words)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def all_finite(a: List[Float32]) -> Bool:
    for i in range(len(a)):  # small-loop(one output's words)
        if not Bool(isfinite(a[i])):
            return False
    return True


# ---------------------------------------------------------------- the kit
struct Kit(Movable):
    """One x_decomp image as mk_image admits it, and the device zero
    mk_finish caches for the next operation on it (row N290)."""
    var entry: EntryRef
    var key: Int
    var capabilities: Int
    var dev_one: Int

    def __init__(out self, entry: EntryRef, key: Int) raises:
        self.entry = entry
        self.key = key
        self.capabilities = entry_ref_call(entry, XD_CAPABILITIES, List[Int](), List[Int](), List[Float64]())
        self.dev_one = -1

    def release(mut self):
        if self.dev_one >= 0:
            try:
                _ = entry_ref_call(self.entry, XD_DEV_FREE, List[Int](), [self.dev_one], List[Float64]())
            except:
                pass
            self.dev_one = -1


def make_kit[Device: Bool]() raises -> Kit:
    var word = entry_ref_word(Int(GLOBAL_NUMERIC_MODE), 0, XD_OPERATION_COUNT)
    comptime if Device:
        return Kit(EntryRef(x_decomp_device_v1, word), 2)
    return Kit(EntryRef(x_decomp_host_v1, word), 1)


def _image(mut run: MRun, kit: Kit) -> Int:
    """mk_image: identical mode, so the Apple FAST ordered LLE cells are off."""
    return run.image(MImage(kit.entry, kit.key, kit.capabilities, False, kit.dev_one))


def _finish(mut run: MRun, mut kit: Kit):
    """mk_finish: a device zero this RUN made is cached on the kit, then every
    other pool id goes back."""
    for k in range(len(run.images)):  # driver-loop: over this operation's images (at most 2)
        if run.images[k].dev_one_made and kit.dev_one < 0:
            kit.dev_one = run.images[k].dev_one
            run.images[k].dev_one_made = False
    run.close()


def _words(mut run: MRun, a: Int) raises -> List[Float32]:
    """mk_out2: a copy of matrix a's words (a device matrix moves to the host)."""
    var count = run.n(a)
    var out = List[Float32](length=count, fill=0.0)
    if count > 0:
        var src = run.addr(a)
        unsafe_memcpy(dest=MK_F32P(unsafe_from_address=Int(out.unsafe_ptr())),
                      src=MK_F32P(unsafe_from_address=src), count=count)
    return out^


def _owner(mut run: MRun, a: Int) raises -> ManifoldMatrix:
    """mk_owner: matrix a as a fitted owner, on the side the fit left it."""
    var ex = run.export(a)
    var owner = ManifoldMatrix(ex.rows, ex.cols)
    if ex.device >= 0:
        owner.device = ex.device
        owner.device_key = run.images[ex.image].key
        owner.device_entry = run.images[ex.image].entry
        owner.on_device = True
    else:
        swap(owner.words, ex.words)  # transfer the field, leaving `ex` whole
        owner.has_words = True
    return owner^


def _fitted(mut run: MRun, owner: ManifoldMatrix) -> Int:
    """mk_fitted: the owner's words' address and pool id, never moved."""
    return run.fit(MFitted(owner.rows, owner.cols, owner.words_address(), owner.device, owner.device_key,
                           owner.device_entry, owner.on_device))


def _keep(mut run: MRun, a: Int, mut owner: ManifoldMatrix):
    """mk_keep: the copies the RUN made of fitted matrix a stay on its owner."""
    var fi = run.mats[a].fitted
    if fi < 0:
        return
    if not owner.has_words:
        var words = run.kept_host(fi)
        if len(words) > 0:
            owner.words = words^
            owner.has_words = True
    if owner.device < 0 and run.fitted[fi].new_device >= 0:
        var k = run.fitted[fi].new_image
        var handle = run.kept_device(fi)
        if handle >= 0:
            owner.device = handle
            owner.device_key = run.images[k].key
            owner.device_entry = run.images[k].entry


# ---------------------------------------------------------------- Isomap
struct IsoModel(Movable):
    """The fitted fields Isomap.transform reads (`_fit_X`, `dist_matrix_m_`,
    `_k_col`, `_k_all`, `_iso_v_m_`, `_knn` / `radius`) and `embedding_`."""
    var fit_x: ManifoldMatrix
    var d: ManifoldMatrix
    var col: ManifoldMatrix
    var allm: ManifoldMatrix
    var iv: ManifoldMatrix
    var knn: Int  # -1 for a radius model
    var radius: Float64
    var radius_text: String
    var embedding: List[Float32]

    def __init__(out self, var fit_x: ManifoldMatrix, var d: ManifoldMatrix, var col: ManifoldMatrix,
                 var allm: ManifoldMatrix, var iv: ManifoldMatrix, knn: Int, radius: Float64,
                 var radius_text: String, var embedding: List[Float32]):
        self.fit_x = fit_x^
        self.d = d^
        self.col = col^
        self.allm = allm^
        self.iv = iv^
        self.knn = knn
        self.radius = radius
        self.radius_text = radius_text^
        self.embedding = embedding^


def iso_fit(mut kit: Kit, x: List[Float32], rows: Int, cols: Int, nn: Int, radius: Float64,
            radius_text: String) raises -> IsoModel:
    """`Isomap(n_neighbors=nn, n_components=2).fit(X)` (nn -1: `n_neighbors=None,
    radius=radius`); metric minkowski p 2 (kind 0), eigen_solver 'auto'."""
    var plan = IsoPlan()
    plan.kind = 0
    plan.pw = 2.0
    var run = MRun()
    try:
        var k = _image(run, kit)
        var m = run.from_snapshot(k, Int(x.unsafe_ptr()), rows, cols, 2, "X")
        if nn < 0:
            plan.use_radius = True
            plan.radius = Float64(Float32(radius))
        else:
            plan.nn = nn
        plan.topk = True
        plan.nc = 2
        var fit = iso_fit_run(run, k, m, plan)
        _ = x  # the admitted X stays alive through the RUN above
        # PACK, in the door's order
        _ = _words(run, fit.d)  # dist_matrix_
        var embedding = _words(run, fit.em)
        var d = _owner(run, fit.d)
        var col = _owner(run, fit.col)
        var allm = _owner(run, fit.allm)
        _ = _owner(run, fit.w)  # eigenvalues_m_ (transform reads _iso_v_m_)
        _ = _owner(run, fit.v)  # eigenvectors_m_
        _ = _owner(run, fit.em)  # embedding_m_
        var fit_x = _owner(run, m)
        _ = _owner(run, fit.kc)  # _Kc (reconstruction_error only)
        var iv = _owner(run, fit.iv)
        var knn = -1 if nn < 0 else nn
        return IsoModel(fit_x^, d^, col^, allm^, iv^, knn, Float64(Float32(radius)), radius_text.copy(), embedding^)
    finally:
        _finish(run, kit)


def _iso_transform_into(mut run: MRun, k: Int, model: IsoModel, mut st: IsoState, q: List[Float32],
                        rows: Int, cols: Int) raises -> Int:
    var qm = run.from_snapshot(k, Int(q.unsafe_ptr()), rows, cols, 2, "X")
    st.fit_x = _fitted(run, model.fit_x)
    st.d = _fitted(run, model.d)
    st.col = _fitted(run, model.col)
    st.allm = _fitted(run, model.allm)
    st.kind = 0
    st.pw = 2.0
    if model.knn < 0:
        st.radius = model.radius
        st.radius_text = model.radius_text.copy()
    else:
        st.knn = model.knn
    st.iv = _fitted(run, model.iv)
    var out = iso_transform_run(run, k, qm, st)
    _ = q  # the admitted query stays alive through the RUN above
    return out


def iso_transform(mut kit: Kit, mut model: IsoModel, q: List[Float32], rows: Int, cols: Int) raises -> List[Float32]:
    """`model.transform(q)`, keeping the copies it made on the fitted owners."""
    var run = MRun()
    try:
        var k = _image(run, kit)
        var st = IsoState()
        var out = _iso_transform_into(run, k, model, st, q, rows, cols)
        var result = _words(run, out)
        _keep(run, st.fit_x, model.fit_x)
        _keep(run, st.d, model.d)
        _keep(run, st.col, model.col)
        _keep(run, st.allm, model.allm)
        _keep(run, st.iv, model.iv)
        return result^
    finally:
        _finish(run, kit)


def iso_transform_shared(kit: Kit, model: IsoModel, q: List[Float32], rows: Int, cols: Int) raises -> List[Float32]:
    """`model.transform(q)` from one of several threads reading the same
    fitted owners: after the first transform kept its copies, a later one
    makes none, so it keeps nothing (the owners are only read)."""
    var run = MRun()
    try:
        var k = _image(run, kit)
        var st = IsoState()
        var out = _iso_transform_into(run, k, model, st, q, rows, cols)
        return _words(run, out)
    finally:
        run.close()


# ---------------------------------------------------------------- LocallyLinearEmbedding
struct LleModel(Movable):
    var fit_x: ManifoldMatrix
    var em: ManifoldMatrix
    var knn: Int
    var embedding: List[Float32]
    var rows: Int
    var cols: Int
    var reconstruction_error: Float64

    def __init__(out self, var fit_x: ManifoldMatrix, var em: ManifoldMatrix, knn: Int,
                 var embedding: List[Float32], rows: Int, cols: Int, reconstruction_error: Float64):
        self.fit_x = fit_x^
        self.em = em^
        self.knn = knn
        self.embedding = embedding^
        self.rows = rows
        self.cols = cols
        self.reconstruction_error = reconstruction_error


def lle_fit(mut kit: Kit, x: List[Float32], rows: Int, cols: Int, nn: Int, nc: Int, seed: Int) raises -> LleModel:
    """`LocallyLinearEmbedding(n_neighbors=nn, n_components=nc,
    random_state=seed).fit(X)`: method 'standard', reg 1e-3, eigen_solver
    'auto', max_iter 100. The helper kit is the same image (identical mode)."""
    var plan = LlePlan()
    var run = MRun()
    try:
        var k = _image(run, kit)
        var hk = _image(run, kit)
        var m = run.from_snapshot(k, Int(x.unsafe_ptr()), rows, cols, 2, "X")
        var n = run.r(m)
        plan.nn = nn
        plan.nc = nc
        require(nn < n, "Expected n_neighbors < n_samples")
        plan.reg = _LLE_REG
        plan.auto_solver = True
        if n > _LLE_ITER_MIN_N and nc + 1 < _LLE_ITER_MAX_K:
            plan.max_iter = _LLE_MAX_ITER
            plan.seed = seed & 0xFFFFFFFF
        var fit = lle_fit_run(run, k, hk, m, plan)
        _ = x  # the admitted X stays alive through the RUN above
        if fit.refusal == 1:
            raise Error("LocallyLinearEmbedding: the shift-invert subspace iteration did not settle (last change "
                        + String(fit.change) + ")")
        if fit.refusal == 2:
            raise Error("LocallyLinearEmbedding: the shift-invert subspace iteration returned non-unit columns")
        var em_rows = run.r(fit.em)
        var em_cols = run.c(fit.em)
        var embedding = _words(run, fit.em)
        var em = _owner(run, fit.em)
        var fit_x = _owner(run, m)
        return LleModel(fit_x^, em^, nn, embedding^, em_rows, em_cols, fit.reconstruction_error)
    finally:
        _finish(run, kit)


def _lle_transform_into(mut run: MRun, k: Int, model: LleModel, mut fit_x: Int, mut em: Int, q: List[Float32],
                        rows: Int, cols: Int) raises -> Int:
    var qm = run.from_snapshot(k, Int(q.unsafe_ptr()), rows, cols, 2, "X")
    fit_x = _fitted(run, model.fit_x)
    em = _fitted(run, model.em)
    var out = lle_transform_run(run, k, qm, fit_x, em, model.knn, _LLE_REG)
    _ = q  # the admitted query stays alive through the RUN above
    return out


def lle_transform(mut kit: Kit, mut model: LleModel, q: List[Float32], rows: Int, cols: Int) raises -> List[Float32]:
    var run = MRun()
    try:
        var k = _image(run, kit)
        var fit_x = -1
        var em = -1
        var out = _lle_transform_into(run, k, model, fit_x, em, q, rows, cols)
        var result = _words(run, out)
        _keep(run, fit_x, model.fit_x)
        _keep(run, em, model.em)
        return result^
    finally:
        _finish(run, kit)


def lle_transform_shared(kit: Kit, model: LleModel, q: List[Float32], rows: Int, cols: Int) raises -> List[Float32]:
    var run = MRun()
    try:
        var k = _image(run, kit)
        var fit_x = -1
        var em = -1
        var out = _lle_transform_into(run, k, model, fit_x, em, q, rows, cols)
        return _words(run, out)
    finally:
        run.close()


# ---------------------------------------------------------------- ClassicalMDS
def cmds_fit_transform(mut kit: Kit, x: List[Float32], rows: Int, cols: Int, nc: Int) raises -> List[Float32]:
    """`ClassicalMDS(n_components=nc).fit_transform(X)` (metric 'euclidean')."""
    var run = MRun()
    try:
        var k = _image(run, kit)
        var m = run.from_snapshot(k, Int(x.unsafe_ptr()), rows, cols, 2, "X")
        var got = cmds_core(run, k, m, MANI_METRIC_EUCLIDEAN, 2.0, nc)
        _ = x  # the admitted X stays alive through the RUN above
        _ = _words(run, got[0])  # dissimilarity_matrix_
        _ = _words(run, got[1])  # eigenvalues_
        var embedding = _words(run, got[2])
        _ = _owner(run, got[2])  # embedding_m_
        return embedding^
    finally:
        _finish(run, kit)


# ---------------------------------------------------------------- fixtures
def repeated_isomap(mut kit: Kit) raises:
    var x = normal_matrix(60, 5, 1)
    var q = normal_matrix(17, 5, 2)
    var model = iso_fit(kit, x, 60, 5, 5, 0.0, "")
    var first = iso_transform(kit, model, q, 17, 5)
    for _ in range(3):  # driver-loop: over the three repeated transforms the original made
        var again = iso_transform(kit, model, q, 17, 5)
        require(same_bits(again, first), "Isomap: a repeated transform returned other bits")
    _ = x^
    _ = q^


def repeated_lle(mut kit: Kit) raises:
    var x = normal_matrix(60, 5, 1)
    var q = normal_matrix(17, 5, 2)
    var model = lle_fit(kit, x, 60, 5, 6, 2, 0)
    var first = lle_transform(kit, model, q, 17, 5)
    for _ in range(3):  # driver-loop: over the three repeated transforms the original made
        var again = lle_transform(kit, model, q, 17, 5)
        require(same_bits(again, first), "LocallyLinearEmbedding: a repeated transform returned other bits")
    _ = x^
    _ = q^


def isomap_fresh_fit(mut kit: Kit) raises:
    var x = normal_matrix(50, 4, 3)
    var q = normal_matrix(9, 4, 4)
    var a = iso_fit(kit, x, 50, 4, 5, 0.0, "")
    var b = iso_fit(kit, x, 50, 4, 5, 0.0, "")
    var ta = iso_transform(kit, a, q, 9, 4)
    var tb = iso_transform(kit, b, q, 9, 4)
    require(same_bits(ta, tb), "Isomap: the transform of an identical fresh fit returned other bits")
    _ = x^
    _ = q^


def isomap_radius(mut kit: Kit) raises:
    var x = normal_matrix(40, 3, 5)
    var q = normal_matrix(6, 3, 6)
    var model = iso_fit(kit, x, 40, 3, -1, 3.0, "3.0")
    var first = iso_transform(kit, model, q, 6, 3)
    var again = iso_transform(kit, model, q, 6, 3)
    require(same_bits(again, first), "Isomap radius 3.0: a repeated transform returned other bits")
    _ = x^
    _ = q^


def cmds_shared_zero(mut kit: Kit) raises:
    var x = normal_matrix(40, 4, 9)
    var a = cmds_fit_transform(kit, x, 40, 4, 2)
    var b = cmds_fit_transform(kit, x, 40, 4, 2)
    require(same_bits(a, b), "ClassicalMDS: a second fit with the shared zero returned other bits")
    _ = x^


def lle_iterative_lifetime(mut kit: Kit) raises:
    """300 rows cross the iterative shift-invert gate (> 200 rows, nc + 1 < 10)."""
    var x = normal_matrix(300, 16, 0)
    var before = x.copy()
    var first = lle_fit(kit, x, 300, 16, 10, 2, 0)
    require(first.rows == 300 and first.cols == 2, "LLE 300x16: embedding shape")
    require(all_finite(first.embedding), "LLE 300x16: embedding finite")
    require(Bool(isfinite(first.reconstruction_error)), "LLE 300x16: reconstruction error finite")
    var second = lle_fit(kit, x, 300, 16, 10, 2, 0)
    require(second.rows == 300 and second.cols == 2, "LLE 300x16: second embedding shape")
    require(all_finite(second.embedding), "LLE 300x16: second embedding finite")
    require(Bool(isfinite(second.reconstruction_error)), "LLE 300x16: second reconstruction error finite")
    require(same_bits(second.embedding, first.embedding), "LLE 300x16: a second fit returned other embedding bits")
    require(second.reconstruction_error == first.reconstruction_error,
            "LLE 300x16: a second fit returned another reconstruction error")
    require(same_bits(x, before), "LLE 300x16: the input changed")
    _ = x^
    _ = before^


def ordered_reads[Lle: Bool](mut kit: Kit) raises:
    """The device column's form of the concurrency fixture: six transforms of
    one fitted model, one after another, against the first."""
    var x = normal_matrix(80, 5, 7)
    var q = normal_matrix(25, 5, 8)
    comptime if Lle:
        var model = lle_fit(kit, x, 80, 5, 6, 2, 0)
        var expected = lle_transform(kit, model, q, 25, 5)
        for _ in range(MANI_CONCURRENT):  # driver-loop: over ordered reads of one fitted model on the one device
            require(same_bits(lle_transform(kit, model, q, 25, 5), expected),
                    "LocallyLinearEmbedding: a later read of one fitted model returned other bits")
    else:
        var model = iso_fit(kit, x, 80, 5, 5, 0.0, "")
        var expected = iso_transform(kit, model, q, 25, 5)
        for _ in range(MANI_CONCURRENT):  # driver-loop: over ordered reads of one fitted model on the one device
            require(same_bits(iso_transform(kit, model, q, 25, 5), expected),
                    "Isomap: a later read of one fitted model returned other bits")
    _ = x^
    _ = q^


def fixture[Device: Bool](unit: Int) raises:
    var kit = make_kit[Device]()
    try:
        if unit == 0:
            repeated_isomap(kit)
        elif unit == 1:
            repeated_lle(kit)
        elif unit == 2:
            isomap_fresh_fit(kit)
        elif unit == 3:
            isomap_radius(kit)
        elif unit == 4:
            cmds_shared_zero(kit)
        elif unit == 5:
            lle_iterative_lifetime(kit)
        elif unit == 6:
            ordered_reads[False](kit)
        else:
            ordered_reads[True](kit)
    finally:
        kit.release()


# ---- The host column ----
# Six fixtures plus twelve concurrent transforms in one batch across host cores.
def _run_cpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var kit = make_kit[False]()
    var x = normal_matrix(80, 5, 7)
    var q = normal_matrix(25, 5, 8)
    var iso = iso_fit(kit, x, 80, 5, 5, 0.0, "")
    var iso_expected = iso_transform(kit, iso, q, 25, 5)
    var lle = lle_fit(kit, x, 80, 5, 6, 2, 0)
    var lle_expected = lle_transform(kit, lle, q, 25, 5)
    comptime UNITS = MANI_REPEAT_FIXTURES + 2 * MANI_CONCURRENT
    var cases = report.cases(UNITS)
    def task(unit: Int) {imm cases, imm iso, imm lle, imm iso_expected, imm lle_expected, imm q}:
        try:
            if unit < MANI_REPEAT_FIXTURES:
                fixture[False](unit)
            else:
                var local = make_kit[False]()
                try:
                    if unit - MANI_REPEAT_FIXTURES < MANI_CONCURRENT:
                        var got = iso_transform_shared(local, iso, q, 25, 5)
                        require(same_bits(got, iso_expected),
                                "Isomap: a concurrent transform of one fitted model returned other bits")
                    else:
                        var got = lle_transform_shared(local, lle, q, 25, 5)
                        require(same_bits(got, lle_expected),
                                "LocallyLinearEmbedding: a concurrent transform of one fitted model returned other bits")
                finally:
                    local.release()
        except error:
            cases.fail(unit, error)
    report.parallel(task, UNITS)
    _ = iso^  # the fitted owners every concurrent unit read
    _ = lle^
    _ = iso_expected^
    _ = lle_expected^
    _ = q^
    _ = x^
    kit.release()


# ---- The device column ----
# One fixture at a time on the one device context; kernels stay parallel.
def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime UNITS = MANI_REPEAT_FIXTURES + 2
    var cases = report.cases(UNITS)
    for unit in range(UNITS):  # driver-loop: over the ordered device fixtures
        try:
            fixture[True](unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("mani_decomp_repeat", cpu=True, gpu=True, serial="", run=run_cases)
