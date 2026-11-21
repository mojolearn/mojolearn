# SPDX-License-Identifier: Apache-2.0
"""Typed SVC transport: the requests of a binary solve and of a machine's
query, and one driver over the SVM binding's two entry tables. Every
provider keeps its existing solver, query, epilogue and glue owners; this
file holds no numerical code and no Python."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

# The device table (svm/svc_native_device.mojo; a GPU binding only): the two
# entries the shared SVC table (svm/operation_metadata.mojo SVC_ABI_NAME)
# does not have.
comptime SVC_DEVICE_ABI = "mojolearn.svc.device.v1"
comptime SVC_DEVICE_FIT = 0
comptime SVC_DEVICE_RESIDENT = 1
comptime SVC_DEVICE_COUNT = 2
comptime SVC_DEVICE_FIT_ADDRESSES = 7
comptime SVC_DEVICE_FIT_INTS = 10
comptime SVC_DEVICE_FIT_FLOATS = 4
comptime SVC_DEVICE_RESIDENT_ADDRESSES = 5
comptime SVC_DEVICE_RESIDENT_INTS = 11
comptime SVC_DEVICE_RESIDENT_FLOATS = 6

# Operations of the shared SVC table (svm/native_descriptor_schema.mojo).
comptime SVC_TABLE_FIT = 0
comptime SVC_TABLE_PREDICT = 1
comptime SVC_TABLE_EPILOGUE = 2
comptime SVC_TABLE_PLATT = 3
comptime SVC_TABLE_SPLITMIX = 4
comptime SVC_TABLE_MODE = 5
comptime SVC_TABLE_FLAGS = 8
comptime SVC_TABLE_SCALE_GAMMA = 9
comptime SVC_TABLE_OVO = 10
comptime SVC_TABLE_GEMV = 11
comptime SVC_TABLE_COUNT = 12


@fieldwise_init
struct SvcFitRequest(Copyable, Movable):
    """One binary C-SVC solve. `device_form`: X is the host words at `x` or
    the fold (`base_id`, `base_rows`, `index_id`) gathered on the device,
    `scale_gamma` asks the entry to resolve gamma='scale' from its device
    copy, and `info` receives eight float64 (the five of the short form,
    then X's finite word, its range word and the gamma solved with).
    Otherwise the short form: host words at `x`, five info words, a flagged
    X refused by the entry. `bounds` 0 = unweighted."""
    var x: Int
    var labels: Int
    var dual: Int
    var support: Int
    var vectors: Int
    var info: Int
    var bounds: Int
    var rows: Int
    var features: Int
    var kernel: Int
    var max_iter: Int
    var nochange_steps: Int
    var degree: Int
    var base_id: Int
    var base_rows: Int
    var index_id: Int
    var scale_gamma: Bool
    var device_form: Bool
    var gamma: Float64
    var c: Float64
    var tol: Float64
    var coef0: Float64


@fieldwise_init
struct SvcQueryRequest(Copyable, Movable):
    """One machine's decisions (or labels, `predict_class`) on `rows`
    queries, written as float32 at `dst`. `resident`: the machine's support
    rows and duals are read from slot `slot` of `lease` (0: uploaded for
    this call only), the query is the host words at `x` or the fold gathered
    on the device, and the entry writes X's finite and range words at
    `status` (two int32)."""
    var x: Int
    var dual: Int
    var vectors: Int
    var dst: Int
    var status: Int
    var rows: Int
    var features: Int
    var supports: Int
    var kernel: Int
    var degree: Int
    var predict_class: Bool
    var lease: Int
    var slot: Int
    var base_id: Int
    var base_rows: Int
    var index_id: Int
    var resident: Bool
    var bias: Float64
    var label0: Float64
    var label1: Float64
    var gamma: Float64
    var cache_mib: Float64
    var coef0: Float64


@fieldwise_init
struct SvcDrivers(Copyable, Movable):
    """The two entries of one SVM binding: `table`, the shared SVC table
    (short-form solve, host-words query, the per-row epilogues and row glue,
    Platt, the fold shuffle, scale_gamma's limbs, the one-vs-one layout, the
    dual GEMV), and `device`, the device table (`has_device`; the device-X
    solve and the resident query). The caller releases the GIL around a run
    of these calls; none of them touches Python."""
    var table: NativeDescriptorCall
    var device: NativeDescriptorCall
    var has_device: Bool

    def fit(self, r: SvcFitRequest) raises -> Int:
        """Returns n_support."""
        if r.device_form:
            var a: Array[Int, 7] = [r.x, r.labels, r.dual, r.support, r.vectors, r.info, r.bounds]
            var p: Array[Int, 10] = [r.rows, r.features, r.kernel, r.max_iter, r.nochange_steps, r.degree,
                r.base_id, r.base_rows, r.index_id, Int(r.scale_gamma)]
            var f: Array[Float64, 4] = [r.gamma, r.c, r.tol, r.coef0]
            return native_descriptor_call_fixed(self.device, SVC_DEVICE_FIT, a, p, f)
        var a: Array[Int, 7] = [r.x, r.labels, r.dual, r.support, r.vectors, r.info, r.bounds]
        var p: Array[Int, 6] = [r.rows, r.features, r.kernel, r.max_iter, r.nochange_steps, r.degree]
        var f: Array[Float64, 4] = [r.gamma, r.c, r.tol, r.coef0]
        return native_descriptor_call_fixed(self.table, SVC_TABLE_FIT, a, p, f)

    def query(self, r: SvcQueryRequest) raises -> Int:
        if r.resident:
            var a: Array[Int, 5] = [r.x, r.dual, r.vectors, r.dst, r.status]
            var p: Array[Int, 11] = [r.rows, r.features, r.supports, r.kernel, Int(r.predict_class), r.degree,
                r.lease, r.slot, r.base_id, r.base_rows, r.index_id]
            var f: Array[Float64, 6] = [r.bias, r.label0, r.label1, r.gamma, r.cache_mib, r.coef0]
            return native_descriptor_call_fixed(self.device, SVC_DEVICE_RESIDENT, a, p, f)
        var a: Array[Int, 4] = [r.x, r.dual, r.vectors, r.dst]
        var p: Array[Int, 6] = [r.rows, r.features, r.supports, r.kernel, Int(r.predict_class), r.degree]
        var f: Array[Float64, 6] = [r.bias, r.label0, r.label1, r.gamma, r.cache_mib, r.coef0]
        return native_descriptor_call_fixed(self.table, SVC_TABLE_PREDICT, a, p, f)

    def epilogue(self, mode: Int, first: Int, second: Int, third: Int, dst: Int,
                 p1: Int, p2: Int, p3: Int, p4: Int, f3: Float64 = 0.0, f4: Float64 = 0.0) raises -> Int:
        """`svc_pair_epilogue`: the per-row epilogues (modes 0 to 5; `f4` is
        label1) and the row glue (GATHER 6, SELECT 7, C_ROWS 8 with C in
        `f3`, FOLD 9, FOLD_FINISH 10) on the binding's own owners."""
        var a: Array[Int, 4] = [first, second, third, dst]
        var p: Array[Int, 5] = [mode, p1, p2, p3, p4]
        var f: Array[Float64, 5] = [0.0, 0.0, 0.0, f3, f4]
        return native_descriptor_call_fixed(self.table, SVC_TABLE_EPILOGUE, a, p, f)

    def platt(self, decisions: Int, labels: Int, dst: Int, count: Int) raises -> Int:
        var a: Array[Int, 3] = [decisions, labels, dst]
        var p: Array[Int, 1] = [count]
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.table, SVC_TABLE_PLATT, a, p, f)

    def splitmix(self, dst: Int, count: Int, seed_lo: Int, seed_hi: Int) raises -> Int:
        var a: Array[Int, 1] = [dst]
        var p: Array[Int, 3] = [count, seed_lo, seed_hi]
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.table, SVC_TABLE_SPLITMIX, a, p, f)

    def mode(self) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.table, SVC_TABLE_MODE, a, p, f)

    def flags(self) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.table, SVC_TABLE_FLAGS, a, p, f)

    def scale_gamma_limbs(self, x: Int, cells: Int, dst: Int) raises -> Int:
        var a: Array[Int, 2] = [x, dst]
        var p: Array[Int, 1] = [cells]
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.table, SVC_TABLE_SCALE_GAMMA, a, p, f)

    def ovo_layout(self, addresses: List[Int], words: List[Int]) raises -> Int:
        """`svc_ovo_layout`: addresses [support out, n_support out, dual
        out, (support, dual) per pair], words [k, n, P, cap, (i, j, m) per
        pair]. Returns the support count."""
        var f = List[Float64]()
        return native_descriptor_call(self.table, SVC_TABLE_OVO, addresses, words, f)

    def dual_gemv(self, dual: Int, vectors: Int, dst: Int, supports: Int, features: Int, sign: Int) raises -> Int:
        var a: Array[Int, 3] = [dual, vectors, dst]
        var p: Array[Int, 3] = [supports, features, sign]
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.table, SVC_TABLE_GEMV, a, p, f)
