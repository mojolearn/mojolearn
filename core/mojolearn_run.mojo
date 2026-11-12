# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn.mojo` (wave 8 run split): the borrowed
buffer address helpers, the KMeans and labels native tables and their slots,
the resident-fold QN table publish, the KMeans fits and the device label
encoder the doors call with the GIL released. Typed Mojo only: no Python
import."""
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, u32_ptr
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.native_descriptor_abi import NativeDescriptorCall
from core.labels_native_api import LABELS_NATIVE_ABI_NAME
from core.labels_native_device import LabelsDeviceKernels, labels_device_native_v2 as labels_native_entry
from glm.logistic_fold_native import LOGISTIC_FOLD_API, logistic_fold_v1
from core.estimators_entry_table import EstimatorsEntryKey, estimators_table_publish, estimators_vendor_word, ESTIMATORS_TABLE_QN_FOLD
from bindings.parallel_base_entry_device import CORE_DEVCTX_SLOT
from core.shared_slot import SharedSlot
from core.kmeans_native_api import KMEANS_NATIVE_ABI
from core.kmeans_native_device import kmeans_device_native_v1
from core.neural_context import process_ctx
from cluster.estimator import kmeans_fit, KMeansFitResult

comptime _DEVCTX_SLOT = CORE_DEVCTX_SLOT


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    """A caller's float32 buffer, borrowed, never owned.

    The origin is untracked because the owner is a NumPy array on the other
    side of the boundary and Mojo cannot see it. The wrapper holds that array
    throughout the call, which is the whole contract.
    """
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return f32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    """A caller's float64 buffer, borrowed, never owned; `_f32_ptr`'s twin
    (DEVIATION 2321). The same helper, spelled the same way, sits in
    `bindings/_mojolearn_gbdt.mojo`, `_mojolearn_estimators.mojo`,
    `_mojolearn_svm.mojo` and `_mojolearn_gp.mojo`."""
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return f64_ptr(addr)


def _u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return u32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return i32_ptr(addr)


struct KMeansDeviceNativeTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(KMEANS_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(kmeans_device_native_v1)


comptime KMEANS_NATIVE_TABLE = SharedSlot[KMeansDeviceNativeTable, ("MojolearnKMeansDeviceNativeIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKMeansDeviceNativeFastV1")]


struct LabelsNativeTable[ABI: StaticString](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(Self.ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(labels_native_entry)


# The slot holds THIS module's dispatch pointer: tier-keyed so a FAST and an
# IDENTICAL build never hand out each other's code (core/shared_slot.mojo).
# TOMBSTONE: LABELS_NATIVE_API_V1_V2 (DELETED: no reader; the label doors admit labels_native_api_v3 only) deleted 2026-11-12 by round2-fixes-2; code recoverable at 8c8e62f56b.
# Restore: git apply experiments/removed/LABELS_NATIVE_API_V1_V2.patch
comptime LABELS_NATIVE_TABLE = SharedSlot[LabelsNativeTable[LABELS_NATIVE_ABI_NAME], ("MojolearnLabelsNativeTableIdenticalV4" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnLabelsNativeTableFastV4")]


def _publish_qn_fold_table() raises:
    # The resident-fold QN entry (core/estimators_entry_table.mojo family
    # ESTIMATORS_TABLE_QN_FOLD, the former `qn_fit_fold` export's body):
    # QNRegressor's owner (bindings/qn_native.mojo) fits a device fold
    # through it. LogisticRegression's door still reaches the same entry
    # through the `logistic_fold_native_api_v1` capsule. A build with no GPU
    # vendor ("none") publishes nothing.
    var word = estimators_vendor_word(String(COMPILED_VENDOR))
    if word < 0:
        return
    var api = LOGISTIC_FOLD_API.get_or_create_ptr()
    estimators_table_publish[ESTIMATORS_TABLE_QN_FOLD](EstimatorsEntryKey(Int(GLOBAL_NUMERIC_MODE), word),
        Int(api[].entries.unsafe_ptr()), logistic_fold_v1)


def kmeans_fit_run(
    xp: MutPointer[Float32, MutUntrackedOrigin], ns: Int, nf: Int, nc: Int,
    cp: MutPointer[Float32, MutUntrackedOrigin], lp: MutPointer[UInt32, MutUntrackedOrigin],
    wp: MutPointer[Float32, MutUntrackedOrigin], nw: Int, mi: Int, tl: Float64, sd: UInt64,
    ninit: Int, ii: Int, mm: Int, ovs: Float64, word_addr: Int,
) raises -> KMeansFitResult:
    """`kmeans_fit_binding`'s fit on the base binding's context, synchronized."""
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = kmeans_fit(
        ctx, xp, ns, nf, nc, cp, lp, wp, nw, mi, tl, sd, ninit, ii, mm,
        0.0, ovs,
        # TOMBSTONE: MOJOLEARN_KMEANS_FAST_LAZY_SHIFT (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
        # Tried: the lazy convergence read (shift tested every KMEANS_LAZY_EVERY iterations) on the KMeans estimator's fits; kmeans istella 1451.5 -> 1527.7 ms, taxi 977.6 -> 974.8, inertia equal.
        # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_FAST_LAZY_SHIFT.patch; record in docs/TOMBSTONES.md.
        lazy_shift=False,
        finite_word=word_addr,
    )
    ctx.synchronize()
    return r


def kmeans_fit_transform_run(
    xp: MutPointer[Float32, MutUntrackedOrigin], ns: Int, nf: Int, nc: Int,
    cp: MutPointer[Float32, MutUntrackedOrigin], lp: MutPointer[UInt32, MutUntrackedOrigin],
    wp: MutPointer[Float32, MutUntrackedOrigin], nw: Int, mi: Int, tl: Float64, sd: UInt64,
    ninit: Int, ii: Int, mm: Int, ovs: Float64, tp: MutPointer[Float32, MutUntrackedOrigin],
    word_addr: Int,
) raises -> KMeansFitResult:
    """`kmeans_fit_transform_binding`'s fit and transform on the base
    binding's context, synchronized."""
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var r = kmeans_fit(
        ctx, xp, ns, nf, nc, cp, lp, wp, nw, mi, tl, sd, ninit, ii, mm,
        0.0, ovs,
        lazy_shift=False,
        transform_out=tp,
        finite_word=word_addr,
    )
    ctx.synchronize()
    return r


def encode_labels_run[dt: DType](source: Int, count: Int, classes: Int, capacity: Int, codes: Int) raises -> Int:
    """`_encode_labels_binding[dt]`'s device encoder: the class count, or -1
    when the caller must take `unique_inverse`."""
    var code = -1
    comptime if dt == DType.float32:
        code = 0
    comptime if dt == DType.float64:
        code = 1
    comptime if dt == DType.int32:
        code = 2
    comptime if dt == DType.int64:
        code = 3
    comptime if dt == DType.uint32:
        code = 4
    comptime if dt == DType.uint8:
        code = 5
    if code < 0:
        raise Error("encode_labels: dtype without a device encoder")
    return LabelsDeviceKernels.encode(source, code, count, classes, capacity, codes)
