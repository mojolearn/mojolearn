# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident KDE fit set (DEVIATION 3003, lane/knn-tiled-distance,
2026-09-17): `KernelDensity.score_samples` without the per-call upload and
validation of the training rows.

`kde_score_samples_host_ptr` (`kde/estimator.mojo`) validated the whole
training set (DEVIATION 604's finiteness scan, the cosine zero-row rule),
copied it into pinned memory and uploaded it on EVERY call: on an RTX 4090
that was 27.5 ms of a 75 ms Istella-S call at 2,000 queries and all of a
one-query call (the floor probe of lane/infer-speed-classical). cuML's
`fit` keeps `X` on the device and `score_samples` reuses it
(`kernel_density.py`, `self.X_`); this file gives the estimator the same
residency in the shape of the k-NN index registry (`neighbors/
resident_index.mojo`, DEVIATION 2921): the FIRST `score_samples` call
validates and uploads the training rows and the weights through
`kde_fit_prepare`, keeps the handle on the Python instance, and every later
call scores through `kde_score_samples_resident`, which validates and
uploads the QUERIES only.

WHAT DOES NOT CHANGE. The refusals are `kde_score_samples_host_ptr`'s, by
name and in its order (kernel and metric names, `kde_fit_validate`, the
training rows), then the query count and the query rows at score time;
the one difference is that a bad training set and a bad query count in the
same call now name the training set first. The score is `kde/impl/kde.mojo::
score_samples` over the same device bytes with the same `sum_weights`, and
the scores come back through the same pinned download, so the bits are the
one-shot entry's (`tools/identity_break.py`, the kde lanes, cuda against the
cpu host route). The context and the buffers are destroyed in DEVIATION
1946's order, buffers before the context.
"""
from core.resident_upload_count import RUC_KDE, resident_upload_note
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.numerics import NUMERIC_FAST

from core.hostptr import copy_f32
from core.host_finite_scan import host_copy_f32_finite
from max.gpu.host import DeviceBuffer, DeviceContext
from core.neural_context import process_ctx
from core.admitted_finite_status import enqueue_finite_word
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojolearnEstimatorsContextIdenticalV3" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnEstimatorsContextFastV3"


from checks.numerics import GLOBAL_NUMERIC_MODE
from core.identity_trace import IdentityTrace
from core.device_fold import device_sum_f32_fixed
from kde.impl.kde import score_samples
# TOMBSTONE: MOJOLEARN_IDN_KDE_PARTIAL_POOL (unmeasured opt-in arm (I20 mixed, no promotion)) deleted 2026-11-15 by lane/expdel2-19-kernel; code recoverable at 3af082d6f4.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_KDE_PARTIAL_POOL.patch; record in docs/TOMBSTONES.md.
from kde.impl.chunk_workspace import KdeChunkWorkspace
from kde.impl.neighbors.kernel_density import (
    KDE_ELEM_TPB,
    KDE_LSE_TPB,
    host_sum_weights,
    kde_fit_validate,
    kde_validate_data_ptr,
    kernel_from_name,
    metric_from_name,
)
from kde.impl.distance.distance_ops import DIST_COSINE_EXPANDED, DIST_L2_EXPANDED
from std.math import sqrt, isinf, isnan
from std.atomic import Atomic
from max.gpu import block_dim, block_idx, thread_idx
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock


def kde_host_rules_needed(metric: Int, finite_tested: Bool) -> Bool:
    """Whether DEVIATION 604's host pass (`kde_validate_data_ptr`) still has
    work for rows whose finiteness is already decided (lane
    a7b-device-x-neighbors): only the sqeuclidean magnitude bound and the
    cosine zero-row rule are not finiteness, so for every other metric a
    tested set (the fit set the door tested in its copy; queries the entry
    tests behind its upload) skips the host pass."""
    return not finite_tested or metric == DIST_COSINE_EXPANDED or metric == DIST_L2_EXPANDED


comptime _RULES_TPB = 256
comptime _RULES_NO_HIT = Int32(2147483647)


def _kde_row_rules_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    hit: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    cosine: Int32,
    bound: Float32,
):
    """`kde_validate_data`'s rules for cosine and sqeuclidean, one thread per row (lane vdf-scope-cluster4; the
    finite rule joined by lane vdf-scope-cluster5 so the refusal keeps the host pass's words).
    cosine: an all-zero row puts its row number into `hit[0]` (`Atomic.min`: the lowest, the row
    `cosine_zero_norm_row` names). Both metrics: the row's first word that is NaN or +-inf, or (sqeuclidean,
    `bound > 0`) has `|v| >= bound`, puts its word index (row * cols + col) into `hit[1]` (`Atomic.min`: the first
    word of the host scan, which tests each word for both in that order). Reads only; integer and compare work,
    no arithmetic on the values."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    var n_cols = Int(n_cols_in)
    var base = row * n_cols
    if cosine != Int32(0):
        var zero = True
        for f in range(n_cols):  # small-loop(one row's words)
            if x.unsafe_load(base + f) != Float32(0.0):
                zero = False
                break
        if zero:
            _ = Atomic.min(hit, Int32(row))
            return
    for f in range(n_cols):  # small-loop(one row's words)
        var v = x.unsafe_load(base + f)
        if isnan(v) or isinf(v) or (bound > Float32(0.0) and abs(v) >= bound):
            _ = Atomic.min(hit + 1, Int32(base + f))
            return


def kde_device_row_rules(ctx: DeviceContext, mut rows: DeviceBuffer[DType.float32], n_rows: Int, n_features: Int,
                         metric: Int, what: String) raises:
    """`kde_validate_data`'s rules for cosine and sqeuclidean over rows ALREADY ON THE DEVICE (`rows`, on `ctx`):
    the cosine zero-row rule, then each word's finite rule and the sqeuclidean magnitude bound in the host scan's
    order, so those two metrics keep the resident-store path (lane vdf-scope-cluster4, 2026-11-24) and refuse in
    the host pass's words, "kde: query contains NaN at row ..." included (lane vdf-scope-cluster5). One launch,
    two words read back (one more for a word refusal, which names the value). Any other metric: nothing to do
    (its finiteness is the caller's admission or the device finite word)."""
    if metric != DIST_COSINE_EXPANDED and metric != DIST_L2_EXPANDED:
        return
    if n_rows <= 0 or n_features <= 0:
        raise Error(
            "kde: " + what + " must have positive dimensions, got "
            + String(n_rows) + " x " + String(n_features)
        )
    var n = n_rows * n_features
    if n >= Int(_RULES_NO_HIT):
        raise Error("kde: " + what + " holds " + String(n) + " words; the device row rules index words in int32")
    if len(rows) < n:
        raise Error("kde: the resident " + what + " rows hold fewer than " + String(n) + " words")
    var cosine = metric == DIST_COSINE_EXPANDED
    var bound = Float32(0.0)
    if not cosine:
        bound = Float32(9.223372036854775808e18 / sqrt(Float64(n_features)))
    var d_hit = ctx.enqueue_create_buffer[DType.int32](2)
    var h_hit = ctx.enqueue_create_host_buffer[DType.int32](2)
    h_hit.unsafe_ptr().unsafe_store(0, _RULES_NO_HIT)
    h_hit.unsafe_ptr().unsafe_store(1, _RULES_NO_HIT)
    ctx.enqueue_copy(dst_buf=d_hit, src_ptr=h_hit.unsafe_ptr())
    ctx.enqueue_function[_kde_row_rules_kernel](
        rows.unsafe_ptr(), d_hit.unsafe_ptr(), Int32(n_rows), Int32(n_features), Int32(1 if cosine else 0), bound,
        grid_dim=((n_rows + _RULES_TPB - 1) // _RULES_TPB, 1, 1),
        block_dim=(_RULES_TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h_hit.unsafe_ptr(), src_buf=d_hit)
    ctx.synchronize()
    var zero_row = Int(h_hit.unsafe_ptr().unsafe_load(0))
    var hit = Int(h_hit.unsafe_ptr().unsafe_load(1))
    _ = d_hit^
    _ = h_hit^
    if cosine and zero_row != Int(_RULES_NO_HIT):
        raise Error(
            "kde: metric='cosine' but " + what + " row " + String(zero_row)
            + " is all zeros; cosine distance divides by ||x|| and is"
            " undefined at the origin (DEVIATION 553)"
        )
    if hit == Int(_RULES_NO_HIT):
        return
    # The refused word's value, for the host pass's sentence (one word read back).
    var h_value = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=h_value.unsafe_ptr(), src_buf=rows.create_sub_buffer[DType.float32](hit, 1))
    ctx.synchronize()
    var v = h_value.unsafe_ptr().unsafe_load(0)
    _ = h_value^
    if isnan(v) or isinf(v):
        # lane vdf-scope-cluster5: the host pass's finite words ("kde: query contains NaN at row ..."), restored.
        raise Error(
            "kde: " + what + " contains " + ("NaN" if isnan(v) else "infinity")
            + " at row " + String(hit // n_features) + ", column "
            + String(hit % n_features) + " (DEVIATION 604)"
        )
    raise Error(
        "kde: metric='sqeuclidean' needs |" + what + "| < 2^63/sqrt("
        + String(n_features) + ") so the expanded identity cannot form"
        " inf - inf; row " + String(hit // n_features) + ", column "
        + String(hit % n_features) + " is " + String(v) + " (DEVIATION 604)"
    )


# Immutable retained fit snapshots and direct upload are separate controls.
# Buffer copies are transport; the GPU score/statistics remain unchanged.
# M3 2026-10-06 F18: one excluded warmup and one score per arm, 509x3,
# 521x7,997x13; bandwidth invalidation, two models, refit, refusal recovery.
# Direct prep cold B/A=0.6349/0.9221/0.8119; repeated totals=0.4917/0.7277/0.9234,
# all captured score quality equal. Promote only FAST Apple direct preparation.
# Evidence ab-20261006/repairs-54c1f35a5/F18.
# Immutable snapshots independently measured faster (cold0.7578/0.9473/0.6224,
# repeated0.4999/0.9233/0.7824) but remain OFF: ownership semantics change and
# combination with direct preparation is unmeasured; one sample limits inference.
comptime KDE_FAST_IMMUTABLE_FIT = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU and is_defined["MOJOLEARN_KDE_FAST_IMMUTABLE_FIT"]()
# TOMBSTONE: MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF (old path beside the measured FAST + Apple default (F18)) deleted 2026-11-15 by lane/expdel2-19-kernel; code recoverable at 9b758c74c1.
# Restore: git apply experiments/removed/MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF.patch; record in docs/TOMBSTONES.md.
comptime KDE_FAST_DIRECT_PREP = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU


struct ResidentKdeFit(Movable):
    """One fitted KDE on the device: the training rows, the weights (or a
    one-element placeholder), their sum, and the shape and names the
    handle was prepared with."""

    var ctx: DeviceContext
    var train: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var has_weights: Bool
    var sum_w: Float32
    var n_train: Int
    var n_features: Int
    var kernel: Int
    var metric: Int
    var bandwidth: Float32
    var partial_pool: Optional[KdeChunkWorkspace]

    def __init__(
        out self,
        train_ptr: MutPointer[Float32, MutUntrackedOrigin],
        n_train: Int,
        n_features: Int,
        bandwidth: Float32,
        kernel: Int,
        metric: Int,
        weights: List[Float32],
        has_weights: Bool,
        finite_tested: Bool = False,
    ) raises:
        # `kde_score_samples_host_ptr`'s refusals for the fit set, in its
        # order: `kde_fit_validate`, then the training rows. `finite_tested`
        # (lane a7b): the door's fit tested the rows in its copy; the serial
        # host pass runs only for the rules that are not finiteness.
        kde_fit_validate(n_train, n_features, bandwidth, kernel, metric, weights, has_weights)
        if kde_host_rules_needed(metric, finite_tested):
            kde_validate_data_ptr(train_ptr, n_train, n_features, metric, "train")
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var n = n_train * n_features
        var train = ctx.enqueue_create_buffer[DType.float32](n)
        var host = ctx.enqueue_create_host_buffer[DType.float32](1 if KDE_FAST_DIRECT_PREP else n)
        comptime if KDE_FAST_DIRECT_PREP:
            # Caller retains the immutable host snapshot through this function's
            # existing completion; no pinned staging copy is needed.
            ctx.enqueue_copy(dst_buf=train, src_ptr=train_ptr)
        else:
            copy_f32(train_ptr, host.unsafe_ptr(), n)
            ctx.enqueue_copy(dst_buf=train, src_ptr=host.unsafe_ptr())
        resident_upload_note(RUC_KDE, n)  # check-only count (compiles to nothing by default)
        var sum_w = Float32(n_train)
        var wbuf: DeviceBuffer[DType.float32]
        if has_weights:
            wbuf = ctx.enqueue_create_buffer[DType.float32](n_train)
            var whost = ctx.enqueue_create_host_buffer[DType.float32](n_train)
            copy_f32(weights.unsafe_ptr(), whost.unsafe_ptr(), n_train)
            ctx.enqueue_copy(dst_buf=wbuf, src_ptr=whost.unsafe_ptr())
            resident_upload_note(RUC_KDE, n_train)
            ctx.synchronize()
            _ = whost^
            sum_w = host_sum_weights(weights)
        else:
            wbuf = ctx.enqueue_create_buffer[DType.float32](1)
            ctx.synchronize()
        _ = host^
        self.n_train = n_train
        self.n_features = n_features
        self.kernel = kernel
        self.metric = metric
        self.bandwidth = bandwidth
        self.partial_pool=Optional[KdeChunkWorkspace]()
        self.has_weights = has_weights
        self.sum_w = sum_w
        self.train = train^
        self.weights = wbuf^
        self.ctx = ctx^

    def __init__(
        out self,
        *,
        device_train: DeviceBuffer[DType.float32],
        n_train: Int,
        n_features: Int,
        bandwidth: Float32,
        kernel: Int,
        metric: Int,
        weights: List[Float32],
        has_weights: Bool,
    ) raises:
        """THE RESIDENT-STORE FIT (lane vdf-scope-cluster2b, 2026-11-24): the
        training rows are already on this binding's context (a fold taken
        on the device out of a resident X, core/estimators_store.mojo), so
        they are copied device to device into the fit's own buffer (the
        store may reuse the slot once the caller frees it) instead of
        uploaded. The same validation (`kde_fit_validate`; the rows were
        admitted finite by the caller; the cosine zero-row rule and the
        sqeuclidean bound run ON THE DEVICE over the copy,
        `kde_device_row_rules`, lane vdf-scope-cluster4), the same weights
        and sum: the same words, so the same bits."""
        kde_fit_validate(n_train, n_features, bandwidth, kernel, metric, weights, has_weights)
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var n = n_train * n_features
        if len(device_train) < n:
            raise Error("kde_fit_prepare_store: the resident slot holds " + String(len(device_train))
                + " words, " + String(n) + " read")
        var train = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_copy(dst_buf=train, src_buf=device_train.create_sub_buffer[DType.float32](0, n))
        kde_device_row_rules(ctx, train, n_train, n_features, metric, "train")
        var sum_w = Float32(n_train)
        var wbuf: DeviceBuffer[DType.float32]
        if has_weights:
            wbuf = ctx.enqueue_create_buffer[DType.float32](n_train)
            ctx.enqueue_copy(dst_buf=wbuf, src_ptr=weights.unsafe_ptr())
            resident_upload_note(RUC_KDE, n_train)
            sum_w = host_sum_weights(weights)
        else:
            wbuf = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.synchronize()
        self.n_train = n_train
        self.n_features = n_features
        self.kernel = kernel
        self.metric = metric
        self.bandwidth = bandwidth
        self.partial_pool = Optional[KdeChunkWorkspace]()
        self.has_weights = has_weights
        self.sum_w = sum_w
        self.train = train^
        self.weights = wbuf^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # I20 explicit owning-context drain/release before the fit buffers
        # and context, including an incomplete/failed score consumer.
        _ = self.partial_pool^
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.weights^
        _ = self.train^
        # DEVIATION 3010 (DEVIATION 2520's drain): the frees enqueued by the
        # releases above must complete before the context is destroyed, or
        # the runtime allocator's lock is left held and the next context's
        # first allocation never returns. Host-side drain; no output bit.
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


struct KdeFitRegistry(Defaultable, Movable):
    var entries: Dict[Int, ResidentKdeFit]
    var next_id: Int
    var metadata_lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ResidentKdeFit]()
        self.next_id = 1
        self.metadata_lock = List[Int64](length=1, fill=0)


comptime KDE_FIT_REGISTRY = SharedSlot[KdeFitRegistry, ( "MojolearnKdeResidentFitIdenticalV2" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKdeResidentFitFastV2" ), 2]


def kde_fit_prepare(
    train_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_train: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: String,
    metric: String,
    weights: List[Float32],
    has_weights: Bool,
    finite_tested: Bool = False,
) raises -> Int:
    """Validate and upload the fit set once; the handle every later score
    names. Refuses, by name, everything `kde_score_samples_host_ptr`
    refuses about the fit set (`finite_tested`: its finiteness is already
    decided by the door's fit)."""
    var k = kernel_from_name(kernel)
    var m = metric_from_name(metric)
    var entry = ResidentKdeFit(train_ptr, n_train, n_features, bandwidth, k, m, weights, has_weights,
                               finite_tested)
    return _kde_fit_register(entry^)


def kde_fit_prepare_device(
    device_train: DeviceBuffer[DType.float32],
    n_train: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: String,
    metric: String,
    weights: List[Float32],
    has_weights: Bool,
) raises -> Int:
    """`kde_fit_prepare` over training rows already on this binding's
    context (the resident-store form, lane vdf-scope-cluster2b): a device
    to device copy instead of the upload; the same handle semantics."""
    var k = kernel_from_name(kernel)
    var m = metric_from_name(metric)
    var entry = ResidentKdeFit(device_train=device_train, n_train=n_train, n_features=n_features,
                               bandwidth=bandwidth, kernel=k, metric=m, weights=weights, has_weights=has_weights)
    return _kde_fit_register(entry^)


def _kde_fit_register(var entry: ResidentKdeFit) raises -> Int:
    """File a prepared fit under a fresh handle."""
    var state = KDE_FIT_REGISTRY.get_or_create_ptr()
    var lock = state[].metadata_lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    shared_slot_lock(lock)
    try:
        if state[].next_id == 9223372036854775807:
            raise Error("resident KDE fit handle space exhausted")
        var handle = state[].next_id
        state[].next_id += 1
        state[].entries[handle] = entry^
        return handle
    finally:
        shared_slot_unlock(lock)


def kde_fit_release(handle: Int) raises:
    """Drop registry ownership; active queries retain their native buffers."""
    var state = KDE_FIT_REGISTRY.get_or_create_ptr()
    var lock = state[].metadata_lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var released: Optional[ResidentKdeFit] = None
    shared_slot_lock(lock)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident KDE fit handle")
        released = state[].entries.pop(handle)
    finally:
        shared_slot_unlock(lock)
    # Pool closure, buffer release and context drain must be outside the lock.
    _ = released^


@fieldwise_init
struct KdeFitLease(Movable):
    # These are owning native handle copies, never additional tensor copies.
    var ctx: DeviceContext
    var train: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var has_weights: Bool
    var sum_w: Float32
    var n_train: Int
    var n_features: Int
    var kernel: Int
    var metric: Int
    var bandwidth: Float32
    var partial_pool: Optional[KdeChunkWorkspace]

    def __deinit__(deinit self):
        _ = self.partial_pool^
        _ = self.weights^
        _ = self.train^
        _ = self.ctx^


def _kde_fit_lease(handle: Int) raises -> KdeFitLease:
    var state = KDE_FIT_REGISTRY.get_or_create_ptr()
    var lock = state[].metadata_lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    shared_slot_lock(lock)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident KDE fit handle")
        ref entry = state[].entries[handle]
        var pool: Optional[KdeChunkWorkspace] = None
        if entry.partial_pool:
            pool = entry.partial_pool.take()
        return KdeFitLease(entry.ctx.copy(), entry.train.copy(), entry.weights.copy(),
            entry.has_weights, entry.sum_w, entry.n_train, entry.n_features,
            entry.kernel, entry.metric, entry.bandwidth, pool^)
    finally:
        shared_slot_unlock(lock)


def _kde_fit_return(handle: Int, mut lease: KdeFitLease, clean: Bool) raises:
    if not lease.partial_pool:
        return
    if not clean:
        # A failed enqueue must not leave a busy/poisoned workspace cached.
        lease.partial_pool.value().close(lease.ctx)
        lease.partial_pool = None
        return
    var state = KDE_FIT_REGISTRY.get_or_create_ptr()
    var lock = state[].metadata_lock.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    shared_slot_lock(lock)
    try:
        if handle in state[].entries:
            ref entry = state[].entries[handle]
            if not entry.partial_pool:
                entry.partial_pool = lease.partial_pool.take()
    finally:
        shared_slot_unlock(lock)
    if lease.partial_pool:
        # A concurrent query returned the cache first, or the handle was
        # released. Keep only one idle workspace, with cleanup outside lock.
        lease.partial_pool.value().close(lease.ctx)
        lease.partial_pool = None



def kde_score_samples_resident(
    handle: Int,
    query: MutPointer[Float32, MutUntrackedOrigin],
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: String,
    metric: String,
    scores: MutPointer[Float32, MutUntrackedOrigin],
    elem_tpb: Int = KDE_ELEM_TPB,
    lse_tpb: Int = KDE_LSE_TPB,
    metric_arg: Float32 = Float32(2.0),
    want_total: Bool = False,
    status_addr: Int = 0,
) raises -> Float32:
    """`kde_score_samples_host_ptr` after its fit-set work: the query count
    and rows are validated as there, the queries staged once into pinned
    memory and uploaded, `score_samples` runs over the handle's training
    rows and weights, and the scores come back through the pinned download
    into `scores`. The names, the bandwidth and the feature count must be
    the handle's; a mismatch is refused rather than served. Returns the
    scores' fixed-order float32 sum on the device (`device_sum_f32_fixed`)
    when `want_total` (KernelDensity.score; lane pyglue-numeric: a host sum
    of the downloaded scores), else 0.

    `status_addr` (not 0): a host int32 word that receives the queries'
    finite status, tested on the bits of each word inside the staging copy
    into pinned memory that the upload makes in any case (`host_copy_f32_finite`,
    lane fb12-kde-finite): no launch, no download, no second pass. On a
    non-finite word nothing is uploaded or launched, `scores` is not
    written, and the door refuses. The host pass `kde_validate_data_ptr`
    then runs only for the sqeuclidean bound and the cosine zero-row rule."""
    var k = kernel_from_name(kernel)
    var m = metric_from_name(metric)
    var entry = _kde_fit_lease(handle)
    var clean = True
    try:
        if entry.kernel != k or entry.metric != m or entry.n_features != n_features or entry.bandwidth != bandwidth:
            raise Error(
                "kde_score_samples_resident: the handle was prepared with kernel "
                + String(entry.kernel) + ", metric " + String(entry.metric)
                + ", " + String(entry.n_features) + " features and bandwidth "
                + String(entry.bandwidth) + "; the call names kernel " + String(k)
                + ", metric " + String(m) + ", " + String(n_features)
                + " features and bandwidth " + String(bandwidth)
            )
        if n_query <= 0:
            raise Error("kde: X must have at least one row (n_query)")
        if kde_host_rules_needed(m, status_addr != 0):
            kde_validate_data_ptr(query, n_query, n_features, m, "query")
        clean = False
        var n = n_query * n_features
        var host = entry.ctx.enqueue_create_host_buffer[DType.float32](n)
        # Admission (lane fb12-kde-finite, 2026-11-17): the finite test rides
        # the staging copy into pinned memory that this upload makes for every
        # query anyway. The copy already loads each word; testing its exponent
        # bits in registers adds no pass, no launch and no download at any
        # size, where the device form (a7b) added a fill launch, a test launch,
        # a device and a pinned status buffer and a status download per call
        # (L40S attribution: launches 3 -> 5, D2H 1 -> 2, +0.22 ms host). The
        # verdict is known before anything is enqueued, so a refused query
        # uploads and computes nothing and `scores` is never written.
        if status_addr != 0:
            if not host_copy_f32_finite(Int(query), Int(host.unsafe_ptr()), n):
                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status_addr).unsafe_store(0, Int32(1))
                _ = host^
                clean = True
                return Float32(0)
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status_addr).unsafe_store(0, Int32(0))
        else:
            copy_f32(query, host.unsafe_ptr(), n)
        var dquery = entry.ctx.enqueue_create_buffer[DType.float32](n)
        entry.ctx.enqueue_copy(dst_buf=dquery, src_ptr=host.unsafe_ptr())
        var total = _kde_resident_tail(entry, dquery, n_query, n_features, bandwidth, k, m, kernel, metric,
                                       scores, elem_tpb, lse_tpb, metric_arg, want_total)
        _ = host^
        _ = dquery^
        clean = True
        return total
    finally:
        if clean:
            _kde_fit_return(handle, entry, True)
        else:
            # Preserve the originating execution failure if draining failed
            # workspace also raises; normal completion reports cleanup errors.
            try:
                _kde_fit_return(handle, entry, False)
            except:
                pass
        _ = entry


def _kde_resident_tail(
    mut entry: KdeFitLease,
    mut dquery: DeviceBuffer[DType.float32],
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    k: Int,
    m: Int,
    kernel: String,
    metric: String,
    scores: MutPointer[Float32, MutUntrackedOrigin],
    elem_tpb: Int,
    lse_tpb: Int,
    metric_arg: Float32,
    want_total: Bool,
) raises -> Float32:
    """The resident score over queries already on the fit's context (`dquery`): `score_samples`, the fixed-order
    total when asked, one download into `scores`. ONE BODY for the host-query and the resident-store forms (lane
    vdf-scope-cluster2b), so both write the same words."""
    var dout = entry.ctx.enqueue_create_buffer[DType.float32](n_query)
    var trace = IdentityTrace()
    trace.header(
        "kde: n_train=" + String(entry.n_train) + " n_query=" + String(n_query)
        + " n_features=" + String(n_features) + " kernel=" + kernel
        + " metric=" + metric + " metric_arg=" + String(metric_arg)
        + " weighted=" + String(entry.has_weights)
    )
    # TOMBSTONE: MOJOLEARN_KDE_SAMPLE_FUSED (DROP) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
    # Tried: the one-drain score + download branch here.
    # Restore: git apply experiments/removed/MOJOLEARN_KDE_SAMPLE_FUSED.patch; record in docs/TOMBSTONES.md.
    score_samples(
        entry.ctx, dquery, entry.train, entry.weights, entry.has_weights, dout,
        n_query, entry.n_train, n_features, bandwidth, entry.sum_w, k, m,
        metric_arg, trace, elem_tpb, lse_tpb,
    )
    var total = Float32(0)
    if want_total:
        total = device_sum_f32_fixed(entry.ctx, dout, n_query)
    var hout = entry.ctx.enqueue_create_host_buffer[DType.float32](n_query)
    entry.ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=dout)
    entry.ctx.synchronize()
    copy_f32(hout.unsafe_ptr(), scores, n_query)
    _ = hout^
    _ = dout^
    return total


def kde_score_samples_resident_device(
    handle: Int,
    mut dquery: DeviceBuffer[DType.float32],
    n_query: Int,
    n_features: Int,
    bandwidth: Float32,
    kernel: String,
    metric: String,
    scores: MutPointer[Float32, MutUntrackedOrigin],
    want_total: Bool = False,
    status_addr: Int = 0,
) raises -> Float32:
    """`kde_score_samples_resident` over queries already on this binding's context (a fold taken on the device
    out of a resident X; lane vdf-scope-cluster2b): no staging copy and no upload. The handle checks are the
    host form's; the queries' finite status (`status_addr`) is tested ON THE DEVICE (`enqueue_finite_word`, one
    word read back with the scores' synchronisation) and a flagged span writes no score. The cosine zero-row
    rule and the sqeuclidean bound on the queries run on the device first (`kde_device_row_rules`, lane
    vdf-scope-cluster4), as the host form runs its host pass before its finite test."""
    var k = kernel_from_name(kernel)
    var m = metric_from_name(metric)
    var entry = _kde_fit_lease(handle)
    var clean = True
    try:
        if entry.kernel != k or entry.metric != m or entry.n_features != n_features or entry.bandwidth != bandwidth:
            raise Error("kde_score_samples_resident_device: the handle was prepared with another kernel, metric,"
                + " feature count or bandwidth")
        if n_query <= 0:
            raise Error("kde: X must have at least one row (n_query)")
        var n = n_query * n_features
        if len(dquery) < n:
            raise Error("kde_score_samples_resident_device: the resident slot holds " + String(len(dquery))
                + " words, " + String(n) + " read")
        kde_device_row_rules(entry.ctx, dquery, n_query, n_features, m, "query")
        clean = False
        if status_addr != 0:
            var d_status = entry.ctx.enqueue_create_buffer[DType.int32](1)
            enqueue_finite_word(entry.ctx, dquery.unsafe_ptr(), n, d_status, 0)
            var h_status = entry.ctx.enqueue_create_host_buffer[DType.int32](1)
            entry.ctx.enqueue_copy(dst_ptr=h_status.unsafe_ptr(), src_buf=d_status)
            entry.ctx.synchronize()
            var word = h_status.unsafe_ptr()[0]
            _ = h_status^
            _ = d_status^
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=status_addr).unsafe_store(0, word)
            if word != 0:
                clean = True
                return Float32(0)
        var total = _kde_resident_tail(entry, dquery, n_query, n_features, bandwidth, k, m, kernel, metric,
                                       scores, KDE_ELEM_TPB, KDE_LSE_TPB, Float32(2.0), want_total)
        clean = True
        return total
    finally:
        if clean:
            _kde_fit_return(handle, entry, True)
        else:
            try:
                _kde_fit_return(handle, entry, False)
            except:
                pass
        _ = entry
