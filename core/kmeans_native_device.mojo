# SPDX-License-Identifier: Apache-2.0
"""Native KMeans plumbing over the existing tuned device operations."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorError
from core.kmeans_native_api import (
    KMeansNativeKernels, KMeansFitRequest, KMeansFitResult, KMeansQueryRequest,
    KMeansQueryResult, kmeans_native_dispatch,
    KMEANS_CAP_JOINT, KMEANS_CAP_RESIDENT, KMEANS_CAP_FOLD,
)
from core.neural_context import process_ctx
from core.msel_device_x import msel_fold_entry_context
from cluster.estimator import kmeans_fit, kmeans_predict, kmeans_transform
from cluster.kmeans_fold import kmeans_fit_fold, kmeans_query_fold
from cluster.resident_centers import (
    kmeans_centers_prepare, kmeans_centers_release,
    kmeans_predict_resident, kmeans_transform_resident,
)

comptime KMEANS_CONTEXT = "MojolearnCoreContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnCoreContextFastV3"


def _f32(address: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def _u32(address: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)


struct KMeansDeviceKernels(KMeansNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def capabilities() -> Int:
        return KMEANS_CAP_JOINT | KMEANS_CAP_RESIDENT | KMEANS_CAP_FOLD

    @staticmethod
    def fit(r: KMeansFitRequest) raises -> KMeansFitResult:
        if r.fold.active():
            if r.weight_count != 0:
                raise Error("kmeans_fit_fold: a fold fit takes unit weights (n_weights 0)")
            var centers = _f32(r.centers)
            var labels = _u32(r.labels)
            var ctx = msel_fold_entry_context()
            var answer = kmeans_fit_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
                r.rows, r.features, r.clusters, centers, labels, r.max_iter, r.tolerance,
                UInt64(r.seed), r.n_init, r.init, r.metric, r.oversampling)
            return KMeansFitResult(answer.fit.inertia, answer.fit.n_iter,
                answer.fit.sum_scale, answer.fit.weight_scale, Int(answer.finite), Int(answer.in_range))
        # Preserve the public entry's pointer admission order and its scratch.
        var source = _f32(r.source)
        var centers = _f32(r.centers)
        var labels = _u32(r.labels)
        var weights = _f32(r.weights)
        var distances = Optional[MutPointer[Float32, MutUntrackedOrigin]]()
        if r.distances != 0:
            distances = Optional[MutPointer[Float32, MutUntrackedOrigin]](_f32(r.distances))
        var word = List[Int32](length=1, fill=0)
        var finite_word = Int(word.unsafe_ptr()) if r.test_finite else 0
        var ctx = process_ctx[KMEANS_CONTEXT]()
        # Same measured kmeans_fit operation, including the fused final
        # transform when distances is supplied: X is still uploaded once.
        var answer = kmeans_fit(ctx, source, r.rows, r.features, r.clusters,
            centers, labels, weights, r.weight_count, r.max_iter, r.tolerance,
            UInt64(r.seed), r.n_init, r.init, r.metric, 0.0, r.oversampling,
            lazy_shift=False, transform_out=distances, finite_word=finite_word)
        ctx.synchronize()
        return KMeansFitResult(answer.inertia, answer.n_iter, answer.sum_scale,
            answer.weight_scale, Int(word[0]), 0)

    @staticmethod
    def query(r: KMeansQueryRequest) raises -> KMeansQueryResult:
        if r.fold.active():
            var centers = _f32(r.centers)
            _ = _u32(r.dst)
            var ctx = msel_fold_entry_context()
            var words = kmeans_query_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
                r.rows, r.features, r.clusters, centers, r.dst, r.metric, r.transform)
            return KMeansQueryResult(Int(words[0]), Int(words[1]))
        var source = _f32(r.source)
        var centers: Optional[MutPointer[Float32, MutUntrackedOrigin]] = None
        if r.handle < 0:
            centers = _f32(r.centers)
        var destination = _u32(r.dst)
        var word = List[Int32](length=1, fill=0)
        var finite_word = Int(word.unsafe_ptr()) if r.test_finite else 0
        if r.handle >= 0:
            if r.transform:
                kmeans_transform_resident(r.handle, source, r.rows, r.features,
                    r.clusters, _f32(r.dst), r.metric, finite_word=finite_word)
            else:
                kmeans_predict_resident(r.handle, source, r.rows, r.features,
                    r.clusters, destination, r.metric, finite_word=finite_word)
        else:
            var center_words = centers.value()
            var ctx = process_ctx[KMEANS_CONTEXT]()
            if r.transform:
                kmeans_transform(ctx, source, r.rows, r.features, r.clusters,
                    center_words, _f32(r.dst), r.metric, finite_word=finite_word)
            else:
                kmeans_predict(ctx, source, r.rows, r.features, r.clusters,
                    center_words, destination, r.metric, finite_word=finite_word)
            ctx.synchronize()
        return KMeansQueryResult(Int(word[0]), 0)

    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int:
        return kmeans_centers_prepare(_f32(centers), clusters, features)

    @staticmethod
    def release(handle: Int) raises:
        kmeans_centers_release(handle)


def kmeans_device_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return kmeans_native_dispatch[KMeansDeviceKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)
