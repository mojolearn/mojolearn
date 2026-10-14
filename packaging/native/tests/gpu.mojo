# SPDX-License-Identifier: Apache-2.0
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys.compile import is_defined
from mojolearn import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from mojolearn.cluster.estimator import kmeans_fit, kmeans_predict
from mojolearn.cluster.impl.kmeans_params import INIT_ARRAY


def main() raises:
    comptime expected = NUMERIC_FAST if is_defined["MOJOLEARN_NUMERIC_FAST"]() else NUMERIC_IDENTICAL
    comptime assert GLOBAL_NUMERIC_MODE == expected, "incorrect package default"
    var ctx = DeviceContext()
    var x: List[Float32] = [-1.0, 0.0, 1.0, 0.0, 9.0, 10.0, 11.0, 10.0]
    var centers: List[Float32] = [0.0, 0.0, 10.0, 10.0]
    var labels = List[UInt32](length=4, fill=99)
    var weights: List[Float32] = [1.0]
    # The native pointer interface borrows these arrays for the call. Keep
    # their owners alive through the final synchronization and result reads.
    var fit = kmeans_fit(ctx, x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 4, 2, 2,
                         centers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                         labels.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                         weights.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 0,
                         max_iter=10, init=INIT_ARRAY)
    var predictions = List[UInt32](length=4, fill=99)
    kmeans_predict(ctx, x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 4, 2, 2,
                   centers.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                   predictions.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    ctx.synchronize()
    var want_centers: List[Float32] = [0.0, 0.0, 10.0, 10.0]
    for i in range(4):
        if labels[i] != UInt32(i // 2) or predictions[i] != labels[i]:
            raise Error("incorrect fit or prediction labels")
        if centers[i] != want_centers[i]:
            raise Error("incorrect fitted centers")
        print("kmeans", i, labels[i], bitcast[DType.uint32](centers[i]))
    if fit.n_iter < 1 or fit.inertia != 4.0:
        raise Error("incorrect fit statistics")
    _ = x^
    _ = weights^
    print("PASS native GPU: k-means fit and predict")
