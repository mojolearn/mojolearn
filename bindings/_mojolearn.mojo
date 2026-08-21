"""CPython extension module for mojolearn.

Built by `bindings/build.sh` into `python/mojolearn/_mojolearn.so`. The public
Python surface is the scikit-learn-style wrapper in `python/mojolearn/`.

**Data crosses as raw buffer addresses plus lengths**, which is the convention
`neighbors/estimator.mojo` and `cluster/estimator.mojo` were written to. The
Python wrapper passes float32 C-contiguous arrays and keeps them alive for the
duration of the call; nothing here retains a Python buffer after it returns.
That contract is the wrapper's to honor and it is stated in
`python/mojolearn/_arrays.py`.

WHAT IS EXPOSED, AND WHAT IS NOT
---------------------------------
`knn_search` and `kmeans_fit`. Those are the algorithms with a caller-facing
surface in THIS extension: they take host pointers, own their device work, and
have checks covering the policy they add.

**GBDT MOVED OUT.** It lives in `bindings/_mojolearn_gbdt.mojo`, built by
`bindings/build_gbdt.sh` into a second extension, for the reason
`bindings/_mojolearn_estimators.mojo` gives for the third: an independently
changing binding should not be a merge point. GBDT is the fastest-moving
surface in this repository and every parameter added to `GbdtFitParams` used
to have to be unpacked in TWO files that could silently disagree about the
order of a flat list -- which is a wrong answer, not a failure. Now there is
one. Do not re-add a gbdt import here.

DBSCAN, PCA and OLS have verified kernels and no such surface yet, so binding
them would mean inventing one here, at the boundary, where no check can see
it. They are named in the wrapper's `__all__` as absent rather than silently
missing.

THE DEVICE CONTEXT IS CREATED PER CALL, AND THAT IS A REAL COST
----------------------------------------------------------------
Each entry point below constructs its own `DeviceContext`. That is not free
and it is not hidden: a caller fitting in a loop pays it every iteration. It
is done this way because a module-global context would have to outlive the
GIL-released regions below and be safe against a caller using mojolearn from
two threads, and neither of those has been checked. **When someone measures
the per-call cost and wants it gone, the fix is a cached context with an
explicit thread contract, not a global slipped in quietly.**

SCALARS ARRIVE AS ONE LIST, WHICH IS NOT A STYLE CHOICE
--------------------------------------------------------
`PythonModuleBuilder.def_function` infers its signature from the function's
arity and stops being able to above roughly nine arguments; mojotrees' widest
binding takes nine and that is not a coincidence. `knn_search` needs ten and
`kmeans_fit` fourteen. So each entry point takes its BUFFER ADDRESSES
positionally, where a mistake is a crash rather than a wrong answer, and its
scalars in one list whose order is written out beside the unpacking below and
mirrored in the wrapper. Both sides name the order in the same words on
purpose: a silent reordering here would be a wrong answer, not a failure.

THE GIL IS RELEASED AROUND THE DEVICE WORK
-------------------------------------------
Both calls hand a buffer address to the GPU and wait. Holding the GIL across
that would block every other Python thread for the whole fit for no reason:
nothing inside touches a Python object, and the caller's arrays are kept alive
by the wrapper on the Python side. The pattern matches mojotrees'
`buffer_has_infinite`.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from cluster.estimator import kmeans_fit
from neighbors.estimator import knn_search


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    """A caller's float32 buffer, borrowed, never owned.

    The origin is untracked because the owner is a NumPy array on the other
    side of the boundary and Mojo cannot see it. The wrapper holds that array
    for the length of the call, which is the whole contract.
    """
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)


def knn_search_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_dist_addr: PythonObject,
    out_idx_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Exact k-NN. Returns the query tile that actually ran.

    `params` is, in this exact order:

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  return_sqrt   (0 or 1)
        5  query_tile

    The return value is not decoration. `plan_query_tile` may lower the tile
    below what was asked for when the workspace cap fires, and a caller
    recording a benchmark number needs to know which configuration produced
    it. The wrapper surfaces it as `used_query_tile`.
    """
    if len(params) != 6:
        raise Error(
            "knn_search: params must hold 6 values, got "
            + String(len(params))
        )
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var dp = _f32_ptr(Int(py=out_dist_addr))
    var xp = _u32_ptr(Int(py=out_idx_addr))
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var nf = Int(py=params[2])
    var kk = Int(py=params[3])
    var sq = Int(py=params[4]) != 0
    var qt = Int(py=params[5])

    var used: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        used = knn_search(ctx, ip, ni, qp, nq, nf, kk, dp, xp, sq, qt)
    return PythonObject(used)


def kmeans_fit_binding(
    x_addr: PythonObject,
    out_centroids_addr: PythonObject,
    out_labels_addr: PythonObject,
    weights_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit k-means. Returns [inertia, n_iter, sum_scale, weight_scale].

    `params` is, in this exact order:

        0  n_samples
        1  n_features
        2  n_clusters
        3  n_weights   (0 means unit weights; weights_addr is then unread)
        4  max_iter
        5  tol         (float)
        6  seed
        7  n_init
        8  init
        9  metric

    All four returns are given because a wrong answer here comes from the two
    scales, and a caller reproducing a result needs them. **`inertia` is 0.0
    when it was NEVER COMPUTED**: `inertia_check` is False by default, per
    cuVS, and with it off the Lloyd loop never forms the cluster cost. Do not
    read a 0.0 as a perfect clustering. The wrapper repeats this warning.
    """
    if len(params) != 10:
        raise Error(
            "kmeans_fit: params must hold 10 values, got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=out_centroids_addr))
    var lp = _u32_ptr(Int(py=out_labels_addr))
    # When n_weights is 0 the estimator never reads this pointer, so the
    # wrapper passes the X address rather than allocating a throwaway array
    # of ones. `_f32_ptr` still refuses a null.
    var wp = _f32_ptr(Int(py=weights_addr))

    var ns = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var nw = Int(py=params[3])
    var mi = Int(py=params[4])
    var tl = Float64(py=params[5])
    var sd = UInt64(Int(py=params[6]))
    var ninit = Int(py=params[7])
    var ii = Int(py=params[8])
    var mm = Int(py=params[9])

    var inertia = Float64(0.0)
    var n_iter = 0
    var sum_scale = Float64(0.0)
    var weight_scale = Float64(0.0)
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var r = kmeans_fit(
            ctx, xp, ns, nf, nc, cp, lp, wp, nw, mi, tl, sd, ninit, ii, mm
        )
        inertia = r.inertia
        n_iter = r.n_iter
        sum_scale = r.sum_scale
        weight_scale = r.weight_scale

    var out = Python.list()
    out.append(PythonObject(inertia))
    out.append(PythonObject(n_iter))
    out.append(PythonObject(sum_scale))
    out.append(PythonObject(weight_scale))
    return out


@export
def PyInit__mojolearn() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn")
        m.def_function[knn_search_binding]("knn_search")
        m.def_function[kmeans_fit_binding]("kmeans_fit")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn module: ", e))
