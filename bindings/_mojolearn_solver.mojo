# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the coordinate-descent and single-linkage kernels.

Kept in a SEPARATE extension, deliberately. `bindings/build.sh` explains the
rule at length and it is the reason this file exists rather than three more
functions in `_mojolearn_estimators.mojo`: an independently changing binding
must not become a merge point. Two lanes land here because they arrived
together and neither has a Python surface of its own yet; both are built by
`bindings/build_solver.sh` and both land in the one wheel.

Arrays cross as BORROWED NumPy addresses. Every device buffer and every
`DeviceContext` lives for exactly one call and no pointer is retained past
the return; the Python side (`python/mojolearn/_solver_impl.py`,
`python/mojolearn/_hierarchy_impl.py`) keeps each array alive across the
call, which is the whole point of `_arrays.py` returning the array beside
its address.

SCALARS ARRIVE AS ONE LIST, NOT AS SEPARATE ARGUMENTS.
`PythonModuleBuilder.def_function` infers its signature from arity and stops
working somewhere around nine arguments, so buffer addresses go positionally
and every scalar goes in one `params` list. THE ORDER OF THAT LIST IS
WRITTEN OUT IN A COMMENT ON BOTH SIDES IN THE SAME WORDS. A silent reorder
here is a WRONG ANSWER, not a crash -- swap `alpha` and `l1_ratio` and both
fits still run and both return plausible coefficients -- so the two comments
are the only thing standing between a caller and a quiet lie. If you change
one, change the other in the same edit.

THE GIL IS RELEASED AROUND THE DEVICE WORK, and nothing inside a
`GILReleased` block touches a `PythonObject`.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from mojo_only.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from hierarchy.estimator import linkage_fit_host
from solver.estimator import cd_fit_host, cd_predict_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def cd_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    coef_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`cdFit` for Lasso / ElasticNet (solver/, DEVIATIONS 610-613, 880).

    Writes `n_cols` float32 coefficients to `coef_addr` and the intercept to
    `info_addr[0]` (float32). Returns `n_iter`, the number of epochs run.

    `x_addr` is COLUMN-MAJOR `n_rows x n_cols` float32 -- cuML's `F` order,
    which `cdFit` requires. The Python layer produces it with
    `np.asfortranarray` and names the copy.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_solver_impl.py`):

        0  n_rows
        1  n_cols
        2  fit_intercept      (0/1)
        3  max_iter           (cdFit's `epochs`)
        4  alpha              (float)
        5  l1_ratio           (float; Lasso passes 1.0)
        6  tol                (float)
        7  shuffle            (0/1; selection='random' is 1 and is REFUSED
                               BY NAME by solver/ported/solver/cd.mojo,
                               DEVIATION 611 reserved and not spent)
        8  has_sample_weight  (0/1; 1 is REFUSED BY NAME by the same file)

    Slots 7 and 8 are plumbed through rather than hardcoded to 0 so that the
    ported refusals stay REACHABLE from this surface. A refusal only the
    Python layer can raise is a refusal the Mojo entry never proves it has.
    """
    if len(params) != 9:
        raise Error(
            "cd_fit: params must contain 9 values, got " + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var cp = _f32_ptr(Int(py=coef_addr))
    var ip = _f32_ptr(Int(py=info_addr))
    var nr = Int(py=params[0])
    var nc = Int(py=params[1])
    var fit_intercept = Int(py=params[2]) != 0
    var epochs = Int(py=params[3])
    var alpha = Float32(Float64(py=params[4]))
    var l1_ratio = Float32(Float64(py=params[5]))
    var tol = Float32(Float64(py=params[6]))
    var shuffle = Int(py=params[7]) != 0
    var has_sw = Int(py=params[8]) != 0
    var n_iter = 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        n_iter = cd_fit_host(
            ctx, xp, yp, cp, ip, nr, nc, fit_intercept, epochs, alpha,
            l1_ratio, tol, shuffle, has_sw,
        )
    return PythonObject(n_iter)


def cd_predict_binding(
    x_addr: PythonObject,
    coef_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`cdPredict` -> `linearRegH`: `pred = X coef + intercept` on the device.

    `x_addr` is COLUMN-MAJOR `n_rows x n_cols`, as `cd_fit` takes it. Writes
    `n_rows` float32 to `out_addr`.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_solver_impl.py`):

        0  n_rows
        1  n_cols
        2  intercept  (float)
    """
    if len(params) != 3:
        raise Error(
            "cd_predict: params must contain n_rows, n_cols, intercept"
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=coef_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=params[0])
    var nc = Int(py=params[1])
    var intercept = Float32(Float64(py=params[2]))
    with GILReleased(Python()):
        var ctx = DeviceContext()
        cd_predict_host(ctx, xp, cp, op, nr, nc, intercept)
    return PythonObject(0)


def linkage_fit_binding(
    x_addr: PythonObject,
    children_addr: PythonObject,
    labels_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Single-linkage agglomerative clustering (hierarchy/, DEVIATIONS
    620-624, 881).

    Writes `(n_rows - 1) * 2` int32 to `children_addr`, `n_rows` int32 to
    `labels_addr`, and two int32 to `info_addr`: `[0]` the Boruvka round
    count, `[1]` `n_connected_components`. Returns the Boruvka round count.

    `x_addr` is ROW-MAJOR `n_rows x n_cols` float32, which is what cuML's
    own estimator passes (`agglomerative.pyx:144-152`, `order="C"`).

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_hierarchy_impl.py`):

        0  n_rows
        1  n_cols
        2  n_clusters
        3  metric    (cuML DistanceType: 1 = L2SqrtExpanded, cuML's
                      'euclidean'/'l2'; 0 = L2Expanded. Every other code is
                      REFUSED BY NAME by hierarchy/ported/cluster/detail/
                      connectivities.mojo)
        4  use_knn   (0/1; 1 is connectivity='knn', rung 2, REFUSED BY NAME
                      by get_distance_graph in the same file)

    Slot 4 is plumbed through rather than hardcoded to 0 for the same reason
    `cd_fit`'s slots 7 and 8 are: the ported refusal has to stay reachable.
    `c` (the k-NN graph's `k = log(n) + c`) is NOT a slot, because it is
    forwarded only when `use_knn` is true (`linkage.cu:40`, `use_knn ? c :
    0`) and that arm raises; the port's default 15 is used.
    """
    if len(params) != 5:
        raise Error(
            "linkage_fit: params must contain 5 values, got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var chp = _i32_ptr(Int(py=children_addr))
    var lp = _i32_ptr(Int(py=labels_addr))
    var ip = _i32_ptr(Int(py=info_addr))
    var nr = Int(py=params[0])
    var nc = Int(py=params[1])
    var k = Int(py=params[2])
    var metric = Int(py=params[3])
    var use_knn = Int(py=params[4]) != 0
    var rounds = 0
    with GILReleased(Python()):
        var ctx = DeviceContext()
        rounds = linkage_fit_host(
            ctx, xp, chp, lp, ip, nr, nc, k, metric, use_knn,
        )
    return PythonObject(rounds)


def solver_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `mojo_only/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_solver() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_solver")
        m.def_function[solver_vendor_binding]("solver_vendor")
        m.def_function[cd_fit_binding]("cd_fit")
        m.def_function[cd_predict_binding]("cd_predict")
        m.def_function[linkage_fit_binding]("linkage_fit")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_solver: ", e))
