# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the IVF-FLAT lane (`mojolearn.IVFIndex`), exposed 2026-09-14.

`ivf/checks/ivf_check.mojo` (the layout sabotage, the large-k and
large-probe checks) reads ALL OK at IDENTICAL on the Apple M4, an NVIDIA
H100 and an AMD MI300X with one card, 1e7c1702, on all three
(bench/results/ivf_embed_km_legs_2026-09-14/README.md). The binding is in
`python/mojolearn/_backend.py`'s `_MODULES` and `_build_script` and in the
four packaging lists `packaging/check_ext_lists.py` holds to it;
`python/mojolearn/_ivf_impl.py` is the Python half and the identity_break
lane is `ivf`.

ONE ENTRY, ONE CARD. `ivf_flat_build_and_search_host` is the entry every
gate uses (the estimator's policy 3): a build and a search under one
identity card. The index itself does not cross, so there is no model
column; `n_probes` has no default at this boundary (policy 1) and
`n_probes > n_lists` RAISES rather than clamps (policy 2), both on the
Mojo host.

THE ABI IS THE GP'S: two length-checked lists, orders written out below
and mirrored in `_ivf_impl.py`.
"""

from std.os import abort
from std.sys.compile import is_defined
from bindings.hostptr import f32_ptr, i32_ptr, copy_f32, read_f32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from ivf.estimator import ivf_flat_build_and_search_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def ivf_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL, 2
    DETERMINISTIC."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def ivf_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


def _ivf_run(
    x: List[Float32],
    n: Int,
    dim: Int,
    n_lists: Int,
    queries: List[Float32],
    m: Int,
    k: Int,
    n_probes: Int,
    kmeans_n_iters: Int,
    metric: Int,
    seed: UInt64,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var ctx = DeviceContext()
    var r = ivf_flat_build_and_search_host(
        ctx,
        x,
        n,
        dim,
        n_lists,
        queries,
        m,
        k,
        n_probes,
        kmeans_n_iters,
        metric,
        seed,
    )
    ctx.synchronize()
    copy_f32(r.distances.unsafe_ptr(), dp, m * k)
    # ORIGINAL row ids, UInt32 on the Mojo side and below 2**31 by
    # construction (n <= 46340 on every lane in this tree), so the int32
    # bytes are the same bytes.
    for i in range(m * k):
        ip.unsafe_store(i, Int32(Int(r.indices[i])))
    comptime if is_defined["MOJOLEARN_IVF_BINDING_SABOTAGE"]():
        # NEVER SHIPPED. Swaps the first two neighbor ids of query 0, the
        # tie-class corruption a wrong (distance, id) order would produce, so
        # the identity_break `ivf` lane is SEEN TO FAIL on its idx part.
        if m * k >= 2:
            var a = ip.unsafe_load(0)
            ip.unsafe_store(0, ip.unsafe_load(1))
            ip.unsafe_store(1, a)
    for i in range(m):
        cp.unsafe_store(i, r.n_candidates[i])
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^


def ivf_flat_build_and_search_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`ivf_flat::build` then `ivf_flat::search` under ONE identity card
    (`ivf_flat_build_and_search_host`, policy 3). Returns 0.

    `addrs`, in this exact order:

        0  x               n * dim float32, row-major, read
        1  queries         m * dim float32, row-major, read
        2  dist_out        m * k float32, WRITTEN
        3  idx_out         m * k int32, WRITTEN (original row ids)
        4  cand_out        m int32, WRITTEN (candidates examined per query)

    `params`, in this exact order:

        0  n
        1  dim
        2  m               n_queries
        3  k
        4  n_lists
        5  n_probes        REQUIRED, no default (policy 1); > n_lists RAISES
        6  kmeans_n_iters  cuVS's 20
        7  metric          0 L2Expanded (squared), 1 L2SqrtExpanded
        8  seed
    """
    if len(addrs) != 5:
        raise Error(
            "ivf_flat_build_and_search: addrs must contain 5 addresses (x,"
            " queries, dist_out, idx_out, cand_out), got "
            + String(len(addrs))
        )
    if len(params) != 9:
        raise Error(
            "ivf_flat_build_and_search: params must contain 9 values (n, dim,"
            " m, k, n_lists, n_probes, kmeans_n_iters, metric, seed), got "
            + String(len(params))
        )
    var n = Int(py=params[0])
    var dim = Int(py=params[1])
    var m = Int(py=params[2])
    var k = Int(py=params[3])
    var n_lists = Int(py=params[4])
    var n_probes = Int(py=params[5])
    var kmeans_n_iters = Int(py=params[6])
    var metric = Int(py=params[7])
    var seed = UInt64(Int(py=params[8]))
    var x = read_f32(Int(py=addrs[0]), max(0, n * dim))
    var queries = read_f32(Int(py=addrs[1]), max(0, m * dim))
    var dp = _f32_ptr(Int(py=addrs[2]))
    var ip = _i32_ptr(Int(py=addrs[3]))
    var cp = _i32_ptr(Int(py=addrs[4]))
    with GILReleased(Python()):
        _ivf_run(
            x,
            n,
            dim,
            n_lists,
            queries,
            m,
            k,
            n_probes,
            kmeans_n_iters,
            metric,
            seed,
            dp,
            ip,
            cp,
        )
    return PythonObject(0)


@export
def PyInit__mojolearn_ivf() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_ivf")
        m.def_function[ivf_vendor_binding]("ivf_vendor")
        m.def_function[ivf_numeric_mode_binding]("ivf_numeric_mode")
        m.def_function[ivf_flat_build_and_search_binding](
            "ivf_flat_build_and_search"
        )
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_ivf: ", e))
