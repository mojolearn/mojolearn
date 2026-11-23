# SPDX-License-Identifier: Apache-2.0
"""The fit operations of the device column of the ANN operations table
(core/neighbors_entry_table_ann.mojo, operations 10 to 15; lane dw-neighbors,
2026-11-21): the typed bodies of the GPU binding's build exports
(bindings/_mojolearn_x_ann.mojo `x_ann_ivf_pq_build`,
`x_ann_ivf_pq_build_resident`, `x_ann_ivf_sq_build`, `x_ann_ivf_rabitq_build`,
`x_ann_cagra_build`), moved here whole: the same checks, the same build
drivers, the same words, so the bits are unchanged. The exports now UNPACK
their lists and call these bodies; the table entry calls them too. Nothing
here touches Python.

Interpreter lock: every `*_build_device_run` is called with the GIL RELEASED
(the exports released it around the build; the row copy of a build that does
not read the caller's rows in place and the `IvfPqDevice` placeholder ran
with it held and no longer do). `ann_ivf_pq_register_built_device_run` is
called with the GIL HELD: it is the only body here that reaches the resident
registry (x_ann/resident.mojo), which the GIL guards.

Each build is one call of a device driver that is parallel inside; a fit is
not split further here.
"""
from core.hostptr import f32_ptr, i32_ptr, read_f32
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import IVF_BUILD_FROM_POINTER
from x_ann.cagra_device import cagra_build_device_ptr
from x_ann.device_ctx import x_ann_ctx
from x_ann.ivf_pq_core import IvfPqIndex
from x_ann.ivf_pq_device import IvfPqDevice, ivf_sq_build_device_ptr, ivf_rabitq_build_device_ptr
from x_ann.mojolearn_x_ann_run import ivf_pq_build_run, ivf_pq_build_resident_run
from x_ann.pending_pq import ann_pending_park, ann_pending_take_built, ann_pending_take_device
from x_ann.resident import x_ann_register_pq_built
from x_ann.stage_timer import AnnStages
from x_ann.switches import IVF_PQ_ONE_UPLOAD, IVF_PQ_RESIDENT_FIT


def _rows(address: Int, count: Int) raises -> List[Float32]:
    """The list copy of the caller's rows a build takes where it does not
    read them in place (lane fg-ivf A3: with IVF_PQ_ONE_UPLOAD and
    IVF_BUILD_FROM_POINTER the coarse build reads the rows at their address
    and this list stays empty)."""
    var x = List[Float32]()
    comptime if not (IVF_PQ_ONE_UPLOAD and IVF_BUILD_FROM_POINTER):
        x = read_f32(address, count)
    return x^


def ann_ivf_pq_build_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_pq_build`: a x, centers, offsets, list_indices, codebooks,
    codes; p n, dim, n_lists, kmeans_n_iters, seed, pq_dim, pq_bits,
    pq_kmeans_n_iters. Returns n. The codes go from the device straight
    into the caller's array (lane fx-ann-fit-nocopy)."""
    if len(a) != 6 or len(p) != 8:
        raise Error("x_ann_ivf_pq_build: need 6 addresses and 8 parameters")
    var n = p[0]
    var dim = p[1]
    var bst = AnnStages("ivf_pq_binding")
    var x = _rows(a[0], n * dim)
    bst.host("copy_in")
    if a[5] == 0:
        raise Error("mojolearn: null int32 buffer address")
    ivf_pq_build_run(x, n, dim, p[2], p[3], p[4], p[5], p[6], p[7], a[5], a[0], bst, a[1], a[2], a[3], a[4])
    return n


def _pq_build_resident(a: List[Int], p: List[Int]) raises -> Int:
    if len(a) != 5 or len(p) != 8:
        raise Error("x_ann_ivf_pq_build_resident: need 5 addresses and 8 parameters")
    var n = p[0]
    var dim = p[1]
    var bst = AnnStages("ivf_pq_binding_resident")
    var x = _rows(a[0], n * dim)
    bst.host("copy_in")
    var ctx = x_ann_ctx()
    var dev = IvfPqDevice(ctx)
    # a placeholder the build below replaces
    var index = IvfPqIndex(
        0, 0, 0, 0, 0, 0, List[Float32](), List[Int32](), List[Int32](), List[Float32](), List[Int32]()
    )
    ivf_pq_build_resident_run(x, a[0], n, dim, p[2], p[3], p[4], p[5], p[6], p[7], dev, index, bst,
                              a[1], a[2], a[3], a[4])
    _ = x^
    var ticket = ann_pending_park(index^, dev^)
    _ = ctx^
    return ticket


def ann_ivf_pq_build_resident_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_pq_build_resident`, first phase (lane fg-ivf A7): the
    build with the fitted index LEFT ON THE DEVICE; the codes are not
    downloaded. a x, centers, offsets, list_indices, codebooks; p the eight
    words of `ann_ivf_pq_build_device_run`. Returns the ticket of the parked
    index (x_ann/pending_pq.mojo) for `ann_ivf_pq_register_built_device_run`,
    or 0, before any work, when this image keeps no resident fit
    (`IVF_PQ_RESIDENT_FIT` off: fit takes the plain build and the first
    search prepares the handle). The door asks this instead of probing the
    export by name."""
    var ticket = 0
    comptime if IVF_PQ_RESIDENT_FIT:
        ticket = _pq_build_resident(a, p)
    return ticket


def ann_ivf_pq_register_built_device_run(ticket: Int) raises -> Int:
    """`x_ann_ivf_pq_build_resident`, second phase, GIL HELD: the parked
    index registered in the resident registry; its handle (the handle space
    and release of `x_ann_index_prepare`)."""
    var bst = AnnStages("ivf_pq_binding_register")
    var built = ann_pending_take_built(ticket)
    var dev = ann_pending_take_device(ticket)
    var handle = x_ann_register_pq_built(built^, dev^)
    bst.host("register")
    return handle


def ann_ivf_sq_build_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_sq_build`: a x, centers, offsets, list_indices, vmin,
    delta, codes; p n, dim, n_lists, kmeans_n_iters, seed[,
    train_rows_per_list] (0 or absent keeps the default training rows).
    Returns n. The six results are written where the caller wants them
    (lane fx-ann-fit-nocopy)."""
    if len(a) != 7 or len(p) < 5:
        raise Error("x_ann_ivf_sq_build: need 7 addresses and 5 or 6 parameters")
    var n = p[0]
    var dim = p[1]
    var train_rows_per_list = p[5] if len(p) > 5 else 0
    var bst = AnnStages("ivf_sq_binding")
    if a[0] == 0:
        raise Error("mojolearn: null float32 buffer address")
    var x = _rows(a[0], n * dim)
    bst.host("copy_in")
    var centers_p = f32_ptr(a[1])
    var offsets_p = i32_ptr(a[2])
    var indices_p = i32_ptr(a[3])
    var vmin_p = f32_ptr(a[4])
    var delta_p = f32_ptr(a[5])
    var codes_p = i32_ptr(a[6])
    ivf_sq_build_device_ptr(x, a[0], n, dim, p[2], p[3], p[4], centers_p, offsets_p, indices_p,
                            vmin_p, delta_p, codes_p, train_rows_per_list)
    bst.host("build")
    return n


def ann_ivf_rabitq_build_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_ivf_rabitq_build`: a x, centers, offsets, list_indices, codes
    (n x words int32 bit words), norms, ips; p n, dim, n_lists,
    kmeans_n_iters, seed. Returns n."""
    if len(a) != 7 or len(p) != 5:
        raise Error("x_ann_ivf_rabitq_build: need 7 addresses and 5 parameters")
    var n = p[0]
    var dim = p[1]
    var bst = AnnStages("ivf_rabitq_binding")
    if a[0] == 0:
        raise Error("mojolearn: null float32 buffer address")
    var x = _rows(a[0], n * dim)
    bst.host("copy_in")
    var centers_p = f32_ptr(a[1])
    var offsets_p = i32_ptr(a[2])
    var indices_p = i32_ptr(a[3])
    var codes_p = i32_ptr(a[4])
    var norms_p = f32_ptr(a[5])
    var ips_p = f32_ptr(a[6])
    ivf_rabitq_build_device_ptr(x, a[0], n, dim, p[2], p[3], p[4], centers_p, offsets_p, indices_p,
                                codes_p, norms_p, ips_p)
    bst.host("build")
    return n


def ann_cagra_build_device_run(a: List[Int], p: List[Int]) raises -> Int:
    """`x_ann_cagra_build`: a x, graph_out; p n, d,
    intermediate_graph_degree, graph_degree. Returns n. The rows are
    uploaded from where they are and the graph is downloaded where the
    caller wants it (lane fx-ann-fit-nocopy)."""
    if len(a) != 2 or len(p) != 4:
        raise Error("x_ann_cagra_build: need 2 addresses and 4 parameters")
    var n = p[0]
    var d = p[1]
    var kdeg = p[2]
    var deg = p[3]
    if n < 2 or d <= 0 or kdeg < 1 or kdeg > n - 1 or deg < 1 or deg > kdeg:
        raise Error("CAGRA: need 1 <= graph_degree <= intermediate_graph_degree <= n - 1")
    var xp = f32_ptr(a[0])
    var gp = i32_ptr(a[1])
    cagra_build_device_ptr(xp, n, d, kdeg, deg, gp)
    return n
