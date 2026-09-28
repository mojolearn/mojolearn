#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GENERATES the neighbors lane's two drivers and two bindings from one op
table, so the GPU column and the CPU column run the SAME items over the SAME
address contract and cannot drift:

    x_neighbors/device_ops.mojo            upload, one GPU thread per item, download
    x_neighbors/host_ops.mojo              a host loop over the same items
    bindings/_mojolearn_x_neighbors.mojo   exports xn_<op>(addrs, ints, floats)
    bindings/_mojolearn_x_neighbors_host.mojo   the same exports + x_neighbors_host_*

Run `python3 x_neighbors/gen.py` after editing OPS; commit the outputs.

An op is (name, module, item, count, params). Buffer params (fin, fout,
finout, iin, iout, iinout, fscr) come first in the item's signature after
`t`, then the scalars (int, float), each in table order. A buffer's size is a
Python-free expression over the scalars. `count` is the number of items (one
GPU thread each); "1" is a sequential solve."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OPS = [
    ("sqdist", "items", "sqdist_item", "n * m",
     [("x", "fin", "n * d"), ("y", "fin", "m * d"), ("res", "fout", "n * m"), ("n", "int"), ("m", "int"), ("d", "int")]),
    ("nan_sqdist", "items", "nan_sqdist_item", "n * m",
     [("x", "fin", "n * d"), ("y", "fin", "m * d"), ("res", "fout", "n * m"), ("n", "int"), ("m", "int"), ("d", "int")]),
    ("l1dist", "items", "l1dist_item", "n * m",
     [("x", "fin", "n * d"), ("y", "fin", "m * d"), ("res", "fout", "n * m"), ("n", "int"), ("m", "int"), ("d", "int")]),
    ("kernel", "items", "kernel_item", "n * m",
     [("x", "fin", "n * d"), ("y", "fin", "m * d"), ("res", "fout", "n * m"), ("n", "int"), ("m", "int"), ("d", "int"),
      ("kind", "int"), ("gamma", "float"), ("coef0", "float"), ("degree", "int")]),
    ("matmul", "items", "matmul_item", "n * m",
     [("a", "fin", "n * k"), ("b", "fin", "k * m"), ("res", "fout", "n * m"), ("n", "int"), ("k", "int"), ("m", "int")]),
    ("rowsum", "items", "rowsum_item", "n",
     [("a", "fin", "n * m"), ("res", "fout", "n"), ("n", "int"), ("m", "int")]),
    ("colsum", "items", "colsum_item", "m",
     [("a", "fin", "n * m"), ("res", "fout", "m"), ("n", "int"), ("m", "int")]),
    ("unary", "items", "unary_item", "count",
     [("x", "fin", "count"), ("res", "fout", "count"), ("count", "int"), ("op", "int"), ("a", "float"), ("b", "float")]),
    ("knn_select", "items", "knn_select_item", "n",
     [("dmat", "fin", "n * m"), ("dist", "fout", "n * k"), ("idx", "iout", "n * k"),
      ("n", "int"), ("m", "int"), ("k", "int"), ("exclude_self", "int")]),
    ("knn_sq", "items", "knn_sq_item", "n",
     [("x", "fin", "n * d"), ("y", "fin", "m * d"), ("dist", "fout", "n * k"), ("idx", "iout", "n * k"),
      ("n", "int"), ("m", "int"), ("d", "int"), ("k", "int"), ("exclude_self", "int")]),
    ("group_mean", "items", "group_mean_item", "g * d",
     [("x", "fin", "n * d"), ("labels", "iin", "n"), ("res", "fout", "g * d"), ("n", "int"), ("d", "int"), ("g", "int")]),
    ("take_rows", "items", "take_rows_item", "n_out * d",
     [("src", "fin", "n_src * d"), ("rows", "iin", "n_out"), ("res", "fout", "n_out * d"),
      ("n_out", "int"), ("d", "int"), ("n_src", "int")]),
    ("take_cols", "items", "take_cols_item", "n * c",
     [("src", "fin", "n * src_c"), ("cols", "iin", "c"), ("res", "fout", "n * c"), ("n", "int"), ("src_c", "int"), ("c", "int")]),
    ("variance", "items", "variance_item", "1",
     [("x", "fin", "count"), ("res", "fout", "1"), ("count", "int")]),
    ("ocsvm", "items", "ocsvm_smo_item", "1",
     [("q", "fin", "n * n"), ("cv", "fin", "n"), ("alpha", "finout", "n"), ("g", "fscr", "n"), ("info", "fout", "1"), ("iters", "iout", "1"),
      ("n", "int"), ("eps", "float"), ("max_iter", "int")]),
    ("lof_lrd", "items", "lof_lrd_item", "n",
     [("dist", "fin", "n * k"), ("idx", "iin", "n * k"), ("fit_dist", "fin", "n_fit * k"), ("lrd", "fout", "n"),
      ("k", "int"), ("n", "int"), ("n_fit", "int")]),
    ("lof_score", "items", "lof_score_item", "n",
     [("idx", "iin", "n * k"), ("fit_lrd", "fin", "n_fit"), ("lrd", "fin", "n"), ("score", "fout", "n"),
      ("k", "int"), ("n", "int"), ("n_fit", "int")]),
    ("kpca_center", "items", "kpca_center_item", "n * m",
     [("k", "fin", "n * m"), ("fit_cols", "fin", "m"), ("pred_rows", "fin", "n"), ("fit_all", "fin", "1"),
      ("res", "fout", "n * m"), ("n", "int"), ("m", "int")]),
    ("scale_div", "items", "scale_div_item", "count",
     [("x", "fin", "count"), ("res", "fout", "count"), ("count", "int"), ("s", "float")]),
    ("svd_flip", "items", "svd_flip_item", "c",
     [("v", "finout", "n * c"), ("n", "int"), ("c", "int")]),
    ("kpca_alpha_scale", "items", "kpca_alpha_scale_item", "n * c",
     [("v", "fin", "n * c"), ("w", "fin", "c"), ("res", "fout", "n * c"), ("n", "int"), ("c", "int"), ("divide", "int")]),
    ("nc_std", "items", "nc_std_item", "d",
     [("x", "fin", "n * d"), ("lab", "iin", "n"), ("cent", "fin", "n_classes * d"), ("std", "fout", "d"),
      ("n", "int"), ("d", "int"), ("n_classes", "int")]),
    ("nc_shrink", "items", "nc_shrink_item", "n_classes * d",
     [("x", "fin", "n * d"), ("cent", "fin", "n_classes * d"), ("nk", "fin", "n_classes"), ("std", "fin", "d"),
      ("res", "fout", "n_classes * d"), ("devs", "fout", "n_classes * d"), ("n", "int"), ("d", "int"),
      ("n_classes", "int"), ("do_shrink", "int"),
      ("med", "float"), ("shrink", "float")]),
    ("nc_decision", "items", "nc_decision_item", "n * n_classes",
     [("q", "fin", "n * d"), ("cent", "fin", "n_classes * d"), ("std", "fin", "d"), ("prior", "fin", "n_classes"),
      ("res", "fout", "n * n_classes"), ("n", "int"), ("d", "int"), ("n_classes", "int")]),
    ("softmax", "items", "softmax_item", "n",
     [("x", "fin", "n * c"), ("res", "fout", "n * c"), ("n", "int"), ("c", "int")]),
    ("log_softmax", "items", "log_softmax_item", "n",
     [("x", "fin", "n * c"), ("res", "fout", "n * c"), ("n", "int"), ("c", "int")]),
    ("pcs", "items", "pcs_item", "n",
     [("x", "fin", "n * d_in"), ("hidx", "iin", "degree * nf"), ("hbit", "iin", "degree * nf"), ("res", "fout", "n * nc"),
      ("scr", "fscr", "n * 2 * nc"), ("n", "int"), ("d_in", "int"), ("nf", "int"), ("nc", "int"), ("degree", "int"),
      ("gamma", "float"), ("coef0", "float")]),
    ("achi2", "items", "achi2_item", "n * d",
     [("x", "fin", "n * d"), ("res", "fout", "n * d * (2 * steps - 1)"), ("n", "int"), ("d", "int"), ("steps", "int"),
      ("interval", "float")]),
    ("skew_weights", "items", "skew_weights_item", "count",
     [("z", "fin", "count"), ("res", "fout", "count"), ("count", "int")]),
    ("skew_transform", "items", "skew_transform_item", "n * nc",
     [("lx", "fin", "n * d"), ("w", "fin", "d * nc"), ("off", "fin", "nc"), ("res", "fout", "n * nc"),
      ("n", "int"), ("d", "int"), ("nc", "int")]),
    ("absdiff_sum", "items", "absdiff_sum_item", "1",
     [("a", "fin", "count"), ("b", "fin", "count"), ("res", "fout", "1"), ("count", "int")]),
    ("row_normalize", "items", "row_normalize_item", "n",
     [("a", "fin", "n * m"), ("res", "fout", "n * m"), ("n", "int"), ("m", "int")]),
    ("lp_clamp", "items", "lp_clamp_item", "n",
     [("ld", "fin", "n * c"), ("ystatic", "fin", "n * c"), ("unlabeled", "iin", "n"), ("res", "fout", "n * c"),
      ("n", "int"), ("c", "int")]),
    ("ls_clamp", "items", "ls_clamp_item", "count",
     [("ld", "fin", "count"), ("ystatic", "fin", "count"), ("res", "fout", "count"), ("count", "int"), ("alpha", "float")]),
    ("ls_laplacian", "items", "ls_laplacian_item", "n * n",
     [("a", "fin", "n * n"), ("res", "fout", "n * n"), ("n", "int")]),
    ("knn_graph", "items", "knn_graph_item", "n",
     [("idx", "iin", "n * k"), ("res", "fout", "n * m"), ("n", "int"), ("m", "int"), ("k", "int")]),
    ("knn_impute", "items", "knn_impute_item", "n * d",
     [("x", "fin", "n * d"), ("fx", "fin", "m * d"), ("best_d", "fscr", "n * d * k"), ("best_i", "iscr", "n * d * k"),
      ("res", "fout", "n * d"), ("n", "int"), ("m", "int"), ("d", "int"), ("k", "int"), ("weights", "int")]),
    ("col_degree", "items", "col_degree_item", "n",
     [("a", "fin", "n * n"), ("res", "fout", "n"), ("n", "int")]),
    ("ls_laplacian_deg", "items", "ls_laplacian_deg_item", "n * n",
     [("a", "fin", "n * n"), ("deg", "fin", "n"), ("res", "fout", "n * n"), ("n", "int")]),
    ("row_all_zero", "items", "row_all_zero_item", "n",
     [("a", "iin", "n * m"), ("res", "iout", "n"), ("n", "int"), ("m", "int")]),
    ("pcs_sketch", "items", "pcs_sketch_item", "n * degree",
     [("x", "fin", "n * d_in"), ("hidx", "iin", "degree * nf"), ("hbit", "iin", "degree * nf"),
      ("sk", "fout", "n * degree * nc"), ("n", "int"), ("d_in", "int"), ("nf", "int"), ("nc", "int"),
      ("degree", "int"), ("gamma", "float"), ("coef0", "float")]),
    ("pcs_conv", "items", "pcs_conv_item", "n * nc",
     [("acc", "fin", "n * nc"), ("sk", "fin", "n * degree * nc"), ("res", "fout", "n * nc"), ("n", "int"),
      ("nc", "int"), ("degree", "int"), ("p", "int")]),
    ("pcs_copy0", "items", "pcs_copy0_item", "n * nc",
     [("sk", "fin", "n * degree * nc"), ("res", "fout", "n * nc"), ("n", "int"), ("nc", "int"), ("degree", "int")]),
    ("knn_impute_cells", "items", "knn_impute_cell_item", "nc",
     [("cells", "iin", "nc"), ("x", "fin", "n * d"), ("fx", "fin", "m * d"), ("best_d", "fscr", "n * d * k"),
      ("best_i", "iscr", "n * d * k"), ("res", "finout", "n * d"), ("n", "int"), ("m", "int"), ("d", "int"),
      ("k", "int"), ("weights", "int"), ("nc", "int")]),
    ("pagerank_step", "items", "pagerank_step_item", "n",
     [("q", "fin", "n * n"), ("x", "fin", "n"), ("p", "fin", "n"), ("dw", "fin", "n"), ("dangling", "iin", "n"), ("res", "fout", "n"),
      ("n", "int"), ("alpha", "float")]),
    ("cc_step", "items", "cc_step_item", "n",
     [("a", "fin", "n * n"), ("lab", "iin", "n"), ("res", "iout", "n"), ("n", "int")]),
    ("louvain", "items", "louvain_item", "1",
     [("a", "fin", "n * n"), ("labels", "iout", "n"), ("info", "fout", "2"), ("w", "fscr", "n * n"), ("w2", "fscr", "n * n"),
      ("comm", "iscr", "n"), ("node_of", "iscr", "n"), ("deg", "fscr", "n"), ("stot", "fscr", "n"), ("k2c", "fscr", "n"),
      ("tmp", "fscr", "n"), ("n", "int"), ("max_level", "int"), ("resolution", "float"), ("threshold", "float")]),
    ("svgp", "items", "svgp_item", "1",
     [("kuu", "fin", "m * m"), ("bmat", "fin", "m * m"), ("b", "fin", "m"), ("y", "fin", "n"), ("alpha", "fout", "m"),
      ("cmat", "fout", "m * m"), ("qmu", "fout", "m"), ("qsqrt", "fout", "m * m"), ("info", "fout", "2"),
      ("luu", "fscr", "m * m"), ("ls", "fscr", "m * m"), ("e", "fscr", "m"), ("col", "fscr", "m"),
      ("m", "int"), ("n", "int"), ("noise", "float"), ("jitter", "float"), ("kdiag", "float")]),
    ("svgp_var", "items", "svgp_var_item", "n",
     [("ksu", "fin", "n * m"), ("cmat", "fin", "m * m"), ("res", "fout", "n"), ("n", "int"), ("m", "int"), ("kdiag", "float")]),
]

#: Hand-written resident drivers (x_neighbors/iter_device.mojo on the GPU,
#: x_neighbors/iter_host.mojo on the CPU): loops of the items above that keep
#: their buffers on the device between steps. Exported like any op.
CUSTOM_OPS = [
    ("lp_iterate",
     [("g", "fin", "n * n"), ("ld", "finout", "n * c"), ("ystatic", "fin", "n * c"), ("unlabeled", "iin", "n"),
      ("info", "iout", "2"), ("n", "int"), ("c", "int"), ("max_iter", "int"), ("variant", "int"),
      ("tol_hi", "int"), ("tol_lo", "int"), ("alpha", "float")]),
    ("pr_iterate",
     [("q", "fin", "n * n"), ("x", "finout", "n"), ("p", "fin", "n"), ("dw", "fin", "n"), ("dangling", "iin", "n"),
      ("info", "iout", "2"), ("n", "int"), ("max_iter", "int"), ("thr_hi", "int"), ("thr_lo", "int"),
      ("alpha", "float")]),
    ("pcs_resident",
     [("x", "fin", "n * d_in"), ("hidx", "iin", "degree * nf"), ("hbit", "iin", "degree * nf"), ("res", "fout", "n * nc"),
      ("n", "int"), ("d_in", "int"), ("nf", "int"), ("nc", "int"), ("degree", "int"), ("gamma", "float"),
      ("coef0", "float")]),
    ("cc_iterate",
     [("a", "fin", "n * n"), ("lab", "iinout", "n"), ("info", "iout", "1"), ("n", "int")]),
]

#: Ops whose GPU driver runs a threadgroup form of the (sequential) item
#: instead of one thread: op -> (module, function, threads constant, the
#: define that restores the one-thread item). The host driver keeps the item.
#: Ops whose GPU binding runs the item loop on the HOST (a sequential solve
#: that one GPU thread runs far slower than one CPU core; the host column is
#: the same statements): op -> the define that restores the one-thread GPU
#: launch.
HOST_RUN = {
    "louvain": "MOJOLEARN_XN_LOUVAIN_GPU",
}

BLOCK_OPS = {
    "ocsvm": ("block_ops", "ocsvm_smo_block", "OCSVM_TPB", "MOJOLEARN_XN_SERIAL_SMO"),
}

HDR = "# SPDX-License-Identifier: Apache-2.0\n# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632\n"
GEN = "# GENERATED by x_neighbors/gen.py from its OPS table; edit the table, not this file.\n"


def split(params):
    bufs = [p for p in params if p[1] not in ("int", "float")]
    scal = [p for p in params if p[1] in ("int", "float")]
    return bufs, scal


def imports(kind):
    mods = {}
    for name, mod, item, count, params in OPS:
        mods.setdefault(mod, []).append(item)
    lines = []
    for mod, items in mods.items():
        lines.append(f"from x_neighbors.{mod} import FP, IP, {', '.join(items)}")
    return "\n".join(lines) + "\n"


def is_int_buf(kind):
    return kind.startswith("i")


def device():
    s = [HDR, GEN, '"""The neighbors lane\'s GPU drivers (see x_neighbors/gen.py)."""\n',
         "from std.gpu import block_idx, block_dim, thread_idx\n",
         "from std.ffi import _Global\n",
         "from max.gpu.host import DeviceBuffer, DeviceContext\n",
         "from std.sys.compile import is_defined\n",
         "from bindings.hostptr import copy_f32\n",
         "from core.host_parallel import host_parallelize\n",
         "from core.host_predict_threads import host_predict_chunk, host_predict_task_count\n",
         "from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL\n", imports("device"),
         "".join(f"from x_neighbors.{m} import {f}, {c}\n" for m, f, c, _ in BLOCK_OPS.values()), """
comptime BLOCK = 128


struct _NeighborsContext(Defaultable, Movable):
    \"\"\"ONE process-lifetime DeviceContext for every x_neighbors driver. A
    context per call let a driver's DeviceBuffers outlive the context they
    were made on (Mojo destroys the context at its last use, the
    synchronize, before the buffers), and a second GPU call in the same
    process hung (LocalOutlierFactor on an RTX 4090, 2026-09-27); a context
    per call also exhausts Metal's per-process command queues. Storage is
    `std.ffi._Global` (x_cnn/device.mojo's pattern), one slot per numeric
    tier so a FAST and an IDENTICAL .so in one process never share it.\"\"\"
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXNeighborsContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXNeighborsContextFast"
comptime X_NEIGHBORS_CONTEXT = _Global[StorageType=_NeighborsContext, name=_CTX_NAME, init_fn=_NeighborsContext.__init__]


def xn_ctx() raises -> DeviceContext:
    \"\"\"The shared context, created on first use.\"\"\"
    var slot = X_NEIGHBORS_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + BLOCK - 1) // BLOCK if count > 0 else 1


def _buf(ctx: DeviceContext, addr: Int, count: Int, upload: Bool) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
    if upload and count > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=FP(unsafe_from_address=addr))
    return buf^


def _buf_i(ctx: DeviceContext, addr: Int, count: Int, upload: Bool) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](count if count > 0 else 1)
    if upload and count > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=IP(unsafe_from_address=addr))
    return buf^


#: lane neighbors-apple2: a large float output comes back through a host
#: staging buffer of XN_OUT_CHUNK floats and is copied into the caller's
#: array over the host cores (the first touch of the caller's fresh pages
#: dominates a plain copy; kernel_methods/estimator.mojo `_download_into`,
#: round one). A copy: no arithmetic. `-D MOJOLEARN_XN_PLAIN_DOWN` keeps the
#: one enqueue_copy.
comptime XN_OUT_CHUNK = 1 << 24
comptime XN_STAGED_MIN = 1 << 22


def _down(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], addr: Int, count: Int) raises:
    if count <= 0:
        return
    comptime if is_defined["MOJOLEARN_XN_PLAIN_DOWN"]():
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=addr), src_buf=buf)
        return
    if count < XN_STAGED_MIN:
        ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=addr), src_buf=buf)
        return
    var c = min(count, XN_OUT_CHUNK)
    var h = ctx.enqueue_create_host_buffer[DType.float32](c)
    var off = 0
    while off < count:
        var m = min(c, count - off)
        var sub = buf.create_sub_buffer[DType.float32](off, m)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=sub)
        ctx.synchronize()
        var src_p = rebind[MutPointer[Float32, MutUntrackedOrigin]](h.unsafe_ptr())
        var dst_p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr + off * 4)
        var tasks = host_predict_task_count(m)
        var part = host_predict_chunk(m, tasks)

        def _part(task: Int) {imm src_p, imm dst_p, imm m, imm part}:
            var lo = task * part
            var hi = min(lo + part, m)
            if hi > lo:
                copy_f32(src_p.unsafe_offset(lo), dst_p.unsafe_offset(lo), hi - lo)

        if tasks == 1:
            _part(0)
        else:
            host_parallelize(_part, tasks)
        _ = sub^
        off += m
    _ = h^


def _down_i(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], addr: Int, count: Int) raises:
    if count > 0:
        ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=addr), src_buf=buf)


@always_inline
def _f(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)


@always_inline
def _i(addr: Int) -> IP:
    return IP(unsafe_from_address=addr)
"""]
    for name, mod, item, count, params in OPS:
        bufs, scal = split(params)
        kp = [f"{b[0]}: {'IP' if is_int_buf(b[1]) else 'FP'}" for b in bufs]
        kp += [f"{p[0]}_: {'Int64' if p[1] == 'int' else 'Float32'}" for p in scal]
        conv = "".join(f"    var {p[0]} = Int({p[0]}_)\n" if p[1] == "int" else f"    var {p[0]} = {p[0]}_\n" for p in scal)
        call = ", ".join(["t"] + [b[0] for b in bufs] + [p[0] for p in scal])
        if name in BLOCK_OPS:
            bm, bf, bc, bd = BLOCK_OPS[name]
            bcall = ", ".join([b[0] for b in bufs] + [p[0] for p in scal])
            s.append(f"\n\ndef {name}_kernel({', '.join(kp)}):\n{conv}    comptime if is_defined[\"{bd}\"]():\n"
                     f"        var t = _tid()\n        if t < {count}:\n            {item}({call})\n"
                     f"    else:\n        {bf}({bcall})\n")
        else:
            s.append(f"\n\ndef {name}_kernel({', '.join(kp)}):\n{conv}    var t = _tid()\n    if t < {count}:\n        {item}({call})\n")
        # driver
        dp = [f"{b[0]}: Int" for b in bufs if b[1] not in ("fscr", "iscr")]
        dp += [f"{p[0]}: {'Int' if p[1] == 'int' else 'Float32'}" for p in scal]
        body = "    var ctx = xn_ctx()\n"
        if name in HOST_RUN:
            hb = host_loop(item, count, bufs, scal)
            for b in bufs:
                if b[1] in ("fscr", "iscr"):
                    hb += f"    _ = s_{b[0]}^\n"
            body = (f"    comptime if not is_defined[\"{HOST_RUN[name]}\"]():\n"
                    + "".join("    " + ln + "\n" for ln in hb.rstrip("\n").split("\n"))
                    + "        return\n" + body)
        for b in bufs:
            up = "True" if b[1] in ("fin", "finout", "iin", "iinout") else "False"
            addr = "0" if b[1] in ("fscr", "iscr") else b[0]
            fn = "_buf_i" if is_int_buf(b[1]) else "_buf"
            body += f"    var d_{b[0]} = {fn}(ctx, {addr}, {b[2]}, {up})\n"
        args = [f"d_{b[0]}.unsafe_ptr()" for b in bufs] + [f"Int64({p[0]})" if p[1] == "int" else p[0] for p in scal]
        if name in BLOCK_OPS:
            bm, bf, bc, bd = BLOCK_OPS[name]
            body += f"    comptime tpb = 1 if is_defined[\"{bd}\"]() else {bc}\n"
            body += f"    ctx.enqueue_function[{name}_kernel](\n        {', '.join(args)},\n        grid_dim=1, block_dim=tpb,\n    )\n"
        else:
            body += f"    ctx.enqueue_function[{name}_kernel](\n        {', '.join(args)},\n        grid_dim=_grid({count}), block_dim=(BLOCK if {count} > 1 else 1),\n    )\n"
        for b in bufs:
            if b[1] in ("fout", "finout", "iout", "iinout"):
                fn = "_down_i" if is_int_buf(b[1]) else "_down"
                body += f"    {fn}(ctx, d_{b[0]}, {b[0]}, {b[2]})\n"
        body += "    ctx.synchronize()\n"
        for b in bufs:
            body += f"    _ = d_{b[0]}^\n"
        body += "    _ = ctx^\n"
        s.append(f"\n\ndef op_{name}({', '.join(dp)}) raises:\n{body}")
    return "".join(s)


def host():
    s = [HDR, GEN, '"""The neighbors lane\'s CPU drivers: host loops over the SAME items (see x_neighbors/gen.py)."""\n',
         "from std.sys.compile import is_defined\n", imports("host"), """
#: the host gate's negative control (`MOJOLEARN_HOST_SABOTAGE`): every op's
#: first float output moves by 1e-3 in its first element
comptime X_NEIGHBORS_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


@always_inline
def _f(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)


@always_inline
def _i(addr: Int) -> IP:
    return IP(unsafe_from_address=addr)
"""]
    for name, mod, item, count, params in OPS:
        bufs, scal = split(params)
        dp = [f"{b[0]}: Int" for b in bufs if b[1] not in ("fscr", "iscr")]
        dp += [f"{p[0]}: {'Int' if p[1] == 'int' else 'Float32'}" for p in scal]
        body = host_loop(item, count, bufs, scal)
        outs = [b for b in bufs if b[1] in ("fout", "finout")]
        if outs:
            b = outs[0]
            body += f"    comptime if X_NEIGHBORS_HOST_SABOTAGE:\n        if ({b[2]}) > 0:\n            _f({b[0]}).unsafe_store(0, _f({b[0]}).unsafe_load(0) + Float32(1e-3))\n"
        for b in bufs:
            if b[1] in ("fscr", "iscr"):
                body += f"    _ = s_{b[0]}^\n"
        s.append(f"\n\ndef op_{name}({', '.join(dp)}) raises:\n{body}")
    return "".join(s)


def host_loop(item, count, bufs, scal):
    """The host driver's body up to the loop: scratch Lists, then the item
    over every t (shared by the host drivers and HOST_RUN device drivers)."""
    if True:
        body = ""
        for b in bufs:
            if b[1] == "fscr":
                body += f"    var s_{b[0]} = List[Float32](length=({b[2]}) if ({b[2]}) > 0 else 1, fill=Float32(0))\n"
            if b[1] == "iscr":
                body += f"    var s_{b[0]} = List[Int32](length=({b[2]}) if ({b[2]}) > 0 else 1, fill=Int32(0))\n"
        ptrs = []
        for b in bufs:
            if b[1] == "fscr":
                ptrs.append(f"FP(unsafe_from_address=Int(s_{b[0]}.unsafe_ptr()))")
            elif b[1] == "iscr":
                ptrs.append(f"IP(unsafe_from_address=Int(s_{b[0]}.unsafe_ptr()))")
            else:
                ptrs.append(f"_i({b[0]})" if is_int_buf(b[1]) else f"_f({b[0]})")
        call = ", ".join(["t"] + ptrs + [p[0] for p in scal])
        body += f"    for t in range({count}):\n        {item}({call})\n"
        return body


def wrappers():
    s = ["""

def _a(v: PythonObject, k: Int) raises -> Int:
    var x = Int(py=v[k])
    if x == 0:
        raise Error("x_neighbors: null buffer address")
    return x


def _n(v: PythonObject, k: Int) raises -> Int:
    var x = Int(py=v[k])
    if x < 0:
        raise Error("x_neighbors: a negative size was passed")
    return x


def _f(v: PythonObject, k: Int) raises -> Float32:
    return Float32(Float64(py=v[k]))


def eigh_binding(a: PythonObject, i: PythonObject, f: PythonObject) raises -> PythonObject:
    var x = _a(a, 0)
    var w = _a(a, 1)
    var v = _a(a, 2)
    var n = _n(i, 0)
    if n <= 0:
        raise Error("x_neighbors: eigh needs n >= 1")
    var sweeps = 0
    with GILReleased(Python()):
        sweeps = op_eigh(x, n, w, v)
    return PythonObject(sweeps)


def x_neighbors_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))
"""]
    for name, params in [(o[0], o[4]) for o in OPS] + CUSTOM_OPS:
        bufs, scal = split(params)
        lines = []
        ai = 0
        for b in bufs:
            if b[1] in ("fscr", "iscr"):
                continue
            lines.append(f"    var v_{b[0]} = _a(a_, {ai})")
            ai += 1
        ii = fi = 0
        for p in scal:
            if p[1] == "int":
                lines.append(f"    var v_{p[0]} = _n(i_, {ii})")
                ii += 1
            else:
                lines.append(f"    var v_{p[0]} = _f(f_, {fi})")
                fi += 1
        args = ["v_" + b[0] for b in bufs if b[1] not in ("fscr", "iscr")] + ["v_" + p[0] for p in scal]
        body = "\n".join(lines)
        s.append(f"\n\ndef {name}_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:\n"
                 f"{body}\n    with GILReleased(Python()):\n        op_{name}({', '.join(args)})\n    return PythonObject(None)\n")
    reg = "".join(f'    m.def_function[{name}_binding]("xn_{name}")\n' for name in [o[0] for o in OPS] + [c[0] for c in CUSTOM_OPS])
    s.append(f"""

def _add_ops(mut m: PythonModuleBuilder) raises:
{reg}    m.def_function[eigh_binding]("xn_eigh")
    m.def_function[x_neighbors_numeric_mode_binding]("x_neighbors_numeric_mode")
""")
    return "".join(s)


BIND_HEAD = """from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_neighbors.eigh import op_eigh
"""


def gpu_binding():
    ops = ", ".join(f"op_{o[0]}" for o in OPS)
    return (HDR + GEN + '"""THE NEIGHBORS EXPANSION LANE\'S GPU BINDING (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md):\nevery export is xn_<op>(addresses, ints, floats) over x_neighbors/device_ops.mojo."""\n'
            + BIND_HEAD + "from checks.vendor import COMPILED_VENDOR\n"
            + f"from x_neighbors.device_ops import {ops}\n"
            + f"from x_neighbors.iter_device import {', '.join('op_' + c[0] for c in CUSTOM_OPS)}\n" + wrappers() + """

def x_neighbors_vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_neighbors() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_neighbors")
        _add_ops(m)
        m.def_function[x_neighbors_vendor_binding]("x_neighbors_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_neighbors: ", e))
""")


def host_binding():
    ops = ", ".join(f"op_{o[0]}" for o in OPS)
    return (HDR + GEN + '"""CPU binding for `_mojolearn_x_neighbors`: the GPU binding\'s export names and\naddress contract over the host drivers x_neighbors/host_ops.mojo. HOST ONLY."""\n'
            + BIND_HEAD + "from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name\n"
            + f"from x_neighbors.host_ops import X_NEIGHBORS_HOST_SABOTAGE, {ops}\n"
            + f"from x_neighbors.iter_host import {', '.join('op_' + c[0] for c in CUSTOM_OPS)}\n" + wrappers() + """

def x_neighbors_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_neighbors_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_neighbors_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_neighbors host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_neighbors_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_NEIGHBORS_HOST_SABOTAGE)


def x_neighbors_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


@export
def PyInit__mojolearn_x_neighbors_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_neighbors_host")
        m.def_function[x_neighbors_host_numeric_mode_binding]("x_neighbors_host_numeric_mode")
        m.def_function[x_neighbors_host_vendor_binding]("x_neighbors_host_vendor")
        m.def_function[x_neighbors_host_column_binding]("x_neighbors_host_column")
        m.def_function[x_neighbors_host_sabotage_binding]("x_neighbors_host_sabotage")
        _add_ops(m)
        m.def_function[x_neighbors_vendor_binding]("x_neighbors_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_neighbors_host: ", e))
""")


if __name__ == "__main__":
    (ROOT / "x_neighbors" / "device_ops.mojo").write_text(device())
    (ROOT / "x_neighbors" / "host_ops.mojo").write_text(host())
    (ROOT / "bindings" / "_mojolearn_x_neighbors.mojo").write_text(gpu_binding())
    (ROOT / "bindings" / "_mojolearn_x_neighbors_host.mojo").write_text(host_binding())
    surf = ROOT / "python" / "mojolearn" / "_surface_neighbors.py"
    t = surf.read_text()
    a = t.index("# BEGIN GENERATED EXPORTS")
    b = t.index("# END GENERATED EXPORTS")
    names = ["x_neighbors_host_numeric_mode", "x_neighbors_host_vendor", "x_neighbors_host_column",
             "x_neighbors_host_sabotage"] + [f"xn_{o[0]}" for o in OPS] + [f"xn_{c[0]}" for c in CUSTOM_OPS] + ["xn_eigh", "x_neighbors_numeric_mode",
                                                                            "x_neighbors_vendor"]
    body = "# BEGIN GENERATED EXPORTS\n" + "".join(f'            "{x}",\n' for x in names) + "            "
    surf.write_text(t[:a] + body + t[b:])
    print(f"x_neighbors/gen.py: {len(OPS)} ops -> device_ops, host_ops and the two bindings")
