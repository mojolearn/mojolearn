# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CLUSTER LANE'S GPU BINDING (algorithm expansion, lane/algos-cluster).

One call, `x_cluster_call(which, x_addr, x_len, a_addr, a_len, ip, fp)`, runs
`x_cluster/entries.mojo::run_entry` on `x_cluster.device_ops.DeviceOps`. The
CPU host binding `_mojolearn_x_cluster_host` exports the same names with the
same contract over `HostOps`."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from x_cluster.device_ops import DeviceOps
from core.device_scan import device_first_nonfinite
from x_cluster.bisect_fast import BISECT_FAST_ZEROCOPY, bisect_entry_ptr
from x_cluster.entries import ENTRY_BISECT, ENTRY_MINIBATCH, run_entry
from x_cluster.minibatch_ptr import MBK_FAST_DEVSCAN, MBK_NONFINITE_MSG, MBK_ZEROCOPY, minibatch_entry_ptr
from x_cluster.out import ClusterOut, py_floats, py_ints
from x_cluster.tree_cut import PY2MOJO_CLUSTER


def call_binding(
    which: PythonObject, x_addr: PythonObject, x_len: PythonObject, a_addr: PythonObject,
    a_len: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    var w = Int(py=which)
    var nx = Int(py=x_len)
    var na = Int(py=a_len)
    # lane/apple-fast-mbkspeed, FAST on Apple, DEFAULT since the M3 A/B
    # (off: -D MOJOLEARN_MBK_ZEROCOPY_OFF):
    # MiniBatchKMeans uploads X from the caller's array, no host copy
    # (x_cluster/minibatch_ptr.mojo); False falls through to the copy below
    comptime if MBK_ZEROCOPY:
        if w == ENTRY_MINIBATCH and nx > 0:
            var a0 = read_f32(Int(py=a_addr), na) if na > 0 else List[Float32]()
            var ints0 = py_ints(ip)
            var floats0 = py_floats(fp)
            var xp = f32_ptr(Int(py=x_addr))
            var res0 = ClusterOut()
            var took = False
            with GILReleased(Python()):
                var ops0 = DeviceOps()
                took = minibatch_entry_ptr(ops0, xp, nx, a0, ints0, floats0, res0)
            if took:
                return res0.to_py()
    # lane/apple-fast-gap-clus3, FAST on Apple, DEFAULT since the M3 A/B
    # (off: -D MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF): BisectingKMeans uploads X from the
    # caller's array and centers it on the device (x_cluster/bisect_fast.mojo);
    # False falls through to the copy below
    comptime if BISECT_FAST_ZEROCOPY:
        if w == ENTRY_BISECT and nx > 0:
            var ints1 = py_ints(ip)
            var floats1 = py_floats(fp)
            var xp1 = f32_ptr(Int(py=x_addr))
            var res1 = ClusterOut()
            var took1 = False
            with GILReleased(Python()):
                var ops1 = DeviceOps()
                took1 = bisect_entry_ptr(ops1, xp1, nx, na > 0, ints1, floats1, res1)
            if took1:
                return res1.to_py()
    var x = read_f32(Int(py=x_addr), nx) if nx > 0 else List[Float32]()
    var a = read_f32(Int(py=a_addr), na) if na > 0 else List[Float32]()
    var ints = py_ints(ip)
    # lane/apple-fast-s-linalg: Python skipped MiniBatchKMeans' host scan
    # (ip[12] = 1) and the zero-copy path did not take the fit: scan X on the
    # device (merge 2026-10-05: was a host walk, refused on the GPU route)
    var scan_dev = False
    comptime if MBK_FAST_DEVSCAN:
        scan_dev = w == ENTRY_MINIBATCH and len(ints) > 12 and ints[12] != 0 and nx > 0
    var floats = py_floats(fp)
    var res = ClusterOut()
    var bad = False
    with GILReleased(Python()):
        var ops = DeviceOps()
        if scan_dev:
            var xd = ops.ctx.enqueue_create_buffer[DType.float32](nx)
            ops.ctx.enqueue_copy(dst_buf=xd, src_ptr=x.unsafe_ptr())
            bad = device_first_nonfinite(ops.ctx, xd, nx) >= 0
            _ = xd^
        if not bad:
            res = run_entry(ops, w, x, a, ints, floats)
    if bad:
        raise Error(MBK_NONFINITE_MSG)
    return res.to_py()


def mbk_devscan_binding() raises -> PythonObject:
    """1 when MiniBatchKMeans' NaN/inf scan runs in this binding
    (-D MOJOLEARN_MBK_FAST_DEVSCAN, x_cluster/minibatch_ptr.mojo), else 0."""
    return PythonObject(1 if MBK_FAST_DEVSCAN else 0)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def py2mojo_binding() raises -> PythonObject:
    """1: the steps `_expansion_cluster.py` / `_hierarchy_impl.py` ran in
    Python run here (lane apple-fast-py2mojo-cluster); 0 under
    `-D MOJOLEARN_PY2MOJO_cluster_OFF`, and Python takes its old path."""
    return PythonObject(1 if PY2MOJO_CLUSTER else 0)


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_cluster() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_cluster")
        m.def_function[call_binding]("x_cluster_call")
        m.def_function[numeric_mode_binding]("x_cluster_numeric_mode")
        m.def_function[mbk_devscan_binding]("x_cluster_mbk_devscan")
        m.def_function[vendor_binding]("x_cluster_vendor")
        m.def_function[py2mojo_binding]("x_cluster_py2mojo")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cluster: ", e))
