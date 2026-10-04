# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fit's finiteness check on the device, every x_linear route
(lane/fam-linear, 2026-10-04).

The binding walked all n x d host words one at a time before every fit
(`bindings/_mojolearn_x_linear.mojo::_finite`); lane/idn-sgd-multiblock moved
the SGD grids' check onto the uploaded words (`device.mojo`
SGD_IDN_DEV_FINITE). This is the same check for the other routes: each grid
calls `xlin_finite_device` on the X buffer it has just uploaded, so X still
crosses the bus once and no host loop touches it. The verdict is the host
walk's (exponent all ones = NaN or infinity), the error text is the
binding's, X is named before y. One witness-guarded unit: the flags are
rebuilt from inputs the launches do not write, so a cut Apple launch reruns.

`-D MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF` (or the master
`MOJOLEARN_IDN_ALL_OFF`) restores the binding's host walk on every vendor.
No bit moves: the check reads, it does not compute a fitted word.
"""

from std.gpu import block_idx, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_linear.ops import FP, IP, ld, sti
from x_linear.witness import Witness, witness_end, WITNESS_TRIES

comptime XLIN_IDN_DEV_FINITE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_XLIN_IDN_DEV_FINITE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime XF_TPB = 256
#: words one thread tests
comptime XF_RUN = 64


def xlin_finite_kernel(p: FP, count: Int32, flag: IP, slot: Int32, wf: IP, woff: Int32, nonce: Int32):
    var q = Int(block_idx.x) * XF_TPB + Int(thread_idx.x)
    var lo = q * XF_RUN
    var hi = min(lo + XF_RUN, Int(count))
    var bad = False
    for i in range(lo, hi):
        var bits = bitcast[DType.uint32](ld(p, i))
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            bad = True
    if bad:
        sti(flag, Int(slot), 1)
    witness_end(wf, woff, nonce)


def _xf_blocks(count: Int) -> Int:
    var tasks = (count + XF_RUN - 1) // XF_RUN
    return (tasks + XF_TPB - 1) // XF_TPB


def xlin_finite_host(p: FP, count: Int, name: String) raises:
    """The binding's scalar walk, for the routes that upload nothing a
    device check could read (and for the shapes no route serves)."""
    for i in range(count):
        var v = p.unsafe_load(i)
        if not (v == v) or v > Float32(3.4028234e38) or v < Float32(-3.4028234e38):
            raise Error(String("mojolearn: ", name, " contains NaN or infinity"))


def xlin_finite_device(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], n_x: Int, y: FP, n_y: Int,
) raises:
    """Raises the binding's error when the uploaded X (`dx`, n_x words) or
    the host y block (n_y words) holds a NaN or an infinity. y is uploaded
    to a buffer of its own here (the grids keep theirs in other layouts);
    it is n_y words, small beside X."""
    var c = ctx.copy()
    var dfl = c.enqueue_create_buffer[DType.int32](2)
    var dyf = c.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var gx = _xf_blocks(n_x) if n_x > 0 else 0
    var gy = _xf_blocks(n_y) if n_y > 0 else 0
    var wit = Witness(c, max(gx + gy, 1))
    var hfl = List[Int32](length=2, fill=Int32(0))
    var tries = 0
    while True:
        var nonce = wit.begin()
        dfl.enqueue_fill(Int32(0))
        if n_x > 0:
            c.enqueue_function[xlin_finite_kernel](
                dx.unsafe_ptr(), Int32(n_x), dfl.unsafe_ptr(), Int32(0), wit.p(), Int32(0), nonce,
                grid_dim=gx, block_dim=XF_TPB,
            )
        if n_y > 0:
            c.enqueue_copy(dst_buf=dyf, src_ptr=y)
            c.enqueue_function[xlin_finite_kernel](
                dyf.unsafe_ptr(), Int32(n_y), dfl.unsafe_ptr(), Int32(1), wit.p(), Int32(gx), nonce,
                grid_dim=gy, block_dim=XF_TPB,
            )
        c.enqueue_copy(dst_ptr=hfl.unsafe_ptr(), src_buf=dfl)
        c.synchronize()
        if wit.ok(c, gx + gy, "x_linear finite check"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    var bx = hfl[0] != 0
    var by = hfl[1] != 0
    _ = hfl^
    _ = dfl^
    _ = dyf^
    _ = wit^
    if bx:
        raise Error("mojolearn: X contains NaN or infinity")
    if by:
        raise Error("mojolearn: y contains NaN or infinity")
