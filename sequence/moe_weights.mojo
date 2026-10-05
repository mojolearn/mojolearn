# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MoEBlock weights kept on the device between forwards (lane
gap-neural-overhead2, 2026-10-02).

The board's MoE cell (x (8192, 1024), E 8, F 2816) uploaded the router,
`gate_up_proj` (184 MB) and `down_proj` (92 MB) on EVERY forward: 277 MB of
PCIe per call for weights that had not changed. `moe_weights_put` copies
them once into device buffers on the sequence lane's context and returns a
handle; the GPU binding's `moe_forward` with that handle reads them in place
(`sequence/pyapi.mojo::moe_forward_run`, the same launches on the same
words). The Python layer owns the handle: it puts a new copy when a weight
array is replaced and frees the old one (`_x_sequence_moe.MoEBlock`).
Copies only: no bit moves."""
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP
from sequence.exec_device import sequence_ctx


struct _MoeWeights(Defaultable, Movable):
    """Per handle: router (E, D), gate_up (E, 2F, D), down (E, D, F)."""
    var router: List[DeviceBuffer[DType.float32]]
    var gate_up: List[DeviceBuffer[DType.float32]]
    var down: List[DeviceBuffer[DType.float32]]
    var ids: List[Int]
    var next_id: Int

    def __init__(out self):
        self.router = List[DeviceBuffer[DType.float32]]()
        self.gate_up = List[DeviceBuffer[DType.float32]]()
        self.down = List[DeviceBuffer[DType.float32]]()
        self.ids = List[Int]()
        self.next_id = 1


comptime _NAME = "MojoXSequenceMoeWeightsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXSequenceMoeWeightsFast"
comptime MOE_WEIGHTS = _Global[StorageType=_MoeWeights, name=_NAME, init_fn=_MoeWeights.__init__]


def _up(ctx: DeviceContext, src: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=b, src_ptr=src)
    return b^


def moe_weights_put(router: FP, gate_up: FP, down: FP, E: Int, D: Int, F: Int) raises -> Int:
    """Device copies of the three weight arrays; their handle (> 0)."""
    if E < 1 or D < 1 or F < 1:
        raise Error("moe_weights: E, D, F >= 1")
    var ctx = sequence_ctx()
    var r = _up(ctx, router, E * D)
    var gu = _up(ctx, gate_up, E * 2 * F * D)
    var dn = _up(ctx, down, E * D * F)
    # the caller's arrays may change once this returns
    ctx.synchronize()
    var s = MOE_WEIGHTS.get_or_create_ptr()
    var h = s[].next_id
    s[].next_id += 1
    s[].router.append(r^)
    s[].gate_up.append(gu^)
    s[].down.append(dn^)
    s[].ids.append(h)
    _ = ctx^
    return h


def _find(h: Int) raises -> Int:
    var s = MOE_WEIGHTS.get_or_create_ptr()
    for i in range(len(s[].ids)):  # small-loop(ids: the live MoE weight handles): a handle lookup, no data
        if s[].ids[i] == h:
            return i
    raise Error("moe_weights: no weights under handle " + String(h))


def moe_weights_ptrs(h: Int) raises -> Tuple[FP, FP, FP]:
    """The device addresses of handle `h`'s router, gate_up and down."""
    var i = _find(h)
    var s = MOE_WEIGHTS.get_or_create_ptr()
    return (
        FP(unsafe_from_address=Int(s[].router[i].unsafe_ptr())),
        FP(unsafe_from_address=Int(s[].gate_up[i].unsafe_ptr())),
        FP(unsafe_from_address=Int(s[].down[i].unsafe_ptr())),
    )


def moe_weights_free(h: Int) raises:
    var i = _find(h)
    # queued forwards may still read them
    var ctx = sequence_ctx()
    ctx.synchronize()
    _ = ctx^
    var s = MOE_WEIGHTS.get_or_create_ptr()
    _ = s[].router.pop(i)
    _ = s[].gate_up.pop(i)
    _ = s[].down.pop(i)
    _ = s[].ids.pop(i)
