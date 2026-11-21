# SPDX-License-Identifier: Apache-2.0
"""The device route of one ExponentialSmoothing estimator, as native words.

`device="gpu"` and `device="cpu"` name two published Holt-Winters operations
tables (core/tsa_entry_table.mojo): the table of the selected vendor and
tier, and the CPU identical table. `device=None` (the default) is this
installation's device; the rule and its words are the linear family's
(core/linear_route.mojo: `linear_device_code`, `linear_device_installed`,
`linear_device_refusal`). A route is resolved ONCE per estimator
(providers/python/tsa_route_boundary_door.mojo) and holds the resolved thin
C entry; every later fit, forecast, predict and resident query calls through
it. No public operation tests the device, reads a table slot or looks an
entry up by name.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from core.tsa_entry_table import (
    TSA_OP_FLAGS, TSA_OP_VENDOR, TSA_OP_MODE, TSA_HAS_FIT, TSA_HAS_RESIDENT, tsa_ops_resolve,
)


struct TsaHwRoute(ImplicitlyCopyable, Movable):
    """Resolved entry of one estimator. `mode`, `vendor` and `flags` are the
    words the compiled entry itself reports (1 identical / 0 fast; 0 CPU,
    1 Metal, 2 CUDA, 3 HIP; TSA_HAS_* bits), not the request. `device` is
    LINEAR_DEVICE_GPU or LINEAR_DEVICE_CPU of core/linear_route.mojo."""
    var device: Int
    var mode: Int
    var vendor: Int
    var flags: Int
    var ops: EntryRef

    def __init__(out self, device: Int):
        self.device = device
        self.mode = -1
        self.vendor = -1
        self.flags = 0
        self.ops = entry_ref_unresolved()


def tsa_hw_route_resolve(device: Int, family: Int, mode: Int, vendor: Int) raises -> TsaHwRoute:
    """Read the operations entry of (`family`, `mode`, `vendor`) once and
    admit it: the compiled entry must report the vendor and tier of the key
    it was published under."""
    var route = TsaHwRoute(device)
    var ops = tsa_ops_resolve(family, mode, vendor)
    var got_vendor = entry_ref_call(ops, TSA_OP_VENDOR, List[Int](), List[Int](), List[Float64]())
    var got_mode = entry_ref_call(ops, TSA_OP_MODE, List[Int](), List[Int](), List[Float64]())
    if got_vendor != vendor:
        raise Error("RuntimeError: Holt-Winters native vendor differs from selected execution")
    if got_mode != mode:
        raise Error("RuntimeError: Holt-Winters native mode differs from selected execution")
    route.flags = entry_ref_call(ops, TSA_OP_FLAGS, List[Int](), List[Int](), List[Float64]())
    route.mode = got_mode
    route.vendor = got_vendor
    route.ops = ops
    return route


@always_inline
def tsa_hw_route_fits(route: TsaHwRoute) -> Bool:
    return (route.flags & TSA_HAS_FIT) != 0


@always_inline
def tsa_hw_route_resident(route: TsaHwRoute) -> Bool:
    return (route.flags & TSA_HAS_RESIDENT) != 0


def tsa_vendor_word(code: Int) -> String:
    """The vendor name of a route's vendor word (`vendor_used`)."""
    if code == 0:
        return String("cpu")
    if code == 1:
        return String("metal")
    if code == 2:
        return String("cuda")
    if code == 3:
        return String("hip")
    return String("none")
