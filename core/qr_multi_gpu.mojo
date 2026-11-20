# SPDX-License-Identifier: Apache-2.0
"""The TSQR panel split over a `DevicePlan` (multi-GPU encapsulation,
2026-11-03, docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md).

`qr_parallel_panels` moved here from `core/householder_qr.mojo` unchanged
except that it takes the plan value instead of reading the thread plan.
`qr_panels_dispatch` is what `qr_factor` calls for its first pass: one owner
runs the unchanged single-device `qr_factor_split`; more owners split the
original shape-defined panels over the devices. The root R-stack fold stays
in `qr_factor` on the caller's context, so the bits of either arm are the
bits the interleaved code produced.
"""

from core.device_plan import DevicePlan, device_plan_current
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.householder_qr import qr_factor_split, qr_split_enqueue
from core.multi_gpu import peer_clone


def qr_plan_current() raises -> DevicePlan:
    """The QR family's plan, read ONCE from the thread plan (or
    MOJOLEARN_QR_DEVICE_COUNT outside one). Only plan-less thin entries call
    this."""
    return device_plan_current("MOJOLEARN_QR_DEVICE_COUNT")


def qr_plan_owner_count(plan: DevicePlan) raises -> Int:
    """The owner count the panel split parsed before the plan value:
    `Int(device_plan_setting("MOJOLEARN_QR_DEVICE_COUNT", "1"))`.
    `plan.single()` is 1; any other setting is parsed exactly as before."""
    if plan.single():
        return 1
    return Int(plan.setting)


def qr_panels_dispatch(plan: DevicePlan, ctx: DeviceContext, mut a: DeviceBuffer[DType.float32],
    mut r_scratch: DeviceBuffer[DType.float32], n_rows: Int, n_cols: Int, ns: Int,
) raises:
    """TSQR's first pass: the `ns` panel R tiles into `r_scratch`. One owner:
    the unchanged single-device `qr_factor_split`; otherwise the panels over
    the plan's devices (`qr_parallel_panels`)."""
    if not qr_parallel_panels(plan,ctx,a,r_scratch,n_rows,n_cols,ns):
        qr_factor_split(ctx, a, r_scratch, n_rows, n_cols, n_cols, ns)


def qr_parallel_panels(plan: DevicePlan, ctx: DeviceContext, mut a: DeviceBuffer[DType.float32],
    mut r_scratch: DeviceBuffer[DType.float32], m: Int,n: Int,ns: Int,
) raises -> Bool:
    """Original shape-defined TSQR panels; no change to the root R-stack fold.

    Every panel keeps exactly its original row interval and QR_TPB lanes.
    Both its destroyed input and R tile return to their original addresses.
    Only row-panel memory is held on workers; the full root matrix remains.
    """
    var count = qr_plan_owner_count(plan)
    if count == 1:
        return False
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel QR requires IDENTICAL and 1..64 devices")
    count = min(count,ns)
    ctx.synchronize()
    var devices = List[DeviceContext]()
    for rank in range(count):
        devices.append(DeviceContext(device_id=plan.device(rank)))
    var panels = List[DeviceBuffer[DType.float32]]()
    var outputs = List[DeviceBuffer[DType.float32]]()
    for panel in range(ns):
        var rank = panel%count
        var first = panel*m//ns
        var rows = (panel+1)*m//ns-first
        var source = first
        comptime if is_defined["MOJOLEARN_QR_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their panel one row early. The
            # length `rows*n` is unchanged, the kernel scalars are re-derived in
            # the launch loop from the true `first`, and both write-backs keep
            # the true offsets. INERT AT ONE DEVICE: `count == 1` returns before
            # this loop, and `panel % 1` is 0 for every panel.
            if rank > 0:
                source = first - 1
        var view = a.create_sub_buffer[DType.float32](source*n,rows*n)
        panels.append(peer_clone(ctx,devices[rank],view))
        outputs.append(devices[rank].enqueue_create_buffer[DType.float32](n*n))
    for rank in range(count):
        devices[rank].synchronize()
    # each panel as the split launches on its device (trailing columns over
    # blocks), enqueued on every device before any wait
    var taus = List[DeviceBuffer[DType.float32]]()
    for panel in range(ns):
        var rank = panel%count
        taus.append(devices[rank].enqueue_create_buffer[DType.float32](1))
    for panel in range(ns):
        var rank = panel%count
        var first = panel*m//ns
        var rows = (panel+1)*m//ns-first
        qr_split_enqueue(
            devices[rank], panels[panel].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            outputs[panel].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            taus[panel].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), rows, n, n, 1,
        )
    for panel in range(ns):
        var rank = panel%count
        devices[rank].synchronize()
        var first = panel*m//ns
        var rows = (panel+1)*m//ns-first
        var av = a.create_sub_buffer[DType.float32](first*n,rows*n)
        var rv = r_scratch.create_sub_buffer[DType.float32](panel*n*n,n*n)
        panels[panel].enqueue_copy_to(av)
        outputs[panel].enqueue_copy_to(rv)
        devices[rank].synchronize()
    _ = taus^
    _ = outputs^
    _ = panels^
    for rank in range(count):
        devices[rank].synchronize()
    _ = devices^
    ctx.synchronize()
    return True
