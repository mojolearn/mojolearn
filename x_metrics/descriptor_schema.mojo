# SPDX-License-Identifier: Apache-2.0
"""Native dispatch for the existing metrics program and range I/O routes.

v2 (lane vdf-xm-transport, 2026-11-24): the image's resident input store
(`x_metrics/device.mojo` `X_METRICS_STORE`, a `core/device_store.mojo`
`DeviceStore` on the image's one device context) is reached through the same
entry: put, free, gather rows and the live count, the operations of the
`x_metrics_dev_put/_free/_take_rows/_live` Python entries deleted as dead on
2026-10-26 (c209716bf), now typed. A slot is consumed by `MP_RUN_RANGES`
(an input triple [lo, hi, slot], core/arena_io.mojo `upload_ranges`: copied
device to device). The two scans (vdf-xm-labels' core/metric_labels_device.mojo)
are the door refusals' finiteness and sign checks of a resident input, run
on the slot in place. The host image publishes no store (`MP_HAS_STORE` clear)
and refuses the store operations."""
comptime METRICS_PROGRAM_ABI = "mojolearn.metrics.program.v2"
comptime MP_MODE = 0
comptime MP_VENDOR = 1
comptime MP_CAPABILITIES = 2
comptime MP_RUN = 3
comptime MP_RUN_OUT = 4
comptime MP_RUN_RANGES = 5
comptime MP_DEV_PUT = 6         # a: [src]  i: [n_words]                  -> slot id
comptime MP_DEV_FREE = 7        # i: [slot]
comptime MP_DEV_TAKE_ROWS = 8   # a: [idx]  i: [src_slot, row_words, n_idx] -> slot id
comptime MP_DEV_LIVE = 9        # (none)                                  -> live slots of the store
comptime MP_DEV_FIRST_NONFINITE = 10  # i: [slot, words] -> first NaN/inf index among the slot's words, or -1
comptime MP_DEV_FIRST_NEGATIVE = 11   # i: [slot, words] -> first negative word's index, or -1
comptime MP_HAS_RANGES = 1
comptime MP_HAS_STORE = 2
comptime MP_OPERATION_COUNT = 12


def metrics_check_spans(op: Int, aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int) raises:
    var addresses = 0
    var ints = 0
    if op == MP_MODE or op == MP_VENDOR or op == MP_CAPABILITIES or op == MP_DEV_LIVE:
        pass
    elif op == MP_RUN:
        addresses = 2
        ints = 2
    elif op == MP_RUN_OUT:
        addresses = 3
        ints = 3
    elif op == MP_RUN_RANGES:
        addresses = 4
        ints = 4
    elif op == MP_DEV_PUT:
        addresses = 1
        ints = 1
    elif op == MP_DEV_FREE:
        ints = 1
    elif op == MP_DEV_TAKE_ROWS:
        addresses = 1
        ints = 3
    elif op == MP_DEV_FIRST_NONFINITE or op == MP_DEV_FIRST_NEGATIVE:
        ints = 2
    else:
        raise Error("x_metrics native ABI: unknown operation")
    if na != addresses or np != ints or nf != 0 or not result:
        raise Error("x_metrics native ABI: invalid request spans")
    if (na and not aa) or (np and not pp):
        raise Error("x_metrics native ABI: null request spans")
    var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
    var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
    if op == MP_DEV_PUT:
        if not a[0] or p[0] < 1:
            raise Error("x_metrics: invalid resident input buffer")
        return
    if op == MP_DEV_FREE:
        if p[0] < 0:
            raise Error("x_metrics: invalid resident slot")
        return
    if op == MP_DEV_TAKE_ROWS:
        if not a[0] or p[0] < 0 or p[1] < 1 or p[2] < 1:
            raise Error("x_metrics: invalid row selection")
        return
    if op == MP_DEV_FIRST_NONFINITE or op == MP_DEV_FIRST_NEGATIVE:
        if p[0] < 0 or p[1] < 0:
            raise Error("x_metrics: invalid resident scan")
        return
    if addresses:
        if not a[0] or not a[1] or p[0] < 0 or p[1] < 0:
            raise Error("x_metrics: invalid program buffers")
        if op == MP_RUN_OUT and (not a[2] or p[2] < 0):
            raise Error("x_metrics: invalid program buffers")
        if op == MP_RUN_RANGES and (p[2] < 0 or p[3] < 0):
            raise Error("x_metrics: invalid program buffers")
