# SPDX-License-Identifier: Apache-2.0
"""Native dispatch for the existing metrics program and range I/O routes.

v3 (lane vdf-xm-wire, 2026-11-24): ONE store for resident metric inputs, the
x_prep image's (x_prep/store_slot.mojo `X_PREP_STORE` on `x_prep_ctx`, which
is also this image's context, x_metrics/device.mojo `metrics_ctx`). A runner
puts, gathers and frees through the x_prep transport entry
(x_prep/transport_abi.mojo `XP_DEV_*`); the v2 store operations of this entry
(`MP_DEV_PUT/_FREE/_TAKE_ROWS/_LIVE` on a second, x_metrics-only store, lane
vdf-xm-transport) are deleted. A slot is consumed by `MP_RUN_RANGES` (an input
triple [lo, hi, slot], core/arena_io.mojo `upload_ranges`: copied device to
device). `MP_HAS_STORE` says the image reads that store. The two scans
(core/metric_labels_device.mojo) are the door refusals' finiteness and sign
checks of a resident input, run on the slot in place. The host image publishes
no store (`MP_HAS_STORE` clear) and refuses the scans."""
comptime METRICS_PROGRAM_ABI = "mojolearn.metrics.program.v3"
comptime MP_MODE = 0
comptime MP_VENDOR = 1
comptime MP_CAPABILITIES = 2
comptime MP_RUN = 3
comptime MP_RUN_OUT = 4
comptime MP_RUN_RANGES = 5
comptime MP_DEV_FIRST_NONFINITE = 6  # i: [slot, words] -> first NaN/inf index among the slot's words, or -1
comptime MP_DEV_FIRST_NEGATIVE = 7   # i: [slot, words] -> first negative word's index, or -1
comptime MP_HAS_RANGES = 1
comptime MP_HAS_STORE = 2
comptime MP_OPERATION_COUNT = 8


def metrics_check_spans(op: Int, aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int) raises:
    var addresses = 0
    var ints = 0
    if op == MP_MODE or op == MP_VENDOR or op == MP_CAPABILITIES:
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
