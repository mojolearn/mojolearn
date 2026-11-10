# SPDX-License-Identifier: Apache-2.0
"""Native dispatch for the existing metrics program and range I/O routes."""
comptime METRICS_PROGRAM_ABI = "mojolearn.metrics.program.v1"
comptime MP_MODE = 0
comptime MP_VENDOR = 1
comptime MP_CAPABILITIES = 2
comptime MP_RUN = 3
comptime MP_RUN_OUT = 4
comptime MP_RUN_RANGES = 5
comptime MP_HAS_RANGES = 1
comptime MP_OPERATION_COUNT = 6


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
    else:
        raise Error("x_metrics native ABI: unknown operation")
    if na != addresses or np != ints or nf != 0 or not result:
        raise Error("x_metrics native ABI: invalid request spans")
    if (na and not aa) or (np and not pp):
        raise Error("x_metrics native ABI: null request spans")
    if addresses:
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        if not a[0] or not a[1] or p[0] < 0 or p[1] < 0:
            raise Error("x_metrics: invalid program buffers")
        if op == MP_RUN_OUT and (not a[2] or p[2] < 0):
            raise Error("x_metrics: invalid program buffers")
        if op == MP_RUN_RANGES and (p[2] < 0 or p[3] < 0):
            raise Error("x_metrics: invalid program buffers")
