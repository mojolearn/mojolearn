# SPDX-License-Identifier: Apache-2.0
"""The x_prep program transport ABI shared by its publishers and its native
callers (lane calltable-prep-decomp, 2026-11-11).

One `abi("C")` entry per x_prep image (`x_prep/transport_device.mojo`,
`x_prep/transport_host.mojo`, published in `core/prep_entry_table.mojo`)
runs every transport operation below on plain words: the bodies of the
Python entries `x_prep_run`, `x_prep_run_scratch`, `x_prep_run_out`,
`x_prep_run_ranges`, `x_prep_dev_put`, `x_prep_dev_free` and
`x_prep_dev_take_rows` with the same checks and the same typed calls.
The caller (`bindings/x_prep_native_owner.mojo`) owns every buffer until the
synchronous call returns and releases the GIL around it, as the entries did.
"""

comptime XP_TRANSPORT_ABI = "mojolearn.x_prep.transport.v1"

comptime XP_RUN = 0            # a: [arena, prog]                  i: [arena_len, stages]
comptime XP_RUN_SCRATCH = 1    # a: [arena, prog]                  i: [arena_len, scratch_len, stages]
comptime XP_RUN_OUT = 2        # a: [arena, prog, out]             i: [arena_len, scratch_len, out_len, stages]
comptime XP_RUN_RANGES = 3     # a: [arena, prog, out, ins, outs]  i: [arena_len, scratch_len, out_len, stages, nins, nouts]
comptime XP_DEV_PUT = 4        # a: [src]                          i: [n_words]            -> slot id
comptime XP_DEV_FREE = 5       # i: [id]
comptime XP_DEV_TAKE_ROWS = 6  # a: [idx]                          i: [src_id, row_words, n_idx]  -> slot id
comptime XP_OP_COUNT = 7


def xp_schema(op: Int) raises -> Tuple[Int, Int]:
    """(addresses, integers) each operation takes, exactly."""
    if op == XP_RUN:
        return (2, 2)
    if op == XP_RUN_SCRATCH:
        return (2, 3)
    if op == XP_RUN_OUT:
        return (3, 4)
    if op == XP_RUN_RANGES:
        return (5, 6)
    if op == XP_DEV_PUT:
        return (1, 1)
    if op == XP_DEV_FREE:
        return (0, 1)
    if op == XP_DEV_TAKE_ROWS:
        return (1, 3)
    raise Error("x_prep transport ABI: unknown operation " + String(op))


def xp_validate_call(op: Int, na: Int, ni: Int, nf: Int, result: Int) raises:
    var want = xp_schema(op)
    if na != want[0] or ni != want[1] or nf != 0:
        raise Error("x_prep transport ABI: operation " + String(op) + " takes " + String(want[0])
                    + " addresses, " + String(want[1]) + " integers and no floats")
    if result == 0:
        raise Error("x_prep transport ABI: null result address")


def xp_vendor_word(vendor: String) -> Int:
    """The table key's vendor word for an x_prep image's `x_prep_vendor`
    string: 0 cpu, 1 metal, 2 cuda, 3 hip, else -1 (nothing published)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1
