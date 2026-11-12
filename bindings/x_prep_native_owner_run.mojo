# SPDX-License-Identifier: Apache-2.0
"""RUN of the x_prep owner (bindings/x_prep_native_owner.mojo): the R3
transport switch read. No Python import (tools/python_free_dirs.sh). Lane
rs-shared-infra-b (2026-11-12): moved word for word. The owner's output-range
merge (`_xprep_merge`) was a copy of `_merge_pairs` kept so the owner did not
import the arena runner; it now calls `_merge_pairs` in the Python-free
bindings/arena_prog_run.mojo."""
from std.os import getenv


def xprep_r3(name: String) -> Bool:
    # These three existing R3 choices default on; OFF wins over ON.
    # p3-array-arena: the switch stays a live read (the transport tests and
    # A/B jobs flip it inside one process), but the usual unset case returns
    # before any parsing or allocation.
    var raw = String(getenv("MOJOLEARN_XPREP_R3_OFF",""))
    if raw.byte_length() == 0:
        return True
    var names = raw.replace("+",",").split(",")
    for i in range(len(names)):  # driver-loop: configured transport-switch names
        if String(names[i]).strip() == name or String(names[i]).strip() == "all":
            return False
    return True


# ---------------------------------------------------------------- typed run (lane encprog-typed, 2026-11-12)
# The body of the owner's program run (bindings/x_prep_native_owner_door.mojo
# `xprep_run`, the transport of every x_prep program the stateless, encoder,
# imputer, label and selector owners build) on typed input records. The door
# resolves the Python-held parts once per run (UNPACK: each input's buffer
# address, its resident cache slot, its DeviceRows fields, the selected
# image's route words) and calls `xprep_execute` with the GIL released. The
# input copies, puts, row gathers, range descriptors and the run call are the
# ones `xprep_run` made in the same order per input: no bit moves.
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef
from core.prep_entry_table import xp_transport_call
from x_prep.transport_abi import XP_RUN, XP_RUN_SCRATCH, XP_RUN_OUT, XP_RUN_RANGES, XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS
from bindings.arena_prog_run import _merge_pairs

comptime XPREP_F32P = MutPointer[Float32, MutUntrackedOrigin]


@fieldwise_init
struct XprepRunInput(ImplicitlyCopyable, Movable):
    """One input as the door resolved it: `words` arena words at `offset`.
    `address`: the host words (or a DeviceRows base) the door keeps alive.
    `source`: a resident slot the door's cache holds (-1: none). `put`: put
    `put_words` words from `address` into a slot this run owns. `rows`: a
    DeviceRows selection (`n_rows` rows of `cols` words, int64 indices at
    `indices`) taken on the device out of the base slot."""
    var offset: Int
    var words: Int
    var address: Int
    var source: Int
    var put: Bool
    var put_words: Int
    var rows: Bool
    var indices: Int
    var n_rows: Int
    var cols: Int


@fieldwise_init
struct XprepRoute(ImplicitlyCopyable, Movable):
    """The run's route words, read by the door from the selected image:
    the range runner, device scratch, the `x_prep_run_out` entry."""
    var ranges: Bool
    var device_scratch: Bool
    var has_out: Bool


def xprep_execute(key: EntryRef, arena: Int, host: Int, scratch: Int, output: Int, out_words: Int,
                  program: List[Int32], inputs: List[XprepRunInput], inout_ranges: List[Int], route: XprepRoute,
                  mut owned: List[Int]) raises:
    """RUN: the inputs into the arena (or onto the device), then one program
    run. Every slot this run acquires is appended to `owned` (the door frees
    them with `xprep_release`). The door keeps every addressed buffer alive
    and has released the GIL."""
    var ins = List[Int32]()
    var outs = List[Int32]()
    var last = 0
    for k in range(len(inputs)):  # driver-loop: over the program's input records
        var r = inputs[k]
        var offset = r.offset
        var words = r.words
        var source = r.source
        if r.put:
            var put_src: List[Int] = [r.address]
            source = xp_transport_call(key, XP_DEV_PUT, put_src, [r.put_words])
            owned.append(source)
        if r.rows:
            var idx: List[Int] = [r.indices]
            source = xp_transport_call(key, XP_DEV_TAKE_ROWS, idx, [source, r.cols, r.n_rows])
            owned.append(source)
        if source < 0:
            unsafe_memcpy(dest=XPREP_F32P(unsafe_from_address=arena + 4 * offset),
                          src=XPREP_F32P(unsafe_from_address=r.address), count=words)
        if route.ranges:
            if source < 0 and len(ins) >= 3 and ins[len(ins) - 1] < 0 and Int(ins[len(ins) - 2]) == offset:
                ins[len(ins) - 2] = Int32(offset + words)
            else:
                ins.append(Int32(offset))
                ins.append(Int32(offset + words))
                ins.append(Int32(source))
            if offset > last:
                outs.append(Int32(last))
                outs.append(Int32(offset))
                outs.append(-1)
                outs.append(1)
            last = offset + words
    if route.ranges and last < host:
        outs.append(Int32(last))
        outs.append(Int32(host))
        outs.append(-1)
        outs.append(1)
    if route.ranges and len(inout_ranges) > 0:
        # the in-out input ranges come back too
        var pairs = List[Int]()
        for i in range(len(outs) // 4):  # driver-loop: over the program's output ranges
            pairs.append(Int(outs[4 * i]))
            pairs.append(Int(outs[4 * i + 1]))
        for j in range(len(inout_ranges)):  # driver-loop: over the program's in-out input ranges
            pairs.append(inout_ranges[j])
        outs = _merge_pairs(pairs^)
    var nst = len(program) // 16
    var pa = Int(program.unsafe_ptr())
    var sc = scratch if route.device_scratch else 0
    if route.ranges:
        var addresses: List[Int] = [arena, pa, output, Int(ins.unsafe_ptr()), Int(outs.unsafe_ptr())]
        _ = xp_transport_call(key, XP_RUN_RANGES, addresses, [host, sc, out_words, nst, len(ins) // 3, len(outs) // 4])
    elif out_words > 0 or (route.device_scratch and route.has_out):
        var addresses: List[Int] = [arena, pa, output]
        _ = xp_transport_call(key, XP_RUN_OUT, addresses, [host, sc, out_words, nst])
    elif route.device_scratch:
        var addresses: List[Int] = [arena, pa]
        _ = xp_transport_call(key, XP_RUN_SCRATCH, addresses, [host, scratch, nst])
    else:
        var addresses: List[Int] = [arena, pa]
        _ = xp_transport_call(key, XP_RUN, addresses, [host + scratch, nst])
    # The address words above are not owners: keep both descriptor lists
    # live until the native call returns.
    _ = ins^
    _ = outs^


def xprep_release(key: EntryRef, owned: List[Int]) -> String:
    """Free every slot one run acquired, newest first, even when one fails;
    the failures as text (empty on success)."""
    var failures = String()
    for i in range(len(owned)):  # driver-loop: attempt every gather/direct release even when an earlier free raises
        var slot = owned[len(owned) - 1 - i]
        try:
            _ = xp_transport_call(key, XP_DEV_FREE, List[Int](), [slot])
        except error:
            failures += ("" if failures.byte_length() == 0 else "; ") + "slot " + String(slot) + ": " + String(error)
    return failures
