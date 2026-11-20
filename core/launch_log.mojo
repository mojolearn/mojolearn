# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The launch log: kernel names in enqueue order, for trace attribution.

WHY THIS EXISTS. Apple's stock 'Metal System Trace' template records the
Shader Timeline instrument Disabled, `xctrace` exposes no way to enable it
from the command line (the `--instrument 'GPU'` spelling records nothing
into `gpu-shader-profiler-interval` either -- measured 2026-08-22), and
MAX exposes no Metal debug-label API. So a trace knows every dispatch's
DURATION but not its NAME: `metal-gpu-intervals` labels rows by encoder,
and this implementation issues one unnamed encoder per launch.

What the trace cannot name, the host can: Metal's compute channel on this
device is a single in-order queue, so DEVICE EXECUTION ORDER IS HOST
ENQUEUE ORDER. Every `enqueue_function` site in the forest path calls
`log_launch("<site name>")` first; with `RF_LAUNCH_LOG=<path>` set, the
names land in that file one per line, and joining line `i` to the trace's
i-th Compute interval (sorted by start) attributes every nanosecond to a
kernel site. `ensemble/bench/profile_metal.py` performs the join and
REFUSES it unless the two counts match exactly -- a misaligned join
mislabels every row after the first slip, so an inexact one is worthless.

THE ACTIVITY RECORD SHARES THE LATCH (lane fx-verify-trace, 2026-11-20).
`python -m mojolearn verify` asks the runtime what it did during a call
(`activity_begin` / `activity_end` below, `core/activity_record.mojo`). The
named launch sites already read this slot once per launch, so the record's
switch lives in the same state and a site still tests one Bool (`watch`).

DISABLED IS THE SHIPPING STATE. DEVIATION 1895: the env read happens
ONCE per process, not once per launch. This used to call
`String(getenv("RF_LAUNCH_LOG"))` on EVERY launch (~9.4k per benchmark
fit) -- a libc environ walk plus a `String` construction on the enqueue
path -- so the disabled path was priced into every measured number. The
path is now read on first use into a process global (the stdlib's
`_Global`, the same mechanism `std.python` keeps its interpreter handle
in) and every later call is one pointer load and one Bool test. The
variable is therefore LATCHED at first launch: flipping RF_LAUNCH_LOG
mid-process no longer changes behavior, which no harness does -- the
profile scripts export it before the run. Enabled runs are TRACE
SUBJECTS, not measurements: an append-per-launch `open()` is a syscall
on the enqueue path and quiet_window would rightly refuse to certify
one.
"""

from std.os import getenv
from core.activity_record import ActivityRecord
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock


struct _LaunchLogState(Defaultable, Movable):
    """The once-read `RF_LAUNCH_LOG` value (DEVIATION 1895) and the activity
    record's switch (lane fx-verify-trace, 2026-11-20). ONE slot for both, so
    a named launch site still reads one pointer and tests one Bool (`watch`)
    when neither is on."""

    var path: String
    var enabled: Bool
    var clock: Bool
    """`RF_LAUNCH_CLOCK=1` beside `RF_LAUNCH_LOG`: `log_launch_ctx` drains
    the queue at every named site and appends `<previous name>\t<ns>`, so
    each line prices the previous site's device work plus the host work up
    to this site. A drain per launch is a DIFFERENT PROGRAM (nothing
    overlaps); read it as attribution, never as a timing."""
    var last_name: String
    var last_ns: Int
    var watch: Bool
    """`enabled or counting`: the one test of a named launch site."""
    var counting: Bool
    """A verification asked for the activity record (`activity_begin`); off
    in every process that did not."""
    var lock: List[Int64]
    """One lock word (`shared_slot_lock`) around every write of `record`:
    the parallel drivers launch from several host threads."""
    var record: ActivityRecord

    def __init__(out self):
        self.path = String(getenv("RF_LAUNCH_LOG"))
        self.enabled = self.path != ""
        self.clock = self.enabled and getenv("RF_LAUNCH_CLOCK") == "1"
        self.last_name = String("")
        self.last_ns = 0
        self.watch = self.enabled
        self.counting = False
        self.lock = List[Int64](length=1, fill=Int64(0))
        self.record = ActivityRecord()


# V2 (lane fx-verify-trace, 2026-11-20): the activity record's fields joined
# the state, so the name and the layout version moved together.
comptime _LAUNCH_LOG = SharedSlot[_LaunchLogState, "MojolearnRFLaunchLogV2", 2]


@always_inline
def _activity_lock_word(
    st: UnsafePointer[_LaunchLogState, MutUntrackedOrigin],
) -> MutPointer[Int64, MutUntrackedOrigin]:
    return rebind[MutPointer[Int64, MutUntrackedOrigin]](st[].lock.unsafe_ptr())


def log_launch(name: StringSlice) raises:
    """Count `name` in the activity record when a verification asked for it,
    and append it to `$RF_LAUNCH_LOG` if that was set at first use; one
    pointer load and one Bool test when neither is on."""
    var st = _LAUNCH_LOG.get_or_create_ptr()
    if not st[].watch:
        return
    if st[].counting:
        var lock = _activity_lock_word(st)
        shared_slot_lock(lock)
        st[].record.launch(String(name))
        shared_slot_unlock(lock)
    if st[].enabled:
        with open(st[].path, "a") as fh:
            fh.write(String(name) + "\n")


# ---------------------------------------------------------------------------
# The activity record (lane fx-verify-trace, 2026-11-20): `core/activity_record
# .mojo` says what is counted at which door. OFF IN EVERY PROCESS THAT DID NOT
# ASK: each entry below reads the slot and tests `counting`, and returns. None
# of them sits in a loop over rows or elements: a host split, a transfer of one
# operand and a host inference entry each happen once per call or per stage.
# ---------------------------------------------------------------------------


def activity_begin() raises:
    """Clear the record and start counting. Called by the verification code
    (bindings/_mojolearn_core_host.mojo `activity_begin`), never by a product
    call."""
    var st = _LAUNCH_LOG.get_or_create_ptr()
    var lock = _activity_lock_word(st)
    shared_slot_lock(lock)
    st[].record = ActivityRecord()
    st[].counting = True
    st[].watch = True
    shared_slot_unlock(lock)


def activity_end() raises -> ActivityRecord:
    """Stop counting and return what was counted since `activity_begin` (an
    empty record when nothing was begun)."""
    var st = _LAUNCH_LOG.get_or_create_ptr()
    var lock = _activity_lock_word(st)
    shared_slot_lock(lock)
    st[].counting = False
    st[].watch = st[].enabled
    var record = st[].record.copy()
    st[].record = ActivityRecord()
    shared_slot_unlock(lock)
    return record^


def activity_uploaded(size: Int) raises:
    """One host-to-device copy of `size` bytes through a shared transfer
    entry."""
    var st = _LAUNCH_LOG.get_or_create_ptr()
    if not st[].counting:
        return
    var lock = _activity_lock_word(st)
    shared_slot_lock(lock)
    st[].record.uploaded(size)
    shared_slot_unlock(lock)


def activity_downloaded(size: Int) raises:
    """One device-to-host copy of `size` bytes through a shared transfer
    entry."""
    var st = _LAUNCH_LOG.get_or_create_ptr()
    if not st[].counting:
        return
    var lock = _activity_lock_word(st)
    shared_slot_lock(lock)
    st[].record.downloaded(size)
    shared_slot_unlock(lock)


def activity_host_split(tasks: Int):
    """One host thread split into `tasks` tasks (`core/host_parallel.mojo`).
    Does not raise: the split it sits in does not."""
    try:
        var st = _LAUNCH_LOG.get_or_create_ptr()
        if not st[].counting:
            return
        var lock = _activity_lock_word(st)
        shared_slot_lock(lock)
        st[].record.host_split(tasks)
        shared_slot_unlock(lock)
    except:
        return


def activity_host(name: StringSlice):
    """One entry of the named host routine. Does not raise."""
    try:
        var st = _LAUNCH_LOG.get_or_create_ptr()
        if not st[].counting:
            return
        var lock = _activity_lock_word(st)
        shared_slot_lock(lock)
        st[].record.host_entry(String(name))
        shared_slot_unlock(lock)
    except:
        return
