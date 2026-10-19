# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A LAUNCH PROFILE for the autograd tape (lane ag-pass2-kernels, 2026-10-19).

MOJOLEARN_AG_PROFILE=1 makes every `DeviceExec` / `HostExec` built while it
is set count its launches per operation code (and the elements launched);
=2 also times each launch on the host clock between an `ex.sync()` before
and after it. The syncs serialise the queue, so level-2 times are a
RELATIVE profile (which launches dominate), never a speed measurement.
The tape marks the pending launches with the node kind that issued them
(autograd/tape.mojo `_record`, the backward entries), so the table also
has rows per (phase, node kind).

The level is read once per executor and once per tape (an environment read
at construction, API glue, no data). Off (the default), a launch costs one
Int test. The table is process-wide (`std.ffi._Global`) so a report can be
read after a fit whose tape and executor are gone (autograd/profile.mojo
`ag_profile_report`). Every loop here walks profile rows (distinct codes
and kinds, a few dozen), never data."""
from std.ffi import _Global
from std.os import getenv

comptime PROFILE_FORWARD = 0
comptime PROFILE_BACKWARD = 1
comptime PROFILE_OTHER = 2
#: a mark that names no single node kind (a whole backward, the launches
#: between graphs: optimizer steps, loss reductions, metric folds)
comptime PROFILE_KIND_ALL = -1


def launch_profile_level() -> Int:
    """0 (off), 1 (count) or 2 (count and time) from MOJOLEARN_AG_PROFILE."""
    var v = String(getenv("MOJOLEARN_AG_PROFILE"))
    if v == "1":
        return 1
    if v == "2":
        return 2
    return 0


struct LaunchProfile(Copyable, Defaultable, Movable):
    """Per operation code: launches, elements, nanoseconds (level 2). Per
    (phase, node kind): launches and nanoseconds. `pending_*` holds the
    launches since the last mark."""
    var ops: List[Int]
    var op_launches: List[Int]
    var op_elements: List[Int]
    var op_ns: List[Int]
    var row_phase: List[Int]
    var row_kind: List[Int]
    var row_launches: List[Int]
    var row_ns: List[Int]
    var pending_launches: Int
    var pending_ns: Int
    var level: Int

    def __init__(out self):
        self.ops = List[Int]()
        self.op_launches = List[Int]()
        self.op_elements = List[Int]()
        self.op_ns = List[Int]()
        self.row_phase = List[Int]()
        self.row_kind = List[Int]()
        self.row_launches = List[Int]()
        self.row_ns = List[Int]()
        self.pending_launches = 0
        self.pending_ns = 0
        self.level = 0

    def record(mut self, op: Int, n: Int, ns: Int, level: Int):
        var i = 0
        while i < len(self.ops) and self.ops[i] != op:  # driver-loop: over profiled operation codes
            i += 1
        if i == len(self.ops):
            self.ops.append(op)
            self.op_launches.append(0)
            self.op_elements.append(0)
            self.op_ns.append(0)
        self.op_launches[i] += 1
        self.op_elements[i] += n
        self.op_ns[i] += ns
        self.pending_launches += 1
        self.pending_ns += ns
        if level > self.level:
            self.level = level

    def mark(mut self, phase: Int, kind: Int):
        if self.pending_launches == 0:
            return
        var i = 0
        while i < len(self.row_kind) and not (self.row_kind[i] == kind and self.row_phase[i] == phase):  # driver-loop: over profile rows
            i += 1
        if i == len(self.row_kind):
            self.row_phase.append(phase)
            self.row_kind.append(kind)
            self.row_launches.append(0)
            self.row_ns.append(0)
        self.row_launches[i] += self.pending_launches
        self.row_ns[i] += self.pending_ns
        self.pending_launches = 0
        self.pending_ns = 0


comptime X_LAUNCH_PROFILE = _Global[StorageType=LaunchProfile, name="MojolearnLaunchProfileV1", init_fn=LaunchProfile.__init__]


def launch_profile_record(op: Int, n: Int, ns: Int, level: Int) raises:
    """One launch of `op` over `n` elements (`ns` host nanoseconds at level 2)."""
    var p = X_LAUNCH_PROFILE.get_or_create_ptr()
    p[].record(op, n, ns, level)


def launch_profile_mark(phase: Int, kind: Int) raises:
    """Attribute the launches since the last mark to (phase, kind)."""
    var p = X_LAUNCH_PROFILE.get_or_create_ptr()
    p[].mark(phase, kind)


def launch_profile_snapshot() raises -> LaunchProfile:
    """A copy of the process-wide table (pending launches still pending)."""
    var p = X_LAUNCH_PROFILE.get_or_create_ptr()
    return p[].copy()


def launch_profile_reset() raises:
    """Clear the process-wide table."""
    var p = X_LAUNCH_PROFILE.get_or_create_ptr()
    p[] = LaunchProfile()
