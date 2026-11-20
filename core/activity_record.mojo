# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The activity record: what the runtime did during one verification call
(lane fx-verify-trace, 2026-11-20).

`python -m mojolearn verify` and the user-data check print, per side, a short
"what ran" block. Its numbers come from here: counters the runtime fills
itself, switched on by the verification code before a call and read once
after it (`core/launch_log.mojo` holds the switch and the process-wide slot;
this file is the record alone, so it has no slot and no import of its own).

WHAT IS COUNTED, AND WHERE. There is no single door every device launch or
copy passes (the launches are `enqueue_function` calls in the family files),
so the record counts at the shared doors that exist and says so by field:

  launches, and the named launch rows   the named launch sites of
                                        `core/launch_log.mojo` (`log_launch`,
                                        `log_launch_ctx`): the forest path.
  upload_*, download_*                  the entries of `core/transfer.mojo`.
  host_splits, host_tasks,              `core/host_parallel.mojo`
  host_tasks_widest                     `host_parallelize`, the one host
                                        thread split.
  the named host rows                   the host inference entries that call
                                        `activity_host`.

A count is what passed its door, never an estimate: a family whose device
work passes none of these doors reads zero here, and the printer says "not
counted on this path" for it instead of printing a zero as a fact.

The field names of the transfer counters are the ones
`sequence/execution_profile.mojo` uses for a program session
(`upload_calls`, `upload_bytes`, `download_calls`, `download_bytes`), so the
two read the same in a report.

Every loop here walks the named rows (at most ACTIVITY_MAX_NAMES), never
data, and runs only while a verification asked for the record."""

#: The most distinct names one record keeps; a name past the bound is counted
#: in `names_dropped` (and its launch still in `launches`).
comptime ACTIVITY_MAX_NAMES = 64

#: Row kinds.
comptime ACTIVITY_KIND_LAUNCH = 0
comptime ACTIVITY_KIND_HOST = 1


struct ActivityRecord(Copyable, Defaultable, Movable):
    """Counts since the record was begun. See the module note for the door
    behind each field."""

    var launches: Int
    var upload_calls: Int
    var upload_bytes: Int
    var download_calls: Int
    var download_bytes: Int
    var host_splits: Int
    var host_tasks: Int
    var host_tasks_widest: Int
    var host_entries: Int
    var names: List[String]
    var kinds: List[Int]
    var counts: List[Int]
    var names_dropped: Int

    def __init__(out self):
        self.launches = 0
        self.upload_calls = 0
        self.upload_bytes = 0
        self.download_calls = 0
        self.download_bytes = 0
        self.host_splits = 0
        self.host_tasks = 0
        self.host_tasks_widest = 0
        self.host_entries = 0
        self.names = List[String]()
        self.kinds = List[Int]()
        self.counts = List[Int]()
        self.names_dropped = 0

    def named(mut self, kind: Int, var name: String):
        """One more entry of the row (`kind`, `name`), created on first sight
        while the bound allows."""
        var i = 0
        while i < len(self.names) and not (self.kinds[i] == kind and self.names[i] == name):  # driver-loop: over the record's named rows
            i += 1
        if i == len(self.names):
            if i >= ACTIVITY_MAX_NAMES:
                self.names_dropped += 1
                return
            self.names.append(name^)
            self.kinds.append(kind)
            self.counts.append(0)
        self.counts[i] += 1

    def launch(mut self, var name: String):
        self.launches += 1
        self.named(ACTIVITY_KIND_LAUNCH, name^)

    def host_entry(mut self, var name: String):
        self.host_entries += 1
        self.named(ACTIVITY_KIND_HOST, name^)

    def host_split(mut self, tasks: Int):
        self.host_splits += 1
        self.host_tasks += tasks
        if tasks > self.host_tasks_widest:
            self.host_tasks_widest = tasks

    def uploaded(mut self, size: Int):
        self.upload_calls += 1
        self.upload_bytes += size

    def downloaded(mut self, size: Int):
        self.download_calls += 1
        self.download_bytes += size
