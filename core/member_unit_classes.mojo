# SPDX-License-Identifier: Apache-2.0
"""THE UNIT CLASSES of a classifier member set (lane mf-del8, 2026-11-23): the
classes a unit's own rows hold, and its codes re-coded onto them.

WHAT THIS REPLACES. A meta-estimator's classifier unit (a stacking cv fold, a
bagging draw) whose rows lack a class of the ensemble used to fail the native
set (core/member_set_run.mojo, the class-cover pass, MEMBER_SET_LACKING) and
was then fitted by its member's Python fit door on the classes it was given,
as scikit-learn fits it. Here the caller reads each distinct unit's classes
ONCE (this pass is the cover pass; the caller sets `skip_lacking`, so the set
makes no second one), re-codes a lacking unit's codes onto the classes it
holds (fold code c = ensemble code `held[c]`), fits the member with that
class count and packs it labelled with the ensemble codes it holds: the
`classes_` its fit door wrote. A unit that holds one class is fitted on one
class: its family's fit decides, as the estimator's own fit does (the
calibrated RUN's rule, xtrees/trees_calibrated_fit_run.mojo).

No Python. The rows are split across the host cores (one mark block per task,
merged in class order), the re-code too. Bits: the member fits the same rows
with the same codes its fit door encoded, so its model is the door's.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime MEMBER_UNIT_OUTSIDE = (
    "RuntimeError: a classifier unit holds a code outside 0 .. classes-1 (invariant: the meta-estimator"
    " encoded every code itself)"
)


@always_inline
def _unit_row(draw: Int, i: Int, wide: Bool) -> Int:
    if draw == 0:
        return i
    if wide:
        return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=draw)[i])
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=draw)[i])


def member_unit_classes(codes: Int, draw: Int, wide: Bool, count: Int, k: Int) raises -> List[Int32]:
    """The ensemble codes the unit's rows hold, ascending, when they are not
    all of 0 .. k-1; EMPTY when the unit holds every class (or has no rows:
    its member's fit decides on them). `codes`: the int32 codes address;
    `draw`: 0 (rows 0 .. count-1) or the address of `count` row numbers
    (int64 when `wide`, else int32). Raises MEMBER_UNIT_OUTSIDE for a code
    outside 0 .. k-1 (an invariant: the caller encoded the codes)."""
    var held = List[Int32]()
    if count <= 0 or k <= 0:
        return held^
    var tasks = max(host_predict_task_count_work(count, count), 1)
    var chunk = host_predict_chunk(count, tasks)
    var seen = List[UInt8](length=tasks * k, fill=UInt8(0))
    var outside = List[UInt8](length=tasks, fill=UInt8(0))
    var seen_at = Int(seen.unsafe_ptr())
    var outside_at = Int(outside.unsafe_ptr())

    def task(t: Int) {imm codes, imm draw, imm wide, imm count, imm k, imm chunk, imm seen_at, imm outside_at}:
        var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
        var marks = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=seen_at + t * k)
        var begin = t * chunk
        var end = min(begin + chunk, count)
        for i in range(begin, end):
            var c = Int(y[_unit_row(draw, i, wide)])
            if c < 0 or c >= k:
                MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=outside_at)[t] = UInt8(1)
            else:
                marks[c] = UInt8(1)

    host_parallelize(task, tasks)
    for t in range(tasks):  # driver-loop: over the tasks (their out-of-range flags)
        if outside[t] != UInt8(0):
            raise Error(String(MEMBER_UNIT_OUTSIDE))
    for c in range(k):  # driver-loop: over the classes (each task's mark)
        var marked = UInt8(0)
        for t in range(tasks):  # driver-loop: over the tasks
            marked = marked | seen[t * k + c]
        if marked != UInt8(0):
            held.append(Int32(c))
    # Integer addresses carry no lifetime: the mark blocks live to here.
    _ = len(seen)
    _ = len(outside)
    if len(held) == k:
        return List[Int32]()
    return held^


def member_unit_recode(codes: Int, n: Int, k: Int, held: List[Int32]) -> List[Int32]:
    """The `n` codes at `codes` re-coded onto `held` (ensemble code
    `held[c]` -> c), every row (a draw then selects the rows it names, as it
    selects the original codes; a row whose code is not held is read by no
    draw of this unit and maps to 0). The rows split across the host cores."""
    var out = List[Int32]()
    if n <= 0:
        return out^
    var table = List[Int32](length=max(k, 1), fill=Int32(0))
    for c in range(len(held)):  # driver-loop: over the held classes (the code table)
        var code = Int(held[c])
        if code >= 0 and code < k:
            table[code] = Int32(c)
    out = List[Int32](length=n, fill=Int32(0))
    var dst = Int(out.unsafe_ptr())
    var table_at = Int(table.unsafe_ptr())
    var tasks = max(host_predict_task_count_work(n, n), 1)
    var chunk = host_predict_chunk(n, tasks)

    def task(t: Int) {imm codes, imm dst, imm table_at, imm chunk, imm n, imm k}:
        var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
        var target = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst)
        var lookup = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=table_at)
        var begin = t * chunk
        var end = min(begin + chunk, n)
        for i in range(begin, end):
            var c = Int(y[i])
            target[i] = lookup[c] if (c >= 0 and c < k) else Int32(0)

    host_parallelize(task, tasks)
    _ = len(table)  # read through its address by the tasks above
    return out^
