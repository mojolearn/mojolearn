# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The YetiRank task table.

GPU-free, shared by the device target (`gbdt/targets/kernel/yeti_rank.mojo`)
and the host oracle, so both walk the same tasks.

Reference: `YetiRankGradientImpl` (`yeti_rank_pointwise.cu:185-247`). A shared
query cursor hands out TASKS: from query `taskQid` at row `offset`, the task
ends at the start of the query that holds row `min(offset + 1024, size)` (or at
`size`), and the cursor moves to that query. Queries are at most 1023 rows
(`InitYetiRank`, `querywise_targets_impl.h:313-321`), so a task is at most
1024 rows and always holds whole queries. Which device thread claims a task
does not change the table: it depends only on the query sizes.
"""

comptime YETI_TASK_POSITIONS = 1024
comptime YETI_MAX_QUERY_SIZE = 1023


struct YetiRankTasks(Movable):
    """One entry per task: its first row, its row count and its first query
    (`taskQid`, which seeds the task's random streams); and each row's query
    index (`ComputeGroupIds`)."""

    var offsets: List[UInt32]
    var sizes: List[UInt32]
    var qids: List[UInt32]
    var query_ids: List[UInt32]

    def __init__(
        out self,
        var offsets: List[UInt32],
        var sizes: List[UInt32],
        var qids: List[UInt32],
        var query_ids: List[UInt32],
    ):
        self.offsets = offsets^
        self.sizes = sizes^
        self.qids = qids^
        self.query_ids = query_ids^

    def count(self) -> Int:
        return len(self.offsets)


def yeti_rank_tasks(group_sizes: List[UInt32], n_rows: Int) raises -> YetiRankTasks:
    """The task table, with `InitYetiRank`'s query size refusal in its words."""
    var q_count = len(group_sizes)
    var q_offsets = List[Int](capacity=q_count)
    var query_ids = List[UInt32](capacity=n_rows)
    var covered = 0
    for q in range(q_count):
        var s = Int(group_sizes[q])
        if s > YETI_MAX_QUERY_SIZE:
            raise Error(
                "Error: max query size supported on GPU is 1023, got " + String(s)
            )
        q_offsets.append(covered)
        for _ in range(s):
            query_ids.append(UInt32(q))
        covered += s
    if covered != n_rows:
        raise Error(
            "YetiRank: the query sizes cover " + String(covered) + " rows of "
            + String(n_rows)
        )
    var offsets = List[UInt32]()
    var sizes = List[UInt32]()
    var qids = List[UInt32]()
    var task_qid = 0
    while task_qid < q_count:
        var offset = q_offsets[task_qid]
        var next_task_offset = offset + YETI_TASK_POSITIONS
        if next_task_offset > n_rows:
            next_task_offset = n_rows
        var next_task_qid = q_count
        if next_task_offset < n_rows:
            next_task_qid = Int(query_ids[next_task_offset])
        if next_task_qid < q_count:
            next_task_offset = q_offsets[next_task_qid]
        else:
            next_task_offset = n_rows
        offsets.append(UInt32(offset))
        sizes.append(UInt32(next_task_offset - offset))
        qids.append(UInt32(task_qid))
        task_qid = next_task_qid
    return YetiRankTasks(offsets^, sizes^, qids^, query_ids^)


def yeti_rank_cuda_seed(seed: UInt64) -> UInt32:
    """`int cudaSeed = ((ui32)seed) + ((ui32)(seed >> 32))`
    (`yeti_rank_pointwise.cu:268`), wrapping in 32 bits."""
    return UInt32(seed & UInt64(0xFFFFFFFF)) + UInt32(seed >> UInt64(32))


def yeti_rank_advance_seed32(seed: UInt32) -> UInt32:
    """`AdvanceSeed32` (`cuda_util/kernel/random_gen.cuh:40-43`)."""
    return UInt32(1664525) * seed + UInt32(1013904223)


def yeti_rank_task_seed(task_qid: UInt32, tid: Int, cuda_seed: UInt32) -> UInt32:
    """The seed of thread `tid` of a task (`yeti_rank_pointwise.cu:224-237`):
    `127 * taskQid + 16807 * tid + 1`, three advances, `+= seed`, three more."""
    var s = UInt32(127) * task_qid + UInt32(16807) * UInt32(tid) + UInt32(1)
    for _ in range(3):
        s = yeti_rank_advance_seed32(s)
    s = s + cuda_seed
    for _ in range(3):
        s = yeti_rank_advance_seed32(s)
    return s
