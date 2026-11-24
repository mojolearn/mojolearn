# SPDX-License-Identifier: Apache-2.0
"""THE FOLD IN PLACE of the meta-estimators' public doors (lane vdf-meta3, 2026-11-24): the index words a meta fit
reads a model-selection fold (`_arena_io.DeviceRows`: a host float32 base and int64 row numbers) with, out of its
base, so that no row of the fold is gathered on the host (bindings/trees_public_meta_door.mojo `_meta_x_fit`).

  * `meta_rows_index`: the fold's int64 row numbers as the int32 row draw every member reads (the member set's
    `index_dtype`), and the inverse map: for each base row, its position in the fold (0 for a row outside it,
    which no draw names).
  * `meta_rows_compose`: a cv split's fold positions as base rows (`draw[rows[i]]`, the inner fold composed
    inside the outer one).
  * `meta_rows_take_u32`: one 4-byte word per fold row (int32 class codes, float32 targets or weights) put at
    the base rows through the inverse map, so a draw over the base reads the fold's own words.

Index and target words only (never X), split across the host's cores (core/host_parallel.mojo). Pure: no Python,
no device; checked by checks/native_tests/vdf_meta3_fold_in_place.mojo.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]

#: The refusal of a fold row number outside its base.
comptime META_ROWS_OUTSIDE = "IndexError: a DeviceRows fold names a row outside its base"


def meta_rows_index(wide: Int, n: Int, base_rows: Int, draw: Int, inv: Int) raises:
    """The fold's `n` int64 row numbers at `wide` as `n` int32 words at `draw`, and at `inv` (`base_rows` int64
    words) the fold position of each base row: zero first, then `inv[wide[i]] = i` (a row listed twice keeps one
    of its positions; both hold the same base row, whose targets are the same). Raises META_ROWS_OUTSIDE for a
    row number outside 0 .. base_rows-1 (after the pass; nothing else is read)."""
    var tasks = host_predict_task_count_work(base_rows, base_rows)
    var chunk = host_predict_chunk(base_rows, tasks)

    def zero(t: Int) {imm inv, imm chunk, imm base_rows}:
        var p = _I64P(unsafe_from_address=inv)
        for i in range(t * chunk, min(t * chunk + chunk, base_rows)):
            p[i] = 0

    if base_rows > 0:
        host_parallelize(zero, max(tasks, 1))
    var bad = List[UInt8](length=1, fill=UInt8(0))
    var bad_at = Int(bad.unsafe_ptr())
    var row_tasks = host_predict_task_count_work(n, n)
    var row_chunk = host_predict_chunk(n, row_tasks)

    def place(t: Int) {imm wide, imm draw, imm inv, imm row_chunk, imm n, imm base_rows, imm bad_at}:
        var s = _I64P(unsafe_from_address=wide)
        var d = _I32P(unsafe_from_address=draw)
        var v = _I64P(unsafe_from_address=inv)
        for i in range(t * row_chunk, min(t * row_chunk + row_chunk, n)):
            var r = Int(s[i])
            if r < 0 or r >= base_rows:
                _U8P(unsafe_from_address=bad_at)[] = UInt8(1)
                d[i] = 0
                continue
            d[i] = Int32(r)
            v[r] = Int64(i)

    if n > 0:
        host_parallelize(place, max(row_tasks, 1))
    var outside = _U8P(unsafe_from_address=bad_at)[] != UInt8(0)
    if outside:
        raise Error(META_ROWS_OUTSIDE)
    _ = bad^  # written through `bad_at` by the tasks above, read through it into `outside`


def meta_rows_compose(rows: Int, count: Int, draw: Int, dst: Int):
    """`count` int32 fold positions at `rows` as base rows at `dst` (int32): `dst[i] = draw[rows[i]]`. The
    positions are a split of the fold (0 .. fold rows-1), as the meta door's `_splits` makes them."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)

    def task(t: Int) {imm rows, imm dst, imm draw, imm chunk, imm count}:
        var s = _I32P(unsafe_from_address=rows)
        var o = _I32P(unsafe_from_address=dst)
        var w = _I32P(unsafe_from_address=draw)
        for i in range(t * chunk, min(t * chunk + chunk, count)):
            o[i] = w[Int(s[i])]

    if count > 0:
        host_parallelize(task, max(tasks, 1))


def meta_rows_take_u32(src: Int, inv: Int, base_rows: Int, dst: Int):
    """One 4-byte word per fold row at `src` put at the base rows: `dst[b] = src[inv[b]]` for every base row
    (`base_rows` words at `dst`). Bitwise: int32 codes and float32 words alike."""
    var tasks = host_predict_task_count_work(base_rows, base_rows)
    var chunk = host_predict_chunk(base_rows, tasks)

    def task(t: Int) {imm src, imm dst, imm inv, imm chunk, imm base_rows}:
        var s = _U32P(unsafe_from_address=src)
        var o = _U32P(unsafe_from_address=dst)
        var v = _I64P(unsafe_from_address=inv)
        for b in range(t * chunk, min(t * chunk + chunk, base_rows)):
            o[b] = s[Int(v[b])]

    if base_rows > 0:
        host_parallelize(task, max(tasks, 1))
