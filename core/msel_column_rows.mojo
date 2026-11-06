# SPDX-License-Identifier: Apache-2.0
"""Column-major block -> row-major matrix copy for the model-selection pandas
materializer (lane p3-msel-parallel, pass 3 rank 106).

`bindings/msel_inputs.mojo::_pandas_materialize` placed every element of a
pandas column block with its own `memcpy` call, serially. This copies each
row range of the block with typed loads and stores, the row ranges split
across the host's cores (`host_parallelize`). Pure storage moves: every
element keeps its bits, and the split never changes a value.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


@always_inline
def _rows_typed[dt: DType](source: Int, out_address: Int, nrows: Int, ncols: Int,
                           dest: MutPointer[Int, MutUntrackedOrigin], width: Int, lo: Int, hi: Int):
    var src = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=source)
    var dst = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=out_address)
    for row in range(lo, hi):
        var out_row = row * ncols
        for col in range(width):
            dst[out_row + dest[col]] = src[col * nrows + row]


def _rows_any(source: Int, out_address: Int, nrows: Int, ncols: Int,
              dest: MutPointer[Int, MutUntrackedOrigin], width: Int, item: Int, lo: Int, hi: Int):
    if item == 8:
        _rows_typed[DType.uint64](source, out_address, nrows, ncols, dest, width, lo, hi)
    elif item == 4:
        _rows_typed[DType.uint32](source, out_address, nrows, ncols, dest, width, lo, hi)
    elif item == 2:
        _rows_typed[DType.uint16](source, out_address, nrows, ncols, dest, width, lo, hi)
    else:
        _rows_typed[DType.uint8](source, out_address, nrows, ncols, dest, width, lo, hi)


def msel_block_columns_to_rows(source: Int, out_address: Int, nrows: Int, ncols: Int,
                               var dest: List[Int], item: Int):
    """Copy a C-ordered (width, nrows) block of `item`-byte elements (one
    pandas column per block row) into columns `dest[c]` of the C-ordered
    (nrows, ncols) matrix at `out_address`. Row ranges run in parallel
    (each output element is written by exactly one task)."""
    var width = len(dest)
    if nrows <= 0 or width == 0:
        return
    var tasks = host_predict_task_count_work(nrows, nrows * width)
    var chunk = host_predict_chunk(nrows, tasks)
    var dp = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(dest.unsafe_ptr()))

    def _task(c: Int) {imm source, imm out_address, imm nrows, imm ncols, imm dp, imm width, imm item, imm chunk}:
        var lo = c * chunk
        var hi = min(lo + chunk, nrows)
        if lo < hi:
            _rows_any(source, out_address, nrows, ncols, dp, width, item, lo, hi)

    if tasks <= 1:
        _task(0)
    else:
        host_parallelize(_task, tasks)
    _ = dest^  # keep the column table alive through every task above
