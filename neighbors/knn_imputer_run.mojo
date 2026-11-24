# SPDX-License-Identifier: Apache-2.0
"""KNNImputer RUN: typed requests in, typed results out.

The boundary (bindings/neighbors_local_native_door.mojo) admits the public
arguments into these requests, releases the GIL once for each function here
and publishes the results. Nothing in this module imports Python. Every
address in a request is admitted storage the boundary keeps alive through
the call; every intermediate (the masked rows, the cell list, the per-column
counts, the imputed rows before column selection) is native scratch of this
module, never a public array. The operations and their order are the public
method's.
"""
from core.neighbors_native_driver import XnDriver
from core.host_parallel import host_parallelize
from neighbors.neighbors_local_run import (
    ImpFitColumns, ImpKeepPlan, imp_cells_length, imp_colmiss_length, imp_read_count, imp_fit_columns,
    imp_cells_operation, local_copy_f32,
)


def imp_scratch_f32(count: Int) -> List[Float32]:
    """Zeroed float32 scratch of `count` words (one word for an empty span,
    so an operation is never handed a null address)."""
    return List[Float32](length=max(count, 1), fill=Float32(0.0))


def imp_i32_words(values: List[Int]) -> List[Int32]:
    """`_i32(list)`: a feature-sized int list as int32 words (one zero word
    for an empty list)."""
    var words = List[Int32](length=max(len(values), 1), fill=Int32(0))
    for i in range(len(values)):  # driver-loop: over a feature-sized int list of model structure
        words[i] = Int32(values[i])
    return words^


#: Rows of one host task of `imp_copy_rows`' identity index.
comptime _IOTA_CHUNK = 16384


def imp_copy_rows(driver: XnDriver, x: Int, dst: Int, rows: Int, features: Int) raises:
    """x (rows x features float32) into the host `dst`: a host copy of a host address, else (THE FOLD X WORD, a
    negative x) the x_neighbors `take_rows` op with the identity index over the slot (one device read, one
    download), bit for bit."""
    var count = rows * features
    if count <= 0:
        return
    if x >= 0:
        local_copy_f32(dst, x, count)
        return
    var index = List[Int32](length=rows, fill=Int32(0))
    var at = Int(index.unsafe_ptr())
    var tasks = (rows + _IOTA_CHUNK - 1) // _IOTA_CHUNK

    def _iota(t: Int) {imm at, imm rows}:
        var o = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=at)
        var lo = t * _IOTA_CHUNK
        var hi = min(rows, lo + _IOTA_CHUNK)
        for r in range(lo, hi):  # the task's rows (one chunk of the split)
            o[r] = Int32(r)

    host_parallelize(_iota, tasks)
    var a: Array[Int, 3] = [x, Int(index.unsafe_ptr()), dst]
    var p: Array[Int, 3] = [rows, features, rows]
    var f: Array[Float64, 0] = []
    _ = driver.run("take_rows", a, p, f)
    _ = index^


def imp_mask_run(driver: XnDriver, x: Int, dst: Int, count: Int, missing: Float64) raises:
    """`KNNImputer._masked`: the cells equal to `missing` (a float32-rounded
    number) become NaN in `dst`."""
    var a: Array[Int, 2] = [x, dst]
    var p: Array[Int, 1] = [count]
    var f: Array[Float64, 1] = [missing]
    _ = driver.run("p2m_mask_value", a, p, f)


@fieldwise_init
struct ImpFitRequest(Copyable, Movable):
    # `masked` is the published rows with the missing number's cells set to
    # NaN (0: X is used as it is). `lean`: the bound image's column-count-only
    # fit entry writes no cell list.
    var x: Int
    var masked: Int
    var rows: Int
    var features: Int
    var missing: Float64
    var lean: Bool


def imp_fit_run(driver: XnDriver, request: ImpFitRequest) raises -> ImpFitColumns:
    """The fit's column masks: valid (some row present) and missing (some
    row NaN), from the per-column NaN counts."""
    var n = request.rows
    var d = request.features
    var source = request.x
    if request.masked != 0:
        imp_mask_run(driver, request.x, request.masked, n * d, request.missing)
        source = request.masked
    var cells = List[Int32](length=imp_cells_length(n, d, request.lean), fill=Int32(0))
    var colmiss = List[Int32](length=imp_colmiss_length(d), fill=Int32(0))
    var info = List[Int32](length=1, fill=Int32(0))
    var a: Array[Int, 4] = [source, Int(cells.unsafe_ptr()), Int(colmiss.unsafe_ptr()), Int(info.unsafe_ptr())]
    var p: Array[Int, 3] = [n, d, 1]
    var f: Array[Float64, 0] = []
    _ = driver.run("nan_cells", a, p, f)
    var columns = imp_fit_columns(Int(colmiss.unsafe_ptr()), d, n)
    _ = cells^
    _ = colmiss^
    _ = info^
    return columns^


@fieldwise_init
struct ImpTransformRequest(Copyable, Movable):
    # `x`: the admitted float32 rows; `mask`: cells equal to `missing` become
    # NaN first. `fit_x`: the fitted rows (`fit_rows` x `features`). `dst`:
    # the public result, `rows` x `imp_result_columns(...)` float32.
    # `uncompact` / `old_items`: the import-time arms MOJOLEARN_XN_UNCOMPACT_IMPUTE
    # and MOJOLEARN_XN_OLD_ITEMS. `keep_empty`: keep_empty_features.
    var x: Int
    var fit_x: Int
    var dst: Int
    var rows: Int
    var fit_rows: Int
    var features: Int
    var neighbors: Int
    var distance: Bool
    var mask: Bool
    var missing: Float64
    var uncompact: Bool
    var old_items: Bool
    var keep_empty: Bool


def imp_selects(rows: Int, features: Int, keep_empty: Bool, columns: ImpKeepPlan) -> Bool:
    """Whether a column step follows the imputation: the empty columns
    zeroed (keep_empty_features) or dropped."""
    if keep_empty:
        return rows != 0 and not columns.all_valid
    return len(columns.keep) != features


def imp_selected_columns(features: Int, keep_empty: Bool, columns: ImpKeepPlan) -> Int:
    """The imputed result's columns before the indicator block."""
    return features if keep_empty else len(columns.keep)


def imp_result_columns(features: Int, keep_empty: Bool, columns: ImpKeepPlan, indicator: Int) -> Int:
    """The public result's columns: the selected ones, then one indicator
    column per feature missing at fit (`indicator` of them; 0 without
    add_indicator or without rows)."""
    return imp_selected_columns(features, keep_empty, columns) + indicator


def imp_transform_run(driver: XnDriver, request: ImpTransformRequest, columns: ImpKeepPlan,
                      indicator: List[Int]) raises:
    """Mask, impute, select the columns, append the indicator block; the
    last step writes `request.dst`, the earlier ones native scratch."""
    var n = request.rows
    var d = request.features
    var count = n * d
    var selects = imp_selects(n, d, request.keep_empty, columns)
    var q = len(indicator)
    var no_f: Array[Float64, 0] = []

    # the rows with the missing number's cells as NaN
    var masked = List[Float32]()
    var source = request.x
    if request.mask and count != 0:
        masked = imp_scratch_f32(count)
        imp_mask_run(driver, request.x, Int(masked.unsafe_ptr()), count, request.missing)
        source = Int(masked.unsafe_ptr())

    # the imputed rows (rows x features)
    var imputed_scratch = List[Float32]()
    var imputed = request.dst
    if selects or q != 0:
        imputed_scratch = imp_scratch_f32(count)
        imputed = Int(imputed_scratch.unsafe_ptr())
    var distance = 1 if request.distance else 0
    if request.uncompact:
        var a: Array[Int, 3] = [source, request.fit_x, imputed]
        var p: Array[Int, 5] = [n, request.fit_rows, d, request.neighbors, distance]
        _ = driver.run("knn_impute", a, p, no_f)
    else:
        # one GPU thread per MISSING cell; a present cell keeps x
        var cells = List[Int32](length=imp_cells_length(n, d, False), fill=Int32(0))
        var colmiss = List[Int32](length=imp_colmiss_length(d), fill=Int32(0))
        var info = List[Int32](length=1, fill=Int32(0))
        var ca: Array[Int, 4] = [source, Int(cells.unsafe_ptr()), Int(colmiss.unsafe_ptr()), Int(info.unsafe_ptr())]
        var cp: Array[Int, 3] = [n, d, 0]
        _ = driver.run("nan_cells", ca, cp, no_f)
        var missing_cells = imp_read_count(Int(info.unsafe_ptr()))
        # The cell op starts `imputed` from x itself (a present cell keeps x; lane vdf-scope-cluster6), so x is
        # read only through the ops' uploads (THE FOLD X WORD for a model-selection fold). With no missing cell
        # the op does not run and `imputed` is x's copy: one `take_rows` read of the rows in order.
        if missing_cells != 0:
            var a: Array[Int, 4] = [Int(cells.unsafe_ptr()), source, request.fit_x, imputed]
            var p: Array[Int, 6] = [n, request.fit_rows, d, request.neighbors, distance, missing_cells]
            _ = driver.run(imp_cells_operation(request.old_items), a, p, no_f)
        elif count != 0:
            imp_copy_rows(driver, source, imputed, n, d)
        _ = cells^
        _ = colmiss^
        _ = info^

    # the column step
    var selected_scratch = List[Float32]()
    var selected = imputed
    var c = imp_selected_columns(d, request.keep_empty, columns)
    if selects:
        selected = request.dst
        if q != 0:
            selected_scratch = imp_scratch_f32(n * c)
            selected = Int(selected_scratch.unsafe_ptr())
        if request.keep_empty:
            var flags = imp_i32_words(columns.flags)
            var a: Array[Int, 3] = [imputed, Int(flags.unsafe_ptr()), selected]
            var p: Array[Int, 2] = [n, d]
            _ = driver.run("p2m_zero_cols", a, p, no_f)
            _ = flags^
        else:
            # `_take_cols(out, keep)`
            var keep = imp_i32_words(columns.keep)
            var a: Array[Int, 3] = [imputed, Int(keep.unsafe_ptr()), selected]
            var p: Array[Int, 3] = [n, d, c]
            _ = driver.run("take_cols", a, p, no_f)
            _ = keep^

    # the indicator block
    if q != 0:
        var missing_columns = imp_i32_words(indicator)
        var second = selected if c != 0 else source
        var a: Array[Int, 4] = [source, second, Int(missing_columns.unsafe_ptr()), request.dst]
        var p: Array[Int, 4] = [n, d, c, q]
        _ = driver.run("p2m_nan_indicator", a, p, no_f)
        _ = missing_columns^
    _ = masked^
    _ = imputed_scratch^
    _ = selected_scratch^
