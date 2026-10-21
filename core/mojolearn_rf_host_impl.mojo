# SPDX-License-Identifier: Apache-2.0
"""Typed loops of the RF host binding, moved word for word from
bindings/_mojolearn_rf_host.mojo. The shell unpacks the Python arguments and
calls these; the session structs are re-exported there under the same names."""
from core.forest_export_binding_impl import copy_forest_export_leaves
from ensemble.host.rf_oracle import RfHostForest, RfHostParams, n_sampled_rows_host


def rf_host_col_major(
    var flat: List[Float32], n_rows: Int, n_cols: Int, row_major: Bool
) -> List[Float32]:
    """The design COLUMN-major: a ROW-major X is transposed here."""
    if not row_major:
        return flat^
    var out = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for r in range(n_rows):
        for c in range(n_cols):
            out[c * n_rows + r] = flat[r * n_cols + c]
    return out^


def rf_host_forest_lists(
    forest: RfHostForest,
    mut offsets: List[Int64], mut colid: List[Int64], mut quesval: List[Float64],
    mut left_child: List[Int64], mut leaves: List[Float64],
):
    """The five model arrays `_forest_out` publishes, widened word for word."""
    for t in range(len(forest.offsets)):
        offsets.append(Int64(Int(forest.offsets[t])))
    for i in range(forest.n_nodes()):
        colid.append(Int64(Int(forest.colid[i])))
        quesval.append(Float64(forest.quesval[i]))
        left_child.append(Int64(Int(forest.left_child[i])))
    for i in range(len(forest.leaves)):
        leaves.append(Float64(forest.leaves[i]))


def _apply_min_weight_fraction(
    mut p: RfHostParams, frac: Float64, weights: List[Float32], n_rows: Int
) -> Float32:
    """`bindings/_mojolearn_rf.mojo::_rf_fold_min_weight_count` and
    `_rf_min_weight_leaf`: on the weighted objective (non-unit weights, no
    bootstrap) returns `frac * total weight` for `rf_host_fit`; on a count
    objective raises `p.min_samples_leaf` to ceil(frac * the tree's row
    count) and returns 0. frac == 0 changes nothing."""
    if not (frac > 0.0):
        return Float32(0)
    if len(weights) > 0 and not p.bootstrap:
        var total = Float64(0)
        for i in range(len(weights)):
            total += Float64(weights[i])
        return Float32(frac * total)
    var n_tree = n_sampled_rows_host(
        p.bootstrap, p.max_samples if p.bootstrap else Float32(1.0), n_rows
    )
    var bound = frac * Float64(n_tree)
    var need = Int(bound)
    if Float64(need) < bound:
        need += 1
    if need > p.min_samples_leaf:
        p.min_samples_leaf = need
    return Float32(0)


def rf_host_weights_from(
    wp: MutPointer[Float32, MutUntrackedOrigin], n_rows: Int, noun: String
) raises -> List[Float32]:
    """The weighted fit's checked Float32 row weights; all-unit weights
    return the empty list (the unweighted fit)."""
    var weights = List[Float32]()
    var total = Float64(0)
    var all_unit = True
    for i in range(n_rows):
        var w = wp[i]
        if not (w >= 0 and w <= Float32(3.4028234663852886e38)):
            raise Error(noun + " weights must be finite and nonnegative")
        weights.append(w)
        total += Float64(w)
        all_unit = all_unit and w == Float32(1)
    if total <= 0:
        raise Error(noun + " weights must have positive total")
    if all_unit:
        return List[Float32]()
    return weights^


struct RfHostSession(Movable):
    var id: Int
    var x: List[Float32]
    var n_rows: Int
    var n_cols: Int

    def __init__(out self, id: Int, var x: List[Float32], n_rows: Int, n_cols: Int):
        self.id = id
        self.x = x^
        self.n_rows = n_rows
        self.n_cols = n_cols


struct RfHostSessionRegistry(Defaultable, Movable):
    var sessions: List[RfHostSession]
    var next_id: Int

    def __init__(out self):
        self.sessions = List[RfHostSession]()
        self.next_id = 1

    def find(self, id: Int) raises -> Int:
        for i in range(len(self.sessions)):
            if self.sessions[i].id == id:
                return i
        raise Error("unknown or closed forest data session handle")


def rf_host_session_rows_x(
    entry: String,
    src_x: List[Float32],
    src_rows: Int,
    rp: MutPointer[Int32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
) raises -> List[Float32]:
    """The column-major gather of the session rows `rp` (checked first)."""
    for i in range(n_rows):
        var r = Int(rp[i])
        if r < 0 or r >= src_rows:
            raise Error(entry + ": a row id is outside the session's X")
    var x = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for c in range(n_cols):
        for i in range(n_rows):
            x[c * n_rows + i] = src_x[c * src_rows + Int(rp[i])]
    return x^


def rf_host_dart_member_xy(
    entry: String,
    rows_a: Int,
    cols_a: Int,
    n_rows: Int,
    n_cols: Int,
    n: Int,
    d: Int,
    c: Int,
    x_addr: Int,
    target: List[Float32],
    mut x: List[Float32],
    mut y: List[Float32],
) raises:
    """The DART member's column-major X and class labels: the row and column
    ids checked against the session, then the device kernels' pure copies."""
    var rows = List[Int](length=n_rows, fill=0)
    for i in range(n_rows):
        rows[i] = i
    if rows_a != 0:
        var rp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=rows_a)
        for i in range(n_rows):
            var r = Int(rp[i])
            if r < 0 or r >= n:
                raise Error(entry + ": a row id is outside the DART session's X")
            rows[i] = r
    var cols = List[Int](length=n_cols, fill=0)
    for j in range(n_cols):
        cols[j] = j
    if cols_a != 0:
        var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=cols_a)
        for j in range(n_cols):
            var q = Int(cp[j])
            if q < 0 or q >= d:
                raise Error(entry + ": a column id is outside the DART session's X")
            cols[j] = q
    # `dart_member_x_kernel` / `dart_member_y_kernel` on the host: pure
    # copies, so the column-major X and the labels are the device's bytes
    if x_addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    var xs = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_addr)
    x = List[Float32](length=n_rows * n_cols, fill=Float32(0.0))
    for j in range(n_cols):
        var col = cols[j]
        for i in range(n_rows):
            x[j * n_rows + i] = xs[rows[i] * d + col]
    y = List[Float32](length=n_rows, fill=Float32(0.0))
    var coff = c * n
    for i in range(n_rows):
        y[i] = target[coff + rows[i]]


def rf_host_export_into(
    model: RfHostForest,
    op: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Int32, MutUntrackedOrigin],
    tp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Int32, MutUntrackedOrigin],
    leaves_addr: Int,
) raises:
    """The retained forest into the caller's five validated arrays."""
    for t in range(len(model.offsets)):
        op[t] = model.offsets[t]
    for i in range(model.n_nodes()):
        cp[i] = model.colid[i]
        tp[i] = model.quesval[i]
        lp[i] = model.left_child[i]
    copy_forest_export_leaves(model.leaves, leaves_addr, 0)
