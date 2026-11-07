# SPDX-License-Identifier: Apache-2.0
# One device copy of X per ensemble call (lane perf-sweep-trees, 2026-11-05).
#
# Each fitted forest member predicts through its resident snapshot, and every
# snapshot of one forest binary borrows that binary's single input workspace
# (core/forest_inference_model.mojo, `ForestIOWorkspace`). Before this every
# member uploaded the same X into that workspace again and scanned it again:
# AdaBoost's one-tree members, a OneVsRest or calibrated forest per class or
# fold, a voting or stacking ensemble of forests. Now the ensemble call stages
# X once in each binary that two or more of its members predict through
# (`forest_stage_input_gpu`), the members' own uploads and input scans are
# skipped by the binary while the staged rows are theirs, and the call
# unstages before it returns (also on an error). Members on other routes
# (non-resident engines, other families, other X) are unaffected; a binary
# without the staging entries or a build that cannot stage answers 0 and the
# members upload as before. Same kernels on the same bits: no output moves.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.trees_public_inputs import trees_table, trees_entry
from bindings.trees_public_forest import forest_effective_mode, forest_kind, _engine, _ordered_export
from bindings.native_model_records import (
    records_stamped_get, records_stamped_put, records_stampable, ContentRecord, RECORDS_TREES,
)

comptime _TABLE_STAMP = "trees_staged_member_table_v1"


def _member_table(member: PythonObject, cols: Int, context: PythonObject) raises -> PythonObject:
    """The forest binary table `member` predicts through on its resident
    snapshot for `cols`-wide rows, or None (not a fitted connected forest,
    another route, another width, a refusal its own predict reports).

    Kept per member (lane ne2-b-members, N073): every Voting, Stacking, OvR,
    Calibrated, Bagging and AdaBoost predict walked each member's class
    ancestry (one table probe per ancestor) and re-resolved its mode, engine
    and binary exports, only to count members per binary. The answer is
    stamped with the member's generation word (bindings/native_model_records.mojo,
    lane ne1-e-records): its class and every field the answer reads
    (`_offsets`, `n_features_in_`, `numeric_mode`, `_fit_numeric_mode`,
    `inference_engine`) live in its instance dictionary, so a refit, a
    `set_params` or any assignment misses. The width, the process default
    mode object and the connected class table are compared too. A
    random-splitter tree (its inner forest decides) and any refusal are not
    stamped: they walk every call, as before."""
    var defaults = context["backend_fields"]["_DEFAULT_MODE"]
    var classes = context["trees_classes"]
    var held = records_stamped_get(RECORDS_TREES, member, _TABLE_STAMP)
    if held is not None and Int(py=held[1]) == cols and held[2] is defaults and held[3] is classes:
        return held[0]
    var found = _member_table_walk(member, cols, context)
    if found[2] is not None:
        # A random-splitter tree: its inner forest decides, unstamped. The
        # recursion stays inside `_member_table` (the walk never calls back):
        # a recursion through the tuple-returning walk is the shape that
        # stopped the Mojo compiler with no diagnostic (lesson 7,
        # docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md section 4).
        return _member_table(found[2], cols, context)
    if Bool(py=found[1]) and records_stampable(member):
        _ = records_stamped_put(RECORDS_TREES, member, _TABLE_STAMP, ContentRecord(),
                                Python.tuple(found[0], PythonObject(cols), defaults, classes))
    return found[0]


def _member_table_walk(member: PythonObject, cols: Int, context: PythonObject) raises -> PythonObject:
    """`(table or None, stampable, inner or None)` for `_member_table`; a
    non-None inner (a random-splitter tree's forest) is resolved there."""
    var b = native_loaded_module("builtins")
    var cls = py_type(member)
    if py_type(cls) is not b.type:
        return Python.tuple(PythonObject(None), PythonObject(False), PythonObject(None))
    var kind = String("")
    try:
        # Once per class (the forest door's own record of the ancestry walk).
        kind = forest_kind(member, context)
    except:
        # Not a connected forest: no table, and the answer holds while the
        # class does (a class reassignment changes the generation).
        return Python.tuple(PythonObject(None), PythonObject(True), PythonObject(None))
    var fields = py_fields(member)
    if kind.startswith("dt_"):
        var inner = py_get(fields, "_random_inner")
        if inner is not None:
            return Python.tuple(PythonObject(None), PythonObject(False), inner)
    if not py_contains(fields, "_offsets"):
        return Python.tuple(PythonObject(None), PythonObject(True), PythonObject(None))
    var width = py_get(fields, "n_features_in_")
    if width is None:
        return Python.tuple(PythonObject(None), PythonObject(True), PythonObject(None))
    try:
        if Int(py=py_call(b.int, width)) != cols:
            return Python.tuple(PythonObject(None), PythonObject(True), PythonObject(None))
        var mode = forest_effective_mode(fields, context)
        var native = trees_table(context, mode, "_mojolearn_trees" if kind.startswith("et_") else "_mojolearn_rf")
        var engine = _engine(fields, mode)
        if engine == "parallel_groves" or (engine == "sequential" and _ordered_export(native)):
            return Python.tuple(native, PythonObject(True), PythonObject(None))
        return Python.tuple(PythonObject(None), PythonObject(True), PythonObject(None))
    except:
        # The member's own predict reports the refusal, in its own words.
        pass
    return Python.tuple(PythonObject(None), PythonObject(False), PythonObject(None))


def trees_stage_members(members: PythonObject, x_addr: Int, rows: Int, cols: Int,
                        context: PythonObject) raises -> PythonObject:
    """Stage the C-order float32 rows at `x_addr` in every forest binary
    that two or more of `members` predict through. Returns the unstage
    entries to pass to `trees_unstage`; the rows must stay alive and
    unchanged until then."""
    var staged = Python.list()
    if rows < 1 or cols < 1 or len(members) < 2:
        return staged
    var tables = Python.list()
    var counts = List[Int]()
    for i in range(len(members)):  # driver-loop: over the member estimators
        var native = PythonObject(None)
        try:
            native = _member_table(members[i], cols, context)
        except:
            # The member's own call reports whatever this read refused.
            pass
        if native is None:
            continue
        var seen = -1
        for j in range(len(tables)):  # small-loop(tables): at most the two forest binaries per mode
            if tables[j] is native:
                seen = j
                break
        if seen < 0:
            _ = py_call(tables.append, native)
            counts.append(1)
        else:
            counts[seen] += 1
    for j in range(len(tables)):  # small-loop(tables): at most the two forest binaries per mode
        if counts[j] < 2:
            continue
        var stage = trees_entry(tables[j], "forest_stage_input_gpu")
        var unstage = trees_entry(tables[j], "forest_unstage_input_gpu")
        if stage is None or unstage is None:
            continue
        # Registered before the call, so the caller's unstage covers it. A
        # failed staging is unstaged here and skipped: staging is optional,
        # the members then upload as before.
        _ = py_call(staged.append, unstage)
        try:
            _ = py_call(stage, PythonObject(x_addr), Python.list(PythonObject(rows), PythonObject(cols)))
        except:
            try:
                _ = py_call(unstage)
            except:
                pass
    return staged


def trees_stage_rounds(member: PythonObject, x_addr: Int, rows: Int, cols: Int,
                       context: PythonObject) raises -> PythonObject:
    """Stage the rows at `x_addr` in the forest binary `member` predicts
    through, for a caller that predicts those same rows through it once per
    round with a new member each round (AdaBoost's fit loop: each round's
    member predicts the training X; lane perf-trees-owed, 2026-11-06). The
    staged rows serve every later member of that binary; one predicting
    other rows unstages them, as in any staging. Returns the unstage entries
    for `trees_unstage`; the rows must stay alive and unchanged until then."""
    var staged = Python.list()
    if rows < 1 or cols < 1:
        return staged
    var native = PythonObject(None)
    try:
        native = _member_table(member, cols, context)
    except:
        pass
    if native is None:
        return staged
    var stage = trees_entry(native, "forest_stage_input_gpu")
    var unstage = trees_entry(native, "forest_unstage_input_gpu")
    if stage is None or unstage is None:
        return staged
    _ = py_call(staged.append, unstage)
    try:
        _ = py_call(stage, PythonObject(x_addr), Python.list(PythonObject(rows), PythonObject(cols)))
    except:
        try:
            _ = py_call(unstage)
        except:
            pass
    return staged


def trees_unstage(staged: PythonObject):
    """End the staging `trees_stage_members` began (quiet: a binary torn
    down at interpreter exit has nothing left staged)."""
    # Each binary is unstaged in its own try: one failed unstage must never
    # skip the next binary's, which would leave its staged mark (keyed on the
    # host address and count only) set for a later call over other bytes at
    # the same address (hazard, p3-trees row 954).
    var n = 0
    try:
        n = len(staged)
    except:
        return
    for i in range(n):  # small-loop(staged): at most the two forest binaries per mode
        try:
            _ = py_call(staged[i])
        except:
            pass
