# SPDX-License-Identifier: Apache-2.0
# The native owner of the x_prep imputers, discretizer and encoders:
# SimpleImputer and KBinsDiscretizer. Each public fit, transform,
# inverse_transform and fit_transform is one call into prep_encode_c, which
# checks the parameters, builds the x_prep program the Python `_Prog` built
# (the same stages, operands and order, so the same bits), runs it through
# the shared transport (bindings/x_prep_native_owner_door.mojo `xprep_run_records`) and
# stores the fitted attributes under the names, types and dtypes the Python
# set. fit_transform converts X once and hands the converted X to the
# transform program.
from std.memory import unsafe_memcpy
from std.math import ceil
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains, py_setattr
from bindings.py_typed_call import py_call_typed
from bindings.classical_warnings_unpack import classical_warning
from bindings.native_caller_frame_pack import native_frame_globals
from bindings.py_cpython_table import cpython_table
from bindings.prep_native_estimator_door import prep_native_split, prep_entry_call
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_address, classical_typed, classical_wrap_array, classical_c_order
from bindings.scaler_native_door import _ptr, _input
from bindings.scaler_lifecycle_native_door import _entries, scaler_entries
from bindings.prep_owner_route_unpack import (
    prep_route_context, prep_route_caps, prep_route_vendor_used)
from bindings.prep_entry_cache_unpack import prep_entry_word, prep_entry_text
from bindings.x_prep_native_owner_door import xprep_storage, xprep_take, xprep_values, xprep_run_records, xprep_caps
from bindings.prep_encode_prog_run import (
    EncProgram, OUT_MIN_WORDS, OUT_MARK, enc_assign_fold_i64, enc_check_train_i64,
    enc_col_stats, enc_mark_missing)
from bindings.prep_encode_imputer_run import (
    ImputerFit, ImputerApply, IMP_MEAN, IMP_MEDIAN, IMP_MOST_FREQUENT, IMP_CONSTANT,
    imputer_fit_stage, imputer_transform_stage)
from bindings.prep_encode_kbins_run import (
    KBinsFit, KBinsApply, KBinsInverse, KB_DRAW_NONE, KB_DRAW_PLAIN, KB_DRAW_WEIGHTED,
    enc_weights_stage, kbins_fit_stage, kbins_transform_stage, kbins_inverse_stage, enc_seed_words)
from bindings.prep_encode_cats_run import (
    CatFit, CatFitPlan, CatBlock, InfrequentFit, CAT_FIT_UNIQUE, CAT_FIT_SCAN, CAT_FIT_PACK_SORT,
    CAT_FIT_PACK_PRESENT, CAT_FIT_INVERSE, cat_fit_stage, cat_codes_stage, cat_i2f_stage, cat_grouped_sizes,
    cat_infrequent_stage)
from bindings.prep_encode_onehot_run import (
    OneHotApply, OneHotInverse, onehot_transform_stage, onehot_inverse_stage)
from bindings.prep_encode_ordinal_run import (
    OrdinalApply, OrdinalInverse, ordinal_transform_stage, ordinal_inverse_stage)
from bindings.prep_encode_target_run import (
    TargetFit, TargetApply, TE_FOLDS_NONE, TE_FOLDS_STAGED, TE_FOLDS_GIVEN, target_fit_stage,
    target_transform_stage)
from bindings.prep_encode_spline_run import (
    SplineFit, SplineApply, SPL_KNOTS_GIVEN, SPL_KNOTS_QUANTILE, SPL_KNOTS_UNIFORM, spline_features,
    spline_fit_stage, spline_transform_stage)
from bindings.prep_encode_select_run import (
    ScoreClassif, ScorePearson, ScorePlan, score_classif_stage, score_pearson_stage, score_topk_stage,
    score_topk_words_stage, score_integral_stage)
from bindings.x_prep_native_owner_run import xprep_r3, XprepCaps
from bindings.public_lifecycle_owner_door import lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result
from bindings.classical_validate_unpack import classical_estimator_name
from core.host_finite_scan import host_all_finite_f32
from bindings.prep_stateless_native_door import _fam, _int_like, _repr, _name, _take_f, _select_transform
from bindings.classical_labels_door import ClassicalLabels, labels_address, labels_cast, classical_encode_labels
from bindings.admitted_input_unpack import admit_c_view, admitted_dtype, admit_f32, admit_ids, admitted_vector
from bindings.classical_arrays_door import classical_address
from core.host_finite_scan import host_all_finite_f64
from bindings.py_optional import py_or_none, py_present

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep_idn_fam bits 2 IDN_CLASS_ONEPASS and 4 IDN_SELECT_BLOCKED.
comptime IDN_CLASS_ONEPASS = 2
comptime IDN_SELECT_BLOCKED = 4
# `_NONE`: a stage's "no operand" word.
comptime NONE = -1
# x_prep_idn_fam bit 1: IDN_STATS_BLOCKED.
comptime IDN_STATS_BLOCKED = 1


# ---------------------------------------------------------------- programs
struct EncProg(Movable):
    """The door of one x_prep program (lane encprog-typed, 2026-11-12): the
    typed program (bindings/prep_encode_prog_run.mojo `EncProgram`: layout,
    stages, input records, native parameter lists) and the Python owner of
    each caller-buffer input record, held for the run. `run` is UNPACK (each
    owner's address, resident slot or DeviceRows fields), the typed run with
    the GIL released (bindings/x_prep_native_owner_run.mojo `xprep_execute`),
    then the arena read back by the callers (PACK)."""
    var prog: EncProgram
    var owners: List[PythonObject]

    def __init__(out self) raises:
        self.prog = EncProgram(xprep_r3("work"))
        self.owners = List[PythonObject]()

    def alloc(mut self, words: Int) -> Int:
        return self.prog.alloc(words)

    def scratch(mut self, words: Int) -> Int:
        return self.prog.scratch(words)

    def work(mut self, words: Int) -> Int:
        return self.prog.work(words)

    def output(mut self, words: Int, entries: PythonObject) raises -> Int:
        # Below OUT_MIN_WORDS, or on a binding without an output region
        # (the host column), the output is host words.
        # TOMBSTONE: MOJOLEARN_XPREP_FAST_OUT_OFF (NOISE: B kept large outputs as host words on a binding with an output region; A/B abn9 NVIDIA L40S, 48 cells 0.91-1.13 arm/base, median 1.00) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
        # Restore: git apply experiments/removed/MOJOLEARN_XPREP_FAST_OUT_OFF.patch; record in docs/TOMBSTONES.md.
        var region = words >= OUT_MIN_WORDS and py_get(entries, "x_prep_run_out") is not None
        return self.prog.output(words, region)

    def _record(mut self, record: PythonObject, words: Int) -> Int:
        self.owners.append(record)
        return self.prog.input(words, len(self.owners) - 1)

    def put(mut self, value: PythonObject, words: Int) raises -> Int:
        # A float32 (or int32-word) contiguous buffer of `words` words.
        if words <= 0:
            return self.prog.alloc(words)
        return self._record(Python.tuple(0, value, PythonObject(None), words, 1), words)

    def own(mut self, value: PythonObject, words: Int) raises -> Int:
        """UNPACK of a caller buffer for a typed stage builder: the float32
        (or int32-word) contiguous buffer `value` of `words` words becomes
        an owner of this program, held for the run. Returns its owner index;
        the builder records it (`enc_input`, bindings/prep_encode_prog_run.mojo)
        where `put` recorded it, so the arena layout is the builder's."""
        self.owners.append(Python.tuple(0, value, PythonObject(None), words, 1))
        return len(self.owners) - 1

    def own_x(mut self, values: PythonObject) -> Int:
        # `own` of X as `xprep_values` converted it (host words or device rows).
        self.owners.append(values)
        return len(self.owners) - 1

    def put_inout(mut self, value: PythonObject, words: Int) raises -> Int:
        # `_Prog.put(arr, inout=True)`: a stage rewrites these words and the
        # caller reads them after the run, so they come back from the device.
        var offset = self.put(value, words)
        self.prog.inout(offset, words)
        return offset

    def put_x(mut self, values: PythonObject) raises -> Int:
        # X as `xprep_values` converted it (host words or device rows).
        return self._record(values, Int(py=values[3]) * Int(py=values[4]))

    def put_floats(mut self, values: List[Float64], context: PythonObject) raises -> Int:
        # `_Prog.put_list`: the values rounded to float32 ([0.0] when empty),
        # program-owned native words.
        return self.prog.put_floats(values)

    def put_ints(mut self, values: List[Int], context: PythonObject) raises -> Int:
        # `_Prog.put_ints`: int32 words ([0] when empty), program-owned.
        return self.prog.put_ints(values)

    def run(mut self, entries: PythonObject, context: PythonObject) raises -> PythonObject:
        """`_Prog.run`: the output region (when there is one) past the host
        arena and the scratch, then the scratch past the host arena; returns
        (arena, output store)."""
        return self.run_caps(entries, context, xprep_caps(entries))

    def run(mut self, entries: PythonObject, context: PythonObject, model: PythonObject) raises -> PythonObject:
        """`run` of an estimator's operation: the image's capabilities and
        its transport entry come from the estimator's device route (one
        record read, one pointer compare) when it holds this selection, in
        place of the by-name reads of `xprep_caps`; read as before otherwise."""
        var caps = prep_route_caps(model, entries)
        return self.run_caps(entries, context, caps if caps.held else xprep_caps(entries))

    def run_caps(mut self, entries: PythonObject, context: PythonObject, caps: XprepCaps) raises -> PythonObject:
        self.prog.check_bound()
        var host = self.prog.host
        var scratch = self.prog.scratch_size
        var out_words = self.prog.out_words
        var device_scratch = scratch > 0 and (caps.has_out or caps.has_scratch)
        var arena = xprep_storage(host + (0 if device_scratch else scratch), context)
        var store = xprep_storage(out_words, context) if out_words > 0 else PythonObject(None)
        xprep_run_records(arena, host, scratch, store, out_words, self.prog.stage_words(), self.prog.inputs,
                          self.owners, entries, context, out_words, self.prog.inout_ranges, caps)
        return Python.tuple(arena, store)


def enc_caps(model: PythonObject, entries: PythonObject) raises -> XprepCaps:
    """The run capabilities of an estimator's operation, once per operation
    at UNPACK: from its device route when the route holds this selection
    (one record read, one pointer compare), else read from `entries` by name
    (`xprep_caps`). These are the reads `EncProg.run(entries, context, model)`
    makes; an operation that takes them here runs with `run_caps`, and its
    typed stage builder gets `has_out` as the plain `region` value."""
    var caps = prep_route_caps(model, entries)
    if caps.held:
        return caps
    return xprep_caps(entries)


def _fitted_flat(owner: PythonObject, name: String, value: PythonObject, context: PythonObject) raises -> PythonObject:
    """(value, flat float32 vector, words) of a fitted field: `_input`'s
    admission and conversion of `value`, kept in the prepared-owner registry
    keyed weakly by the model (one slot per field name, never in the model's
    `__dict__`) and reused while the field holds the same object (perf2-prep).
    The record is kept only when the admission returned `value` itself (an
    exact float32 C-order Array): the program then reads the live words on
    every call, so an in-place write is always seen. A converted copy (a
    float64 or strided array assigned by the user) is never kept: it is
    converted per call, so an in-place write into the source is seen too
    (hazard class A, p3-prep)."""
    var mods = NativeModules()
    var slot = String("prep_encode_fitted_v1:") + name
    var record = prepared_get(mods, slot, owner)
    if record is not None and record[0] is value:
        return record
    var flat = _input(value, 1, False, context)
    record = Python.tuple(value, flat, Int(py=flat.size))
    prepared_put(mods, slot, owner, record if flat is value else PythonObject(None))
    return record


def _floats(values: List[Float64], context: PythonObject) raises -> PythonObject:
    var out = classical_typed(Python.tuple(max(len(values), 1)), "<f4", context["output_context"])
    var dest = _ptr(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over a program's parameter words
        dest[i] = Float32(values[i])
        _ = out  # keep the Python object alive through the native call above
    return out


def _word(ran: PythonObject, offset: Int) raises -> Float32:
    return F32P(unsafe_from_address=Int(py=ran[0][1]))[offset]


def _iword(ran: PythonObject, offset: Int) raises -> Int:
    return Int(I32P(unsafe_from_address=Int(py=ran[0][1]))[offset])


def _take1(ran: PythonObject, offset: Int, count: Int, context: PythonObject) raises -> PythonObject:
    # `_Prog.get(off, count)`: a 1-D float32 Array.
    if offset == OUT_MARK:
        return xprep_take(ran[1][0], 0, count, 0, context["output_context"])
    return xprep_take(ran[0][0], offset, count, 0, context["output_context"])


def _take2(ran: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # `_Prog.get(off, (rows, cols))`: a C-order float32 Array.
    if rows * cols == 0:
        return classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(rows, cols), "<f4", "C", context["output_context"])
    if offset == OUT_MARK:
        return xprep_take(ran[1][0], 0, rows, cols, context["output_context"])
    return xprep_take(ran[0][0], offset, rows, cols, context["output_context"])


def _int_list(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over fitted per-column metadata
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _list_text(values: List[Int]) -> String:
    # A Python list's str of ints.
    var text = String("[")
    for i in range(len(values)):  # driver-loop: over reported row or column indices
        text += (", " if i > 0 else "") + String(values[i])
    return text + "]"


# ---------------------------------------------------------------- shared stages
def _cs(mut p: EncProg, mode: PythonObject, entries: PythonObject, xo: Int, n: Int, d: Int, out_buf: Int, variance: Bool = True) raises:
    """`_cs`: col_stats' six rows of d at `out`. UNPACK of the image's
    family word (IDN_STATS_BLOCKED, IDENTICAL), then the typed stages
    (bindings/prep_encode_prog_run.mojo `enc_col_stats`, which reads the
    MOJOLEARN_XPREP_BLOCKED run-time switch)."""
    enc_col_stats(p.prog, (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0, xo, n, d, out_buf, variance)


def _bad_rows(ran: PythonObject, staged: Tuple[Int, Int], limit: Int) raises -> List[Int]:
    var k = _iword(ran, staged[1])
    var rows = List[Int]()
    for i in range(min(k, limit)):  # driver-loop: over at most `limit` reported rows
        rows.append(_iword(ran, staged[0] + i))
    return rows^


def _seed64(value: PythonObject) raises -> Tuple[Int, Int]:
    # (0 if None else int(value)) & 0xFFFFFFFFFFFFFFFF as its (low, high) words.
    var b = native_loaded_module("builtins")
    var seed = PythonObject(0) if value is None else py_call_typed(b.int, value)
    var mask = PythonObject(0xFFFFFFFF)
    var lo = Int(py=py_call(b.int.__dict__["__and__"], seed, mask))
    var hi = Int(py=py_call(b.int.__dict__["__and__"], py_call(b.int.__dict__["__rshift__"], seed, PythonObject(32)), mask))
    return (lo, hi)


def _seed_words(lo: Int, hi: Int) -> List[Int]:
    # `_seed_words` / `put_ints`: the words as signed int32.
    return enc_seed_words(lo, hi)


# ---------------------------------------------------------------- shared checks
def _truth(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(native_loaded_module("builtins").bool, value))


def _is_str(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    return Bool(py=py_call(b.isinstance, value, b.str))


def _is_float(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    return Bool(py=py_call(b.isinstance, value, b.float))


def _is_nan_value(value: PythonObject) raises -> Bool:
    # `_is_nan_value`: a Python float (or subclass) that is NaN.
    if not _is_float(value):
        return False
    var v = Float64(py=value)
    return v != v


def _real_like(value: PythonObject, context: PythonObject) raises -> Bool:
    # numbers.Real without its Python __instancecheck__: Python ints, bools
    # and floats, and NumPy integer and floating scalars when connected.
    var b = native_loaded_module("builtins")
    if Bool(py=py_call(b.isinstance, value, b.int)) or Bool(py=py_call(b.isinstance, value, b.float)):
        return True
    var integer = py_get(context, "integer")
    if integer is not None and Bool(py=py_call(b.isinstance, value, integer)):
        return True
    var floats = py_get(context, "float_types")
    return floats is not None and Bool(py=py_call(b.isinstance, value, floats))


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call_typed(native_loaded_module("builtins").float, value))


def _str_text(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").str, value))


def _fitted(model: PythonObject) raises:
    # `_check_fitted`.
    if not py_contains(py_fields(model), "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")


def _check_width(model: PythonObject, d: Int) raises:
    # `_check_width`.
    var fitted = py_fields(model)["n_features_in_"]
    if py_type(fitted) is not native_loaded_module("builtins").int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + _name(model) + " was fitted with " + _str_text(fitted))


def _i64_values(value: PythonObject, context: PythonObject) raises -> List[Int]:
    # A fitted int64 vector's values (`n_bins_.tolist()`).
    if String(value.dtype) != "<i8" or Int(py=value.ndim) != 1 or not classical_c_order(value, context):
        raise Error("TypeError: a fitted int64 vector must remain a contiguous int64 Array")
    var count = Int(py=value.size)
    var source = I64P(unsafe_from_address=Int(py=classical_address(value)))
    var out = List[Int]()
    for i in range(count):  # driver-loop: over a fitted per-column vector
        out.append(Int(source[i]))
    return out^


def _default_mode(context: PythonObject) raises -> PythonObject:
    # `_mode()`: the process numeric mode.
    return context["backend_fields"]["_DEFAULT_MODE"]


# ---------------------------------------------------------------- SimpleImputer
def _missing_mark(missing: PythonObject) raises -> Tuple[Bool, Float64]:
    # UNPACK of `missing_values`: (a numeric value is marked NaN, the value);
    # nothing is marked for None or NaN.
    if missing is None or _is_nan_value(missing):
        return (False, 0.0)
    return (True, _float(missing))


def _mark_missing(mut p: EncProg, xo: Int, count: Int, missing: PythonObject, context: PythonObject) raises -> Int:
    """`_mark_missing`: a numeric missing_values marked NaN (mark_missing);
    the input itself when missing_values is None or NaN. UNPACK of the
    value, then the typed stage (`enc_mark_missing`)."""
    var mark = _missing_mark(missing)
    return enc_mark_missing(p.prog, xo, count, mark[0], mark[1])


def _qselect(mode: PythonObject, entries: PythonObject, context: PythonObject) raises -> Bool:
    # `_prep2_qselect(mode, 1)`: MOJOLEARN_X_PREP_FAST_QSELECT=1 (read at
    # import) on the FAST tier of a Metal device binding.
    if not Bool(py=context["prep_encode_qselect"]):
        return False
    var text = py_call(native_loaded_module("builtins").str, mode)
    if String(py_call(py_call(text.strip).lower)) != "fast":
        return False
    if py_get(entries, "x_prep_host_column") is not None:
        return False
    return prep_entry_text(entries, "x_prep_vendor") == "metal"  # read once per binding


def _imputer_fit(model: PythonObject, values: PythonObject, context: PythonObject) raises:
    # UNPACK: the parameters, read once into the typed request; the route's
    # image, its capabilities and its switches; X as an owner of the program.
    var fields = py_fields(model)
    var strategy_field = fields["strategy"]
    # A callable strategy would be user Python code run per column in fit; it
    # is refused with every other value that is not one of the four names.
    var name = String(strategy_field) if _is_str(strategy_field) else String("")
    var strategy = IMP_MEAN
    if name == "median":
        strategy = IMP_MEDIAN
    elif name == "most_frequent":
        strategy = IMP_MOST_FREQUENT
    elif name == "constant":
        strategy = IMP_CONSTANT
    elif name != "mean":
        raise Error("mojolearn: SimpleImputer strategy " + _repr(strategy_field) + " is not valid")
    var konst = strategy == IMP_CONSTANT
    var fill_value = fields["fill_value"]
    if konst and fill_value is not None and not _real_like(fill_value, context):
        raise Error("TypeError: mojolearn: SimpleImputer fill_value must be numeric")
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var mark = _missing_mark(fields["missing_values"])
    var keep_empty = _truth(fields["keep_empty_features"])
    var add_indicator = _truth(fields["add_indicator"])
    var fill = 0.0
    if konst and fill_value is not None:
        fill = _float(fill_value)
    var p = EncProg()
    var request = ImputerFit(n, d, p.own_x(values), mark[0], mark[1], strategy,
                             strategy == IMP_MEDIAN and _qselect(mode, entries, context),
                             (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0, fill, keep_empty)
    # RUN: the typed stages (bindings/prep_encode_imputer_run.mojo), then the typed run.
    var plan = imputer_fit_stage(p.prog, request)
    var ran = p.run_caps(entries, context, caps)
    # PACK: the statistics, the kept and the indicator columns.
    var counts = List[Int]()
    var any_empty = False
    for j in range(d):  # driver-loop: over the per-column non-missing counts (feature metadata)
        counts.append(Int(_word(ran, plan.stats + j)))
        any_empty = any_empty or counts[j] == 0
    var keep = List[Int]()
    var indicator = List[Int]()
    if konst or any_empty:
        py_setitem(fields, "statistics_", _take1(ran, plan.statistics, d, context))
        py_setitem(fields, "_fill", _take1(ran, plan.fill, d, context))
    else:
        var statistics = _take1(ran, plan.source, d, context)
        py_setitem(fields, "statistics_", statistics)
        py_setitem(fields, "_fill", statistics)
    for j in range(d):  # driver-loop: over the kept columns (feature metadata)
        if keep_empty or counts[j] != 0:
            keep.append(j)
    if add_indicator:
        for j in range(d):  # driver-loop: over the indicator columns (feature metadata)
            if counts[j] < n:
                indicator.append(j)
    py_setitem(fields, "_keep", _int_list(keep))
    py_setitem(fields, "_indicator", _int_list(indicator))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = ran^


def _index_list(value: PythonObject) raises -> List[Int]:
    # A fitted list of Python ints (`_keep`, `_indicator`).
    var b = native_loaded_module("builtins")
    var items = py_call(b.list, value)
    var out = List[Int]()
    for i in range(len(items)):  # driver-loop: over fitted column indices (feature metadata)
        out.append(Int(py=py_call(b.int, items[i])))
    return out^


def _to_floats(values: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(values)):  # driver-loop: over parameter words
        out.append(Float64(values[i]))
    return out^


def _imputer_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK: the fitted fields, read once into the typed request; X and the
    # fitted fill vector as owners of the program.
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var entries = scaler_entries(model, fields["numeric_mode_"], context)
    var caps = enc_caps(model, entries)
    var keep = _index_list(fields["_keep"])
    var indicator = _index_list(fields["_indicator"])
    var cols = len(keep) + len(indicator)
    var mark = _missing_mark(fields["missing_values"])
    var fill = _fitted_flat(model, "_fill", fields["_fill"], context)
    var fill_words = Int(py=fill[2])
    var p = EncProg()
    var x = p.own_x(values)
    var fill_owner = p.own(fill[1], fill_words)
    var request = ImputerApply(n, d, x, mark[0], mark[1], fill_owner, fill_words, keep^, indicator^)
    # RUN: the typed stages, then the typed run.
    var out = imputer_transform_stage(p.prog, request, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _take2(ran, out, n, cols, context)


# ---------------------------------------------------------------- KBinsDiscretizer
def _kbins_strategy(fields: PythonObject) raises -> Int:
    var strategy = fields["strategy"]
    var name = String(strategy) if _is_str(strategy) else String("")
    if name == "uniform":
        return 0
    if name == "kmeans":
        return 3
    if name == "quantile":
        var method = fields["quantile_method"]
        var text = String(method) if _is_str(method) else String("")
        if text == "averaged_inverted_cdf":
            return 1
        if text == "linear":
            return 2
        if text == "inverted_cdf":
            return 4
        if text == "closest_observation":
            return 5
        if text == "interpolated_inverted_cdf":
            return 6
        if text == "hazen":
            return 7
        if text == "weibull":
            return 8
        if text == "median_unbiased":
            return 9
        if text == "normal_unbiased":
            return 10
        raise Error("mojolearn: invalid quantile_method " + _repr(method))
    raise Error("mojolearn: invalid strategy " + _repr(strategy))


def _check_weights(source: PythonObject, n: Int, who: String, mode: PythonObject, entries: PythonObject,
                   caps: XprepCaps, context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """`_check_weights`: sample_weight as float32 words (nonnegative, length
    n) and its count of positive entries, from one device program
    (bindings/prep_encode_kbins_run.mojo `enc_weights_stage`) on the image
    and capabilities of the estimator's operation (`entries`, `caps`: read
    once by the caller)."""
    # UNPACK
    var w = _input(source, 1, False, context)
    if Int(py=w.size) != n:
        raise Error("mojolearn: sample_weight has " + String(Int(py=w.size)) + " entries; X has " + String(n) + " rows")
    var p = EncProg()
    var owner = p.own(w, n)
    # RUN
    var plan = enc_weights_stage(p.prog, owner, n, (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    var count = _word(ran, plan.stats)
    var low = _word(ran, plan.stats + 3)
    var m = _iword(ran, plan.positives)
    # a NaN is not counted by col_stats
    if Int(count) != n or low < 0:
        raise Error("mojolearn: " + who + " sample_weight must be nonnegative")
    if m == 0:
        raise Error("mojolearn: " + who + " sample_weight is all zero")
    _ = ran^
    return (w, m)


def _kbins_finite(values: PythonObject) raises:
    """scikit-learn's check_array for KBinsDiscretizer: X with a NaN or an
    infinity is refused before any program runs. `values` is xprep_values'
    tuple; a host value (kind 0) is the C-order float32 (rows, columns)
    buffer the program uploads, scanned once here (the input admission scan
    the family doors share). A DeviceRows selection (kind 1) stays on the
    device; its base was admitted by the model-selection door."""
    if Int(py=values[0]) != 0:
        return
    var count = Int(py=values[3]) * Int(py=values[4])
    if not host_all_finite_f32(F32P(unsafe_from_address=Int(py=classical_address(values[1]))), count):
        raise Error("Input X contains NaN or infinity (KBinsDiscretizer).")


def _kbins_fit(model: PythonObject, values: PythonObject, weight_source: PythonObject, context: PythonObject) raises:
    # UNPACK: the parameters, read once into the typed request; the route's
    # image and capabilities; X and the weights as owners of the program.
    var b = native_loaded_module("builtins")
    var operator = native_loaded_module("operator")
    var fields = py_fields(model)
    var encode = fields["encode"]
    var encode_text = String(encode) if _is_str(encode) else String("")
    if encode_text != "onehot" and encode_text != "onehot-dense" and encode_text != "ordinal":
        raise Error("mojolearn: invalid encode " + _repr(encode))
    var strat = _kbins_strategy(fields)
    _kbins_finite(values)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var weights = PythonObject(None)
    var given = weight_source is not None
    var positive = 0
    if given:
        var checked = _check_weights(weight_source, n, "KBinsDiscretizer", mode, entries, caps, context)
        weights = checked[0]
        positive = checked[1]
    var subsample = fields["subsample"]
    var resample = subsample is not None and Bool(py=py_call(operator.gt, PythonObject(n), subsample))
    var draw = KB_DRAW_NONE
    var draws = 0
    var seed_lo = 0
    var seed_hi = 0
    if resample:
        var seed = _seed64(fields["random_state"])
        seed_lo = seed[0]
        seed_hi = seed[1]
        draws = Int(py=py_call_typed(b.int, subsample))
        # a weighted resample with replacement is drawn in the fit's program,
        # and the edges of its rows are then unweighted
        draw = KB_DRAW_WEIGHTED if given else KB_DRAW_PLAIN
    var weighted = given and not resample
    var nb = List[Int]()
    var n_bins = fields["n_bins"]
    if _int_like(n_bins, context):
        var k = Int(py=py_call_typed(b.int, n_bins))
        for _ in range(d):  # driver-loop: over the per-column n_bins argument
            nb.append(k)
    else:
        var items = py_call_typed(b.list, n_bins)
        for i in range(len(items)):  # driver-loop: over the per-column n_bins argument
            nb.append(Int(py=py_call_typed(b.int, items[i])))
    var smallest = nb[0] if len(nb) > 0 else 0
    var nbmax = 0
    for i in range(len(nb)):  # driver-loop: over the per-column n_bins argument
        smallest = min(smallest, nb[i])
        nbmax = max(nbmax, nb[i])
    if len(nb) != d or smallest < 2:
        raise Error("mojolearn: n_bins must be >= 2 per feature")
    if weighted and strat != 0 and strat != 1 and strat != 3 and strat != 4:
        raise Error("mojolearn: When fitting with strategy='quantile' and sample weights, quantile_method "
                    + "should either be set to 'averaged_inverted_cdf' or 'inverted_cdf', got "
                    + "quantile_method='" + _str_text(fields["quantile_method"]) + "' instead.")
    var p = EncProg()
    var x = p.own_x(values)
    var w = p.own(weights, n) if given else NONE
    var request = KBinsFit(n, d, x, strat, nb^, nbmax, draw, draws, seed_lo, seed_hi, weighted, w, positive,
                           (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0)
    # RUN: the typed stages (bindings/prep_encode_kbins_run.mojo), then the typed run.
    var plan = kbins_fit_stage(p.prog, request)
    var ran = p.run_caps(entries, context, caps)
    # PACK: the per-column edges and bin counts.
    var edges = plan.edges
    var ne = plan.counts
    var output = context["output_context"]
    var bin_edges = Python.list()
    var n_bins_ = classical_typed(Python.tuple(d), "<i8", output)
    var dest = I64P(unsafe_from_address=Int(py=classical_address(n_bins_)))
    _ = n_bins_  # keep the Python object alive through the native call above
    for j in range(d):  # driver-loop: over the per-column edge counts (feature metadata)
        var count = Int(_word(ran, ne + j))
        _ = py_call(bin_edges.append, _take1(ran, edges + j * (nbmax + 1), count, context))
        dest[j] = Int64(count - 1)
        _ = n_bins_  # keep the Python object alive through the native call above
    py_setitem(fields, "bin_edges_", bin_edges)
    py_setitem(fields, "n_bins_", n_bins_)
    py_setitem(fields, "_edges", _take1(ran, edges, d * (nbmax + 1), context))
    py_setitem(fields, "_ne", _take1(ran, ne, d, context))
    py_setitem(fields, "_stride", nbmax + 1)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = weights^
    _ = ran^


def _kbins_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK: the fitted fields, read once into the typed request; X and the
    # fitted edge vectors as owners of the program.
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    _kbins_finite(values)
    var entries = scaler_entries(model, fields["numeric_mode_"], context)
    var caps = enc_caps(model, entries)
    var stride = Int(py=fields["_stride"])
    var edges = _fitted_flat(model, "_edges", fields["_edges"], context)
    var counts = _fitted_flat(model, "_ne", fields["_ne"], context)
    var edges_words = Int(py=edges[2])
    var counts_words = Int(py=counts[2])
    var encode = fields["encode"]
    var onehot = not (_is_str(encode) and String(encode) == "ordinal")
    var widths = List[Int]()
    var cols = d
    if onehot:
        widths = _i64_values(fields["n_bins_"], context)
        cols = 0
        for j in range(len(widths)):  # driver-loop: over the fitted per-column bin counts
            cols += widths[j]
    var p = EncProg()
    var x = p.own_x(values)
    var edges_owner = p.own(edges[1], edges_words)
    var counts_owner = p.own(counts[1], counts_words)
    var request = KBinsApply(n, d, x, edges_owner, edges_words, counts_owner, counts_words, stride, onehot, widths^)
    # RUN: the typed stages, then the typed run.
    var result = kbins_transform_stage(p.prog, request, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _take2(ran, result, n, cols, context)


def _kbins_inverse(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK: the fitted fields, read once into the typed request; X and the
    # fitted edges as owners of the program.
    var fields = py_fields(model)
    var widths = _i64_values(fields["n_bins_"], context)
    var d = len(widths)
    var n = Int(py=values[3])
    var cols = Int(py=values[4])
    var entries = scaler_entries(model, fields["numeric_mode_"], context)
    var caps = enc_caps(model, entries)
    var encode = fields["encode"]
    var ordinal = _is_str(encode) and String(encode) == "ordinal"
    var why = String("hold codes that name no bin")
    if ordinal:
        _check_width(model, cols)
    else:
        var total = 0
        for j in range(d):  # driver-loop: over the fitted per-column bin counts
            total += widths[j]
        if cols != total:
            raise Error("mojolearn: X has " + String(cols) + " columns, expected " + String(total))
        why = "can not be inverted because they contain all zeros"
    var edges = _fitted_flat(model, "_edges", fields["_edges"], context)
    var edges_words = Int(py=edges[2])
    var stride = Int(py=fields["_stride"])
    var p = EncProg()
    var x = p.own_x(values)
    var edges_owner = p.own(edges[1], edges_words)
    var request = KBinsInverse(n, cols, x, ordinal, widths^, edges_owner, edges_words, stride)
    # RUN: the typed stages, then the typed run.
    var plan = kbins_inverse_stage(p.prog, request, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK: the refusal's first rows, or the bin centers.
    var bad = _bad_rows(ran, (plan.bad_rows, plan.bad_count), 10)
    if len(bad) > 0:
        raise Error("mojolearn: samples " + _list_text(bad) + " " + why)
    return _take2(ran, plan.centers, n, d, context)


# ---------------------------------------------------------------- categories
def _fast_unique(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_fast_on("UNIQUE", mode)`: the FAST binding exports x_prep_fast_unique.
    return String(mode) == "fast" and py_get(entries, "x_prep_fast_unique") is not None


def _cls2_cat(mode: PythonObject, entries: PythonObject) raises -> Int:
    # `_cls2_cat`: bit 1 PACK, bit 2 PRESENT on the FAST binding built with them.
    if String(mode) != "fast":
        return 0
    return prep_entry_word(entries, "x_prep_cls2_cat")  # read once per binding


def _cat_fit_pack(ran: PythonObject, plan: CatFitPlan, n: Int, d: Int, context: PythonObject) raises -> PythonObject:
    """PACK of a category fit: the list of the d float32 category Arrays;
    None when a packed plan did not hold the values (a column that does not
    suit the presence flags, or more values than the packed region)."""
    var out = Python.list()
    if not plan.packed:
        for c in range(d):  # driver-loop: slices the per-column category outputs (feature count)
            _ = py_call(out.append, _take1(ran, plan.values + c * n, Int(_word(ran, plan.counts + c)), context))
        return out
    if plan.flags >= 0:
        for j in range(d):  # driver-loop: over the per-column overflow flags (feature metadata)
            if _iword(ran, plan.flags + j) != 0:
                return PythonObject(None)
    var counts = List[Int]()
    var total = 0
    for c in range(d):  # driver-loop: over the per-column category counts (feature metadata)
        counts.append(Int(_word(ran, plan.counts + c)))
        total += counts[c]
    if total > plan.cap:
        return PythonObject(None)
    var o = 0
    for c in range(d):  # driver-loop: slices the per-column category outputs (feature count)
        _ = py_call(out.append, _take1(ran, plan.values + o, counts[c], context))
        o += counts[c]
    return out


def _cat_fit_try(values: PythonObject, words: List[Float64], n: Int, d: Int, route: Int, entries: PythonObject,
                 caps: XprepCaps, context: PythonObject) raises -> PythonObject:
    """One category fit program on `route`: X as `xprep_values` converted it
    (an owner of the program), or, when `values` is None, the n label words
    `words` as program-owned float32 words (one column)."""
    # UNPACK
    var p = EncProg()
    var x = NONE
    var held = NONE
    if values is None:
        held = p.prog.put_floats(words)
    else:
        x = p.own_x(values)
    # RUN: the typed stages (bindings/prep_encode_cats_run.mojo), then the typed run.
    var plan = cat_fit_stage(p.prog, CatFit(n, d, x, held, route))
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _cat_fit_pack(ran, plan, n, d, context)


def _cat_fit(values: PythonObject, words: List[Float64], n: Int, d: Int, mode: PythonObject, entries: PythonObject,
             caps: XprepCaps, context: PythonObject) raises -> PythonObject:
    # The image's route, read once: the packed program first where the FAST
    # image has it, then the sort with a run scan or with unique_cols.
    var fast = _fast_unique(mode, entries)
    var cls2 = _cls2_cat(mode, entries)
    if (cls2 & 1) != 0 and fast:
        var got = _cat_fit_try(values, words, n, d, CAT_FIT_PACK_PRESENT if (cls2 & 2) != 0 else CAT_FIT_PACK_SORT,
                               entries, caps, context)
        if got is not None:
            return got
    return _cat_fit_try(values, words, n, d, CAT_FIT_SCAN if fast else CAT_FIT_UNIQUE, entries, caps, context)


def _fit_categories_caps(values: PythonObject, mode: PythonObject, entries: PythonObject, caps: XprepCaps,
                         context: PythonObject) raises -> PythonObject:
    """`_fit_categories`: per column the sorted distinct values (-0.0 folded
    into 0.0), on the image and capabilities of the estimator's operation."""
    return _cat_fit(values, List[Float64](), Int(py=values[3]), Int(py=values[4]), mode, entries, caps, context)


def _fit_category_words(words: List[Float64], mode: PythonObject, entries: PythonObject, caps: XprepCaps,
                        context: PythonObject) raises -> PythonObject:
    """`_fit_categories` of one column of label words (a numeric Python
    list's values, rounded to float32 as program-owned words): the float32
    Array of its sorted distinct values."""
    return _cat_fit(PythonObject(None), words, len(words), 1, mode, entries, caps, context)[0]


def _fit_categories(values: PythonObject, mode: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # `_fit_categories_caps` for a stateless caller (no estimator route): the
    # image's capabilities read from `entries` by name.
    return _fit_categories_caps(values, mode, entries, xprep_caps(entries), context)


def _fit_categories_with_codes(values: PythonObject, entries: PythonObject, caps: XprepCaps,
                               context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_fit_categories_with_codes` (C08): the categories and the (n, d)
    inverse from one program (sort_cols, unique_inverse)."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var p = EncProg()
    var x = p.own_x(values)
    var plan = cat_fit_stage(p.prog, CatFit(n, d, x, NONE, CAT_FIT_INVERSE))
    var ran = p.run_caps(entries, context, caps)
    return (_cat_fit_pack(ran, plan, n, d, context), _take2(ran, plan.inverse, n, d, context))


def _c08_route(mode: PythonObject, entries: PythonObject, bit: Int) raises -> Bool:
    # `_c08_route`: the IDENTICAL binding's x_prep_c08_routes bit.
    if String(mode) != "identical":
        return False
    return (prep_entry_word(entries, "x_prep_c08_routes") & bit) != 0  # read once per binding


def _user_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # A user list argument's values as a Python list (`v.tolist()` for an
    # array type, a mojolearn Array read as its float32 words).
    var b = native_loaded_module("builtins")
    if py_type(value) is context["array_class"]:
        var out = Python.list()
        if Int(py=value.size) == 0:
            return out
        var flat = _input(value, 1, False, context)
        var source = _ptr(flat)
        _ = flat  # keep the Python object alive through the native call above
        for i in range(Int(py=flat.size)):  # driver-loop: over one user category or drop list
            _ = py_call(out.append, PythonObject(Float64(source[i])))
            _ = flat  # keep the Python object alive through the native call above
        return out
    if Bool(py=py_call(b.hasattr, value, "tolist")):
        return py_call(b.list, py_call(value.tolist))
    return py_call(b.list, value)


def _f32(value: PythonObject) raises -> Float32:
    # array.array("f", [float(v)])[0]
    return Float32(_float(value))


def _cat_values(cats: PythonObject, context: PythonObject) raises -> List[Float32]:
    # A fitted category Array's float32 words.
    var out = List[Float32]()
    var count = Int(py=cats.size)
    if count == 0:
        return out^
    var flat = _input(cats, 1, False, context)
    var source = _ptr(flat)
    _ = flat  # keep the Python object alive through the native call above
    for i in range(count):  # driver-loop: over one column's fitted categories
        out.append(source[i])
        _ = flat  # keep the Python object alive through the native call above
    return out^


def _f32_array(values: List[Float32], context: PythonObject) raises -> PythonObject:
    # Array.from_list(values, "<f4"); an empty list is an empty Array.
    if len(values) == 0:
        return classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(0), "<f4", "C", context["output_context"])
    var out = classical_typed(Python.tuple(len(values)), "<f4", context["output_context"])
    var dest = _ptr(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # driver-loop: over one column's category words
        dest[i] = values[i]
        _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- fitted category record
# The fitted category state a transform, inverse_transform or infrequent
# step reads (the (d, kmax) float32 category block, kmax, the per-column
# sizes and their float32 words) used to be rebuilt and re-admitted on every
# call: per column a full `_input` admission of the category Array (about
# 40-60 CPython calls each, ~700-900 per 16-column transform). It is built
# once per fitted `categories_` and kept in the prepared-owner registry
# (bindings/native_prepared_owner.mojo) keyed weakly by the model, outside
# its `__dict__` (perf2-prep). The record is valid while `categories_` is the
# same object holding the same column objects (identity compares, no Python
# protocol runs); a refit, set_params + fit, a reassignment or an item
# replacement misses and rebuilds. In-place writes (hazard class A, p3-prep):
# the record is only kept when every column is an exact mojolearn Array, and
# those Arrays are made read-only at fit and when the record is taken
# (`_category_freeze`), so `enc.categories_[0][1] = 9.0` through the public
# NumPy view raises NumPy's read-only error instead of silently diverging
# from the record. A column that cannot be frozen (a NumPy array or a list
# assigned by the user) keeps no record: the block is rebuilt per call. A
# copied or unpickled model has no record and builds one on first use
# (pickle keeps the read-only flag). Same float32 words as before.
comptime _CAT_SLOT = "prep_encode_category_block_v1"


def _category_freeze(categories: PythonObject, context: PythonObject) raises -> Bool:
    """Marks every fitted category column read-only; False (nothing kept)
    when a column is not an exact mojolearn Array (its writes cannot be
    refused, so no copy of it may be served later)."""
    var b = native_loaded_module("builtins")
    var kind = py_type(categories)
    if kind is not b.list and kind is not b.tuple:
        return False
    var d = len(categories)
    for j in range(d):  # driver-loop: type check of the per-column category arrays
        if py_type(categories[j]) is not context["array_class"]:
            return False
    for j in range(d):  # driver-loop: freezes the per-column category arrays
        py_setattr(categories[j], "_readonly", PythonObject(True))
    return True


def _category_build(categories: PythonObject, context: PythonObject) raises -> PythonObject:
    """(categories, column objects, block, kmax, sizes, size words, block
    words): `_category_block`'s block and `_sizes` built once."""
    var b = native_loaded_module("builtins")
    var d = len(categories)
    var items = Python.list()
    var sizes = List[Int]()
    var kmax = 0
    for j in range(d):  # driver-loop: over the per-column category arrays
        var cats = categories[j]
        _ = py_call(items.append, cats)
        var k = Int(py=cats.size)
        sizes.append(k)
        kmax = max(kmax, k)
    var words = max(d * kmax, 1)
    var block = classical_typed(Python.tuple(words), "<f4", context["output_context"])
    var dest = _ptr(block)
    _ = block  # keep the Python object alive through the native call above
    for j in range(d):  # driver-loop: over the per-column category arrays
        var cats = _cat_values(items[j], context)
        for i in range(len(cats)):  # driver-loop: over one column's categories
            dest[j * kmax + i] = cats[i]
            _ = block  # keep the Python object alive through the native call above
    var size_list = Python.list()
    for j in range(d):  # driver-loop: over the per-column category counts
        _ = py_call(size_list.append, PythonObject(sizes[j]))
    return Python.tuple(categories, py_call(b.tuple, items), block, kmax, py_call(b.tuple, size_list),
                        _floats(_to_floats(sizes), context), words)


def _category_same(record: PythonObject, categories: PythonObject) raises -> Bool:
    if record is None or record[0] is not categories:
        return False
    var held = record[1]
    var d = len(held)
    if len(categories) != d:
        return False
    for j in range(d):  # driver-loop: identity compare of the per-column category arrays
        if categories[j] is not held[j]:
            return False
    return True


def _category_record(owner: PythonObject, categories: PythonObject, context: PythonObject) raises -> PythonObject:
    """The category record of `categories` (the fitted `categories_` of
    `owner`); without an owner (a fit's fresh lists) it is built for this
    call only."""
    if owner is None:
        return _category_build(categories, context)
    var mods = NativeModules()
    var record = prepared_get(mods, _CAT_SLOT, owner)
    if _category_same(record, categories):
        return record
    record = _category_build(categories, context)
    if _category_freeze(categories, context):
        prepared_put(mods, _CAT_SLOT, owner, record)
    else:
        prepared_put(mods, _CAT_SLOT, owner, PythonObject(None))
    return record


def _cat_block(mut p: EncProg, record: PythonObject, sizes: Bool) raises -> CatBlock:
    """UNPACK of a category record (`_category_record`) for a typed stage
    builder: its (d, kmax) block and, with `sizes`, its per-column size
    words become owners of this program."""
    var words = Int(py=record[6])
    var block = p.own(record[2], words)
    var count = max(len(record[4]), 1)
    return CatBlock(block, words, Int(py=record[3]), p.own(record[5], count) if sizes else NONE, count if sizes else 0)


def _group_lists(grouping: PythonObject) raises -> List[List[Int]]:
    # The fitted `_grouping` as native lists: column j's grouped code of each
    # category, empty for a column that is not grouped (None).
    var groups = List[List[Int]]()
    for j in range(len(grouping)):  # driver-loop: over the per-column grouping lists (feature metadata)
        if grouping[j] is None:
            groups.append(List[Int]())
        else:
            groups.append(_index_list(grouping[j]))
    return groups^


def _sizes(categories: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for j in range(len(categories)):  # driver-loop: over the per-column category arrays
        out.append(Int(py=categories[j].size))
    return out^


def _record_column(record: PythonObject, j: Int) raises -> List[Float32]:
    # Column j's float32 category words (`_cat_values`) from the record's
    # block (row j, the column's size words).
    var size = Int(py=record[4][j])
    var kmax = Int(py=record[3])
    var block = record[2]
    var source = _ptr(block)
    var out = List[Float32]()
    for i in range(size):  # driver-loop: over one column's fitted categories
        out.append(source[j * kmax + i])
        _ = block  # keep the Python object alive through the reads above
    return out^


def _record_last_is_nan(record: PythonObject, j: Int) raises -> Bool:
    # `_last_is_nan` of column j from the record.
    var values = _record_column(record, j)
    return len(values) > 0 and values[len(values) - 1] != values[len(values) - 1]


def _record_sizes(record: PythonObject) raises -> List[Int]:
    # `_sizes` from a category record (exact ints, no attribute reads).
    var held = record[4]
    var out = List[Int]()
    for j in range(len(held)):  # driver-loop: over the per-column category counts
        out.append(Int(py=held[j]))
    return out^


def _codes(mut p: EncProg, values: PythonObject, categories: PythonObject, context: PythonObject, device_only: Bool,
           owner: PythonObject) raises -> Tuple[Int, Int]:
    """`_codes`: each element's category index (or -1) and, unless
    device_only, each column's unknown count. Returns (codes, neg).
    `owner`: the fitted model whose `categories_` these are (its category
    record is reused), or None. UNPACK of X and the record, then the typed
    stages (bindings/prep_encode_cats_run.mojo `cat_codes_stage`)."""
    var record = _category_record(owner, categories, context)
    var x = p.own_x(values)
    var block = _cat_block(p, record, True)
    var staged = cat_codes_stage(p.prog, Int(py=values[3]), Int(py=values[4]), x, block, device_only)
    return (staged.codes, staged.unknown)


def _unknown_columns(ran: PythonObject, neg: Int, d: Int) raises -> List[Int]:
    # The columns whose unknown count the device set.
    var bad = List[Int]()
    for j in range(d):  # driver-loop: over the per-column unknown flags (feature metadata)
        if _word(ran, neg + j) > 0:
            bad.append(j)
    return bad^


def _given_categories(categories: PythonObject, values: PythonObject, entries: PythonObject, caps: XprepCaps,
                      check_unknown: Bool, who: String, context: PythonObject) raises -> PythonObject:
    """`_given_categories`: one numeric list per column, sorted ascending with
    at most a NaN last, stored as float32; with check_unknown a training
    value outside its column's list is refused."""
    var b = native_loaded_module("builtins")
    var d = Int(py=values[4])
    var lists = py_call(b.list, categories)
    if len(lists) != d:
        raise Error("mojolearn: " + who + " categories has " + String(len(lists)) + " lists; X has " + String(d) + " features")
    var out = Python.list()
    for j in range(d):  # driver-loop: validates the user categories argument (user category lists)
        var raw = _user_values(lists[j], context)
        var f32 = List[Float32]()
        for i in range(len(raw)):  # driver-loop: over one user category list
            f32.append(_f32(raw[i]))
        var nums = List[Float32]()
        for i in range(len(f32)):  # driver-loop: over one user category list
            if f32[i] == f32[i]:
                nums.append(f32[i])
        var total = len(f32)
        if len(nums) < total - 1 or (len(nums) < total and f32[total - 1] == f32[total - 1]):
            raise Error("mojolearn: " + who + " categories[" + String(j) + "]: nan must be the last category")
        for i in range(1, len(nums)):  # driver-loop: over one user category list
            if nums[i] < nums[i - 1]:
                raise Error("mojolearn: " + who + " unsorted categories are not supported for numerical categories")
        for i in range(1, len(nums)):  # driver-loop: over one user category list
            if nums[i] == nums[i - 1]:
                raise Error("mojolearn: " + who + " categories[" + String(j) + "] has values equal in float32")
        if total == 0:
            raise Error("mojolearn: " + who + " categories[" + String(j) + "] is empty")
        var canon = List[Float32]()
        for i in range(len(nums)):  # driver-loop: over one user category list
            canon.append(Float32(0.0) if nums[i] == 0 else nums[i])
        if len(nums) < total:
            canon.append(Float32(Float64(py=py_call(b.float, "nan"))))
        _ = py_call(out.append, _f32_array(canon, context))
    if check_unknown:
        var p = EncProg()
        var staged = _codes(p, values, out, context, False, PythonObject(None))
        var ran = p.run_caps(entries, context, caps)
        var bad = _unknown_columns(ran, staged[1], d)
        if len(bad) > 0:
            raise Error("mojolearn: " + who + " found unknown categories in column(s) " + _list_text(bad) + " during fit")
    return out


# ---------------------------------------------------------------- infrequent categories
def _integral(value: PythonObject, context: PythonObject) raises -> Bool:
    # numbers.Integral: Python ints (bool included), NumPy integers.
    return _int_like(value, context)


def _is_bool(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    return Bool(py=py_call(b.isinstance, value, b.bool))


def _cmp(name: String, a: PythonObject, b: PythonObject) raises -> Bool:
    # A Python comparison (operator.<name>) of two numbers.
    var operator = native_loaded_module("operator")
    return _truth(py_call(py_get(operator.__dict__, PythonObject(name)), a, b))


def _check_infrequent_params(fields: PythonObject, context: PythonObject) raises -> Bool:
    """`_check_infrequent_params`: the reference's constraints; True when
    grouping is on."""
    var mf = fields["min_frequency"]
    var mc = fields["max_categories"]
    if mf is not None:
        var ok = (_integral(mf, context) and not _is_bool(mf) and _cmp("ge", mf, PythonObject(1))) or (
            _real_like(mf, context) and not _integral(mf, context) and _cmp("lt", PythonObject(0), mf) and _cmp("lt", mf, PythonObject(1)))
        if not ok:
            raise Error("mojolearn: min_frequency must be an int >= 1 or a float in (0, 1), got " + _repr(mf))
    if mc is not None and not (_integral(mc, context) and not _is_bool(mc) and _cmp("ge", mc, PythonObject(1))):
        raise Error("mojolearn: max_categories must be an int >= 1, got " + _repr(mc))
    return mc is not None or mf is not None


def _infrequent_threshold(n: Int, min_frequency: PythonObject, context: PythonObject) raises -> Int:
    """`_infrequent_threshold`: an int min_frequency as is; a fraction f,
    ceil(n * f) in binary64; none, 0."""
    if min_frequency is None:
        return 0
    var b = native_loaded_module("builtins")
    if _integral(min_frequency, context):
        return Int(py=py_call(b.int, min_frequency))
    return Int(ceil(Float64(n) * _float(min_frequency)))


def _fit_infrequent(model: PythonObject, values: PythonObject, entries: PythonObject, caps: XprepCaps,
                    ignore_missing: Bool, inverse: PythonObject, context: PythonObject) raises:
    """`_fit_infrequent`: one program (code_counts, c2_inf_m0, c2_inf_m1,
    c2_inf_map) whose masks and codes come back as the fitted per-column
    `_infrequent`, `_grouping` and `infrequent_categories_`."""
    # UNPACK
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var cats_all = fields["categories_"]
    # The fitted category record (built here once; the transforms reuse it):
    # sizes and each column's float32 words come from it, not from a fresh
    # admission of every category array.
    var record = _category_record(model, cats_all, context)
    var sizes = _record_sizes(record)
    var ks = List[Int]()
    var kmax = 0
    for j in range(len(sizes)):  # driver-loop: one size per column
        var size = sizes[j]
        ks.append(size - 1 if (ignore_missing and size > 0 and _record_last_is_nan(record, j)) else size)
        kmax = max(kmax, size)
    var max_categories = fields["max_categories"]
    var maxc = -1 if max_categories is None else Int(py=py_call(native_loaded_module("builtins").int, max_categories))
    var threshold = _infrequent_threshold(n, fields["min_frequency"], context)
    var p = EncProg()
    var from_inverse = inverse is not None
    var x = NONE
    var held = NONE
    var block = CatBlock(NONE, 0, 0, NONE, 0)
    if from_inverse:
        held = p.own(inverse, n * d)
    else:
        x = p.own_x(values)
        block = _cat_block(p, record, True)
    var request = InfrequentFit(n, d, x, block.copy(), from_inverse, held, ks.copy(), kmax, threshold, maxc)
    # RUN
    var plan = cat_infrequent_stage(p.prog, request)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    var m1 = plan.mask
    var mp = plan.map
    var infrequent = Python.list()
    var grouping = Python.list()
    var categories = Python.list()
    for j in range(len(ks)):  # driver-loop: the device's tables as the fitted per-column lists
        var inf = List[Int]()
        for i in range(ks[j]):  # driver-loop: over one column's categories
            if _iword(ran, m1 + j * kmax + i) != 0:
                inf.append(i)
        if len(inf) == 0:
            _ = py_call(infrequent.append, PythonObject(None))
            _ = py_call(grouping.append, PythonObject(None))
            _ = py_call(categories.append, PythonObject(None))
            continue
        _ = py_call(infrequent.append, _int_list(inf))
        var group = List[Int]()
        for i in range(ks[j]):  # driver-loop: over one column's categories
            group.append(_iword(ran, mp + j * kmax + i))
        _ = py_call(grouping.append, _int_list(group))
        var values_j = _record_column(record, j)
        var picked = List[Float32]()
        for i in range(len(inf)):  # driver-loop: over one column's infrequent categories
            picked.append(values_j[inf[i]])
        _ = py_call(categories.append, _f32_array(picked, context))
    py_setitem(fields, "_infrequent", infrequent)
    py_setitem(fields, "_grouping", grouping)
    py_setitem(fields, "infrequent_categories_", categories)
    _ = ran^


def _num_text(value: PythonObject) raises -> String:
    # A number in an f-string (format(value, "")).
    return String(py_call(native_loaded_module("builtins").format, value, ""))


# ---------------------------------------------------------------- OrdinalEncoder
def _ordinal_fit(model: PythonObject, values: PythonObject, capture: Bool, context: PythonObject) raises -> PythonObject:
    """OrdinalEncoder.fit; with `capture` (the C08 fit_transform) the fit's
    (n, d) inverse is returned for the transform, else None."""
    var fields = py_fields(model)
    var grouping = _check_infrequent_params(fields, context)
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    if handle_text != "error" and handle_text != "use_encoded_value":
        raise Error("mojolearn: invalid handle_unknown " + _repr(handle))
    var unknown_value = fields["unknown_value"]
    if handle_text == "use_encoded_value" and not _real_like(unknown_value, context):
        raise Error("TypeError: mojolearn: unknown_value must be a number when handle_unknown='use_encoded_value'")
    var emv = fields["encoded_missing_value"]
    if not _real_like(emv, context):
        raise Error("TypeError: mojolearn: encoded_missing_value must be a number (or NaN)")
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var auto = _is_str(fields["categories"]) and String(fields["categories"]) == "auto"
    var inverse = PythonObject(None)
    if capture and auto:
        var got = _fit_categories_with_codes(values, entries, caps, context)
        py_setitem(fields, "categories_", got[0])
        inverse = got[1]
    elif auto:
        py_setitem(fields, "categories_", _fit_categories_caps(values, mode, entries, caps, context))
    else:
        py_setitem(fields, "categories_", _given_categories(fields["categories"], values, entries, caps, handle_text == "error", "OrdinalEncoder", context))
    var cats = fields["categories_"]
    _ = _category_freeze(cats, context)  # hazard A: fitted categories are read-only (p3-prep)
    # The fitted category record, built here once (the transforms reuse it):
    # the sizes and the NaN slots come from it, not from an admission of
    # every category array.
    var record = _category_record(model, cats, context)
    var sizes = _record_sizes(record)
    var missing = List[Int]()
    var any_missing = False
    for j in range(len(sizes)):  # driver-loop: per-column NaN category slot (fitted per-column category counts)
        missing.append(sizes[j] - 1 if sizes[j] > 0 and _record_last_is_nan(record, j) else -1)
        any_missing = any_missing or missing[j] >= 0
    py_setitem(fields, "_missing", _int_list(missing))
    py_setitem(fields, "_infrequent", PythonObject(None))
    py_setitem(fields, "_grouping", PythonObject(None))
    if grouping:
        _fit_infrequent(model, values, entries, caps, True, inverse, context)
    var cards = List[Int]()
    for j in range(len(sizes)):  # driver-loop: per-column category cardinalities (fitted per-column category counts)
        cards.append(sizes[j] - (1 if missing[j] >= 0 else 0))
    if grouping:
        cards = cat_grouped_sizes(cards, _group_lists(fields["_grouping"]))
    if handle_text == "use_encoded_value" and not _is_nan_value(unknown_value):
        for j in range(len(cards)):  # driver-loop: validates unknown_value against cardinalities
            if _cmp("le", PythonObject(0), unknown_value) and _cmp("lt", unknown_value, PythonObject(cards[j])):
                raise Error("mojolearn: the used value for unknown_value " + _num_text(unknown_value) + " is one of the "
                            + "values already used for encoding the seen categories.")
    if any_missing and not _is_nan_value(emv):
        var bad = List[Int]()
        for j in range(len(cards)):  # driver-loop: validates encoded_missing_value argument
            if missing[j] >= 0 and _cmp("le", PythonObject(0), emv) and _cmp("lt", emv, PythonObject(cards[j])):
                bad.append(j)
        if len(bad) > 0:
            raise Error("mojolearn: encoded_missing_value (" + _num_text(emv) + ") is already "
                        + "used to encode a known category in features: " + _list_text(bad))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    return inverse


def _ordinal_transform(model: PythonObject, values: PythonObject, inverse: PythonObject, context: PythonObject) raises -> PythonObject:
    """OrdinalEncoder `_transform_encoded`: lookup (or the C08 fit's
    inverse), the grouping remap, where_neg and where_code."""
    # UNPACK: the parameters and fitted fields, read once into the typed
    # request; X (or the fit's inverse) and the category block as owners.
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var entries = scaler_entries(model, fields["numeric_mode_"], context)
    var caps = enc_caps(model, entries)
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    var use_unknown = handle_text == "use_encoded_value"
    var unknown_value = _float(fields["unknown_value"]) if use_unknown else 0.0
    var missing = _index_list(fields["_missing"])
    var any_missing = False
    for j in range(len(missing)):  # driver-loop: checks for a NaN category slot (per-column NaN category slots)
        any_missing = any_missing or missing[j] >= 0
    var encoded_missing = _float(fields["encoded_missing_value"]) if any_missing else 0.0
    var grouping = fields["_grouping"]
    var grouped = grouping is not None
    var groups = List[List[Int]]()
    if grouped:
        groups = _group_lists(grouping)
    var p = EncProg()
    var from_inverse = inverse is not None
    var x = NONE
    var held = NONE
    var block = CatBlock(NONE, 0, 0, NONE, 0)
    if from_inverse:
        held = p.own(inverse, Int(py=inverse.size))
    else:
        var record = _category_record(model, fields["categories_"], context)
        x = p.own_x(values)
        block = _cat_block(p, record, True)
    var request = OrdinalApply(n, d, x, block.copy(), from_inverse, held, grouped, groups^, use_unknown, unknown_value,
                               any_missing, missing^, encoded_missing)
    # RUN: the typed stages (bindings/prep_encode_ordinal_run.mojo), then the typed run.
    var plan = ordinal_transform_stage(p.prog, request)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    if handle_text == "error":
        var bad = _unknown_columns(ran, plan.unknown, d)
        if len(bad) > 0:
            raise Error("mojolearn: OrdinalEncoder found unknown categories in column(s) " + _list_text(bad) + " during transform")
    return _take2(ran, plan.encoded, n, d, context)


def _ordinal_inverse(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """OrdinalEncoder.inverse_transform: ord_inverse (`_inverse_codes`), the
    inverse grouping remap, cat_gather and the bad-row compaction."""
    # UNPACK: the parameters and fitted fields, read once into the typed
    # request; X and the category block as owners.
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var entries = scaler_entries(model, fields["numeric_mode_"], context)
    var caps = enc_caps(model, entries)
    var handle = fields["handle_unknown"]
    var encoded = _is_str(handle) and String(handle) == "use_encoded_value"
    var record = _category_record(model, fields["categories_"], context)
    var sizes = _record_sizes(record)
    var grouping = fields["_grouping"]
    var grouped = grouping is not None
    var groups = List[List[Int]]()
    if grouped:
        groups = _group_lists(grouping)
        sizes = cat_grouped_sizes(sizes, groups)
    var missing = _index_list(fields["_missing"])
    var encoded_missing = _float(fields["encoded_missing_value"])
    var unknown_value = _float(fields["unknown_value"]) if encoded else 0.0
    var p = EncProg()
    var x = p.own_x(values)
    var block = _cat_block(p, record, False)
    var request = OrdinalInverse(n, d, x, block.copy(), grouped, groups^, sizes^, missing^, encoded_missing, encoded,
                                 unknown_value)
    # RUN: the typed stages, then the typed run.
    var plan = ordinal_inverse_stage(p.prog, request)
    var ran = p.run_caps(entries, context, caps)
    # PACK: the refusal's first rows, or the category values.
    var bad = _bad_rows(ran, (plan.bad_rows, plan.bad_count), 10)
    if len(bad) > 0:
        raise Error("mojolearn: rows " + _list_text(bad) + " hold codes that name no category")
    return _take2(ran, plan.values, n, d, context)


# ---------------------------------------------------------------- OneHotEncoder
def _grouped_sizes(fields: PythonObject, owner: PythonObject, context: PythonObject) raises -> List[Int]:
    # With an `owner` (a fitted model), the sizes come from its category record.
    var sizes = _sizes(fields["categories_"]) if owner is None else _record_sizes(_category_record(owner, fields["categories_"], context))
    var groups = fields["_grouping"]
    if groups is None:
        return sizes^
    return cat_grouped_sizes(sizes, _group_lists(groups))


def _set_drop_idx(model: PythonObject, context: PythonObject) raises:
    """`_set_drop_idx`: `_drop_after` in grouped codes, drop_idx_ in category
    indices (None, or one int or None per column)."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var sizes = _grouped_sizes(fields, model, context)
    var d = len(sizes)
    var groups = fields["_grouping"]
    var drop = fields["drop"]
    if drop is None:
        py_setitem(fields, "_drop_after", PythonObject(None))
        py_setitem(fields, "drop_idx_", PythonObject(None))
        return
    var after = Python.list()
    var text = String(drop) if _is_str(drop) else String("")
    if text == "first":
        for _ in range(d):  # driver-loop: over the per-column drop indices
            _ = py_call(after.append, PythonObject(0))
    elif text == "if_binary":
        for j in range(d):  # driver-loop: drop index for binary columns
            _ = py_call(after.append, PythonObject(0) if sizes[j] == 2 else PythonObject(None))
    else:
        var vals = _user_values(drop, context)
        if len(vals) != d:
            raise Error("mojolearn: `drop` should have length equal to the number of features ("
                        + String(d) + "), got " + String(len(vals)))
        var missing = String()
        var any_missing = False
        var cats = fields["categories_"]
        for j in range(d):  # driver-loop: validates the user drop argument (per-column drop arguments)
            var v = vals[j]
            var cl = _cat_values(cats[j], context)
            var hit = -1
            if _is_nan_value(v):
                if len(cl) > 0 and cl[len(cl) - 1] != cl[len(cl) - 1]:
                    hit = len(cl) - 1
            else:
                var fv = _f32(v)
                for i in range(len(cl)):  # driver-loop: finds the user drop category in its column
                    if cl[i] == fv:
                        hit = i
                        break
            if hit < 0:
                missing += ("\n" if any_missing else "") + "Category: " + _str_text(v) + ", Feature: " + String(j)
                any_missing = True
                continue
            var i = hit
            if groups is not None and groups[j] is not None:
                var infrequent = _index_list(fields["_infrequent"][j])
                for k in range(len(infrequent)):  # driver-loop: over one column's infrequent categories
                    if infrequent[k] == i:
                        raise Error("mojolearn: Unable to drop category " + _repr(PythonObject(Float64(cl[i]))) + " from feature "
                                    + String(j) + " because it is infrequent")
                i = _index_list(groups[j])[i]
            _ = py_call(after.append, PythonObject(i))
        if any_missing:
            raise Error("mojolearn: The following categories were supposed to be dropped, but were not "
                        + "found in the training data.\n" + missing)
    py_setitem(fields, "_drop_after", after)
    var idx = Python.list()
    for j in range(d):  # driver-loop: maps drop indices through the grouping (per-column drop indices)
        var a = after[j]
        if a is None or groups is None or groups[j] is None:
            _ = py_call(idx.append, a)
            continue
        var g = _index_list(groups[j])
        var target = Int(py=a)
        var found = -1
        for i in range(len(g)):  # driver-loop: over one column grouping
            if g[i] == target:
                found = i
                break
        if found < 0:
            raise Error(String(target) + " is not in list")
        _ = py_call(idx.append, PythonObject(found))
    py_setitem(fields, "drop_idx_", idx)


def _onehot_drops(fields: PythonObject) raises -> List[Int]:
    # `_widths`' drop indices per column in grouped codes (-1: none).
    var d = len(fields["categories_"])
    var after = fields["_drop_after"]
    var drops = List[Int]()
    for j in range(d):  # driver-loop: per-column drop indices
        var dr = PythonObject(None) if after is None else after[j]
        drops.append(-1 if dr is None else Int(py=dr))
    return drops^


def _onehot_widths(fields: PythonObject) raises -> List[Int]:
    # `_widths`: per-column one-hot block widths.
    return _onehot_widths_of(_grouped_sizes(fields, PythonObject(None), PythonObject(None)), _onehot_drops(fields))


def _onehot_widths_of(sizes: List[Int], drops: List[Int]) -> List[Int]:
    # `_widths` from sizes and drops the caller already read (the transform
    # and inverse read the drops once, not twice).
    var widths = List[Int]()
    for j in range(len(sizes)):  # driver-loop: per-column one-hot block widths
        widths.append(sizes[j] - (0 if drops[j] < 0 else 1))
    return widths^


def _onehot_fit(model: PythonObject, values: PythonObject, capture: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var grouping = _check_infrequent_params(fields, context)
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    if handle_text != "error" and handle_text != "ignore" and handle_text != "infrequent_if_exist" and handle_text != "warn":
        raise Error("mojolearn: OneHotEncoder handle_unknown=" + _repr(handle) + " is not valid")
    var drop = fields["drop"]
    if _is_str(drop) and String(drop) != "first" and String(drop) != "if_binary":
        raise Error("mojolearn: OneHotEncoder drop must be None, 'first', 'if_binary' or one category "
                    + "per feature")
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var auto = _is_str(fields["categories"]) and String(fields["categories"]) == "auto"
    var inverse = PythonObject(None)
    if capture and auto:
        var got = _fit_categories_with_codes(values, entries, caps, context)
        py_setitem(fields, "categories_", got[0])
        inverse = got[1]
    elif auto:
        py_setitem(fields, "categories_", _fit_categories_caps(values, mode, entries, caps, context))
    else:
        py_setitem(fields, "categories_", _given_categories(fields["categories"], values, entries, caps, handle_text == "error", "OneHotEncoder", context))
    _ = _category_freeze(fields["categories_"], context)  # hazard A: fitted categories are read-only (p3-prep)
    py_setitem(fields, "_infrequent", PythonObject(None))
    py_setitem(fields, "_grouping", PythonObject(None))
    if grouping:
        _fit_infrequent(model, values, entries, caps, False, inverse, context)
    _set_drop_idx(model, context)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    return inverse


def _onehot_caller_location() raises -> PythonObject:
    """(filename, line, module) of the code that called the public method
    (the Python frame below the shell's), read through public CPython frame
    accessors (no sys._getframe or frame attribute audit hook); None when
    the stack is not that deep (the context's default location is used)."""
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not pointer:
        return PythonObject(None)
    var current = PythonObject(from_borrowed=pointer)
    pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetBack")(current._obj_ptr)
    _ = current^
    if not pointer:
        if cpython_table()[].PyErr_Occurred():
            raise cpy.unsafe_get_error()
        return PythonObject(None)
    var frame = PythonObject(from_owned=pointer)
    pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetCode")(frame._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    var code = PythonObject(from_owned=pointer)
    var line = cpy.lib.get_function[Int32]("PyFrame_GetLineNumber")(frame._obj_ptr)
    var module = py_get(native_frame_globals(frame), "__name__", "")
    if py_type(module) is not native_loaded_module("builtins").str:
        module = PythonObject("")
    return Python.tuple(code.co_filename, Int(line), module)


def _onehot_transform(model: PythonObject, values: PythonObject, inverse: PythonObject, context: PythonObject) raises -> PythonObject:
    """OneHotEncoder `_transform_encoded`: lookup (or the C08 fit's inverse),
    the grouping remap with the unknown slots, and onehot."""
    # UNPACK: the parameters and fitted fields, read once into the typed
    # request; X (or the fit's inverse) and the category block as owners.
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var entries = scaler_entries(model, fields["numeric_mode_"], context)
    var caps = enc_caps(model, entries)
    var record = _category_record(model, fields["categories_"], context)
    var grouping = fields["_grouping"]
    var grouped = grouping is not None
    var groups = List[List[Int]]()
    var sizes = _record_sizes(record)
    if grouped:
        groups = _group_lists(grouping)
        sizes = cat_grouped_sizes(sizes, groups)
    var drops = _onehot_drops(fields)
    var widths = _onehot_widths_of(sizes, drops)
    var total = 0
    for j in range(len(widths)):  # driver-loop: total one-hot output width
        total += widths[j]
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    var p = EncProg()
    var from_inverse = inverse is not None
    var x = NONE
    var held = NONE
    var block = CatBlock(NONE, 0, 0, NONE, 0)
    if from_inverse:
        held = p.own(inverse, Int(py=inverse.size))
    else:
        x = p.own_x(values)
        block = _cat_block(p, record, True)
    var request = OneHotApply(n, d, x, block.copy(), from_inverse, held, grouped, groups^,
                              handle_text == "infrequent_if_exist" or handle_text == "warn", widths^, drops^)
    # RUN: the typed stages (bindings/prep_encode_onehot_run.mojo), then the typed run.
    var plan = onehot_transform_stage(p.prog, request, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    var dropping = fields["drop"] is not None
    if handle_text == "error":
        var bad = _unknown_columns(ran, plan.unknown, d)
        if len(bad) > 0:
            raise Error("mojolearn: OneHotEncoder found unknown categories in column(s) " + _list_text(bad) + " during transform")
    elif handle_text == "warn" or (dropping and (handle_text == "ignore" or handle_text == "infrequent_if_exist")):
        var bad = _unknown_columns(ran, plan.unknown, d)
        if len(bad) > 0:
            var where = String("encoded as the infrequent category") if handle_text != "ignore" else String("encoded as all zeros")
            # TOMBSTONE: P1_ONEHOT_WARN_IN_NATIVE (removed: warnings.warn through Python inside the native transform; filters and any showwarning hook ran in the operation) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
            # Restore: experiments/removed/P1_ONEHOT_WARN_IN_NATIVE.patch.
            # Delivered by the native warning path (bindings/classical_warnings
            # .mojo: the filters read natively, only the standard display
            # hooks admitted, a custom showwarning refused), attributed to the
            # caller of the public method.
            classical_warning("Found unknown categories in columns " + _list_text(bad)
                              + " during transform. These unknown categories will be " + where + ".",
                              b.UserWarning, context, _onehot_caller_location(), "UserWarning")
    return _take2(ran, plan.encoded, n, total, context)


def _onehot_inverse(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """OneHotEncoder.inverse_transform: block_argmax, the inverse grouping
    remap, cat_gather and the strict bad-row compaction."""
    # UNPACK: the parameters and fitted fields, read once into the typed
    # request; X and the category block as owners.
    var fields = py_fields(model)
    var record = _category_record(model, fields["categories_"], context)
    var grouping = fields["_grouping"]
    var grouped = grouping is not None
    var groups = List[List[Int]]()
    var sizes = _record_sizes(record)
    if grouped:
        groups = _group_lists(grouping)
        sizes = cat_grouped_sizes(sizes, groups)
    var drops = _onehot_drops(fields)
    var widths = _onehot_widths_of(sizes, drops)
    var n = Int(py=values[3])
    var cols = Int(py=values[4])
    var d = len(widths)
    var total = 0
    for j in range(d):  # driver-loop: over the per-column block widths
        total += widths[j]
    if cols != total:
        raise Error("mojolearn: X has " + String(cols) + " columns, expected " + String(total))
    var entries = scaler_entries(model, fields["numeric_mode_"], context)
    var caps = enc_caps(model, entries)
    # an all-zero block is unknown (NaN) under 'ignore', and under
    # 'infrequent_if_exist' / 'warn' for a column with no infrequent
    # category; anywhere else it cannot be inverted
    var handle = fields["handle_unknown"]
    var handle_text = String(handle) if _is_str(handle) else String("")
    var infrequent = fields["_infrequent"]
    var strict = List[Float64]()
    for j in range(d):  # driver-loop: over the per-column strictness flags
        var hard = handle_text == "error" or (handle_text != "ignore" and infrequent is not None and infrequent[j] is not None)
        strict.append(1.0 if hard else 0.0)
    var p = EncProg()
    var x = p.own_x(values)
    var block = _cat_block(p, record, False)
    var request = OneHotInverse(n, x, block.copy(), widths^, drops^, grouped, groups^, strict^)
    # RUN: the typed stages, then the typed run.
    var plan = onehot_inverse_stage(p.prog, request)
    var ran = p.run_caps(entries, context, caps)
    # PACK: the refusal's first rows, or the category values.
    var bad = _bad_rows(ran, (plan.bad_rows, plan.bad_count), 10)
    if len(bad) > 0:
        raise Error("mojolearn: samples " + _list_text(bad) + " can not be inverted when drop=None and "
                    + "handle_unknown='error' because they contain all zeros")
    return _take2(ran, plan.values, n, d, context)


# ---------------------------------------------------------------- TargetEncoder
def _target_scratch(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_target_scratch`: the FAST binding's x_prep_target_scratch.
    if String(mode) != "fast":
        return False
    return prep_entry_word(entries, "x_prep_target_scratch") != 0  # read once per binding


def _target_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """TargetEncoder.transform: lookup (device-only codes when the binding
    keeps them in scratch) and te_apply over the fitted encodings."""
    # UNPACK: the fitted fields, read once into the typed request; X, the
    # category block and the fitted encodings as owners of the program.
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var mode = fields["numeric_mode_"]
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var T = Int(py=fields["_T"])
    var cmax = Int(py=fields["_cmax"])
    var record = _category_record(model, fields["categories_"], context)
    var enc = _fitted_flat(model, "_enc", fields["_enc"], context)
    var meta = _fitted_flat(model, "_meta", fields["_meta"], context)
    var enc_words = Int(py=enc[2])
    var meta_words = Int(py=meta[2])
    var p = EncProg()
    var x = p.own_x(values)
    var block = _cat_block(p, record, True)
    var request = TargetApply(n, d, T, cmax, x, block.copy(), _target_scratch(mode, entries), p.own(enc[1], enc_words),
                              enc_words, p.own(meta[1], meta_words), meta_words)
    # RUN: the typed stages (bindings/prep_encode_target_run.mojo), then the typed run.
    var result = target_transform_stage(p.prog, request, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK
    return _take2(ran, result, n, d * T, context)


# ---------------------------------------------------------------- SplineTransformer
def _spline_fused(mode: PythonObject, entries: PythonObject) raises -> Bool:
    # `_spline_fused`: the FAST Apple build with -D MOJOLEARN_SPLINE_FAST_FUSED
    # (it alone exports x_prep_spline_fused).
    var text = py_call(native_loaded_module("builtins").str, mode)
    if String(py_call(py_call(text.strip).lower)) != "fast":
        return False
    return py_get(entries, "x_prep_spline_fused") is not None


def _extrapolation(value: PythonObject) raises -> Int:
    # `SplineTransformer._EXTRAP`.
    if _is_str(value):
        var text = String(value)
        if text == "constant":
            return 0
        if text == "continue" or text == "error":
            return 1
        if text == "linear":
            return 2
        if text == "periodic":
            return 3
    raise Error("mojolearn: invalid extrapolation " + _repr(value))


def _spline_checks(fields: PythonObject) raises:
    """The parameter refusals `_fit` raised before converting X."""
    _ = _extrapolation(fields["extrapolation"])
    var missing = fields["handle_missing"]
    if not (_is_str(missing) and (String(missing) == "error" or String(missing) == "zeros")):
        raise Error("mojolearn: invalid handle_missing " + _repr(missing))
    var order = fields["order"]
    if not (_is_str(order) and (String(order) == "C" or String(order) == "F")):
        raise Error("mojolearn: invalid order " + _repr(order))
    if _truth(fields["sparse_output"]):
        raise Error("NotImplementedError: mojolearn: SplineTransformer sparse_output is not implemented "
                    + "(there is no sparse Array)")
    var k = Int(py=py_call_typed(native_loaded_module("builtins").int, fields["degree"]))
    if k < 0 or k > 7:
        raise Error("mojolearn: SplineTransformer needs 0 <= degree <= 7")


def _no_nan(fields: PythonObject, ran: PythonObject, st: Int, d: Int, n: Int) raises:
    # `_no_nan`: handle_missing='error' refuses a column with a NaN.
    var missing = fields["handle_missing"]
    if not (_is_str(missing) and String(missing) == "error"):
        return
    for j in range(d):  # driver-loop: over the per-column non-NaN counts (feature metadata)
        if Int(_word(ran, st + j)) != n:
            raise Error("mojolearn: Input X contains NaN values and `SplineTransformer` is configured to "
                        + "error in this case (handle_missing='error'). To avoid this error, set "
                        + "handle_missing='zeros' to encode missing values as splines with value 0 or ensure "
                        + "no missing values in X.")


def _knot_rows(knots: PythonObject, context: PythonObject) raises -> PythonObject:
    # The user knots argument's rows, each a Python list of numbers.
    var b = native_loaded_module("builtins")
    if py_type(knots) is context["array_class"] and Int(py=knots.ndim) == 2:
        var rows = Python.list()
        var r = Int(py=knots.shape[0])
        var c = Int(py=knots.shape[1])
        if r * c == 0:
            for _ in range(r):  # driver-loop: over the user knots rows
                _ = py_call(rows.append, Python.list())
            return rows
        var flat = _input(knots, 2, False, context)
        var source = _ptr(flat)
        _ = flat  # keep the Python object alive through the native call above
        for i in range(r):  # driver-loop: over the user knots rows
            var row = Python.list()
            for j in range(c):  # driver-loop: over one user knots row
                _ = py_call(row.append, PythonObject(Float64(source[i * c + j])))
                _ = flat  # keep the Python object alive through the native call above
            _ = py_call(rows.append, row)
        return rows
    var outer = _user_values(knots, context)
    var rows = Python.list()
    for i in range(len(outer)):  # driver-loop: over the user knots rows
        _ = py_call(rows.append, _user_values(outer[i], context))
    return rows


def _spline_fit(model: PythonObject, values: PythonObject, weight_source: PythonObject, fuse: Bool,
                context: PythonObject) raises -> PythonObject:
    """SplineTransformer `_fit`: the knots program (col_stats, the given,
    quantile or uniform base knots, spline_knots) and, with `fuse`, the
    spline_apply of X in the same program; returns the output or None."""
    # UNPACK: the parameters, read once into the typed request; the route's
    # image and capabilities; X and the weights as owners of the program.
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    _spline_checks(fields)
    var extrap = _extrapolation(fields["extrapolation"])
    var k = Int(py=py_call_typed(b.int, fields["degree"]))
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required by SplineTransformer.")
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var w = PythonObject(None)
    var wl = 0
    var weighted = weight_source is not None
    if weighted:
        var checked = _check_weights(weight_source, n, "SplineTransformer", mode, entries, caps, context)
        w = checked[0]
        wl = checked[1]
    var knots_arg = fields["knots"]
    var given = not _is_str(knots_arg)
    var knot_kind = String(knots_arg) if not given else String("")
    if not given and knot_kind != "uniform" and knot_kind != "quantile":
        raise Error("mojolearn: invalid knots " + _repr(knots_arg))
    var nk = 0
    var cols = List[Float64]()
    if given:
        var rows = _knot_rows(knots_arg, context)
        nk = len(rows)
        if nk < 2:
            raise Error("mojolearn: Number of knots, knots.shape[0], must be >= 2.")
        var table = List[List[Float32]]()
        for i in range(nk):  # driver-loop: over the user knots rows
            if len(rows[i]) != d:
                raise Error("mojolearn: knots.shape[1] == n_features is violated.")
            var row = List[Float32]()
            for c in range(d):  # driver-loop: over one user knots row
                row.append(_f32(rows[i][c]))
            table.append(row^)
        for c in range(d):  # driver-loop: transposes the user knots argument (feature count)
            for i in range(nk):  # driver-loop: over one knots column
                if i > 0 and not (table[i][c] > table[i - 1][c]):
                    raise Error("mojolearn: knots must be sorted without duplicates.")
        for c in range(d):  # driver-loop: packs the user knots argument column-major
            for i in range(nk):  # driver-loop: over one knots column
                cols.append(Float64(table[i][c]))
    else:
        nk = Int(py=py_call_typed(b.int, fields["n_knots"]))
        if nk < 2:
            raise Error("mojolearn: SplineTransformer needs n_knots >= 2")
    var periodic = extrap == 3
    if periodic and nk <= k:
        raise Error("mojolearn: Periodic splines require degree < n_knots. Got n_knots=" + String(nk) + " and "
                    + "degree=" + String(k) + ".")
    var bias = _truth(fields["include_bias"])
    var W = spline_features(d, nk, k, periodic, bias)
    var wd = nk + 2 * k
    var kind = SPL_KNOTS_GIVEN if given else (SPL_KNOTS_QUANTILE if knot_kind == "quantile" else SPL_KNOTS_UNIFORM)
    var p = EncProg()
    var x = p.own_x(values)
    var request = SplineFit(n, d, x, k, nk, kind, cols^, extrap, bias, weighted, p.own(w, n) if weighted else NONE, wl,
                            _spline_fused(mode, entries), (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0, fuse)
    # RUN: the typed stages (bindings/prep_encode_spline_run.mojo), then the typed run.
    var plan = spline_fit_stage(p.prog, request, caps.has_out)
    var st = plan.stats
    var knots = plan.knots
    var out = plan.applied
    var ran = p.run_caps(entries, context, caps)
    # PACK: the missing-value refusal, then the fitted knots.
    _no_nan(fields, ran, st, d, n)
    var output = context["output_context"]
    py_setitem(fields, "_knots", _take1(ran, knots, d * wd, context))
    var bsplines = Python.list()
    var lows = Python.list()
    var highs = Python.list()
    for c in range(d):  # driver-loop: per-column fitted spline knots (feature count)
        _ = py_call(bsplines.append, _take1(ran, knots + c * wd, wd, context))
        _ = py_call(lows.append, PythonObject(Float64(_word(ran, knots + c * wd + k))))
        _ = py_call(highs.append, PythonObject(Float64(_word(ran, knots + c * wd + k + nk - 1))))
    py_setitem(fields, "bsplines_", bsplines)
    py_setitem(fields, "_lo", lows)
    py_setitem(fields, "_hi", highs)
    py_setitem(fields, "_nk", nk)
    py_setitem(fields, "_k", k)
    py_setitem(fields, "n_features_out_", W)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    _ = w^
    if fuse:
        return _take2(ran, out, n, W, context)
    return PythonObject(None)


def _spline_transform(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """SplineTransformer.transform: spline_apply over the fitted knots (with
    the column statistics when a refusal reads them) and, for order 'F',
    p2m_transpose into the output."""
    # UNPACK: the parameters and fitted fields, read once into the typed
    # request; X and the fitted knots as owners of the program.
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    _check_width(model, d)
    var mode = fields["numeric_mode_"]
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var W = Int(py=fields["n_features_out_"])
    var extrap = _extrapolation(fields["extrapolation"])
    var fortran = _is_str(fields["order"]) and String(fields["order"]) == "F"
    var missing_error = _is_str(fields["handle_missing"]) and String(fields["handle_missing"]) == "error"
    var extrap_error = _is_str(fields["extrapolation"]) and String(fields["extrapolation"]) == "error"
    var check = extrap_error or missing_error
    var knots = _fitted_flat(model, "_knots", fields["_knots"], context)
    var knots_words = Int(py=knots[2])
    var p = EncProg()
    var x = p.own_x(values)
    var request = SplineApply(n, d, x, p.own(knots[1], knots_words), knots_words, Int(py=fields["_nk"]),
                              Int(py=fields["_k"]), extrap, W, _truth(fields["include_bias"]), fortran, check,
                              check and _spline_fused(mode, entries),
                              check and (_fam(mode, entries) & IDN_STATS_BLOCKED) != 0)
    # RUN: the typed stages (bindings/prep_encode_spline_run.mojo), then the typed run.
    var plan = spline_transform_stage(p.prog, request, caps.has_out)
    var ran = p.run_caps(entries, context, caps)
    # PACK: the refusals, then the block in the requested order.
    var st = plan.stats
    if check:
        _no_nan(fields, ran, st, d, n)
    if extrap_error:
        var lows = fields["_lo"]
        var highs = fields["_hi"]
        for j in range(d):  # driver-loop: over the per-column knot limits (feature metadata)
            if Float64(_word(ran, st + 3 * d + j)) < Float64(py=lows[j]) or Float64(_word(ran, st + 4 * d + j)) > Float64(py=highs[j]):
                raise Error("mojolearn: X contains values beyond the limits of the knots")
    if plan.transposed != NONE:
        if plan.transposed == OUT_MARK:
            return _take_f(ran[1][0], 0, n, W, context)
        return _take_f(ran[0][0], plan.transposed, n, W, context)
    if fortran:
        # an empty block
        return classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(n, W), "<f4", "F", context["output_context"])
    return _take2(ran, plan.applied, n, W, context)


# ---------------------------------------------------------------- targets
struct TargetInfo(Copyable, Movable):
    """`_target_arrays` / `_target_kind`: the kind, the class list (None
    for a continuous target), the target words (a float32 vector, or the
    int32 class codes) and T (1, or the class count of a multiclass target)."""
    var kind: String
    var classes: PythonObject
    var target: PythonObject
    var T: Int
    var size: Int

    def __init__(out self, kind: String, classes: PythonObject, target: PythonObject, T: Int, size: Int):
        self.kind = kind
        self.classes = classes
        self.target = target
        self.T = T
        self.size = size


def _view_or_none(y: PythonObject, context: PythonObject) -> Optional[PythonObject]:
    # `_materialize(y)`: y in its own dtype, C order (bindings/admitted_input
    # admit_c_view; lane a1b: was labels_view), or None where it refuses
    # (str labels, objects, a dtype without an Array typestr).
    try:
        var view = admit_c_view(y, "y", context)
        _ = admitted_dtype(view, context)
        return view
    except:
        return None


def _v_type(view: PythonObject, context: PythonObject) raises -> String:
    return admitted_dtype(view, context)


def _v_size(view: PythonObject) raises -> Int:
    return Int(py=view.size)


def _v_addr(view: PythonObject) raises -> Int:
    return Int(py=classical_address(view))


def _numeric_dtype(typestr: String) -> Bool:
    # `_numeric_target`'s dtypes.
    return (typestr == "<f4" or typestr == "<f8" or typestr == "<i4" or typestr == "<i8" or typestr == "<u4"
            or typestr == "<u1" or typestr == "<i2" or typestr == "<u2" or typestr == "<i1")


def _is_vector(view: PythonObject) raises -> Bool:
    var big = 0
    var nd = Int(py=view.ndim)
    for k in range(nd):  # driver-loop: over array rank metadata
        big = max(big, Int(py=view.shape[k]))
    return nd >= 1 and _v_size(view) == big


def _flat_copy(view: PythonObject, context: PythonObject) raises -> PythonObject:
    # The flat Array of a contiguous label vector, its own dtype (`num`).
    var n = _v_size(view)
    var out = admitted_vector(n, _v_type(view, context), context)
    if n > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=labels_address(out)),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=_v_addr(view)),
               count=n * Int(py=view.itemsize))
        _ = out  # keep the Python object alive through the native call above
    return out


def _f32_target(view: PythonObject, context: PythonObject) raises -> PythonObject:
    # `num.astype("<f4")`: the flat float32 target (cast_elements).
    # fb-boost-copy (2026-11-17): a float32 target is the C-order view
    # itself (`admit_c_view`); the program only reads it (EncProg.put), so
    # it is no longer copied into a zero-filled vector per fit (O(n)).
    var t = _v_type(view, context)
    if t == "<f4":
        return view
    var n = _v_size(view)
    var out = admitted_vector(n, "<f4", context)
    labels_cast(_v_addr(view), t, labels_address(out), "<f4", n, context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _all_integral(model: PythonObject, view: PythonObject, vector: Bool, context: PythonObject) raises -> Bool:
    """`_all_integral`: every float32 / float64 value finite and an integer,
    on the estimator's image (c2_integral by chunks, c2_isum, one word read
    back). UNPACK (the view's words as an owner; the entry selection and the
    run capabilities the estimator's route holds), the typed stages
    (bindings/prep_encode_select_run.mojo `score_integral_stage`), the typed
    run, one word read back."""
    var t = _v_type(view, context)
    var kind = 0 if t == "<f4" else 4
    var per = 1 if t == "<f4" else 2
    var n = _v_size(view)
    # `vector`: `_target_arrays` tests y as it is; `_target_kind` its flat copy
    if (t != "<f4" and t != "<f8") or n == 0 or (vector and not _is_vector(view)) or n * per > 1073741824:
        raise Error("TypeError: mojolearn: the integral test takes a float32 or float64 vector")
    # fb-boost-copy (2026-11-17): a float32 view (C order, `admit_c_view`)
    # is read in place by the scan; only a float64 one is copied, as its
    # n x 2 float32 words (the program's record size is the owner's `size`
    # in float32 words). It was copied into a zero-filled store per call.
    var words = view
    if per != 1:
        words = classical_typed(Python.tuple(n * per), "<f4", context["output_context"])
        unsafe_memcpy(dest=_ptr(words), src=F32P(unsafe_from_address=_v_addr(view)), count=n * per)
        _ = words  # keep the Python object alive through the native call above
    # The fit's own selection (the same mode object), held by the route.
    var entries = scaler_entries(model, _default_mode(context), context)
    var caps = enc_caps(model, entries)
    var p = EncProg()
    var held = p.own(words, n * per)
    var tot = score_integral_stage(p.prog, held, n * per, kind, n)
    var ran = p.run_caps(entries, context, caps)
    return _iword(ran, tot) == 0


def _class_target(labels: ClassicalLabels, target_type: String) raises -> TargetInfo:
    var K = len(labels.classes)
    if target_type == "binary" or (target_type == "auto" and K <= 2):
        return TargetInfo("binary", labels.classes, labels.codes, 1, labels.n)
    return TargetInfo("multiclass", labels.classes, labels.codes, K, labels.n)


def _target_info(model: PythonObject, y: PythonObject, target_type: String, context: PythonObject) raises -> TargetInfo:
    """`_target_arrays`, else `_target_kind`: a numeric buffer vector's own
    words (continuous: float32; else its class codes), then the label input
    step's codes for anything else."""
    var b = native_loaded_module("builtins")
    var cls = py_type(y)
    var listed = cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes
    var found = _view_or_none(y, context)
    var has_view = Bool(found)
    if not listed and has_view:
        # `_target_arrays`
        var view = found.value()
        var t = _v_type(view, context)
        var kind = String(t[byte=1]) if t.byte_length() > 1 else String("")
        if _v_size(view) > 0 and _is_vector(view) and (kind == "f" or kind == "i" or kind == "u"):
            var floating = kind == "f"
            var ok = True
            if floating:
                # A label vector's finite test: the one host scan.
                ok = ((t == "<f4" and host_all_finite_f32(F32P(unsafe_from_address=_v_addr(view)), _v_size(view)))
                      or (t == "<f8" and host_all_finite_f64(
                          MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=_v_addr(view)), _v_size(view))))
                _ = view  # keep the Python object alive through the native call above
            if ok:
                var cont = target_type == "continuous" or (target_type == "auto" and floating and not _all_integral(model, view, True, context))
                if cont:
                    return TargetInfo("continuous", PythonObject(None), _f32_target(view, context), 1, _v_size(view))
                return _class_target(classical_encode_labels(y, context), target_type)
    # `_target_kind`
    var numeric = has_view and _numeric_dtype(_v_type(found.value(), context))
    var continuous = target_type == "continuous"
    if not continuous and target_type == "auto" and numeric:
        var view = found.value()
        var vt = _v_type(view, context)
        if _v_size(view) > 0 and (vt == "<f4" or vt == "<f8"):
            continuous = not _all_integral(model, view, False, context)
    if continuous:
        if not numeric:
            raise Error("mojolearn: a continuous TargetEncoder target must be numeric")
        var view = found.value()
        return TargetInfo("continuous", PythonObject(None), _f32_target(view, context), 1, _v_size(view))
    if numeric and listed:
        # a list's numbers as `_materialize` packed them (a bool leaf is the int 0 or 1)
        return _class_target(classical_encode_labels(_flat_copy(found.value(), context), context), target_type)
    return _class_target(classical_encode_labels(y, context), target_type)


# ---------------------------------------------------------------- TargetEncoder fit
def _target_check(fields: PythonObject, context: PythonObject) raises:
    """TargetEncoder `_check`."""
    var b = native_loaded_module("builtins")
    var categories = fields["categories"]
    var auto = _is_str(categories) and String(categories) == "auto"
    if not auto and not Bool(py=py_call(b.isinstance, categories, Python.tuple(b.list, b.tuple))):
        raise Error("mojolearn: TargetEncoder categories must be 'auto' or a list of lists")
    var tt = fields["target_type"]
    var text = String(tt) if _is_str(tt) else String("")
    if text != "auto" and text != "binary" and text != "continuous" and text != "multiclass":
        raise Error("mojolearn: invalid target_type " + _repr(tt))
    var smooth = fields["smooth"]
    if not ((_is_str(smooth) and String(smooth) == "auto") or (_real_like(smooth, context) and _cmp("ge", smooth, PythonObject(0)))):
        raise Error("mojolearn: invalid smooth " + _repr(smooth))
    var cv = fields["cv"]
    if _int_like(cv, context) and _cmp("lt", cv, PythonObject(2)):
        raise Error("mojolearn: TargetEncoder cv must be an integer >= 2, a splitter or an iterable")


def _put_codes(mut p: EncProg, codes: PythonObject) raises -> Int:
    # `_Prog.put_codes`: int32 codes as words, then their floats (i2f):
    # UNPACK of the codes as an owner, then the typed stage.
    var count = Int(py=codes.size)
    var held = p.own(codes, count)
    return cat_i2f_stage(p.prog, count, held)


def _target_run(model: PythonObject, values: PythonObject, info: TargetInfo, folds: PythonObject, n_folds: Int,
                lo: Int, hi: Int, shuffle: Bool, apply: Bool, context: PythonObject) raises -> PythonObject:
    """TargetEncoder `_run`: the categories, then one program: the codes,
    the target words, the folds (staged, or the given int32 folds, or none),
    te_global, the bucketed te_enc and, with `apply`, te_apply over the
    training rows. `folds` None with n_folds > 0: the staged integer folds."""
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var T = info.T
    if info.size * T != n * T:
        raise Error("mojolearn: X and y have different numbers of rows")
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    var caps = enc_caps(model, entries)
    var auto = _is_str(fields["categories"]) and String(fields["categories"]) == "auto"
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var inverse = Optional[PythonObject]()
    var cats = PythonObject(None)
    if _c08_route(mode, entries, 1) and auto:
        # C08 reuses only unsupervised category IDs
        var got = _fit_categories_with_codes(values, entries, caps, context)
        cats = got[0]
        inverse = py_present(got[1])
    elif auto:
        cats = _fit_categories_caps(values, mode, entries, caps, context)
    else:
        cats = _given_categories(fields["categories"], values, entries, caps, False, "TargetEncoder", context)
    var F = n_folds
    var p = EncProg()
    var from_inverse = Bool(inverse)
    var x = NONE
    var held = NONE
    var block = CatBlock(NONE, 0, 0, NONE, 0)
    var sizes = List[Int]()
    if from_inverse:
        var coded = inverse.value()
        held = p.own(coded, Int(py=coded.size))
        sizes = _sizes(cats)
    else:
        # The category record of the lists this fit stores as `categories_`
        # (kept for the model: its transforms reuse it).
        var record = _category_record(model, cats, context)
        sizes = _record_sizes(record)
        x = p.own_x(values)
        block = _cat_block(p, record, True)
    var cmax = 0
    for j in range(len(sizes)):  # driver-loop: largest category count (per-column category counts)
        cmax = max(cmax, sizes[j])
    var continuous = info.kind == "continuous"
    var target_words = n if continuous else Int(py=info.target.size)
    var target = p.own(info.target, target_words)
    var folds_kind = TE_FOLDS_GIVEN
    var folds_owner = NONE
    var folds_words = 0
    if folds is None:
        folds_kind = TE_FOLDS_STAGED if F > 0 else TE_FOLDS_NONE
    else:
        folds_words = Int(py=folds.size)
        folds_owner = p.own(folds, folds_words)
    var smooth = fields["smooth"]
    var smooth_value = Float64(-1.0) if (_is_str(smooth) and String(smooth) == "auto") else _float(smooth)
    var request = TargetFit(n, d, T, x, block.copy(), _target_scratch(mode, entries), from_inverse, held, continuous,
                            target, target_words, len(info.classes) if info.classes is not None else 0, folds_kind, F,
                            folds_owner, folds_words, lo, hi, shuffle, sizes.copy(), cmax, smooth_value,
                            py_get(entries, "x_prep_host_column") is not None, apply)
    # RUN: the typed stages (bindings/prep_encode_target_run.mojo), then the typed run.
    var plan = target_fit_stage(p.prog, request, caps.has_out)
    var enc = plan.enc
    var meta = plan.meta
    var flag = plan.flag
    var out = plan.applied
    var ran = p.run_caps(entries, context, caps)
    # PACK: the refusal of too few class members, then the fitted fields.
    if flag >= 0 and _word(ran, flag) != 0:
        raise Error("mojolearn: n_splits=" + String(F) + " cannot be greater than the number of members in "
                    + "each class")
    py_setitem(fields, "categories_", cats)
    _ = _category_freeze(cats, context)  # hazard A: fitted categories are read-only (p3-prep)
    py_setitem(fields, "target_type_", PythonObject(info.kind))
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "classes_", info.classes)
    py_setitem(fields, "_T", T)
    py_setitem(fields, "_cmax", cmax)
    var base = F * d * cmax * T
    py_setitem(fields, "_enc", _take1(ran, enc + base, d * cmax * T, context))
    py_setitem(fields, "_meta", _take1(ran, meta + 2 * F * T, 2 * T, context))
    var encodings = Python.list()
    for j in range(d):  # driver-loop: slices the per-column encodings outputs (feature count)
        _ = py_call(encodings.append, _take1(ran, enc + base + (j * cmax) * T, sizes[j] * T, context))
    py_setitem(fields, "encodings_", encodings)
    if T == 1:
        py_setitem(fields, "target_mean_", _take1(ran, meta + 2 * F * T, 1, context))
    else:
        var means = List[Float32]()
        for t in range(T):  # driver-loop: over the target's classes
            means.append(_word(ran, meta + 2 * F * T + 2 * t))
        py_setitem(fields, "target_mean_", _f32_array(means, context))
    if apply:
        return _take2(ran, out, n, d * T, context)
    return PythonObject(None)


def _target_fit(model: PythonObject, values: PythonObject, y: PythonObject, context: PythonObject) raises:
    # TargetEncoder.fit: `_run(arr, y, None, 0, False)`.
    var fields = py_fields(model)
    var tt = String(fields["target_type"])
    var info = _target_info(model, y, tt, context)
    _ = _target_run(model, values, info, PythonObject(None), 0, 0, 0, False, False, context)


struct IndexI64(Copyable, Movable):
    """`_buffer.as_index_i64(values)`: the int64 words of a user index list
    (`array('q', values)`, in C) or of a numeric buffer (its own int64 words,
    else the native `astype('<i8')` cast), with the owner kept alive."""
    var keep: PythonObject
    var addr: Int
    var size: Int

    def __init__(out self, values: PythonObject, name: String, context: PythonObject) raises:
        var b = native_loaded_module("builtins")
        var cls = py_type(values)
        if cls is b.list or cls is b.tuple or cls is b.range:
            var store = py_call(native_loaded_module("array").array, "q", values)
            self.keep = store
            self.size = len(store)
            self.addr = Int(py=py_call(store.buffer_info)[0])
            return
        # The one integer admission (lane a1b: was labels_view + labels_cast).
        var out = admit_ids(values, name, context)
        self.keep = out
        self.addr = labels_address(out)
        self.size = Int(py=out.size)


def _target_splitter_folds(model: PythonObject, x: PythonObject, y: PythonObject, n: Int,
                           context: PythonObject) raises -> PythonObject:
    """`TargetEncoder._splitter_folds`: (row -> fold int32 Array, n_splits)
    from a user cv splitter (`cv.split(X, y)`) or (train, test) iterable: the
    test folds cover every row exactly once, and each fold's training rows
    are every other row."""
    var b = native_loaded_module("builtins")
    var cv = py_fields(model)["cv"]
    # Only data and native splitters cross: a list or tuple of (train, test)
    # index arrays, or a mojolearn splitter, whose folds come from the
    # model-selection owner's `family_msel_split` operation directly (the
    # operation its Python `split` calls; that method is not called). Any
    # other splitter or iterable is user Python code and is refused by name.
    var cls = py_type(cv)
    var splits = PythonObject(None)
    if cls is b.list or cls is b.tuple:
        splits = py_call(b.list, cv)
    elif String(py_call(b.getattr, cls, "__module__", "")).startswith("mojolearn.") and Bool(py=py_call(b.hasattr, cv, "split")):
        splits = py_call(b.list, prep_native_split(cv, x, y, context))
    else:
        raise Error("TypeError: mojolearn: TargetEncoder cv must be an integer, a mojolearn splitter or a list of "
                    + "(train, test) index arrays; got " + String(py_call(b.getattr, cls, "__name__")))
    var cover = String("mojolearn: Validation indices from `cv` must cover each sample index exactly once "
                       + "with no overlap. Pass a splitter with non-overlapping validation folds as `cv`.")
    var F = len(splits)
    if F < 1:
        raise Error(cover)
    var fold = admitted_vector(n, "<i4", context)
    var fp = I32P(unsafe_from_address=labels_address(fold))
    for i in range(n):  # native data loop (not a driver loop): the row -> fold table cleared to -1 before the user splits fill it (no CPython call)
        fp[i] = -1
    var trains = List[IndexI64]()
    for k in range(F):  # api-bound-loop: over the folds the user's cv splitter yielded (public type: iterable of (train, test) pairs); one tuple unpack and one native index read per fold
        var pair = py_call(b.tuple, splits[k])
        if len(pair) > 2:
            raise Error("too many values to unpack (expected 2)")
        if len(pair) < 2:
            raise Error("not enough values to unpack (expected 2, got " + String(len(pair)) + ")")
        var te = IndexI64(pair[1], "test", context)
        if enc_assign_fold_i64(te.addr, te.size, n, k, fp) != 0:
            raise Error(cover)
        _ = te^
        trains.append(IndexI64(pair[0], "train", context))
    var sizes = List[Int](length=F, fill=0)
    for i in range(n):  # cpu-route: the row -> fold table the user splits filled (rows)
        if fp[i] < 0:
            raise Error(cover)
        sizes[Int(fp[i])] += 1
    for k in range(F):  # cpu-route: user cv splitter train index lists, one per fold
        if trains[k].size != n - sizes[k] or (trains[k].size > 0 and not enc_check_train_i64(trains[k].addr, trains[k].size, n, k, fp)):
            raise Error("NotImplementedError: mojolearn: TargetEncoder cv folds whose training rows are not every "
                        + "row outside the test fold are not implemented")
    _ = trains^
    return Python.tuple(fold, F)


def _target_fit_transform(model: PythonObject, values: PythonObject, y: PythonObject, splits: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """TargetEncoder.fit_transform: `splits` (fold int32 Array, n_splits)
    from a user cv splitter, or None: cv an integer, the folds staged on the
    device in the fit's own program."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var n = Int(py=values[3])
    var tt = String(fields["target_type"])
    if splits is not None:
        var info = _target_info(model, y, tt, context)
        return _target_run(model, values, info, splits[0], Int(py=splits[1]), 0, 0, False, True, context)
    var cv = 5 if fields["cv"] is None else Int(py=py_call_typed(b.int, fields["cv"]))
    if n < cv:
        raise Error("mojolearn: cv=" + String(cv) + " folds need at least " + String(cv) + " rows")
    var seed = _seed64(fields["random_state"])
    var shuffle = _truth(fields["shuffle"])
    if tt == "binary" and xprep_r3("te_arrays"):
        # `_target_binary_codes`: an explicit binary target's codes as they are
        var labels = classical_encode_labels(y, context)
        if labels.n != n:
            raise Error("mojolearn: X and y have different numbers of rows")
        var info = TargetInfo("binary", labels.classes, labels.codes, 1, labels.n)
        return _target_run(model, values, info, PythonObject(None), cv, seed[0], seed[1], shuffle, True, context)
    var info = _target_info(model, y, tt, context)
    if info.size != n:
        raise Error("mojolearn: X and y have different numbers of rows")
    return _target_run(model, values, info, PythonObject(None), cv, seed[0], seed[1], shuffle, True, context)


# ---------------------------------------------------------------- SelectKBest
def _scores_classif(mut p: EncProg, values: PythonObject, y: PythonObject, chi: Bool, mode: PythonObject,
                    entries: PythonObject, context: PythonObject) raises -> ScorePlan:
    """`_scores_classif`: UNPACK (the label input step's class codes, the
    image's family word as three Bool fields, X and the codes as owners),
    then the typed stages (bindings/prep_encode_select_run.mojo
    `score_classif_stage`). The plan's `extra`: chi2's column statistics."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var labels = classical_encode_labels(y, context)
    if labels.n != n:
        raise Error("mojolearn: X and y have different numbers of rows")
    var fam = _fam(mode, entries)
    var x = p.own_x(values)
    var count = Int(py=labels.codes.size)
    var codes = p.own(labels.codes, count)
    var request = ScoreClassif(n, d, x, codes, count, len(labels.classes), chi, (fam & IDN_STATS_BLOCKED) != 0,
                               (fam & IDN_CLASS_ONEPASS) != 0, (fam & IDN_SELECT_BLOCKED) != 0)
    return score_classif_stage(p.prog, request)


def _pearson(mut p: EncProg, values: PythonObject, y: PythonObject, mode: PythonObject, entries: PythonObject,
             context: PythonObject, center: Bool = True, force_finite: Bool = True) raises -> ScorePlan:
    """`_pearson(X, y, center, force_finite)`: UNPACK (the admitted float32
    y, the image's family word as a Bool field, X and y as owners), then the
    typed stages (`score_pearson_stage`). The plan's `extra`: the correlations."""
    var n = Int(py=values[3])
    var d = Int(py=values[4])
    var target = admit_f32(y, 1, "y", context, finite=False)
    if target.rows != n:
        raise Error("mojolearn: X and y have different numbers of rows")
    # fb-boost-copy (2026-11-17): the program reads the admitted C-order
    # float32 y in place (its owner is held for the run); it was copied
    # into a zero-filled vector per f_regression call (O(n)).
    var x = p.own_x(values)
    var held = p.own(target.array, n)
    var request = ScorePearson(n, d, x, held, center, force_finite, (_fam(mode, entries) & IDN_SELECT_BLOCKED) != 0)
    var plan = score_pearson_stage(p.prog, request)
    _ = target^
    return plan^


def _score_words(scores: PythonObject, d: Int, context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """`_key_words`: a float32 / int32 / int64 / float64 score vector's own
    words and its c2_key64 kind; a user score function's other answer
    converted once to float64."""
    var b = native_loaded_module("builtins")
    var cls = py_type(scores)
    if not (cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes):
        var found = _view_or_none(scores, context)
        if found:
            var view = found.value()
            var t = _v_type(view, context)
            var kind = -1
            if t == "<f4":
                kind = 0
            elif t == "<i4":
                kind = 1
            elif t == "<i8":
                kind = 3
            elif t == "<f8":
                kind = 4
            if kind >= 0 and _v_size(view) == d and _is_vector(view):
                var per = 1 if kind <= 1 else 2
                var words = classical_typed(Python.tuple(d * per), "<f4", context["output_context"])
                unsafe_memcpy(dest=_ptr(words), src=F32P(unsafe_from_address=_v_addr(view)), count=d * per)
                _ = words  # keep the Python object alive through the native call above
                _ = view  # keep the Python object alive through the native call above
                return (words, kind)
    var vals = py_call(b.list, py_call(scores.tolist)) if Bool(py=py_call(b.hasattr, scores, "tolist")) else py_call(b.list, scores)
    if len(vals) != d:
        raise Error("mojolearn: expected " + String(d) + " scores, got " + String(len(vals)))
    var words = classical_typed(Python.tuple(2 * d), "<f4", context["output_context"])
    var dest = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(words)))
    _ = words  # keep the Python object alive through the native call above
    for i in range(d):  # cpu-route: a user score function's answer, converted once (feature count)
        dest[i] = _float(vals[i])
        _ = words  # keep the Python object alive through the native call above
    return (words, 4)


def _topk_from(p_ran: PythonObject, mask: Int, d: Int) raises -> PythonObject:
    var out = Python.list()
    for j in range(d):  # driver-loop: the device's mask as the fitted list (feature count)
        _ = py_call(out.append, PythonObject(_word(p_ran, mask + j) != 0))
    return out


def _select_fit(model: PythonObject, x: PythonObject, values: PythonObject, y: PythonObject, context: PythonObject) raises:
    """SelectKBest.fit: f_classif, chi2, f_regression and r_regression run
    their programs here with the top-k stages in the same program (the
    scores stay in the arena); mutual_info_classif and mutual_info_regression
    (their defaults) run as the encode owner's own operation, re-entered
    through its compiled entry. Any other score_func would be user Python code
    in fit and is refused."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var d = Int(py=values[4])
    var mode = _default_mode(context)
    var entries = scaler_entries(model, mode, context)
    # the run capabilities of the estimator's route, once for the fit's programs
    var caps = enc_caps(model, entries)
    var score_func = fields["score_func"]
    var builtin = py_get(context["prep_encode_scores"], score_func)
    if builtin is None:
        raise Error("TypeError: mojolearn: SelectKBest score_func must be f_classif, chi2, f_regression, "
                    + "r_regression, mutual_info_classif or mutual_info_regression; a user callable is not supported")
    var mutual = String(builtin).startswith("mutual_info_")
    var k_arg = fields["k"]
    var keep_all = _is_str(k_arg) and String(k_arg) == "all"
    var k = 0
    var k_ok = True
    if not keep_all:
        k = Int(py=py_call_typed(b.int, k_arg))
        k_ok = 0 <= k and k <= d
    var output = context["output_context"]
    var mask = PythonObject(None)
    if not mutual:
        var name = String(builtin)
        var p = EncProg()
        var plan = ScorePlan(0, 0, NONE)
        if name == "f_classif" or name == "chi2":
            plan = _scores_classif(p, values, y, name == "chi2", mode, entries, context)
        else:
            plan = _pearson(p, values, y, mode, entries, context)
        var score_at = plan.extra if name == "r_regression" else plan.scores
        # the scores' float32 words, read by c2_key64 where the device wrote them
        var mo = -1
        if not keep_all and k_ok:
            mo = score_topk_stage(p.prog, score_at, 0, d, k)
        # RUN: the typed run on the estimator's image, then PACK
        var ran = p.run_caps(entries, context, caps)
        if name == "chi2":
            for j in range(d):  # driver-loop: over the per-column minima (feature metadata)
                if _word(ran, plan.extra + 3 * d + j) < 0:
                    raise Error("mojolearn: Input X must be non-negative.")
        if name == "r_regression":
            py_setitem(fields, "scores_", _take1(ran, plan.extra, d, context))
            py_setitem(fields, "pvalues_", PythonObject(None))
        else:
            py_setitem(fields, "scores_", _take1(ran, plan.scores, d, context))
            py_setitem(fields, "pvalues_", _take1(ran, plan.pvalues, d, context))
        if mo >= 0:
            mask = _topk_from(ran, mo, d)
    else:
        # `score_func(X, y)`: mutual_info_*(X, y) with its defaults
        # (discrete_features='auto', n_neighbors=3, random_state=None), the
        # operation the public function passes to the same entry.
        # `context` is the connected context of this estimator's device
        # route (the door top), and the mutual-information programs run on
        # the encode transport of the context they are handed (their entry
        # selection, label step and storage all come from it), so the
        # scores are computed on the estimator's device. Nothing here reads
        # `device`.
        var entry = context["prep_encode_entry"]
        if py_type(entry) is not py_type(b.len):
            raise Error("ImportError: mojolearn: the encode owner's entry is unavailable")
        var out = prep_entry_call(entry, Python.tuple(PythonObject(None), builtin,
                                                     Python.tuple(x, y, "auto", 3, PythonObject(None)), context))
        var scores = out
        var pvals = PythonObject(None)
        if Bool(py=py_call(b.isinstance, out, Python.tuple(b.tuple, b.list))):
            # `scores, pvals = out`
            if len(out) > 2:
                raise Error("too many values to unpack (expected 2)")
            if len(out) < 2:
                raise Error("not enough values to unpack (expected 2, got " + String(len(out)) + ")")
            scores = out[0]
            pvals = out[1]
        py_setitem(fields, "scores_", scores)
        py_setitem(fields, "pvalues_", pvals)
        if not keep_all and k_ok:
            var p = EncProg()
            var got = _score_words(scores, d, context)
            _ = scores  # keep the Python object alive through the native call above
            var count = Int(py=got[0].size)
            var held = p.own(got[0], count)
            var mo = score_topk_words_stage(p.prog, held, count, got[1], d, k)
            var ran = p.run_caps(entries, context, caps)
            mask = _topk_from(ran, mo, d)
    if keep_all:
        mask = Python.list()
        for _ in range(d):  # driver-loop: the fitted support (feature count)
            _ = py_call(mask.append, PythonObject(True))
    elif not k_ok:
        raise Error("mojolearn: k should be 0 <= k <= n_features = " + String(d) + "; got " + String(k))
    py_setitem(fields, "_mask", mask)
    py_setitem(fields, "numeric_mode_", mode)
    py_setitem(fields, "n_features_in_", d)


def prep_encode_scores_call(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """f_classif(X, y), chi2(X, y), f_regression(X, y, center, force_finite)
    and r_regression(X, y, center, force_finite): `_scores_classif` /
    `_pearson`'s one program, the scores (and p-values) as float32 Arrays."""
    var mode = _default_mode(context)
    var entries = _entries(mode, context)
    var values = xprep_values(args[0], entries, context)
    var d = Int(py=values[4])
    var p = EncProg()
    var result = PythonObject(None)
    if operation == "f_classif" or operation == "chi2":
        var chi = operation == "chi2"
        var plan = _scores_classif(p, values, args[1], chi, mode, entries, context)
        var ran = p.run(entries, context)
        if chi:
            for j in range(d):  # driver-loop: over the per-column minima (feature metadata)
                if _word(ran, plan.extra + 3 * d + j) < 0:
                    raise Error("mojolearn: Input X must be non-negative.")
        result = Python.tuple(_take1(ran, plan.scores, d, context), _take1(ran, plan.pvalues, d, context))
    else:
        var plan = _pearson(p, values, args[1], mode, entries, context, _truth(args[2]), _truth(args[3]))
        var ran = p.run(entries, context)
        if operation == "r_regression":
            result = _take1(ran, plan.extra, d, context)
        else:
            result = Python.tuple(_take1(ran, plan.scores, d, context), _take1(ran, plan.pvalues, d, context))
    _ = values^
    return result


# ---------------------------------------------------------------- the door
def _kind(model: PythonObject, context: PythonObject) raises -> String:
    # The admitted class this model is (or subclasses), by its MRO. The
    # model's own class is the MRO's first row: an exact public class is
    # answered by that one lookup, without the MRO read (perf2-prep).
    var kinds = context["prep_encode_kinds"]
    var own = py_get(kinds, py_type(model))
    if own is not None:
        return String(own)
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: the native encode owner does not serve this class")


def _fit_weights(model: PythonObject, args: PythonObject, operation: String) raises -> PythonObject:
    """The fit's sample_weight: fit(X, y, sample_weight); fit_transform(X,
    y, fit_params) passes the keywords the Python fit would have taken."""
    if operation == "fit":
        return args[2] if len(args) > 2 else PythonObject(None)
    var params = args[2]
    var b = native_loaded_module("builtins")
    if py_type(params) is not b.dict:
        raise Error("TypeError: native fit keywords require an exact dictionary")
    var string_type = b.str
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var weights = Optional[PythonObject]()
    ref cpy = Python().cpython()
    var position = 0
    var raw_key = PyObjectPtr()
    var raw_value = PyObjectPtr()
    # PyDict_Next in main's compiled form (causal_model_config.mojo); the dict
    # holds at most a few fit keywords, so the lookup per call is negligible.
    for i in range(len(params)):  # driver-loop: native keyword insertion order, without iterator/list dispatch
        if cpy.lib.get_function[Int32]("PyDict_Next")(params._obj_ptr, MutPointer(to=position), MutPointer(to=raw_key), MutPointer(to=raw_value)) == 0:
            break
        var key = PythonObject(from_owned=cpy.Py_NewRef(raw_key))
        # Check before String/repr: a supplied name subclass must not invoke
        # its conversion hooks merely to report an unexpected fit keyword.
        if py_type(key) is not string_type:
            raise Error("TypeError: native fit keyword names must be exact strings")
        if String(key) == "sample_weight":
            weights = PythonObject(from_owned=cpy.Py_NewRef(raw_value))
        else:
            raise Error("TypeError: " + _name(model) + ".fit() got an unexpected keyword argument " + _repr(key))
    _ = params  # owns borrowed values through their final NewRef
    return py_or_none(weights)


def _no_fit_params(model: PythonObject, args: PythonObject) raises:
    # fit(X, y=None) takes no keywords: fit_transform's refuse them as fit would.
    var params = args[2]
    var b = native_loaded_module("builtins")
    if py_type(params) is not b.dict:
        raise Error("TypeError: native fit keywords require an exact dictionary")
    ref cpy = Python().cpython()
    var position = 0
    var raw_key = PyObjectPtr()
    var raw_value = PyObjectPtr()
    # Only the first keyword can be consumed: any keyword refuses. Native
    # iteration reads no value and invokes no mapping/key protocol hooks.
    if cpy.lib.get_function[Int32]("PyDict_Next")(
            params._obj_ptr, MutPointer(to=position), MutPointer(to=raw_key), MutPointer(to=raw_value)) != 0:
        var key = PythonObject(from_owned=cpy.Py_NewRef(raw_key))
        _ = params
        if py_type(key) is not b.str:
            raise Error("TypeError: native fit keyword names must be exact strings")
        raise Error("TypeError: " + _name(model) + ".fit() got an unexpected keyword argument " + _repr(key))


def _precheck(model: PythonObject, kind: String, context: PythonObject) raises:
    """The parameter refusals the Python fit raised before converting X."""
    var fields = py_fields(model)
    if kind == "simple_imputer":
        var strategy = fields["strategy"]
        var name = String(strategy) if _is_str(strategy) else String("")
        if name != "mean" and name != "median" and name != "most_frequent" and name != "constant":
            raise Error("mojolearn: SimpleImputer strategy " + _repr(strategy) + " is not valid")
        if name == "constant" and fields["fill_value"] is not None and not _real_like(fields["fill_value"], context):
            raise Error("TypeError: mojolearn: SimpleImputer fill_value must be numeric")
    elif kind == "ordinal":
        _ = _check_infrequent_params(fields, context)
        var handle = fields["handle_unknown"]
        if not (_is_str(handle) and (String(handle) == "error" or String(handle) == "use_encoded_value")):
            raise Error("mojolearn: invalid handle_unknown " + _repr(handle))
        if _is_str(handle) and String(handle) == "use_encoded_value" and not _real_like(fields["unknown_value"], context):
            raise Error("TypeError: mojolearn: unknown_value must be a number when handle_unknown='use_encoded_value'")
        if not _real_like(fields["encoded_missing_value"], context):
            raise Error("TypeError: mojolearn: encoded_missing_value must be a number (or NaN)")
    elif kind == "onehot":
        _ = _check_infrequent_params(fields, context)
        var handle = fields["handle_unknown"]
        var text = String(handle) if _is_str(handle) else String("")
        if text != "error" and text != "ignore" and text != "infrequent_if_exist" and text != "warn":
            raise Error("mojolearn: OneHotEncoder handle_unknown=" + _repr(handle) + " is not valid")
        var drop = fields["drop"]
        if _is_str(drop) and String(drop) != "first" and String(drop) != "if_binary":
            raise Error("mojolearn: OneHotEncoder drop must be None, 'first', 'if_binary' or one category "
                        + "per feature")
    elif kind == "spline":
        _spline_checks(fields)
    elif kind == "kbins":
        var encode = fields["encode"]
        var text = String(encode) if _is_str(encode) else String("")
        if text != "onehot" and text != "onehot-dense" and text != "ordinal":
            raise Error("mojolearn: invalid encode " + _repr(encode))
        _ = _kbins_strategy(fields)


def _fit(model: PythonObject, kind: String, values: PythonObject, weights: PythonObject, capture: Bool,
         context: PythonObject) raises -> PythonObject:
    # The fit; an encoder's C08 fit (`capture`) returns its (n, d) inverse.
    if kind == "simple_imputer":
        _imputer_fit(model, values, context)
    elif kind == "kbins":
        _kbins_fit(model, values, weights, context)
    elif kind == "ordinal":
        return _ordinal_fit(model, values, capture, context)
    elif kind == "onehot":
        return _onehot_fit(model, values, capture, context)
    elif kind == "spline":
        _ = _spline_fit(model, values, weights, False, context)
    else:
        raise Error("TypeError: the native encode owner does not fit this class")
    return PythonObject(None)


def _transform(model: PythonObject, kind: String, values: PythonObject, inverse: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "simple_imputer":
        return _imputer_transform(model, values, context)
    if kind == "kbins":
        return _kbins_transform(model, values, context)
    if kind == "ordinal":
        _check_width(model, Int(py=values[4]))
        return _ordinal_transform(model, values, inverse, context)
    if kind == "onehot":
        _check_width(model, Int(py=values[4]))
        return _onehot_transform(model, values, inverse, context)
    if kind == "target":
        return _target_transform(model, values, context)
    if kind == "spline":
        return _spline_transform(model, values, context)
    raise Error("TypeError: the native encode owner does not transform this class")


def _encode_dispatch(model: PythonObject, kind: String, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    var result = PythonObject(None)
    if (operation == "fit" or operation == "fit_transform") and kind == "target":
        # TargetEncoder: fit(X, y); fit_transform(X, y, splits or None)
        _target_check(py_fields(model), context)
        var values = xprep_values(args[0], scaler_entries(model, _default_mode(context), context), context)
        if operation == "fit":
            _target_fit(model, values, args[1], context)
            result = model
        else:
            # fit_transform(X, y, None): a cv that is neither None nor an
            # integer is a user splitter, its folds read here once
            var cv = py_fields(model)["cv"]
            var splits = PythonObject(None)
            # numbers.Integral by exact type (_int_like), never the ABC's
            # Python __instancecheck__ (census d95f805cce: 2 frames per call).
            if cv is not None and not _int_like(cv, context):
                splits = _target_splitter_folds(model, args[0], args[1], Int(py=values[3]), context)
            result = _target_fit_transform(model, values, args[1], splits, context)
        _ = values^
    elif (operation == "fit" or operation == "fit_transform") and kind == "select_kbest":
        # SelectKBest: fit(X, y); fit_transform(X, y) keeps the support's columns
        var values = xprep_values(args[0], scaler_entries(model, _default_mode(context), context), context)
        var x = args[0] if Int(py=values[0]) == 1 else values[1]
        _select_fit(model, x, values, args[1], context)
        result = model
        if operation == "fit_transform":
            result = _select_transform(model, values, context)
        _ = values^
    elif operation == "fit" or operation == "fit_transform":
        var weights = PythonObject(None)
        if kind == "kbins":
            weights = _fit_weights(model, args, operation)
        elif kind == "spline":
            # fit(X, y, sample_weight) and fit_transform(X, y, sample_weight)
            weights = args[2]
        elif operation == "fit_transform":
            _no_fit_params(model, args)
        _precheck(model, kind, context)
        var mode = _default_mode(context)
        var entries = scaler_entries(model, mode, context)
        var values = xprep_values(args[0], entries, context)
        # The C08 fit_transform (OneHotEncoder bit 2, OrdinalEncoder bit
        # 4, categories='auto'): the fit's inverse is the transform's codes.
        var fields = py_fields(model)
        var auto = py_contains(fields, "categories") and _is_str(fields["categories"]) and String(fields["categories"]) == "auto"
        var capture = operation == "fit_transform" and auto and (
            (kind == "onehot" and _c08_route(mode, entries, 2)) or (kind == "ordinal" and _c08_route(mode, entries, 4)))
        var fused = False
        if kind == "spline" and operation == "fit_transform":
            # with x_prep_spline_fused one program fits and applies; 'error'
            # extrapolation and order 'F' keep fit + transform
            var extrap = fields["extrapolation"]
            var order = fields["order"]
            fused = not (_is_str(extrap) and String(extrap) == "error") and _is_str(order) and String(order) == "C" and _spline_fused(mode, entries)
        var inverse = PythonObject(None)
        if fused:
            result = _spline_fit(model, values, weights, True, context)
        else:
            inverse = _fit(model, kind, values, weights, capture, context)
            result = model
        if operation == "fit_transform" and not fused:
            # The X converted for the fit is the transform's input.
            result = _transform(model, kind, values, inverse, context)
        _ = values^
    elif operation == "transform" or operation == "inverse_transform":
        _fitted(model)
        var values = xprep_values(args[0], scaler_entries(model, py_fields(model)["numeric_mode_"], context), context)
        if operation == "transform":
            result = _transform(model, kind, values, PythonObject(None), context)
        elif kind == "kbins":
            result = _kbins_inverse(model, values, context)
        elif kind == "ordinal":
            result = _ordinal_inverse(model, values, context)
        elif kind == "onehot":
            result = _onehot_inverse(model, values, context)
        else:
            raise Error("TypeError: the native encode owner does not invert this class")
        _ = values^
    else:
        raise Error("unknown native encode prep operation")
    return result


def prep_encode_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var root = call[3]
        var kind = _kind(model, root)
        # The connected context of this estimator's device route (one record
        # read and one pointer compare; the route is resolved once per
        # estimator, bindings/prep_owner_route_unpack.mojo). No operation
        # below reads or compares `device`.
        var context = prep_route_context(model, root)
        classical_connect_numpy(context)
        # The public fit/transform methods run the former `_mode` wrappers'
        # lifecycle (bindings/public_lifecycle_owner.mojo) around the owner.
        if not (operation == "fit" or operation == "fit_transform" or operation == "transform"
                or operation == "inverse_transform"):
            if operation == "vendor_used":
                # the image of this estimator's device route, read back
                return prep_route_vendor_used(model, root).steal_data()
            return _encode_dispatch(model, kind, operation, args, context).steal_data()
        var depth = lifecycle_begin(model, operation, args[0], True, False, True, context)
        var result = PythonObject(None)
        try:
            if kind == "target" and depth == 0 and (operation == "fit" or operation == "fit_transform") and args[1] is None:
                # TargetEncoder's target tag `required` (`_sk_validate._check_y`).
                raise Error(classical_estimator_name(model) + " requires y to be passed, but the target y is None.")
            result = _encode_dispatch(model, kind, operation, args, context)
        finally:
            lifecycle_end(model, operation, depth)
        lifecycle_complete(model, operation, args[0], False, context)
        return lifecycle_result(model, operation, result, "", context).steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("UserWarning:"):
            # an "error" warning filter on the unknown-category warning
            return raise_python_exception(error_without_class(error, "UserWarning"), ExceptionType("PyExc_UserWarning"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
