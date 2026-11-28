# SPDX-License-Identifier: Apache-2.0
"""The door of the count-data transformers (lane sc-prep-1, 2026-11-28), reached from
bindings/prep_stateless_native_door.mojo for the kinds that start with "sc_":

  sc_rowtotal  TotalCountNormalizer   each row divided by its total over a column subset, times target_sum
  sc_log       Log1pTransformer       log(1 + x), optionally in another base
  sc_scale     ClippedStandardScaler  (x - mean) / std with ddof 1, clipped at max_value
  sc_hvg       HighlyVariableGenes    the seurat / cell_ranger / seurat_v3 variable-column recipes
  sc_regress   RegressOut             the residuals of a least-squares fit of every column on covariates

UNPACK: X as dense float32 rows (`xprep_values`) or as the caller's CSR buffers (`indptr`, `indices`, `data`, read
as they are: no SciPy call, no import), the parameter
words, the fitted vectors (`prep_fitted_vector`). RUN: the x_prep programs of bindings/prep_scprep_run.mojo (units
x_prep/sc_prep.mojo), each one call of the program runner with the GIL released. PACK: the fitted attributes and the
result, a CSR input giving a CSR result where the arithmetic keeps zeros at zero (the row normalizer, the log, the
uncentred scale, the column selection); a CSR input is never densified without saying so: the operations whose
result is dense by its arithmetic (the centred scale refuses, the regression returns its dense residuals as
documented).
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_arrays_door import classical_address, classical_dtype_str, classical_typed, classical_c_order
from bindings.prep_encode_prog_run import enc_input
from core.preprocessing_program import prep_program_stage
from bindings.admitted_input_unpack import classical_view
from bindings.family_common_pack import family_array
from bindings.scaler_native_door import _ptr
from bindings.x_prep_native_owner_door import xprep_values
from bindings.p3_prep_fitted_unpack import prep_fitted_vector
from bindings.classical_labels_door import classical_encode_labels
from bindings.prep_classify_common_door import is_sparse, _i32_vector
from bindings.admitted_input_unpack import admit_x
from bindings.prep_encode_native_door import EncProg, _take1, _take2
from bindings.prep_stateless_route_unpack import stateless_entries
from bindings.prep_scprep_run import (
    SpInput, RowTotalFit, RowTotalApply, ColBlock, HvgDispersion, HvgV3Loess, HvgV3Select, ScaleApply,
    RegressFit, RegressApply, sp_chunk_width, rowtotal_fit_stage, rowtotal_apply_stage, log1p_stage,
    col_block_stage, hvg_dispersion_stage, hvg_v3_loess_stage, hvg_v3_select_stage, keep_stage, select_dense_stage,
    select_csr_stage, scale_apply_stage, regress_fit_stage, regress_apply_stage, SP_DEFAULT_BINS,
)
from x_prep.sc_prep import SP_PRE_NONE, SP_PRE_EXPM1, SP_SEURAT, SP_CELL_RANGER, SP_SEURAT_V3

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


# ---------------------------------------------------------------- UNPACK helpers


# `b` in the helpers below is the builtins module the public call admitted once (scprep_dispatch, or the caller
# of `_SpX` / `scprep_csr_like`), passed by reference: no helper admits a module itself (tools/check_admitted_call.py).


def _name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def _repr(value: PythonObject, b: PythonObject) raises -> String:
    return String(py_call(b.repr, value))


def _is_none(value: PythonObject) -> Bool:
    return value is None


def _float(value: PythonObject, what: String, b: PythonObject) raises -> Float64:
    if py_type(value) is b.bool or py_type(value) is b.str:
        raise Error("TypeError: " + what + " must be a real number, got " + _repr(value, b))
    return Float64(py=py_call(b.float, value))


def _int(value: PythonObject, what: String, b: PythonObject) raises -> Int:
    if py_type(value) is b.bool or py_type(value) is b.float or py_type(value) is b.str:
        raise Error("TypeError: " + what + " must be an integer, got " + _repr(value, b))
    return Int(py=py_call(b.int, value))


def _truth(value: PythonObject, b: PythonObject) raises -> Bool:
    return Bool(py=py_call(b.bool, value))


def _text(value: PythonObject, what: String, b: PythonObject) raises -> String:
    if py_type(value) is not b.str:
        raise Error("ValueError: " + what + " must be a string, got " + _repr(value, b))
    return String(value)


def _not_fitted(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "n_features_in_"):
        raise Error("RuntimeError: mojolearn: this " + _name(model) + " instance is not fitted yet")


def _region(entries: PythonObject) raises -> Bool:
    return py_get(entries, "x_prep_run_out") is not None


def _mode(context: PythonObject) raises -> PythonObject:
    return context["backend_fields"]["_DEFAULT_MODE"]


def scprep_csr_like(source: PythonObject) raises -> Bool:
    """The form below for a caller that holds no admitted builtins (bindings/prep_stateless_native_door.mojo): one
    admission through the per-call module handle."""
    if is_sparse(source):
        return True
    var mods = NativeModules()
    return scprep_csr_like(source, mods.module("builtins"))


def scprep_csr_like(source: PythonObject, b: PythonObject) raises -> Bool:
    """A sparse X in CSR form: a scipy.sparse container (by its defining module), or any object exposing a
    CSR form (`format == "csr"` and `shape`, `indptr`, `indices`, `data`), read as buffers only."""
    if is_sparse(source):
        return True
    if not Bool(py=py_call(b.hasattr, source, PythonObject("format"))):
        return False
    var fmt = py_call(b.getattr, source, PythonObject("format"))
    if py_type(fmt) is not b.str or String(fmt) != "csr":
        # another sparse layout with index buffers (a CSC-form object) is refused here, never read as dense
        if Bool(py=py_call(b.hasattr, source, PythonObject("indptr"))):
            raise Error("ValueError: a sparse X must be in CSR form; convert it with .tocsr() first")
        return False
    return (Bool(py=py_call(b.hasattr, source, PythonObject("indptr")))
            and Bool(py=py_call(b.hasattr, source, PythonObject("indices")))
            and Bool(py=py_call(b.hasattr, source, PythonObject("data")))
            and Bool(py=py_call(b.hasattr, source, PythonObject("shape"))))


struct _SpX(Movable):
    """X of one call, held for its programs: dense (`values`, `xprep_values`' tuple) or CSR (`indptr`, `indices`,
    `data`, `csr_input`'s int32 / float32 vectors)."""
    var csr: Bool
    var cls: PythonObject
    var n: Int
    var d: Int
    var nnz: Int
    var values: PythonObject
    var indptr: PythonObject
    var indices: PythonObject
    var data: PythonObject

    def __init__(out self, source: PythonObject, entries: PythonObject, b: PythonObject,
                 context: PythonObject) raises:
        self.csr = scprep_csr_like(source, b)
        self.cls = py_type(source)
        self.values = PythonObject(None)
        self.indptr = PythonObject(None)
        self.indices = PythonObject(None)
        self.data = PythonObject(None)
        self.nnz = 0
        if self.csr:
            # UNPACK of the caller's CSR buffers only (no SciPy call): a sparse X must already be in CSR form
            if String(py_call(b.str, py_call(b.getattr, source, PythonObject("format")))) != "csr":
                raise Error("TypeError: " + String(self.cls.__name__) + " input must be in CSR form (csr_matrix or "
                            "csr_array); convert it with .tocsr() first")
            var shape = py_call(b.getattr, source, PythonObject("shape"))
            self.n = Int(py=py_call(b.int, shape[0]))
            self.d = Int(py=py_call(b.int, shape[1]))
            self.indptr = _i32_vector(py_call(b.getattr, source, PythonObject("indptr")), "indptr", context)
            self.indices = _i32_vector(py_call(b.getattr, source, PythonObject("indices")), "indices", context)
            if Int(py=self.indptr.size) != self.n + 1:
                raise Error("ValueError: the CSR indptr must hold n_samples + 1 offsets")
            self.nnz = Int(py=self.indices.size)
            var data = py_call(b.getattr, source, PythonObject("data"))
            self.data = (admit_x(data, 1, "data", context) if self.nnz > 0
                         else classical_typed(Python.tuple(1), "<f4", context["output_context"]))
        else:
            self.values = xprep_values(source, entries, context)
            self.n = Int(py=self.values[3])
            self.d = Int(py=self.values[4])

    def own(self, mut p: EncProg) raises -> SpInput:
        """The owners of X in program `p` (one program, one EncProg)."""
        if self.csr:
            var ip = p.own(self.indptr, self.n + 1)
            var ix = p.own(self.indices, self.nnz)
            var dv = p.own(self.data, self.nnz)
            return SpInput(self.n, self.d, True, -1, ip, ix, dv, self.nnz)
        var xo = p.own_x(self.values)
        return SpInput(self.n, self.d, False, xo, -1, -1, -1, 0)


def _check_width(model: PythonObject, fields: PythonObject, x: _SpX, b: PythonObject) raises:
    var fitted = fields["n_features_in_"]
    if py_type(fitted) is not b.int or Int(py=fitted) != x.d:
        raise Error("ValueError: mojolearn: X has " + String(x.d) + " features, " + _name(model)
                    + " was fitted with " + String(py_call(b.str, fitted)))


def _arena(ran: PythonObject) raises -> Int:
    """The address of a run's host arena."""
    return Int(py=ran[0][1])


def _word(ran: PythonObject, offset: Int) raises -> Float32:
    return _F32P(unsafe_from_address=_arena(ran))[offset]


def _iword(ran: PythonObject, offset: Int) raises -> Int:
    return Int(_I32P(unsafe_from_address=_arena(ran))[offset])


def _i32_out(count: Int, context: PythonObject) raises -> PythonObject:
    return family_array(Python.tuple(count), "<i4", context["output_context"])


def _take_i32(ran: PythonObject, offset: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """`count` int words of a run's host arena as an int32 Array."""
    var result = _i32_out(count, context)
    if count > 0:
        var dest = _I32P(unsafe_from_address=Int(py=classical_address(result)))
        unsafe_memcpy(dest=dest, src=_I32P(unsafe_from_address=_arena(ran)) + offset, count=count)
    _ = ran
    return result


def _copy_i32(source: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    """A fresh int32 Array holding `count` words of the int32 vector `source` (a CSR result's structure is its
    own, never the caller's arrays)."""
    var result = _i32_out(count, context)
    if count > 0:
        var dest = _I32P(unsafe_from_address=Int(py=classical_address(result)))
        unsafe_memcpy(dest=dest, src=_I32P(unsafe_from_address=Int(py=classical_address(source))), count=count)
    _ = source
    return result


def _csr_matrix(cls: PythonObject, data: PythonObject, indices: PythonObject, indptr: PythonObject, n: Int,
                d: Int) raises -> PythonObject:
    """PACK of a CSR result: the caller's own sparse class built from the three finished buffers,
    `cls((data, indices, indptr), shape=(n, d))`; nothing is imported and nothing is computed outside Mojo."""
    return py_call(cls, Python.tuple(data, indices, indptr), shape=Python.tuple(n, d))


def _csr_same_structure(x: _SpX, ran: PythonObject, values_at: Int, context: PythonObject) raises -> PythonObject:
    """A CSR result with X's structure and the program's `values_at` values."""
    var data = _take1(ran, values_at, x.nnz, context) if x.nnz > 0 else classical_typed(Python.tuple(0), "<f4",
                                                                                         context["output_context"])
    return _csr_matrix(x.cls, data, _copy_i32(x.indices, x.nnz, context), _copy_i32(x.indptr, x.n + 1, context),
                       x.n, x.d)


def _dense_or_csr(x: _SpX, ran: PythonObject, at: Int, context: PythonObject) raises -> PythonObject:
    if x.csr:
        return _csr_same_structure(x, ran, at, context)
    return _take2(ran, at, x.n, x.d, context)


# ---------------------------------------------------------------- TotalCountNormalizer


def _column_indices(value: PythonObject, d: Int, context: PythonObject) raises -> PythonObject:
    """UNPACK of `columns`: a boolean mask of length n_features or column indices, as an int32 Array of the
    indices (ascending for a mask, as given for indices)."""
    var view = classical_view(value, context, "columns")
    if Int(py=view.ndim) != 1:
        raise Error("ValueError: columns must be a 1-D boolean mask or a 1-D list of column indices")
    var count = Int(py=view.shape[0])
    var typestr = classical_dtype_str(view, context)
    if not classical_c_order(view, context):
        raise Error("ValueError: columns must be a contiguous 1-D array or a list")
    var address = Int(py=classical_address(view))
    if typestr == "|b1":
        if count != d:
            raise Error("ValueError: a boolean columns mask must have one entry per feature (" + String(d)
                        + "), got " + String(count))
        var mask = _U8P(unsafe_from_address=address)
        var kept = 0
        for j in range(d):  # small-loop(columns: the caller's column mask, admission)
            if mask[j] != 0:
                kept += 1
        var out = _i32_out(max(kept, 1), context)
        var dest = _I32P(unsafe_from_address=Int(py=classical_address(out)))
        var w = 0
        for j in range(d):  # small-loop(columns: the caller's column mask, admission)
            if mask[j] != 0:
                dest[w] = Int32(j)
                w += 1
        _ = view
        return Python.tuple(out, kept)
    if typestr != "<i8" and typestr != "<i4":
        raise Error("TypeError: columns must be a boolean mask or integer column indices, got dtype " + typestr)
    var out = _i32_out(max(count, 1), context)
    var dest = _I32P(unsafe_from_address=Int(py=classical_address(out)))
    for j in range(count):  # small-loop(columns: the caller's column indices, admission)
        var c: Int
        if typestr == "<i8":
            c = Int(_I64P(unsafe_from_address=address)[j])
        else:
            c = Int(_I32P(unsafe_from_address=address)[j])
        if c < 0 or c >= d:
            raise Error("ValueError: column index " + String(c) + " is out of range for " + String(d) + " features")
        dest[j] = Int32(c)
    _ = view
    return Python.tuple(out, count)


def _rowtotal_fit(model: PythonObject, args: PythonObject, b: PythonObject, context: PythonObject) raises -> _SpX:
    var fields = py_fields(model)
    var mode = _mode(context)
    var entries = stateless_entries(model, mode, context)
    var target = fields["target_sum"]
    var median = target is None
    var target_value = 0.0
    if not median:
        target_value = _float(target, "target_sum", b)
        if not (target_value > 0.0):
            raise Error("ValueError: target_sum must be above zero or None, got " + _repr(target, b))
    var rule = _truth(fields["exclude_highly_expressed"], b)
    var frac = _float(fields["max_fraction"], "max_fraction", b)
    if rule and not (frac > 0.0 and frac <= 1.0):
        raise Error("ValueError: max_fraction must be in (0, 1], got " + _repr(fields["max_fraction"], b))
    var x = _SpX(args[0], entries, b, context)
    var columns = fields["columns"]
    var idx = PythonObject(None)
    var n_idx = 0
    if columns is not None:
        var unpacked = _column_indices(columns, x.d, context)
        idx = unpacked[0]
        n_idx = Int(py=unpacked[1])
    var p = EncProg()
    var xin = x.own(p)
    var owner = p.own(idx, max(n_idx, 1)) if columns is not None else -1
    var plan = rowtotal_fit_stage(p.prog, RowTotalFit(xin^, owner, n_idx, rule, frac, median))
    var ran = p.run(entries, context, model)
    var excl = _take1(ran, plan.excl, x.d, context)
    if median:
        var med = _word(ran, plan.median)
        if med != med:
            raise Error("ValueError: target_sum=None takes the median of the positive row totals, and every row "
                        "total is zero")
        target_value = Float64(med)
    py_setitem(fields, "target_sum_", target_value)
    py_setitem(fields, "excluded_", excl)
    py_setitem(fields, "n_features_in_", x.d)
    py_setitem(fields, "numeric_mode_", mode)
    _ = ran
    _ = p^
    _ = idx
    return x^


def _rowtotal_transform(model: PythonObject, x: _SpX, entries: PythonObject, b: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var excl = prep_fitted_vector(fields["excluded_"], context)
    if Int(py=excl.size) != x.d:
        raise Error("RuntimeError: excluded_ does not match n_features_in_")
    var target = _float(fields["target_sum_"], "target_sum_", b)
    var p = EncProg()
    var xin = x.own(p)
    var eo = p.own(excl, x.d)
    var o = rowtotal_apply_stage(p.prog, RowTotalApply(xin^, eo, target, _region(entries)))
    var ran = p.run(entries, context, model)
    var result = _dense_or_csr(x, ran, o, context)
    _ = excl
    _ = p^
    return result


# ---------------------------------------------------------------- Log1pTransformer


def _log_base(fields: PythonObject, b: PythonObject) raises -> Tuple[Bool, Float64]:
    var base = fields["base"]
    if base is None:
        return (False, 0.0)
    var v = _float(base, "base", b)
    if not (v > 0.0) or v == 1.0:
        raise Error("ValueError: base must be above zero and not 1, or None, got " + _repr(base, b))
    return (True, v)


def _log_transform(model: PythonObject, x: _SpX, entries: PythonObject, b: PythonObject, context: PythonObject) raises -> PythonObject:
    var base = _log_base(py_fields(model), b)
    var p = EncProg()
    var xin = x.own(p)
    var o = log1p_stage(p.prog, xin, base[1], base[0], _region(entries))
    var ran = p.run(entries, context, model)
    var result = _dense_or_csr(x, ran, o, context)
    _ = p^
    return result


# ---------------------------------------------------------------- the column statistics


struct _Moments(Movable):
    """The per-segment column moments of the gene blocks, gathered into two host Arrays of segments x d words
    (segment-major), and the batch row starts."""
    var mean: PythonObject
    var var_: PythonObject
    var start: PythonObject
    var segments: Int

    def __init__(out self, mean: PythonObject, var_: PythonObject, start: PythonObject, segments: Int):
        self.mean = mean
        self.var_ = var_
        self.start = start
        self.segments = segments


def _block_programs(model: PythonObject, x: _SpX, codes: PythonObject, batches: Int, pre: Int, all_rows: Bool,
                    clip: PythonObject, entries: PythonObject, context: PythonObject) raises -> _Moments:
    """The gene-block programs (bindings/prep_scprep_run.mojo `col_block_stage`): each block's two segment-major
    results copied into the two whole-gene host blocks by one memcpy per segment."""
    var d = x.d
    var segments = batches + (1 if all_rows and clip is None else 0)
    var out_context = context["output_context"]
    var first = classical_typed(Python.tuple(segments * d), "<f4", out_context)
    var second = classical_typed(Python.tuple(segments * d), "<f4", out_context)
    var fp = _ptr(first)
    var sp = _ptr(second)
    var start = _i32_out(batches + 1, context)
    var width = sp_chunk_width(x.n, d)
    var c0 = 0
    while c0 < d:  # driver-loop: over the gene blocks of the block copy's memory bound
        var dc = min(width, d - c0)
        var p = EncProg()
        var xin = x.own(p)
        var co = p.own(codes, x.n) if codes is not None else -1
        var cl = p.own(clip, batches * d) if clip is not None else -1
        var plan = col_block_stage(p.prog, ColBlock(xin^, co, batches, c0, dc, pre, all_rows, cl))
        var ran = p.run(entries, context, model)
        var arena = _arena(ran)
        if c0 == 0:
            unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(start))),
                          src=_I32P(unsafe_from_address=arena) + plan.start, count=batches + 1)
        var src = _F32P(unsafe_from_address=arena) + plan.out
        for s in range(segments):  # driver-loop: over the segments (batches) of one block
            unsafe_memcpy(dest=fp + s * d + c0, src=src + s * dc, count=dc)
            unsafe_memcpy(dest=sp + s * d + c0, src=src + segments * dc + s * dc, count=dc)
        _ = ran
        _ = p^
        c0 += dc
    _ = first
    _ = second
    return _Moments(first, second, start, segments)


# ---------------------------------------------------------------- ClippedStandardScaler


def _scale_options(fields: PythonObject, b: PythonObject) raises -> Tuple[Bool, Bool, Float64]:
    var zc = _truth(fields["zero_center"], b)
    var mv = fields["max_value"]
    if mv is None:
        return (zc, False, 0.0)
    var v = _float(mv, "max_value", b)
    if not (v > 0.0):
        raise Error("ValueError: max_value must be above zero or None, got " + _repr(mv, b))
    return (zc, True, v)


def _scale_fit(model: PythonObject, args: PythonObject, b: PythonObject, context: PythonObject) raises -> _SpX:
    var fields = py_fields(model)
    var mode = _mode(context)
    var entries = stateless_entries(model, mode, context)
    _ = _scale_options(fields, b)
    var x = _SpX(args[0], entries, b, context)
    if x.n < 2:
        raise Error("ValueError: ClippedStandardScaler needs at least two rows (the variance has ddof 1)")
    var m = _block_programs(model, x, PythonObject(None), 1, SP_PRE_NONE, False, PythonObject(None), entries,
                            context)
    # scale_ = sqrt(var_), a zero deviation counting as one: one program over the d words
    var p = EncProg()
    var vo = p.own(m.var_, x.d)
    var sc = _scale_of(p, vo, x.d)
    var ran = p.run(entries, context, model)
    py_setitem(fields, "mean_", m.mean)
    py_setitem(fields, "var_", m.var_)
    var scale = _take1(ran, sc, x.d, context)
    py_setitem(fields, "scale_", scale)
    # Scanpy's var['std'] (the same words: sqrt(var_ with ddof 1), a zero deviation as one)
    py_setitem(fields, "std_", scale)
    py_setitem(fields, "n_samples_seen_", x.n)
    py_setitem(fields, "n_features_in_", x.d)
    py_setitem(fields, "numeric_mode_", mode)
    _ = p^
    return x^


def _scale_of(mut p: EncProg, var_owner: Int, d: Int) raises -> Int:
    """x_prep op 102 `std_scale` (StandardScaler's own: zero -> one, else sqrt) over the fitted variances."""
    var vin = enc_input(p.prog, d, var_owner)
    var sc = p.prog.alloc(d)
    prep_program_stage(p.prog.program, 102, d, vin, sc)
    return sc


def _scale_transform(model: PythonObject, x: _SpX, entries: PythonObject, b: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var opts = _scale_options(fields, b)
    if x.csr and opts[0]:
        raise Error("ValueError: zero_center=True makes a sparse X dense (every zero becomes -mean / std); pass "
                    "zero_center=False to keep it sparse, or a dense X")
    var mean = prep_fitted_vector(fields["mean_"], context)
    var std = prep_fitted_vector(fields["scale_"], context)
    var p = EncProg()
    var xin = x.own(p)
    var mo = p.own(mean, x.d)
    var so = p.own(std, x.d)
    var o = scale_apply_stage(p.prog, ScaleApply(xin^, mo, so, opts[0], opts[1], opts[2], _region(entries)))
    var ran = p.run(entries, context, model)
    var result = _dense_or_csr(x, ran, o, context)
    _ = mean
    _ = std
    _ = p^
    return result


# ---------------------------------------------------------------- HighlyVariableGenes


def _flavor(fields: PythonObject, b: PythonObject) raises -> Int:
    var name = _text(fields["flavor"], "flavor", b)
    if name == "seurat":
        return SP_SEURAT
    if name == "cell_ranger":
        return SP_CELL_RANGER
    if name == "seurat_v3":
        return SP_SEURAT_V3
    raise Error("ValueError: flavor must be 'seurat', 'cell_ranger' or 'seurat_v3', got '" + name + "'")


def _hvg_fit(model: PythonObject, args: PythonObject, b: PythonObject, context: PythonObject) raises -> _SpX:
    var fields = py_fields(model)
    var mode = _mode(context)
    var entries = stateless_entries(model, mode, context)
    var flavor = _flavor(fields, b)
    var n_top = -1
    if fields["n_top_genes"] is not None:
        n_top = _int(fields["n_top_genes"], "n_top_genes", b)
        if n_top < 1:
            raise Error("ValueError: n_top_genes must be a positive integer or None")
    if flavor == SP_SEURAT_V3 and n_top < 0:
        raise Error("ValueError: flavor='seurat_v3' requires n_top_genes")
    var n_bins = _int(fields["n_bins"], "n_bins", b)
    if n_bins < 1:
        raise Error("ValueError: n_bins must be a positive integer")
    var span = _float(fields["span"], "span", b)
    if not (span > 0.0 and span <= 1.0):
        raise Error("ValueError: span must be in (0, 1], got " + _repr(fields["span"], b))
    var cuts: List[Float64] = [_float(fields["min_mean"], "min_mean", b), _float(fields["max_mean"], "max_mean", b),
                               _float(fields["min_disp"], "min_disp", b), _float(fields["max_disp"], "max_disp", b)]
    var x = _SpX(args[0], entries, b, context)
    if x.n < 2:
        raise Error("ValueError: HighlyVariableGenes needs at least two rows (the variance has ddof 1)")
    # the batches (Scanpy's batch_key): y's labels, encoded as f_classif's classes
    var codes = PythonObject(None)
    var batches = 1
    var y = args[1]
    if y is not None:
        var labels = classical_encode_labels(y, context)
        if labels.n != x.n:
            raise Error("ValueError: X has " + String(x.n) + " rows but y (the batches) has " + String(labels.n))
        codes = labels.codes
        batches = Int(py=py_call(b.len, labels.classes))
    var d = x.d
    var out_context = context["output_context"]
    if flavor != SP_SEURAT_V3:
        var pre = SP_PRE_EXPM1 if flavor == SP_SEURAT else SP_PRE_NONE
        var m = _block_programs(model, x, codes, batches, pre, False, PythonObject(None), entries, context)
        var p = EncProg()
        var mo = p.own(m.mean, batches * d)
        var vo = p.own(m.var_, batches * d)
        var plan = hvg_dispersion_stage(p.prog, HvgDispersion(d, batches, flavor,
                                                              n_bins if flavor == SP_SEURAT else 20, mo, vo, n_top,
                                                              cuts^))
        var keep = keep_stage(p.prog, d, plan.hv)
        var ran = p.run(entries, context, model)
        py_setitem(fields, "means_", _take1(ran, plan.means, d, context))
        py_setitem(fields, "dispersions_", _take1(ran, plan.disp, d, context))
        py_setitem(fields, "dispersions_norm_", _take1(ran, plan.disp_norm, d, context))
        py_setitem(fields, "highly_variable_", _take_i32(ran, plan.hv, d, context))
        py_setitem(fields, "highly_variable_nbatches_",
                   _take_i32(ran, plan.nbatches, d, context) if plan.nbatches >= 0 else PythonObject(None))
        _keep_fields(fields, ran, keep.idx, keep.inv, keep.count, d, context)
        _ = m^
        _ = p^
    else:
        var m = _block_programs(model, x, codes, batches, SP_PRE_NONE, True, PythonObject(None), entries, context)
        var p = EncProg()
        var mo = p.own(m.mean, batches * d)
        var vo = p.own(m.var_, batches * d)
        var so = p.own(m.start, batches + 1)
        var lp = hvg_v3_loess_stage(p.prog, HvgV3Loess(d, batches, mo, vo, so, span))
        var ran = p.run(entries, context, model)
        var reg = _take1(ran, lp.regstd, batches * d, context)
        var clip = _take1(ran, lp.clip, batches * d, context)
        var ncnt = _take_i32(ran, lp.ncnt, batches, context)
        _ = p^
        var c = _block_programs(model, x, codes, batches, SP_PRE_NONE, False, clip, entries, context)
        var q = EncProg()
        var qm = q.own(m.mean, batches * d)
        var qr = q.own(reg, batches * d)
        var qs = q.own(c.mean, batches * d)
        var qq = q.own(c.var_, batches * d)
        var qn = q.own(ncnt, batches)
        var sel = hvg_v3_select_stage(q.prog, HvgV3Select(d, batches, qm, qr, qs, qq, qn, n_top))
        var keep = keep_stage(q.prog, d, sel.hv)
        var ran2 = q.run(entries, context, model)
        # the whole-data moments (the segment after the batches)
        var means = classical_typed(Python.tuple(d), "<f4", out_context)
        var variances = classical_typed(Python.tuple(d), "<f4", out_context)
        unsafe_memcpy(dest=_ptr(means), src=_ptr(m.mean) + batches * d, count=d)
        unsafe_memcpy(dest=_ptr(variances), src=_ptr(m.var_) + batches * d, count=d)
        py_setitem(fields, "means_", means)
        py_setitem(fields, "variances_", variances)
        py_setitem(fields, "variances_norm_", _take1(ran2, sel.varnorm, d, context))
        py_setitem(fields, "highly_variable_rank_", _take1(ran2, sel.rank, d, context))
        py_setitem(fields, "highly_variable_nbatches_", _take_i32(ran2, sel.nbatches, d, context))
        py_setitem(fields, "highly_variable_", _take_i32(ran2, sel.hv, d, context))
        _keep_fields(fields, ran2, keep.idx, keep.inv, keep.count, d, context)
        _ = reg
        _ = clip
        _ = ncnt
        _ = c^
        _ = m^
        _ = q^
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "numeric_mode_", mode)
    return x^


def _keep_fields(fields: PythonObject, ran: PythonObject, idx: Int, inv: Int, count: Int, d: Int,
                 context: PythonObject) raises:
    var k = _iword(ran, count)
    py_setitem(fields, "n_highly_variable_", k)
    py_setitem(fields, "_keep_idx", _take_i32(ran, idx, max(k, 1), context))
    py_setitem(fields, "_keep_inv", _take_i32(ran, inv, d, context))


def _hvg_transform(model: PythonObject, x: _SpX, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var k = Int(py=fields["n_highly_variable_"])
    var p = EncProg()
    var xin = x.own(p)
    if x.csr:
        var io = p.own(fields["_keep_inv"], x.d)
        var plan = select_csr_stage(p.prog, xin, io)
        var ran = p.run(entries, context, model)
        var nnz = _iword(ran, plan.indptr + x.n)
        var data = _take1(ran, plan.data, nnz, context) if nnz > 0 else classical_typed(Python.tuple(0), "<f4",
                                                                                         context["output_context"])
        var result = _csr_matrix(x.cls, data, _take_i32(ran, plan.indices, nnz, context),
                                 _take_i32(ran, plan.indptr, x.n + 1, context), x.n, k)
        _ = p^
        return result
    var io = p.own(fields["_keep_idx"], max(k, 1))
    var o = select_dense_stage(p.prog, xin, io, k, _region(entries))
    var ran = p.run(entries, context, model)
    var result = _take2(ran, o, x.n, k, context)
    _ = p^
    return result


# ---------------------------------------------------------------- RegressOut


struct _Cov(Movable):
    var values: PythonObject
    var k: Int

    def __init__(out self, source: PythonObject, n: Int, entries: PythonObject, b: PythonObject,
                 context: PythonObject) raises:
        if source is None:
            raise Error("ValueError: RegressOut needs the covariates of X's rows: pass them as y (n_samples x "
                        "n_covariates)")
        if scprep_csr_like(source, b) or is_sparse(source):
            raise Error("TypeError: RegressOut takes dense covariates (n_samples x n_covariates)")
        self.values = xprep_values(source, entries, context)
        if Int(py=self.values[3]) != n:
            raise Error("ValueError: X has " + String(n) + " rows but the covariates have "
                        + String(Int(py=self.values[3])))
        self.k = Int(py=self.values[4])
        if self.k < 1:
            raise Error("ValueError: RegressOut needs at least one covariate column")


def _regress_fit(model: PythonObject, args: PythonObject, residuals: Bool, b: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var mode = _mode(context)
    var entries = stateless_entries(model, mode, context)
    var x = _SpX(args[0], entries, b, context)
    var cov = _Cov(args[1], x.n, entries, b, context)
    var p = EncProg()
    var xin = x.own(p)
    var co = p.own_x(cov.values)
    var plan = regress_fit_stage(p.prog, RegressFit(xin^, co, cov.k, residuals, _region(entries)))
    var ran = p.run(entries, context, model)
    py_setitem(fields, "covariate_mean_", _take1(ran, plan.cmean, cov.k, context))
    py_setitem(fields, "covariate_scale_", _take1(ran, plan.cstd, cov.k, context))
    py_setitem(fields, "mean_", _take1(ran, plan.xmean, x.d, context))
    py_setitem(fields, "coef_", _take2(ran, plan.coef, cov.k, x.d, context))
    py_setitem(fields, "_coef", _take1(ran, plan.coef, cov.k * x.d, context))
    py_setitem(fields, "n_covariates_", cov.k)
    py_setitem(fields, "n_features_in_", x.d)
    py_setitem(fields, "numeric_mode_", mode)
    var result = model
    if residuals:
        result = _take2(ran, plan.out, x.n, x.d, context)
    _ = cov^
    _ = x^
    _ = p^
    return result


def _regress_transform(model: PythonObject, args: PythonObject, b: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var entries = stateless_entries(model, fields["numeric_mode_"], context)
    var x = _SpX(args[0], entries, b, context)
    _check_width(model, fields, x, b)
    var y = args[1] if len(args) > 1 else PythonObject(None)
    var cov = _Cov(y, x.n, entries, b, context)
    var k = Int(py=fields["n_covariates_"])
    if cov.k != k:
        raise Error("ValueError: the covariates have " + String(cov.k) + " columns, RegressOut was fitted with "
                    + String(k))
    var cm = prep_fitted_vector(fields["covariate_mean_"], context)
    var cs = prep_fitted_vector(fields["covariate_scale_"], context)
    var xm = prep_fitted_vector(fields["mean_"], context)
    var coef = prep_fitted_vector(fields["_coef"], context)
    if Int(py=cm.size) != k or Int(py=cs.size) != k or Int(py=xm.size) != x.d or Int(py=coef.size) != k * x.d:
        raise Error("RuntimeError: RegressOut's fitted arrays do not match n_covariates_ and n_features_in_")
    var p = EncProg()
    var xin = x.own(p)
    var co = p.own_x(cov.values)
    var cmo = p.own(cm, k)
    var cso = p.own(cs, k)
    var xmo = p.own(xm, x.d)
    var cfo = p.own(coef, k * x.d)
    var request = RegressApply(xin^, co, k, cmo, cso, xmo, cfo, _region(entries))
    var o = regress_apply_stage(p.prog, request)
    var ran = p.run(entries, context, model)
    var result = _take2(ran, o, x.n, x.d, context)
    _ = cm
    _ = cs
    _ = xm
    _ = coef
    _ = cov^
    _ = p^
    return result


# ---------------------------------------------------------------- dispatch


def scprep_dispatch(model: PythonObject, kind: String, operation: String, args: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    """One public operation of a count-data transformer: fit, transform or fit_transform."""
    var fields = py_fields(model)
    # The one builtins admission of the call, passed to every helper.
    var mods = NativeModules()
    var b = mods.module("builtins")
    if kind == "sc_regress":
        if operation == "fit":
            return _regress_fit(model, args, False, b, context)
        if operation == "fit_transform":
            return _regress_fit(model, args, True, b, context)
        if operation == "transform":
            _not_fitted(model, fields)
            return _regress_transform(model, args, b, context)
        raise Error("unknown native count-data operation")
    if operation == "fit" or operation == "fit_transform":
        var x: _SpX
        if kind == "sc_rowtotal":
            x = _rowtotal_fit(model, args, b, context)
        elif kind == "sc_scale":
            x = _scale_fit(model, args, b, context)
        elif kind == "sc_hvg":
            x = _hvg_fit(model, args, b, context)
        elif kind == "sc_log":
            var mode = _mode(context)
            _ = _log_base(fields, b)
            x = _SpX(args[0], stateless_entries(model, mode, context), b, context)
            py_setitem(fields, "n_features_in_", x.d)
            py_setitem(fields, "numeric_mode_", mode)
        else:
            raise Error("unknown native count-data transformer")
        if operation == "fit":
            _ = x^
            return model
        var entries = stateless_entries(model, fields["numeric_mode_"], context)
        mods.released()  # the fit above ran its programs (a GIL window): re-check on this request
        b = mods.module("builtins")
        var result = _transform(model, kind, x, entries, b, context)
        _ = x^
        return result
    if operation == "transform":
        _not_fitted(model, fields)
        var entries = stateless_entries(model, fields["numeric_mode_"], context)
        var x = _SpX(args[0], entries, b, context)
        _check_width(model, fields, x, b)
        var result = _transform(model, kind, x, entries, b, context)
        _ = x^
        return result
    raise Error("unknown native count-data operation")


def _transform(model: PythonObject, kind: String, x: _SpX, entries: PythonObject, b: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "sc_rowtotal":
        return _rowtotal_transform(model, x, entries, b, context)
    if kind == "sc_log":
        return _log_transform(model, x, entries, b, context)
    if kind == "sc_scale":
        return _scale_transform(model, x, entries, b, context)
    if kind == "sc_hvg":
        return _hvg_transform(model, x, entries, context)
    raise Error("unknown native count-data transformer")
