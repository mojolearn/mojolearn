# SPDX-License-Identifier: Apache-2.0
# Complete PCA fit/fit_transform owner. Only this native operation orchestrates
# admission, scalar snapshots, native solvers and publication of finished state.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.native_loaded_module import native_loaded_module
from bindings.pca_exact import pca_too_small, pca_whiten_entries, pca_refusal
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.pca_input_pyapi import pca_input_snapshot
from providers.python.pca_fit_model_pyapi import pca_fit_fields, pca_fit_entries, pca_fit_host_only, pca_fit_preflight, pca_fit_number


struct PCAFitOutput(Copyable, Movable):
    var keep: PythonObject
    var address: Int

    def __init__(out self, rows: Int, cols: Int, context: PythonObject) raises:
        self.keep = sequential_array(Python.tuple(rows, cols) if cols else Python.tuple(rows), context["float32"], context)
        self.address = Int(py=sequential_address(self.keep))

    def prefix(self, rows: Int, cols: Int, context: PythonObject) raises -> Self:
        var result = Self(rows, cols, context)
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=result.address),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.address),
               count=rows * (cols if cols else 1) * 4)
        _ = self.keep  # retain the original output owner through the native copy
        return result^


def pca_fit_operation(model: PythonObject, args: PythonObject, project: Bool,
                       context: PythonObject) raises -> PythonObject:
    var state = classical_family_admit(model)
    if len(args) != 2:
        raise Error("TypeError: native PCA fitting requires X and ignored y")
    # The ignored target keeps its existing native input admission. X uses
    # PCA's dedicated sparse/pandas/Torch/buffer provider, not the narrower
    # sequential-value admission that formerly preempted its supported paths.
    sequential_admit_value(args[1], context)
    if py_get(context, "ndarray") is None:
        raise Error("ImportError: mojolearn: PCA requires optional NumPy support. Install it with: "
                    + 'python -m pip install "mojolearn[numpy]" ("mojolearn[verify]" also includes NumPy).')
    var operation = String("fit_transform") if project else String("fit")
    var fields = pca_fit_fields(state, operation, context)
    var entries = pca_fit_entries(fields, context)
    if pca_fit_host_only(model):
        raise Error("ImportError: mojolearn: no CPU implementation of the PCA fit on the host class; "
                    + "the host PCA transforms a saved model only")
    var solver = pca_fit_preflight(fields, operation, context)
    var whiten = Bool(py=fields["whiten"])
    if whiten:
        pca_whiten_entries(entries)
    var source = args[0]
    var x = pca_input_snapshot(source, context)
    var rows = x.shape[0]
    var cols = x.shape[1]
    var b = native_loaded_module("builtins")
    var source_type = py_type(source)
    # The former pca_copied reports native C-f32 ndarray/Array/buffer inputs
    # as uncopied; sequences, sparse, pandas and Torch are materialized. The
    # conversion bit is captured by the same descriptor used for execution.
    var direct = source_type is context["ndarray"] or source_type is context["array_class"] or source_type is b.memoryview or source_type is context["array_type"] or source_type is b.bytearray
    var copied = not direct or x.copied
    var randomized = solver == "randomized"
    var dense = solver == "full" and cols >= 2
    if dense and py_get(entries, "pca_fit_full") is None:
        raise Error("NotImplementedError: mojolearn PCA: svd_solver='full' needs the dense arm and this build of "
                    + "_mojolearn_estimators does not export it; rebuild it with bindings/build_estimators.sh")
    if rows < 2 or (randomized and cols < 2):
        raise Error(pca_too_small(rows, cols))
    if dense and rows < cols and Int(py=entries["_pca_numeric_mode"]) != 1:
        raise Error("NotImplementedError: mojolearn PCA: svd_solver='full' needs at least as many samples as "
                    + "features and got " + String(rows) + " x " + String(cols) + ". The dense route is an R-SVD, "
                    + "which needs a tall matrix; svd_solver='covariance_eigh' handles this shape "
                    + "(decomposition/NOT_IMPLEMENTED.tsv, DEVIATION 593)")
    var requested = fields["n_components"]
    var keep_mode = 0
    var fraction = Float64(0)
    var count = min(rows, cols)
    if not randomized and py_type(requested) is b.str and String(requested) == "mle":
        if rows < cols:
            raise Error("n_components='mle' is only supported if n_samples >= n_features")
        keep_mode = 2
    elif not randomized and (py_type(requested) is b.float or py_type(requested) is py_get(context, "float64")):
        requested = pca_fit_number(requested, operation, context, False)
        fraction = Float64(py=requested)
        if not (fraction > 0 and fraction < 1):
            raise Error("mojolearn PCA: a float n_components must be in (0, 1)")
        keep_mode = 1
    else:
        if requested is not None:
            requested = pca_fit_number(requested, operation, context, True)
            count = Int(py=requested)
        if count < 1 or count > cols:
            raise Error("mojolearn n_components must be in [1, " + String(cols) + "], got " + String(count))
    if dense and count > min(rows, cols):
        raise Error("full SVD n_components cannot exceed min(n_samples, n_features)")
    if randomized and count >= min(rows, cols):
        raise Error("svd_solver='randomized' needs n_components < min(n_samples, n_features)")
    var oversamples = 0
    var iterations = 0
    var auto = False
    var seed = PythonObject(0)
    if randomized:
        var normalizer = fields["power_iteration_normalizer"]
        if py_type(normalizer) is not b.str or (String(normalizer) != "auto" and String(normalizer) != "QR" and String(normalizer) != "LU" and String(normalizer) != "none"):
            raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
        seed = fields["random_state"]
        if seed is None:
            seed = PythonObject(0)
        elif py_type(seed) is not b.int:
            raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
        var extra = pca_fit_number(fields["n_oversamples"], operation, context, True)
        oversamples = Int(py=extra)
        if oversamples < 0:
            raise Error("n_oversamples must be nonnegative")
        var power = fields["iterated_power"]
        auto = py_type(power) is b.str and String(power) == "auto"
        if not auto:
            power = pca_fit_number(power, operation, context, True)
            iterations = Int(py=power)
        if py_get(entries, "pca_fit_randomized") is None:
            raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not export "
                        + "pca_fit_randomized; rebuild it with bindings/build_estimators.sh")
        # Exact immutable native integer, frozen before any input/reader work.
        seed = py_call(b.int.__dict__["__and__"], seed, PythonObject(0xFFFFFFFF))
    elif py_get(entries, "pca_fit_owned") is None:
        raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not export "
                    + "pca_fit_owned; rebuild it with bindings/build_estimators.sh")
    var components = PCAFitOutput(count, cols, context)
    var mean = PCAFitOutput(cols, 0, context)
    var variance = PCAFitOutput(count, 0, context)
    var ratio = PCAFitOutput(count, 0, context)
    var singular = PCAFitOutput(count, 0, context)
    # No dummy numerical output is allocated for fit. A missing projection
    # is represented by a native null address, as in the existing solver ABI.
    var projection = PythonObject(None)
    var projection_address = 0
    if project:
        projection = sequential_array(Python.tuple(rows, count), context["float32"], context)
        projection_address = Int(py=sequential_address(projection))
    var noise = PythonObject(None)
    var keep = count
    if randomized:
        try:
            noise = py_call_typed(entries["pca_fit_randomized"], x.source.addr, components.address,
                mean.address, variance.address, ratio.address, singular.address, projection_address,
                Python.list(rows, cols, count, oversamples, iterations, Int(auto), Int(whiten), seed))
        except error:
            raise pca_refusal(error^)
    else:
        var fitted = py_call_typed(entries["pca_fit_owned"], x.source.addr, components.address,
            mean.address, variance.address, ratio.address, singular.address, projection_address,
            Python.list(rows, cols, count, Int(dense), keep_mode, Int(whiten), fraction))
        keep = Int(py=fitted[0])
        noise = fitted[1]
        if keep != count:
            components = components.prefix(keep, cols, context)
            variance = variance.prefix(keep, 0, context)
            ratio = ratio.prefix(keep, 0, context)
            singular = singular.prefix(keep, 0, context)
            if project:
                var cropped = PCAFitOutput(rows, keep, context)
                memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=cropped.address),
                       src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=projection_address),
                       count=rows * keep * 4)
                projection = cropped.keep
    # Publish only after the solver and all finished-output allocations pass.
    # Writes target the admitted original dict; no live settings are reread.
    py_setitem(state, "input_copied_", copied)
    py_setitem(state, "components_", components.keep)
    py_setitem(state, "mean_", mean.keep)
    py_setitem(state, "explained_variance_", variance.keep)
    py_setitem(state, "explained_variance_ratio_", ratio.keep)
    py_setitem(state, "singular_values_", singular.keep)
    py_setitem(state, "noise_variance_", py_call(b.float, noise))
    py_setitem(state, "n_components_", keep)
    py_setitem(state, "n_features_in_", cols)
    py_setitem(state, "n_samples_", rows)
    py_setitem(state, "_mojolearn_fitted", True)
    _ = fields^
    _ = x^
    return projection if project else model
