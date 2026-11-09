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
from bindings.decomp_inputs import decomp_admit
from bindings.admitted_input import admitted_finite_refuse
from bindings.device_rows_door import door_rows_refuse
from providers.python.pca_fit_scalar_pyapi import PCAFitScalar, pca_fit_scalar_capture
from providers.python.pca_fit_model_pyapi import pca_fit_fields, pca_fit_entries, pca_fit_host_only, pca_fit_preflight


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
    # Freeze mutable scalar storage before Torch/native input preparation
    # can release the GIL. Conversion/errors stay at their prior stages.
    var settings = List[PCAFitScalar]()
    var count_setting = pca_fit_scalar_capture(fields["n_components"], settings, context)
    var extra_setting = -1
    var power_setting = -1
    if solver == "randomized":
        extra_setting = pca_fit_scalar_capture(fields["n_oversamples"], settings, context)
        power_setting = pca_fit_scalar_capture(fields["iterated_power"], settings, context)
    var source = args[0]
    # The one admission path: borrowed when X already is C float32, else one
    # parallel cast into uninitialised storage (scikit-learn wording, est PCA).
    # Lane a7d: the exact solvers' entry (pca_fit_owned) tests a borrowed X
    # on the device after its upload and returns the status word
    # (defer_finite): no host scan of X. Lane a8d: the randomized driver
    # returns the word too (its centred-square total, read before the
    # decomposition), so no solver scans X here.
    # Lane a8d: an identical exact fit (no projection) of a cross-validation
    # fold resident on the device is gathered there by the base binding's
    # `pca_fit_fold` (no host gather, no upload of X; finite and range
    # words from the gather); otherwise a fold's rows are gathered on the
    # host by the admission.
    var fold_entry = py_get(context, "pca_fit_fold")
    var mode = py_get(fields, "numeric_mode")
    var identical = (mode is not None and py_type(mode) is py_type(PythonObject(""))
                     and String(py_call(py_call(mode.strip).lower)) == "identical")
    var fold_ok = fold_entry is not None and identical and not project and solver != "randomized"
    var x = decomp_admit(source, "PCA", context, device_x=fold_ok, defer_finite=True)
    var on_device = x.store.active()
    var rows = x.rows
    var cols = x.cols
    var b = native_loaded_module("builtins")
    var source_type = py_type(source)
    # The former pca_copied reports native C-f32 ndarray/Array/buffer inputs
    # as uncopied; sequences, sparse, pandas and Torch are materialized. The
    # conversion bit is captured by the same descriptor used for execution.
    var direct = source_type is context["ndarray"] or source_type is context["array_class"] or source_type is b.memoryview or source_type is context["array_type"] or source_type is b.bytearray
    var copied = not direct or not x.borrowed
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
    elif not randomized and (py_type(requested) is b.float or py_type(requested) is context["float64"].type):
        requested = settings[count_setting].convert(operation, context, False)
        fraction = Float64(py=requested)
        if not (fraction > 0 and fraction < 1):
            raise Error("mojolearn PCA: a float n_components must be in (0, 1)")
        keep_mode = 1
    else:
        if requested is not None:
            requested = settings[count_setting].convert(operation, context, True)
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
        var extra = settings[extra_setting].convert(operation, context, True)
        oversamples = Int(py=extra)
        if oversamples < 0:
            raise Error("n_oversamples must be nonnegative")
        var power = fields["iterated_power"]
        auto = py_type(power) is b.str and String(power) == "auto"
        if not auto:
            power = settings[power_setting].convert(operation, context, True)
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
        var rparams = Python.list(rows, cols, count, oversamples, iterations, Int(auto), Int(whiten), seed)
        if not x.finite:
            _ = py_call(rparams.append, PythonObject(1))
        var got: PythonObject
        try:
            got = py_call_typed(entries["pca_fit_randomized"], x.address, components.address,
                mean.address, variance.address, ratio.address, singular.address, projection_address,
                rparams)
        except error:
            raise pca_refusal(error^)
        if not x.finite:
            admitted_finite_refuse(Int(py=got[1]), "PCA")
            noise = got[0]
        else:
            noise = got
    elif on_device:
        var fold = py_call_typed(fold_entry, Python.list(x.store.base_id, x.store.base_rows, x.store.index_id),
            Python.list(components.address, mean.address, variance.address, ratio.address, singular.address),
            Python.list(rows, cols, count, Int(dense), keep_mode, fraction))
        door_rows_refuse(Int(py=fold[2]), Int(py=fold[3]), "PCA", x.store.base_rows)
        keep = Int(py=fold[0])
        noise = fold[1]
        if keep != count:
            components = components.prefix(keep, cols, context)
            variance = variance.prefix(keep, 0, context)
            ratio = ratio.prefix(keep, 0, context)
            singular = singular.prefix(keep, 0, context)
    else:
        var fit_params = Python.list(rows, cols, count, Int(dense), keep_mode, Int(whiten), fraction)
        if not x.finite:
            _ = py_call(fit_params.append, PythonObject(1))
        var fitted = py_call_typed(entries["pca_fit_owned"], x.address, components.address,
            mean.address, variance.address, ratio.address, singular.address, projection_address,
            fit_params)
        if not x.finite:
            admitted_finite_refuse(Int(py=fitted[2]), "PCA")
        keep = Int(py=fitted[0])
        noise = fitted[1]
        if keep != count:
            components = components.prefix(keep, cols, context)
            variance = variance.prefix(keep, 0, context)
            ratio = ratio.prefix(keep, 0, context)
            singular = singular.prefix(keep, 0, context)
            if project:
                # MLE returns zero below two eigenvalues. Keep the public
                # projection a matrix even when its component axis is empty.
                var cropped = sequential_array(Python.tuple(rows, keep), context["float32"], context)
                var cropped_address = Int(py=sequential_address(cropped))
                if keep > 0:
                    memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=cropped_address),
                           src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=projection_address),
                           count=rows * keep * 4)
                # Keep the full projection alive through the solver call and
                # the copy above. Its last read was the address taken before
                # the solver; reassigning below is no read, so Mojo's ASAP
                # destruction freed it before the solver wrote into it (the
                # fit_transform rows of n_components=0.8/'mle' came back with
                # the allocator's free-list words in their first 16 bytes).
                _ = projection
                projection = cropped
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
    _ = settings^
    _ = fields^
    _ = x^
    return projection if project else model
