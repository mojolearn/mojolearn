# SPDX-License-Identifier: Apache-2.0
# Native orchestration of PCA's existing exact covariance/full solvers. No
# replacement decomposition or Python helper is called. Fitted buffers stay
# mutable at the Python ABI; native shape descriptors are rebuilt on admission
# so replacing a supported public array preserves its documented semantics.
from std.memory import memcpy, bitcast
from core.pca_exact_impl import pca_check_state
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_array, classical_host_f32, classical_word
from bindings.sequential_inputs import sequential_input
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_description import sequential_fields
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense


struct PCADimensions(Copyable, Movable):
    var rows: Int
    var features: Int
    var components: Int

    def __init__(out self, rows: Int, features: Int, components: Int):
        self.rows = rows
        self.features = features
        self.components = components


def pca_copied(source: PythonObject, context: PythonObject) raises -> Bool:
    var b = Python.import_module("builtins")
    if py_type(source) is context["ndarray"] or py_type(source) is context["array_class"]:
        var view = sequential_input(source, context)
        return not (view.dtype is context["float32"] and Bool(py=view.flags.c_contiguous))
    if py_type(source) is b.memoryview or py_type(source) is context["array_type"] or py_type(source) is b.bytearray:
        var view = py_call(b.memoryview, source)
        return not (Int(py=view.itemsize) == 4 and String(view.format) == "f" and Bool(py=view.c_contiguous))
    # Nested native sequences are materialized. Pandas/Torch publication and
    # feature-name semantics remain outside this private dense-buffer slice.
    return True


def pca_whiten_entries(entries: PythonObject) raises:
    if py_get(entries, "pca_whiten_transform") is None or py_get(entries, "pca_whiten_inverse_transform") is None:
        raise Error("NotImplementedError: PCA whiten=True requires the compiled whitened transform pair")


def pca_state_array(fields: PythonObject, name: String, shape: PythonObject,
                    context: PythonObject, finite: Bool = False, nonnegative: Bool = False) raises -> PythonObject:
    if not py_contains(fields, name):
        raise Error("PCA fitted state is missing " + name)
    var value = sequential_input(fields[name], context)
    if value.dtype is not context["float32"] or value.shape != shape or not Bool(py=value.flags.c_contiguous):
        raise Error("PCA requires C-order float32 " + name + " with the fitted shape")
    if finite or nonnegative:
        var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
        pca_check_state(ptr, Int(py=value.size), name, finite, nonnegative)
    return value


def pca_prefix(value: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    var result = classical_array(rows, cols, context)
    var count = rows * (cols if cols else 1)
    var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
    var dest = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
    memcpy(dest=dest, src=source, count=count * 4)
    return result


def pca_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """X as C-order float32: an exact SciPy sparse container is densified
    from its storage (bindings/host_sparse_native.mojo), as the public PCA
    always densified a sparse input; anything else is the shared host
    conversion."""
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        var dense = host_sparse_dense(source, format, context)
        return classical_host_f32(dense, 2, "X", context)
    return classical_host_f32(source, 2, "X", context)


def pca_refusal(var error: Error) -> Error:
    """A compiled PCA entry's refusal. A builtin class prefix is kept; a
    plain refusal of the data or the fitted state (non-finite values, a
    numerically zero matrix) becomes a ValueError naming PCA, as the public
    shell's `_native` raised it."""
    var message = String(error)
    if (message.startswith("ValueError: ") or message.startswith("TypeError: ") or message.startswith("IndexError: ")
            or message.startswith("AttributeError: ") or message.startswith("RuntimeError: ")
            or message.startswith("NotImplementedError: ") or message.startswith("ZeroDivisionError: ")
            or message.startswith("OverflowError: ") or message.startswith("ArithmeticError: ")
            or message.startswith("ImportError: ") or message.startswith("FileNotFoundError: ")):
        return error^
    return Error("ValueError: mojolearn PCA: " + message)


def pca_too_small(rows: Int, cols: Int) -> String:
    # scikit-learn's check_array wording for a block below 2 x 2.
    if rows < 2:
        return ("Found array with " + String(rows) + " sample(s) (shape=(" + String(rows) + ", " + String(cols)
                + ")) while a minimum of 2 is required by PCA.")
    return ("Found array with " + String(cols) + " feature(s) (shape=(" + String(rows) + ", " + String(cols)
            + ")) while a minimum of 2 is required by PCA.")


def pca_solver_checks(fields: PythonObject, entries: PythonObject) raises -> String:
    """The solver name and tol refusals, before any input is read."""
    var b = Python.import_module("builtins")
    var value = fields["svd_solver"]
    var solver = String("")
    if py_type(value) is b.str:
        solver = String(value)
    if (solver != "auto" and solver != "covariance_eigh" and solver != "jacobi" and solver != "full"
            and solver != "randomized"):
        raise Error("NotImplementedError: mojolearn PCA: svd_solver=" + String(py_call(b.repr, value))
                    + " is not implemented; this class runs ('auto', 'covariance_eigh', 'jacobi', 'full', "
                    + "'randomized'). 'arpack' is scikit-learn's implicitly restarted Lanczos, a third "
                    + "algorithm; accepting the name and running an exact arm would be a silent substitution, "
                    + "which is why this raises (decomposition/NOT_IMPLEMENTED.tsv)")
    if Float64(py=py_call(b.float, fields["tol"])) != 0:
        raise Error("NotImplementedError: mojolearn PCA: tol is ARPACK's convergence tolerance and "
                    + "svd_solver='arpack' is not implemented; leave tol=0.0")
    if Bool(py=fields["whiten"]):
        pca_whiten_entries(entries)
    return solver


def pca_keep_fit(fields: PythonObject, source: PythonObject, components: PythonObject, mean: PythonObject,
                 variance: PythonObject, ratio: PythonObject, singular: PythonObject, noise: PythonObject,
                 keep: Int, cols: Int, rows: Int, context: PythonObject) raises:
    # Commit a complete state only after the one fit call returned.
    var b = Python.import_module("builtins")
    py_setitem(fields, "input_copied_", pca_copied(source, context))
    py_setitem(fields, "components_", components)
    py_setitem(fields, "mean_", mean)
    py_setitem(fields, "explained_variance_", variance)
    py_setitem(fields, "explained_variance_ratio_", ratio)
    py_setitem(fields, "singular_values_", singular)
    py_setitem(fields, "noise_variance_", py_call(b.float, noise))
    py_setitem(fields, "n_components_", keep)
    py_setitem(fields, "n_features_in_", cols)
    py_setitem(fields, "n_samples_", rows)
    py_setitem(fields, "_mojolearn_fitted", True)


def pca_exact_fit(model: PythonObject, args: PythonObject, solver: String, project: Bool,
                  entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`PCA.fit` / `fit_transform` for the exact solvers: ONE compiled call
    (`pca_fit_owned`, decomposition/impl/pca_owner.mojo) fits, picks the kept
    count (every component, the variance fraction or Minka's MLE), takes the
    dropped tail's noise variance and, with `project`, writes X's projection
    from the same resident data. The same entry, arguments and order the
    Python shell used, so the bits are unchanged. Returns the model, or the
    projection when `project`."""
    var b = Python.import_module("builtins")
    var fields = sequential_fields(model)
    var x = pca_input(args[0], context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    # One feature takes the covariance arm (its 1 x 1 covariance without a
    # rotation); at one column the R-SVD and the covariance eigenvalue are
    # the same number, and the R-SVD kernels keep their two-column floor.
    var dense = solver == "full" and cols >= 2
    if dense and py_get(entries, "pca_fit_full") is None:
        raise Error("NotImplementedError: mojolearn PCA: svd_solver='full' needs the dense arm and this build of "
                    + "_mojolearn_estimators does not export it; rebuild it with bindings/build_estimators.sh")
    if rows < 2:
        raise Error(pca_too_small(rows, cols))
    if dense and rows < cols and Int(py=entries["estimators_numeric_mode"]()) != 1:
        # DEVIATION 593: IDENTICAL takes the transposed QR route for a wide
        # matrix; the other mode keeps its refusal.
        raise Error("NotImplementedError: mojolearn PCA: svd_solver='full' needs at least as many samples as "
                    + "features and got " + String(rows) + " x " + String(cols) + ". The dense route is an R-SVD, "
                    + "which needs a tall matrix; svd_solver='covariance_eigh' handles this shape "
                    + "(decomposition/NOT_IMPLEMENTED.tsv, DEVIATION 593)")
    var requested = fields["n_components"]
    var keep_mode = 0
    var frac = Float64(0)
    var nc = min(rows, cols)
    if py_type(requested) is b.str and String(requested) == "mle":
        if rows < cols:
            raise Error("n_components='mle' is only supported if n_samples >= n_features")
        keep_mode = 2
    elif py_type(requested) is not b.bool and Bool(py=py_call(b.isinstance, requested, b.float)):
        frac = Float64(py=requested)
        if not (frac > 0 and frac < 1):
            raise Error("mojolearn PCA: a float n_components must be in (0, 1)")
        keep_mode = 1
    else:
        if requested is not None:
            nc = Int(py=py_call(b.int, requested))
        if nc < 1 or nc > cols:
            raise Error("mojolearn n_components must be in [1, " + String(cols) + "], got " + String(nc))
    if dense and nc > min(rows, cols):
        raise Error("full SVD n_components cannot exceed min(n_samples, n_features)")
    if py_get(entries, "pca_fit_owned") is None:
        raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not export "
                    + "pca_fit_owned; rebuild it with bindings/build_estimators.sh")
    var components = classical_array(nc, cols, context)
    var mean = classical_array(cols, 0, context)
    var variance = classical_array(nc, 0, context)
    var ratio = classical_array(nc, 0, context)
    var singular = classical_array(nc, 0, context)
    var z = classical_array(rows, nc, context) if project else PythonObject(None)
    var fitted = py_call_typed(entries["pca_fit_owned"], sequential_address(x), sequential_address(components),
        sequential_address(mean), sequential_address(variance), sequential_address(ratio),
        sequential_address(singular), sequential_address(z) if project else PythonObject(0),
        Python.list(rows, cols, nc, Int(dense), keep_mode, Int(Bool(py=fields["whiten"])), frac))
    var keep = Int(py=fitted[0])
    if keep != nc:
        # The kernel writes the kept components (and the projection) as the
        # prefix of each output.
        components = pca_prefix(components, keep, cols, context)
        variance = pca_prefix(variance, keep, 0, context)
        ratio = pca_prefix(ratio, keep, 0, context)
        singular = pca_prefix(singular, keep, 0, context)
        if project:
            z = pca_prefix(z, rows, keep, context)
    pca_keep_fit(fields, args[0], components, mean, variance, ratio, singular, fitted[1], keep, cols, rows, context)
    _ = x^
    return z if project else model


def pca_transform(model: PythonObject, source: PythonObject, inverse: Bool,
                   entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var dims = PCADimensions(Int(py=fields["n_samples_"]), Int(py=fields["n_features_in_"]), Int(py=fields["n_components_"]))
    if dims.features < 1 or dims.components < 0:
        raise Error("PCA fitted dimensions are invalid")
    var x = pca_input(source, context)
    var rows = Int(py=x.shape[0])
    if Int(py=x.shape[1]) != (dims.components if inverse else dims.features):
        if inverse:
            raise Error("mojolearn PCA component count differs from fit")
        raise Error("mojolearn PCA feature count differs from fit")
    var whiten = Bool(py=fields["whiten"])
    var components = pca_state_array(fields, "components_", Python.tuple(dims.components, dims.features), context, whiten)
    var mean = pca_state_array(fields, "mean_", Python.tuple(dims.features), context, whiten)
    var output = sequential_array(Python.tuple(rows, dims.features if inverse else dims.components), context["float32"], context)
    if whiten:
        pca_whiten_entries(entries)
        if dims.rows < 2:
            raise Error("PCA whitening requires fit rows >=2")
        var singular = pca_state_array(fields, "singular_values_", Python.tuple(dims.components), context, True, True)
        var variance = pca_state_array(fields, "explained_variance_", Python.tuple(dims.components), context, True, True)
        # `n_samples_` and not this call's row count (DEVIATION 580), so
        # transform(X[:k]) equals transform(X)[:k].
        try:
            if inverse:
                _ = py_call_typed(entries["pca_whiten_inverse_transform"], sequential_address(x), sequential_address(components),
                        sequential_address(singular), sequential_address(mean), sequential_address(output),
                        Python.list(rows, dims.features, dims.components, dims.rows))
            else:
                _ = py_call_typed(entries["pca_whiten_transform"], sequential_address(x), sequential_address(mean),
                        sequential_address(components), sequential_address(singular), sequential_address(output),
                        Python.list(rows, dims.features, dims.components, dims.rows))
        except error:
            raise pca_refusal(error^)
        _ = singular^
        _ = variance^
    elif inverse:
        _ = py_call_typed(entries["inverse_transform"], sequential_address(x), sequential_address(components),
                sequential_address(mean), sequential_address(output), Python.list(rows, dims.features, dims.components, 1))
    else:
        _ = py_call_typed(entries["pca_transform"], sequential_address(x), sequential_address(mean), sequential_address(components),
                                    sequential_address(output), Python.list(rows, dims.features, dims.components))
    _ = x^
    _ = components^
    _ = mean^
    return output
