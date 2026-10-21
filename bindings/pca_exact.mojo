# SPDX-License-Identifier: Apache-2.0
# Native orchestration of PCA's existing exact covariance/full solvers. No
# replacement decomposition or Python helper is called. Fitted buffers stay
# mutable at the Python ABI; native shape descriptors are rebuilt on admission
# so replacing a supported public array preserves its documented semantics.
from std.memory import memcpy, bitcast
from core.pca_exact_impl import pca_check_state, pca_widen_spectrum
from std.python import Python, PythonObject
from bindings.classical_arrays import classical_array, classical_host_f32, classical_word
from bindings.sequential_inputs import sequential_input
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.sequential_description import sequential_fields


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
    if b.type(source) is context["ndarray"] or b.type(source) is context["array_class"]:
        var view = sequential_input(source, context)
        return not (view.dtype is context["float32"] and Bool(py=view.flags.c_contiguous))
    if b.type(source) is b.memoryview or b.type(source) is context["array_type"] or b.type(source) is b.bytearray:
        var view = b.memoryview(source)
        return not (Int(py=view.itemsize) == 4 and String(view.format) == "f" and Bool(py=view.c_contiguous))
    # Nested native sequences are materialized. Pandas/Torch publication and
    # feature-name semantics remain outside this private dense-buffer slice.
    return True


def pca_whiten_entries(entries: PythonObject) raises:
    if entries.get("pca_whiten_transform") is None or entries.get("pca_whiten_inverse_transform") is None:
        raise Error("NotImplementedError: PCA whiten=True requires the compiled whitened transform pair")


def pca_state_array(fields: PythonObject, name: String, shape: PythonObject,
                    context: PythonObject, finite: Bool = False, nonnegative: Bool = False) raises -> PythonObject:
    if not Bool(py=fields.__contains__(name)):
        raise Error("PCA fitted state is missing " + name)
    var value = sequential_input(fields[name], context)
    if value.dtype is not context["float32"] or value.shape != shape or not Bool(py=value.flags.c_contiguous):
        raise Error("PCA requires C-order float32 " + name + " with the fitted shape")
    if finite or nonnegative:
        var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
        pca_check_state(ptr, Int(py=value.size), name, finite, nonnegative)
    return value


def pca_mle(spectrum: PythonObject, rows: Int, entries: PythonObject, context: PythonObject) raises -> Int:
    var count = Int(py=spectrum.size)
    if count < 2:
        return 0
    var ranks = entries.get("rank_bindings")
    if Python.import_module("builtins").type(ranks) is not Python.import_module("builtins").dict:
        raise Error("ImportError: native PCA rank entries are unavailable")
    var wide = classical_array(count, 0, context, True)
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(spectrum)))
    var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(wide)))
    pca_widen_spectrum(src, dst, count)
    var rank = 0
    if String(context["vendor"]) == "cpu":
        if ranks.get("x_decomp_pca_mle_rank") is None:
            raise Error("ImportError: native CPU PCA MLE rank entry is unavailable")
        rank = Int(py=ranks["x_decomp_pca_mle_rank"](sequential_address(wide), Python.list(count, rows)))
    else:
        var required = Python.tuple("x_decomp_dev_alloc", "x_decomp_dev_free", "x_decomp_dev_upload",
                                    "x_decomp_dev_download", "x_decomp_dev_pca_mle_rank")
        for i in range(len(required)):  # driver-loop: over required resident MLE entry names
            if ranks.get(required[i]) is None:
                raise Error("ImportError: native device PCA MLE rank entry is unavailable")
        var device_spectrum = Int(py=ranks["x_decomp_dev_alloc"](2 * count))
        var device_rank = -1
        try:
            device_rank = Int(py=ranks["x_decomp_dev_alloc"](1))
            _ = ranks["x_decomp_dev_upload"](device_spectrum, sequential_address(wide), 2 * count)
            _ = ranks["x_decomp_dev_pca_mle_rank"](device_spectrum, device_rank, Python.list(count, rows))
            var result = classical_array(1, 0, context)
            _ = ranks["x_decomp_dev_download"](device_rank, sequential_address(result), 1)
            rank = Int(classical_word(result))
        finally:
            if device_rank >= 0:
                _ = ranks["x_decomp_dev_free"](device_rank)
            _ = ranks["x_decomp_dev_free"](device_spectrum)
    _ = wide^
    return rank


def pca_prefix(value: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    var result = classical_array(rows, cols, context)
    var count = rows * (cols if cols else 1)
    var source = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(value)))
    var dest = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(result)))
    memcpy(dest=dest, src=source, count=count * 4)
    return result


def pca_exact_fit(model: PythonObject, args: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var fields = sequential_fields(model)
    var solver = String(fields["svd_solver"])
    if solver != "auto" and solver != "covariance_eigh" and solver != "jacobi" and solver != "full":
        # Public PCA retains its randomized implementation until that complete
        # native owner is qualified; this private slice never substitutes it.
        raise Error("NotImplementedError: the private exact PCA owner does not implement this solver")
    if Float64(py=fields["tol"]) != 0:
        raise Error("NotImplementedError: PCA tol is ARPACK's tolerance; leave tol=0.0")
    var whiten = Bool(py=fields["whiten"])
    if whiten:
        pca_whiten_entries(entries)
    var x = classical_host_f32(args[0], 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if rows < 2:
        raise Error("Found array with " + String(rows) + " sample(s) while a minimum of 2 is required by PCA")
    var dense = solver == "full" and cols >= 2
    if dense and entries.get("pca_fit_full") is None:
        raise Error("NotImplementedError: PCA svd_solver='full' requires its compiled dense arm")
    if dense and rows < cols and Int(py=entries["estimators_numeric_mode"]()) != 1:
        raise Error("NotImplementedError: wide full PCA requires identical mode")
    var requested = fields["n_components"]
    var mle = b.type(requested) is b.str and String(requested) == "mle"
    var fraction = Bool(py=b.isinstance(requested, b.float))
    var frac = Float64(0)
    var nc = min(rows, cols)
    if mle:
        if rows < cols:
            raise Error("n_components='mle' is only supported if n_samples >= n_features")
    elif fraction:
        frac = Float64(py=requested)
        if not (frac > 0 and frac < 1):
            raise Error("a float n_components must be in (0, 1)")
    else:
        nc = min(rows, cols) if requested is None else Int(py=b.int(requested))
        if nc < 1 or nc > cols:
            raise Error("n_components must be between 1 and n_features")
    if dense and nc > min(rows, cols):
        raise Error("full SVD n_components cannot exceed min(n_samples, n_features)")
    var components = classical_array(nc, cols, context)
    var mean = classical_array(cols, 0, context)
    var variance = classical_array(nc, 0, context)
    var ratio = classical_array(nc, 0, context)
    var singular = classical_array(nc, 0, context)
    var fit = entries["pca_fit_full"] if dense else entries["pca_fit"]
    var noise = fit(sequential_address(x), sequential_address(components), sequential_address(mean),
                    sequential_address(variance), sequential_address(ratio), sequential_address(singular),
                    Python.list(rows, cols, nc))
    if fraction or mle:
        var keep_in = pca_mle(variance, rows, entries, context) if mle else -1
        var finish = entries["pca_rank_finish"](sequential_address(ratio), sequential_address(variance),
                                                Python.list(nc, keep_in), PythonObject(frac))
        nc = Int(py=finish[0])
        noise = finish[1]
        components = pca_prefix(components, nc, cols, context)
        variance = pca_prefix(variance, nc, 0, context)
        ratio = pca_prefix(ratio, nc, 0, context)
        singular = pca_prefix(singular, nc, 0, context)
    # Commit a complete state only after all native calls and rank selection.
    fields["components_"] = components
    fields["mean_"] = mean
    fields["explained_variance_"] = variance
    fields["explained_variance_ratio_"] = ratio
    fields["singular_values_"] = singular
    fields["noise_variance_"] = b.float(noise)
    fields["n_components_"] = nc
    fields["n_features_in_"] = cols
    fields["n_samples_"] = rows
    fields["input_copied_"] = pca_copied(args[0], context)
    fields["_mojolearn_fitted"] = True
    _ = x^
    return model


def pca_transform(model: PythonObject, source: PythonObject, inverse: Bool,
                   entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not Bool(py=fields.__contains__("components_")):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var dims = PCADimensions(Int(py=fields["n_samples_"]), Int(py=fields["n_features_in_"]), Int(py=fields["n_components_"]))
    if dims.features < 1 or dims.components < 0:
        raise Error("PCA fitted dimensions are invalid")
    var x = classical_host_f32(source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    if Int(py=x.shape[1]) != (dims.components if inverse else dims.features):
        raise Error("X columns do not match PCA's fitted input space")
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
        if inverse:
            _ = entries["pca_whiten_inverse_transform"](sequential_address(x), sequential_address(components),
                    sequential_address(singular), sequential_address(mean), sequential_address(output),
                    Python.list(rows, dims.features, dims.components, dims.rows))
        else:
            _ = entries["pca_whiten_transform"](sequential_address(x), sequential_address(mean),
                    sequential_address(components), sequential_address(singular), sequential_address(output),
                    Python.list(rows, dims.features, dims.components, dims.rows))
        _ = singular^
        _ = variance^
    elif inverse:
        _ = entries["inverse_transform"](sequential_address(x), sequential_address(components),
                sequential_address(mean), sequential_address(output), Python.list(rows, dims.features, dims.components, 1))
    else:
        _ = entries["pca_transform"](sequential_address(x), sequential_address(mean), sequential_address(components),
                                    sequential_address(output), Python.list(rows, dims.features, dims.components))
    _ = x^
    _ = components^
    _ = mean^
    return output
