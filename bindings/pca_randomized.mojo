# SPDX-License-Identifier: Apache-2.0
# PCA(svd_solver='randomized').fit / fit_transform: the argument checks of
# scikit-learn's `_fit_truncated`, then ONE compiled call (`pca_fit_randomized`,
# decomposition/impl/pca_owner.mojo `pca_randomized_owned`) that centers,
# sketches, iterates, takes the small SVD, flips signs, derives the spectrum
# and (fit_transform) projects, with every intermediate resident on the
# device. The same entry and arguments the Python shell passed, so the bits
# are unchanged. (The earlier ClassicalKit transcription made one native call
# per step with host round trips between them; it is replaced by this.)
from std.python import Python, PythonObject
from bindings.classical_arrays import classical_array
from bindings.sequential_arrays import sequential_address
from bindings.sequential_description import sequential_fields
from bindings.py_typed_call import py_call_typed
from bindings.pca_exact import pca_input, pca_too_small, pca_keep_fit, pca_refusal
from bindings.py_balanced import py_call, py_get, py_type


def pca_randomized_fit(model: PythonObject, args: PythonObject, project: Bool, entries: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var b = Python.import_module("builtins")
    var x = pca_input(args[0], context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if rows < 2 or cols < 2:
        raise Error(pca_too_small(rows, cols))
    var requested = fields["n_components"]
    var nc = min(rows, cols) if requested is None else Int(py=py_call(b.int, requested))
    if nc < 1 or nc > cols:
        raise Error("mojolearn n_components must be in [1, " + String(cols) + "], got " + String(nc))
    if nc >= min(rows, cols):
        raise Error("svd_solver='randomized' needs n_components < min(n_samples, n_features)")
    var normalizer = fields["power_iteration_normalizer"]
    var name = String(normalizer) if py_type(normalizer) is b.str else String("")
    if name != "auto" and name != "QR" and name != "LU" and name != "none":
        raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    var seed = fields["random_state"]
    if seed is None:
        seed = PythonObject(0)
    elif py_type(seed) is b.bool or not Bool(py=py_call(b.isinstance, seed, b.int)):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    var oversamples = Int(py=py_call(b.int, fields["n_oversamples"]))
    if oversamples < 0:
        raise Error("n_oversamples must be nonnegative")
    var power = fields["iterated_power"]
    var auto = py_type(power) is b.str and String(power) == "auto"
    var iterations = 0 if auto else Int(py=py_call(b.int, power))
    if py_get(entries, "pca_fit_randomized") is None:
        raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not export "
                    + "pca_fit_randomized; rebuild it with bindings/build_estimators.sh")
    # The Philox seed is taken mod 2^32, with Python integer semantics.
    var word = py_call(b.int.__dict__["__and__"], seed, PythonObject(0xFFFFFFFF))
    var components = classical_array(nc, cols, context)
    var mean = classical_array(cols, 0, context)
    var variance = classical_array(nc, 0, context)
    var ratio = classical_array(nc, 0, context)
    var singular = classical_array(nc, 0, context)
    var z = classical_array(rows, nc, context) if project else PythonObject(None)
    var noise = PythonObject(None)
    try:
        noise = py_call_typed(entries["pca_fit_randomized"], sequential_address(x), sequential_address(components),
            sequential_address(mean), sequential_address(variance), sequential_address(ratio),
            sequential_address(singular), sequential_address(z) if project else PythonObject(0),
            Python.list(rows, cols, nc, oversamples, iterations, Int(auto), Int(Bool(py=fields["whiten"])), word))
    except error:
        raise pca_refusal(error^)
    pca_keep_fit(fields, args[0], components, mean, variance, ratio, singular, noise, nc, cols, rows, context)
    _ = x^
    return z if project else model
