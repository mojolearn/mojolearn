# SPDX-License-Identifier: Apache-2.0
"""The door of `Harmony.fit(X, batch)` and `Harmony.fit_transform(X, batch)` (lane sc-stats-1, 2026-11-28), reached
from bindings/cluster_fit_public_native_pack.mojo for the codes `family_cfa_harmony_fit` /
`family_cfa_harmony_fit_transform` with the binding the family door resolved.

UNPACK: the estimator's fields and their refusals, X as float32 rows (`admit_f32`), each batch covariate's labels as
int32 codes (`classical_encode_labels`; one memcpy per covariate into the covariate-major code block) with its first
global level, theta and lamb per level. RUN: core/harmony_api.mojo `HarmonyDriver.fit` on the binding's
`harmony_native_api_v1` table, the GIL released (x_cluster/harmony.mojo on that binding's column). PACK: the fitted
attributes."""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import NativeModules
from bindings.py_balanced import py_call, py_type
from bindings.classical_arrays_door import classical_address
from bindings.admitted_input_unpack import admit_f32
from bindings.classical_labels_door import classical_encode_labels
from bindings.family_common_array_pack import _xc_array
from bindings.cluster_fit_public_native_unpack import _set, _attr, _seed
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from providers.python.x_cluster_out_pyapi_pack import _py_array
from core.harmony_api import HARMONY_ABI, HARMONY_COUNT, HarmonyDriver, HarmonyRequest, HarmonyResultSlot


@fieldwise_init
struct HarmonyConnection(Movable):
    var provider: PythonObject
    var driver: HarmonyDriver


def harmony_connection(binding: PythonObject) raises -> HarmonyConnection:
    var factory = neural_descriptor_entry(binding, "harmony_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(HARMONY_ABI), HARMONY_COUNT)
    var driver = HarmonyDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    var mode = driver.mode()
    if mode != 0 and mode != 1:
        raise Error("harmony native provider: unsupported compiled numeric mode")
    return HarmonyConnection(provider, driver^)


def _float(model: PythonObject, name: String, b: PythonObject) raises -> Float64:
    return Float64(py=py_call(b.float, _attr(model, name)))


def _int(model: PythonObject, name: String, b: PythonObject) raises -> Int:
    return Int(py=py_call(b.int, _attr(model, name)))


def _per_covariate(value: PythonObject, c: Int, name: String, b: PythonObject) raises -> List[Float64]:
    """A number for every covariate, or a list / tuple of c numbers. `b` is
    the builtins module harmony_fit admitted for the call."""
    var out_values = List[Float64]()
    if py_type(value) is b.list or py_type(value) is b.tuple:
        if Int(py=py_call(b.len, value)) != c:
            raise Error("ValueError: Harmony: " + name + " must be a number or one number per batch covariate")
        for cc in range(c):  # small-loop(covariates: one parameter per batch covariate)
            out_values.append(Float64(py=py_call(b.float, value[cc])))
    else:
        var v = Float64(py=py_call(b.float, value))
        for _ in range(c):  # small-loop(covariates: one parameter per batch covariate)
            out_values.append(v)
    return out_values^


def _f32_array(values: List[Float32], shape: PythonObject, module: PythonObject,
               context: PythonObject) raises -> PythonObject:
    # `module`: the `array` module harmony_fit admitted once for its three results.
    var flat = _py_array(module, "f", values.unsafe_ptr().bitcast[UInt8](), len(values) * 4)
    return _xc_array(flat, shape, "<f4", "f", context)


def harmony_fit(model: PythonObject, binding: PythonObject, source: PythonObject, batch: PythonObject,
                context: PythonObject) raises:
    # UNPACK
    # One admission of each module for the call (tools/check_admitted_call.py):
    # builtins here, `array` after the GIL window, both passed to the helpers.
    var mods = NativeModules()
    var b = mods.module("builtins")
    var x = admit_f32(source, 2, "X", context)
    var n = x.rows
    var d = x.cols
    if n < 2 or d < 1:
        raise Error("ValueError: Harmony: X must have at least two rows and one column")
    var covariates = Python.list()
    if py_type(batch) is b.list or py_type(batch) is b.tuple:
        for cc in range(Int(py=py_call(b.len, batch))):  # small-loop(covariates: the batch columns given)
            _ = covariates.append(batch[cc])
    else:
        _ = covariates.append(batch)
    var c = Int(py=py_call(b.len, covariates))
    if c < 1:
        raise Error("ValueError: Harmony: batch must hold at least one covariate")
    var codes = List[Int32](length=c * n, fill=Int32(0))
    var offsets = List[Int32](length=c, fill=Int32(0))
    var levels = List[Int](length=c, fill=0)
    var nb = 0
    for cc in range(c):  # small-loop(covariates: one label encoding per batch covariate)
        var labels = classical_encode_labels(covariates[cc], context)
        if labels.n != n:
            raise Error("ValueError: Harmony: batch has " + String(labels.n) + " labels for " + String(n) + " rows")
        var levels_cc = Int(py=py_call(b.len, labels.classes))
        unsafe_memcpy(dest=codes.unsafe_ptr() + cc * n,
                      src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(labels.codes))),
                      count=n)
        offsets[cc] = Int32(nb)
        levels[cc] = levels_cc
        nb += levels_cc
        _ = labels^
    var theta_c = _per_covariate(_attr(model, "theta"), c, "theta", b)
    var lamb_c = _per_covariate(_attr(model, "lamb"), c, "lamb", b)
    var theta = List[Float64]()
    var lamb = List[Float64]()
    for cc in range(c):  # small-loop(covariates: their levels' parameters)
        if theta_c[cc] < 0.0:
            raise Error("ValueError: Harmony: theta must be non-negative")
        if not (lamb_c[cc] > 0.0):
            raise Error("ValueError: Harmony: lamb must be positive")
        for _ in range(levels[cc]):  # small-loop(levels: one parameter per batch level)
            theta.append(theta_c[cc])
            lamb.append(lamb_c[cc])
    var n_clusters = _attr(model, "n_clusters")
    var k: Int
    if n_clusters is None:
        # harmonypy: min(round(N / 30), 100), at least one
        k = max(1, min(Int(Float64(n) / 30.0 + 0.5), 100))
    else:
        k = Int(py=py_call(b.int, n_clusters))
    if k < 1 or k > n:
        raise Error("ValueError: Harmony: n_clusters must be in [1, n_samples]")
    var sigma = _float(model, "sigma", b)
    var tau = _float(model, "tau", b)
    var block_size = _float(model, "block_size", b)
    var max_iter_harmony = _int(model, "max_iter_harmony", b)
    var max_iter_kmeans = _int(model, "max_iter_kmeans", b)
    var epsilon_cluster = _float(model, "epsilon_cluster", b)
    var epsilon_harmony = _float(model, "epsilon_harmony", b)
    var seed = _seed(_attr(model, "random_state"))
    if not (sigma > 0.0):
        raise Error("ValueError: Harmony: sigma must be positive")
    if tau < 0.0:
        raise Error("ValueError: Harmony: tau must be non-negative")
    if not (block_size > 0.0 and block_size <= 1.0):
        raise Error("ValueError: Harmony: block_size must be in (0, 1]")
    if max_iter_harmony < 1 or max_iter_kmeans < 1:
        raise Error("ValueError: Harmony: max_iter_harmony and max_iter_kmeans must be positive")
    var request = HarmonyRequest(x.address, Int(codes.unsafe_ptr()), Int(offsets.unsafe_ptr()),
                                 Int(theta.unsafe_ptr()), Int(lamb.unsafe_ptr()), n, d, k, c, nb, max_iter_harmony,
                                 max_iter_kmeans, seed, sigma, tau, block_size, epsilon_cluster,
                                 epsilon_harmony)
    var connection = harmony_connection(binding)
    var result = HarmonyResultSlot()
    # RUN
    with GILReleased(Python()):
        connection.driver.fit(request, result)
    _ = x^  # keep the admitted X alive through the native call above
    _ = codes^
    _ = offsets^
    _ = theta^
    _ = lamb^
    # PACK
    mods.released()  # the GIL window above ended: the next request re-checks
    var array_module = mods.module("array")
    var objective = Python.list()
    for i in range(len(result.storage[0].objective)):  # small-loop(rounds: one objective value per Harmony round)
        _ = objective.append(PythonObject(result.storage[0].objective[i]))
    _set(model, "embedding_", _f32_array(result.storage[0].z_corr, Python.tuple(n, d), array_module, context))
    _set(model, "R_", _f32_array(result.storage[0].r, Python.tuple(n, k), array_module, context))
    _set(model, "cluster_centers_", _f32_array(result.storage[0].y, Python.tuple(k, d), array_module, context))
    _set(model, "objective_", objective)
    _set(model, "n_iter_", PythonObject(result.storage[0].n_iter))
    _set(model, "converged_", PythonObject(result.storage[0].converged))
    _set(model, "n_clusters_", PythonObject(k))
    _set(model, "n_features_in_", PythonObject(d))
    _ = result^
