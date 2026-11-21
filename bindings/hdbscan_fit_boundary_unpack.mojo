# SPDX-License-Identifier: Apache-2.0
"""HDBSCAN fit UNPACK: provider admission, the compiled-mode check, the
estimator's settings read once into `HdbscanFitSettings` and narrowed to the
typed `HdbscanFitOptions`, and the address admission of a borrowed buffer.
RUN is the typed driver call (core/hdbscan_fit_api.mojo) with the number
refusals of providers/python/hdbscan_public_lifecycle_run.mojo; PACK is
bindings/hdbscan_fit_boundary_pack.mojo."""
from std.python import Python, PythonObject
from std.memory import unsafe_memcpy
from core.hdbscan_fit_api import HdbscanFitOptions
from core.mode_answer import MODE_FAMILY_CLUSTER
from bindings.mode_answer_door import fast_answers_compiled_code
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from bindings.family_common_unpack import (
    family_repr, family_is_str, family_is_int, family_isinstance, family_type_name,
)
from bindings.family_common_pack import family_call
from bindings.py_balanced import py_get, py_contains, py_fields, py_item
from bindings.cluster_keys_unpack import cluster_keys
from providers.python.estimator_binding_pyapi_door import _estimator_mode_word
from providers.python.hdbscan_public_lifecycle_run import hdbscan_epsilon_refused
from core.hdbscan_fit_api import HDBSCAN_FIT_ABI, HDBSCAN_FIT_COUNT, HDBSCAN_FIT_HOST, HdbscanFitDriver
from core.kmeans_variants_api import VariantsSpan
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from bindings.family_common_unpack import FAMILY_I32P
from bindings.family_common_pack import family_array
from bindings.classical_arrays_door import classical_address


@fieldwise_init
struct HdbscanFitConnection(Movable):
    var provider: PythonObject
    var driver: HdbscanFitDriver
    var host: Bool


def hdbscan_fit_connection(binding: PythonObject) raises -> HdbscanFitConnection:
    var factory = neural_descriptor_entry(binding, "hdbscan_fit_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(HDBSCAN_FIT_ABI), HDBSCAN_FIT_COUNT)
    var driver = HdbscanFitDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    var host = driver.metadata(HDBSCAN_FIT_HOST)
    if host != 0 and host != 1:
        raise Error("HDBSCAN native provider: invalid host snapshot capability")
    return HdbscanFitConnection(provider, driver^, host != 0)


def hdbscan_buffer_span(value: PythonObject, count: Int) raises -> VariantsSpan:
    # Keep the old unconditional address admission, including zero logical
    # counts backed by the public max(count, 1) exporters. Do not introduce
    # a new size/property read or turn their non-null address into zero.
    return VariantsSpan(Int(py=classical_address(value)), count)


def _truthy(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(native_loaded_module("builtins").bool, value))


def _int(value: PythonObject) raises -> PythonObject:
    """`int(value)`."""
    return family_call(native_loaded_module("builtins").int, value)


def _float(value: PythonObject) raises -> PythonObject:
    """`float(value)`."""
    return family_call(native_loaded_module("builtins").float, value)


def _hdbscan_int_param(model: PythonObject, name: String) raises -> PythonObject:
    var value = family_raw_get(model, name)
    if not family_is_int(value):
        raise Error("TypeError: mojolearn HDBSCAN: " + name + " must be an int, got " + family_type_name(value))
    return value


def _hdbscan_mode(model: PythonObject, binding: PythonObject, context: PythonObject) raises:
    """`HDBSCAN._extension`: the binary's own compile-time mode must be the
    one requested."""
    var want = family_raw_get_or(model, "numeric_mode", PythonObject(None))
    var nk = cluster_keys()
    if want is None or not _truthy(want):
        want = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var getter = neural_descriptor_entry(binding, "hdbscan_numeric_mode", False)
    if getter is not None:
        # Perf (lane perf3-cluster-gp-gate): the compiled mode word is a
        # constant of the loaded binary, read once per (module, getter) for
        # the process, as GaussianMixture's _check_mode reads it
        # (bindings/mixture_public_native_pack.mojo), not called on every fit and query.
        var got = _estimator_mode_word(binding, getter)
        var codes = py_item(context, nk[].k_family_cfa_hdbscan_mode_codes)
        var wanted = py_get(codes, want, PythonObject(None))
        # The word `fast` answered by the identical set (core/mode_answer.mojo,
        # family CLUSTER) matches the identical binary: tested on the
        # mismatch branch only, so a matching call pays nothing.
        if (wanted is None or Int(py=wanted) != got) and not fast_answers_compiled_code(
                want, codes, got, MODE_FAMILY_CLUSTER):
            raise Error("RuntimeError: mojolearn HDBSCAN: numeric_mode=" + family_repr(want) + " was requested but "
                        + String(py_item(py_fields(binding), nk[].k___name__)) + " reports compile-time "
                        + "mode code " + String(got) + "; rebuild it with bash bindings/build_hdbscan.sh")


struct HdbscanFitSettings(Movable):
    """The estimator's settings as the fit read them, before narrowing to
    machine words: Python values only (UNPACK state, never passed to RUN)."""
    var min_samples: PythonObject
    var min_cluster_size: PythonObject
    var max_cluster_size: PythonObject
    var method: PythonObject
    var metric: PythonObject
    var alpha: PythonObject
    var epsilon: PythonObject
    var allow_single: Bool
    var prediction_data: Bool

    def __init__(out self):
        self.min_samples = PythonObject(None)
        self.min_cluster_size = PythonObject(None)
        self.max_cluster_size = PythonObject(None)
        self.method = PythonObject(None)
        self.metric = PythonObject(None)
        self.alpha = PythonObject(None)
        self.epsilon = PythonObject(None)
        self.allow_single = False
        self.prediction_data = False


def hdbscan_fit_settings(model: PythonObject, context: PythonObject) raises -> HdbscanFitSettings:
    """Every setting's type refusal, in the public order and wording."""
    var b = native_loaded_module("builtins")
    var settings = HdbscanFitSettings()
    settings.min_cluster_size = _hdbscan_int_param(model, "min_cluster_size")
    settings.max_cluster_size = _hdbscan_int_param(model, "max_cluster_size")
    var ms_field = family_raw_get(model, "min_samples")
    if ms_field is None:
        settings.min_samples = _int(settings.min_cluster_size)
    elif not family_is_int(ms_field):
        raise Error("TypeError: mojolearn HDBSCAN: min_samples must be an int or None")
    else:
        settings.min_samples = _int(ms_field)
    var metric_field = family_raw_get(model, "metric")
    var nk = cluster_keys()
    if family_is_str(metric_field):
        var metrics = py_item(context, nk[].k_family_cfa_hdbscan_metrics)
        if not py_contains(metrics, metric_field):
            raise Error("mojolearn HDBSCAN: metric must be 'euclidean' (cuML's L2SqrtExpanded, the one "
                        + "fit_hdbscan accepts), got " + family_repr(metric_field))
        settings.metric = py_item(metrics, metric_field)
    else:
        settings.metric = _int(metric_field)
    var method_field = family_raw_get(model, "cluster_selection_method")
    if family_is_str(method_field):
        var selections = py_item(context, nk[].k_family_cfa_hdbscan_selection)
        if not py_contains(selections, method_field):
            raise Error("mojolearn HDBSCAN: cluster_selection_method must be 'eom' or 'leaf', got "
                        + family_repr(method_field))
        settings.method = py_item(selections, method_field)
    else:
        settings.method = _int(method_field)
    settings.alpha = family_raw_get(model, "alpha")
    if not family_isinstance(settings.alpha, Python.tuple(b.int, b.float)) or family_isinstance(settings.alpha, b.bool):
        raise Error("TypeError: mojolearn HDBSCAN: alpha must be a real number")
    settings.epsilon = family_raw_get(model, "cluster_selection_epsilon")
    if not family_isinstance(settings.epsilon, Python.tuple(b.int, b.float)) or family_isinstance(settings.epsilon, b.bool):
        raise Error("TypeError: mojolearn HDBSCAN: cluster_selection_epsilon must be a real number")
    if hdbscan_epsilon_refused(Float64(py=_float(settings.epsilon))):
        raise Error("mojolearn HDBSCAN: cluster_selection_epsilon must be a finite number >= 0, got "
                    + family_repr(settings.epsilon) + " (scikit-learn's range)")
    var want_pd_field = family_raw_get(model, "prediction_data")
    if not family_isinstance(want_pd_field, b.bool):
        raise Error("TypeError: mojolearn HDBSCAN: prediction_data must be a bool")
    settings.prediction_data = Bool(py=want_pd_field)
    return settings^


def hdbscan_fit_stage(mut settings: HdbscanFitSettings, model: PythonObject) raises:
    """Every scalar conversion before any machine-word narrowing, in the
    public order (a large int is refused before a later invalid parameter)."""
    settings.min_cluster_size = _int(settings.min_cluster_size)
    settings.max_cluster_size = _int(settings.max_cluster_size)
    settings.alpha = _float(settings.alpha)
    settings.allow_single = _truthy(family_raw_get(model, "allow_single_cluster"))
    settings.epsilon = _float(settings.epsilon)


def hdbscan_fit_options(settings: HdbscanFitSettings, rows: Int, features: Int) raises -> HdbscanFitOptions:
    """The typed options of one fit: the last Python read of the settings."""
    return HdbscanFitOptions(rows, features, Int(py=settings.min_samples), Int(py=settings.min_cluster_size),
        Int(py=settings.max_cluster_size), settings.allow_single, Int(py=settings.method), Int(py=settings.metric),
        settings.prediction_data, Float64(py=settings.alpha), Float64(py=settings.epsilon))
