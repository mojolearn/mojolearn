# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of LogisticRegression's native owner
(bindings/logistic_native_door.mojo) and its fixed ABI."""

_ESTIMATOR_ENTRIES = ("logistic_native_api_v1",)
_X_TREES_ENTRIES = ("label_weights_native_api_v1",)


# One refusal spelling for the import-time bridges (lane py-dedupe, 2026-11-15).
from ._native_labels_bridge import _refusal  # noqa: E402


def connect():
    # glue: compiled entries and plain metadata are retained during import
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context
    from ._native_labels_bridge import label_context
    from .tokenizer import _binding

    context = _state_context()
    context.update(label_context())
    # glue: reuse native warning delivery for NumPy scalar configuration
    # arithmetic; this captures metadata only and does not call its hooks.
    from ._native_classical_warnings import connect_warnings
    warning = connect_warnings()
    warning.update(warning_filename=__file__, warning_lineno=1,
                   warning_module=__name__, warning_source_line="")
    context.update(sequence_array_warnings=warning,
                   module_type=context["native_module_type"], lr_complex_type=complex)
    estimators = {}
    x_trees = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding("_mojolearn_estimators", mode)
            entries = {}
            for name in _ESTIMATOR_ENTRIES:  # glue: fixed native exports
                entry = getattr(module, name, None)
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"LogisticRegression requires compiled {name}")
                entries[name] = entry
            estimators[mode] = entries
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            estimators[mode] = _refusal(error)
        try:
            module = _backend.binding("_mojolearn_x_trees", mode)
            table = {}
            for name in _X_TREES_ENTRIES:  # glue: fixed native exports
                entry = getattr(module, name, None)
                if isinstance(entry, BuiltinFunctionType):
                    table[name] = entry
            x_trees[mode] = table
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            x_trees[mode] = _refusal(error)
    call = _binding()._module.logistic_native_call
    if not isinstance(call, BuiltinFunctionType):
        raise TypeError("LogisticRegression requires a compiled logistic_native_call")
    context.update(lr_bindings=estimators, lr_x_trees=x_trees, lr_host_contexts={},
                   logistic_fold_native_api_v1=_fold_entry('logistic_fold_native_api_v1'))
    return call, context


def connect_host(cls, context, estimators_host, host_context):
    # glue: called only by _classical_host at its explicit CPU API setup. The
    # explicit host class fits and scores on the CPU estimators binding
    # (identical only, `_HostBound._bind`); its predict is the classical host
    # inference owner, which its score calls too.
    from types import BuiltinFunctionType
    host = dict(context)
    entries = {}
    try:
        if isinstance(estimators_host, Exception):
            raise estimators_host
        # The descriptor owns all numerical dispatch and native capabilities.
        for name in _ESTIMATOR_ENTRIES:  # glue: fixed native exports
            entry = getattr(estimators_host, name, None)
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError(f"{cls.__name__} requires compiled {name}")
            entries[name] = entry
    except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
        entries = _refusal(error)
    refusal = (f"ImportError: mojolearn: the host {cls.__name__} serves "
               "_mojolearn_estimators only, not _mojolearn_x_trees")
    host.update(lr_bindings={"identical": entries}, lr_x_trees={"identical": refusal},
                lr_host_model=True, lr_host_score_context=host_context)
    context.setdefault("lr_host_contexts", {})[cls] = host


#: The constructor's parameters, in its order. `device` is last: it is stored
#: as given, checked when it is given and read once into the estimator's
#: native route owner (providers/python/logistic_route_boundary_door.mojo).
_PARAMETERS = ('penalty', 'tol', 'C', 'fit_intercept', 'class_weight', 'max_iter',
               'linesearch_max_iter', 'l1_ratio', 'solver', 'numeric_mode', 'device')
#: The parameters every operation admits again before it reads them. `device`
#: is not one: no operation reads it after the route is resolved.
_CHECKED_PARAMETERS = _PARAMETERS[:-1]

#: The CPU binaries `device="cpu"` runs on a GPU installation: the estimators
#: host (the quasi-Newton entry) and the trees expansion host (the class-weight
#: programs of a weighted fit).
_CPU_DEVICE_HOSTS = ("_mojolearn_estimators_host", "_mojolearn_x_trees_host")


def connect_cpu_device(context):
    """Register the CPU entries of `device="cpu"` beside the device entries:
    they load at the first device="cpu" admission beside a GPU set
    (fx-host-import, `_native_family_bridge.cpu_load_later`), never at
    import; an estimator restored without construction connects them at its
    first CPU route (providers/python/logistic_route_boundary_door.mojo).
    Nothing is registered in a CPU-only process."""
    # glue: import-time registration, once
    if context["vendor"] == "cpu":
        return
    from ._native_family_bridge import cpu_load_later
    cpu_load_later(context, lambda: _load_cpu_device(context))


def _load_cpu_device(context):
    """Connect the CPU entries of `device="cpu"` once, beside the device
    entries (`context["lr_bindings"]`).

    The compiled owner resolves an estimator's route against the device
    entries or these, once per estimator. In a CPU-only process the selected
    entries already are the CPU entries and nothing is loaded. A host binary
    that is not built is kept as its refusal text, said when a `device="cpu"`
    estimator first runs (the class-weight binary: when a `device="cpu"` fit
    first takes class weights)."""
    # glue: load the CPU images once and retain their compiled entries
    from types import BuiltinFunctionType
    from . import _backend
    if "lr_cpu_bindings" in context:
        return
    errors = (ImportError, AttributeError, TypeError, ValueError, RuntimeError)
    estimators, x_trees = _CPU_DEVICE_HOSTS
    try:
        module = _backend.load_host_module(estimators)
        entries = {}
        for name in _ESTIMATOR_ENTRIES:  # glue: fixed native exports
            entry = getattr(module, name, None)
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError(f"LogisticRegression(device='cpu') requires compiled {name}")
            entries[name] = entry
    except errors as error:  # recorded, raised at the first device="cpu" operation
        entries = _refusal(error)
    try:
        module = _backend.load_host_module(x_trees)
        table = {}
        for name in _X_TREES_ENTRIES:  # glue: fixed native exports
            entry = getattr(module, name, None)
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError(f"LogisticRegression(device='cpu') requires compiled {name}")
            table[name] = entry
    except errors as error:  # recorded, raised by the first device="cpu" fit with class weights
        table = _refusal(error)
    context.update(lr_cpu_bindings={"identical": entries}, lr_cpu_x_trees={"identical": table})
    # The label steps of a device="cpu" classifier run the host label kernels
    # (bindings/labels_descriptor_boundary_door.mojo `labels_scope`): its CPU
    # label scope, built once here, last.
    from ._native_labels_bridge import cpu_scope
    cpu_scope(context)


def connect_public(cls, context, install=True):
    # glue: install native attribute binding and constructor metadata once
    import sys
    from types import BuiltinFunctionType, MethodType
    from .tokenizer import _binding
    from ._native_classical_bridge import _layout
    from ._labels import ClassLabels
    owner = context.get('lr_host_contexts', {}).get(cls, context)
    module = _binding()._module
    attribute = module.standard_scaler_attribute_call
    descriptor_entry = module.classical_attribute_descriptor
    if type(attribute) is not BuiltinFunctionType or type(descriptor_entry) is not BuiltinFunctionType:
        raise TypeError('LogisticRegression requires compiled native attribute entries')
    if install:
        descriptor = descriptor_entry(MethodType(attribute, owner))
        cls.__getattribute__ = descriptor
    else:
        descriptor = vars(cls)['__getattribute__']
    owner.update(classical_getframe=sys._getframe, classical_public_installed=True,
                 classical_attribute_descriptor=descriptor,
                 classical_public_layouts={cls: _layout(cls, descriptor)},
                 classifier_labels_class=ClassLabels, classifier_labels_namespace=dict(vars(ClassLabels)),
                             classifier_labels_mro=type.__dict__["__mro__"].__get__(ClassLabels),
                 lr_parameters=_CHECKED_PARAMETERS, lr_constructor_parameters=_PARAMETERS,
                 lr_sorted_parameters=tuple(sorted(_PARAMETERS)),  # glue: fixed import-time parameter metadata
                 lr_consumed_fields=('_w', 'classes_', 'n_features_in_', '_l1_ratio', 'lbfgs_memory',
                                     'penalty_normalized', '_mojolearn_fitted', '_mojolearn_fitting'))
    return owner


def _fold_entry(name):
    # glue (lanes a7c, a8c): the QN fold entry (`logistic_fold_native_api_v1`:
    # fit and inference of a cross-validation fold resident on the device),
    # exported by the identical GPU base binding next to the fold store it
    # read (bindings/msel_device_unpack.mojo); a build without them (the CPU
    # column) leaves None and the door gathers host rows.
    from types import BuiltinFunctionType
    from ._buffer import _native
    try:
        fold = _native(name)
    except (ImportError, AttributeError, TypeError, RuntimeError):
        fold = None
    return fold if isinstance(fold, BuiltinFunctionType) else None
