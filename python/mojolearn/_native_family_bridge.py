# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled owner of the classical estimator
families' public methods (bindings/family_native_door.mojo): k-means,
neighbors, kernels, model selection and the rest of lane
migrate-classical-b. A public method of a connected class is one call,
`_NATIVE(self, "family_<op>", args, _NATIVE_CONTEXT)`; the native owner
validates, prepares, runs the family's compiled kernels and stores or
returns the result. Python here only retains the compiled entry and the
shared admission context, once, at import; each module adds its fixed
lookup tables to the context at its own import."""

_ENTRY = None
_CONTEXT = None


def _connect_context_module():
    # glue: archive admission retains the initialized shared family owner,
    # including the explicit unavailable-owner state on partial installations.
    import sys
    from ._native_loaded_modules import connect_module
    connect_module(__name__, sys.modules[__name__], ('_CONTEXT',))


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; a connected method then explains what is missing.
    message = f"mojolearn: this estimator needs the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def shared_context():
    """Return the family owners' shared admission and metadata context."""
    global _CONTEXT
    if _CONTEXT is None:
        from types import BuiltinFunctionType
        from ._native_classical_bridge import _state_context
        from ._labels import ClassLabels
        context = _state_context()
        # The classifier label class every family door's label admission reads
        # (bindings/label_admission_native_unpack.mojo, logistic_admission_unpack.mojo,
        # prep_lda_admission_door.mojo, trees_public_state_unpack.mojo). It was
        # written by the SVM and sequence-forecast imports, so a lazily loaded
        # neighbors or semi-supervised family read no key (lane lazy-import-fix).
        context.update(family_builtin_type=BuiltinFunctionType,
                       classifier_labels_class=ClassLabels,
                       classifier_labels_namespace=dict(vars(ClassLabels)),
                       classifier_labels_mro=type.__dict__["__mro__"].__get__(ClassLabels))
        # The CPU metrics connection every family's device route shares
        # (providers/python/metrics_route_boundary_door.mojo
        # `device_route_metrics_cpu`): on a GPU installation, the R^2 /
        # accuracy programs of a device="cpu" estimator's score.
        context["metrics_cpu_cells"] = _cpu_metrics_cells()
        _CONTEXT = context
        _connect_context_module()
    return _CONTEXT


def _cpu_metrics_cells():
    # glue: import-time connection, once, of the CPU metrics program table
    # beside the device one; the refusal text when it cannot be connected. A
    # CPU-only process has no second table (its own is the CPU one).
    from . import _backend
    from ._native_classical_bridge import _native_table_cells, _CPU_DEVICE_METRICS_HOST
    if _backend._CPU_ONLY is not None:
        return "ImportError: this installation is CPU only; its own metrics table is the CPU one"
    try:
        return _native_table_cells(_backend.load_host_module(_CPU_DEVICE_METRICS_HOST), "x_metrics")  # import-time: the CPU metrics image beside the device one
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError, LookupError) as error:
        return type(error).__name__ + ": " + str(error)


def connect():
    """`(entry, context)`: the compiled family owner and the context every
    connected class shares. Connected once; later calls return the same
    pair."""
    global _ENTRY, _CONTEXT
    if _ENTRY is not None:
        return _ENTRY, shared_context()
    from types import BuiltinFunctionType
    try:
        from .tokenizer import _binding
        entry = _binding()._module.classical_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("the family owner requires the compiled classical_native_call entry")
        context = shared_context()
    except ImportError as error:
        _ENTRY = _unavailable(error)
        if _CONTEXT is None:
            _CONTEXT = {"family_unavailable": str(error)}
            _connect_context_module()
        return _ENTRY, _CONTEXT
    _ENTRY, _CONTEXT = entry, context
    return _ENTRY, _CONTEXT


def tables(**fixed):
    """Add one module's fixed lookup tables (its `family_*` names) to the
    shared context, once, at that module's import."""
    context = shared_context()
    context.update(fixed)


#: Binding names the host surface routes no CPU module for, whose CPU
#: identical image is loaded by name beside the device one (the metrics
#: programs a device="cpu" estimator scores on beside a GPU set). One place:
#: no family bridge loads such an image itself.
_CPU_DEVICE_IMAGES = {"_mojolearn_x_metrics": "_mojolearn_x_metrics_host"}


def _connect_cpu_binding(_backend, name):
    # glue: import-time connection of one binding name's CPU identical module
    # on a GPU installation: `(module, module name, mode reader name)`, or a
    # one-item tuple holding the refusal a device="cpu" estimator of the
    # family raises. A CPU-only process connects nothing here: its own set is
    # already the CPU one (`family_binding_connections`).
    if _backend._CPU_ONLY is not None:
        return ('ImportError: this installation is CPU only; its own bindings are the CPU bindings',)
    basename = _backend._HOST_MODULES.get(name) or _CPU_DEVICE_IMAGES.get(name)
    if basename is None:
        return ("ValueError: device='cpu' is not available: " + name + ' has no CPU implementation. '
                "Leave device=None to run on this installation's device",)
    try:
        module = _backend.load_host_module(basename)  # import-time: the CPU module beside the device module, never a GPU fallback
        return (module, module.__name__, _backend._host_prefix(basename) + '_numeric_mode')
    except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
        return (type(error).__name__ + ': ' + str(error),)


def connect_bindings(*names):
    """Connect estimator bindings during module import, never during execution."""
    from . import _backend
    context = shared_context()
    connections = context.setdefault('family_binding_connections', {})
    offerings = context.setdefault('family_binding_offerings', {})
    readers = context.setdefault('family_binding_readers', {})
    # The CPU identical module of each binding name, beside the device one: what
    # an estimator constructed with device="cpu" resolves to on a GPU
    # installation (providers/python/estimator_binding_pyapi_door.mojo
    # `_estimator_binding_cpu`). Connected here, at import, because a public
    # operation never calls back into Python to load a module.
    cpu_connections = context.setdefault('family_binding_connections_cpu', {})
    context['family_mode_set_type'] = _backend._ModeSet
    context['family_mode_set_fields'] = vars(_backend._ModeSet)['__dict__'].__get__
    for name in names:  # glue: fixed module-import binding names, never user data or product execution
        modes = {}
        readers[name] = _backend._vendor_fn(name).removesuffix('_vendor') + '_numeric_mode'
        offerings[name] = {}
        for mode in ('identical', 'fast'):  # glue: fixed module-import binding modes, never product execution
            offerings[name][mode] = _backend._offers(name, mode)
            try:
                binding = _backend.binding(name, mode)
                reader = _backend._vendor_fn(name).removesuffix('_vendor') + '_numeric_mode'
                if type(binding) is _backend._HostBinding:
                    if _backend._CPU_ONLY is None:
                        raise TypeError("host binding facade cannot serve a GPU operation")
                    basename = vars(binding)['_HostBinding__basename']
                    binding = _backend.load_host_module(basename)  # import-time: import connection of an explicitly selected CPU-only binding
                    reader = _backend._host_prefix(basename) + '_numeric_mode'
                elif type(binding) is _backend._NoGpuBinding:
                    raise ImportError(_backend._no_cpu_implementation(
                        name, 'native provider', vars(binding)['_NoGpuBinding__reason']))
                modes[mode] = (binding, binding.__name__, reader)
            except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
                modes[mode] = (type(error).__name__ + ': ' + str(error),)
        connections[name] = modes
        if name not in cpu_connections:
            cpu_connections[name] = _connect_cpu_binding(_backend, name)
