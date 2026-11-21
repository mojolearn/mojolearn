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
# Set by the first `device="cpu"` estimator beside a GPU set
# (`connect_cpu`); the host images load then, never at import.
_CPU_WANTED = False
_CPU_DONE = False
# Other family contexts whose CPU label scope waits for the same moment
# (`cpu_scope_later`): the decomposition kit's (EllipticEnvelope).
_CPU_LATER = []


def cpu_scope_later(context, load=None):
    """Build `context`'s CPU label scope (_native_labels_bridge `cpu_scope`)
    at the first device="cpu" admission beside a GPU set, or now when that
    already happened: a family context connected at import that should not
    load the host label images at import. `load(context)`, when given,
    connects the family's own CPU image into `context` first (the
    decomposition kit's x_decomp host image), at the same moment."""
    # glue: import-time registration, once per family context
    if _CPU_DONE:
        if load is not None:
            load(context)
        from ._native_labels_bridge import cpu_scope
        cpu_scope(context)
    else:
        _hold_cpu_connect(context)
        _CPU_LATER.append((context, load))


def cpu_load_later(context, load):
    """Run `load()` at the first device="cpu" admission beside a GPU set, or
    now when that already happened: a family module's host-image load that
    must not run at import (lane fx-audit item 9: `import mojolearn` on a GPU
    box loads no host image until then). `context` is the family context
    the loaded tables go into; it gets `device_cpu_connect` so an estimator
    restored without construction connects the same way at its first CPU
    route (`device_cpu_connect_late`). A CPU-only installation loads
    nothing: its own images already are the CPU ones."""
    # glue: import-time registration, once per family context
    if _CPU_DONE:
        load()
    else:
        _hold_cpu_connect(context)
        _CPU_LATER.append((None, load))


def _hold_cpu_connect(context):
    """Give a lazily connected family context `device_cpu_connect`
    (`connect_cpu`): a device="cpu" estimator restored without construction
    (unpickle, copy, archive load) admits the CPU device at its first label
    step or score route through it, once, as construction does
    (providers/python/device_route_boundary_door.mojo
    `device_cpu_connect_late`). A CPU-only installation needs none."""
    # glue: import-time registration, once per family context
    from . import _backend
    if _backend._CPU_ONLY is None:
        context["device_cpu_connect"] = connect_cpu


def _connect_cpu(context):
    # glue: once per process, at the first construction or set_params of a
    # device="cpu" estimator beside a GPU set (never at import, never per
    # call). The CPU metrics connection every family's device route shares
    # (providers/python/metrics_route_boundary_door.mojo
    # `device_route_metrics_cpu`): the R^2 / accuracy programs of a
    # device="cpu" estimator's score. The label steps of a device="cpu"
    # classifier of any shared-context family (SVC, neighbors,
    # semi-supervised, GP) run the host label kernels
    # (bindings/labels_descriptor_boundary_door.mojo `labels_scope`). Later
    # `tables()` imports grow this context; the scope is copied again at the
    # first label step after that.
    context["metrics_cpu_cells"] = _cpu_metrics_cells()
    from ._native_labels_bridge import cpu_scope
    cpu_scope(context)


def connect_cpu():
    """Connect the host label and metrics images to the shared family
    context: called by the compiled device check
    (providers/python/device_route_boundary_door.mojo `device_admit_class`)
    when an estimator is given `device="cpu"` beside a GPU set. Once per
    process; later calls return at the first flag."""
    # glue: lazy connection of the CPU route's images, once
    global _CPU_WANTED, _CPU_DONE
    if _CPU_DONE:
        return None
    _CPU_WANTED = True
    if _CONTEXT is not None:
        _connect_cpu(_CONTEXT)
    _CPU_DONE = True
    from ._native_labels_bridge import cpu_scope
    for context, load in _CPU_LATER:  # glue: once, over the registered family contexts (at most a few)
        if context is None:
            # `cpu_load_later`: a host-image load, no label scope.
            load()
            continue
        if load is not None:
            load(context)
        cpu_scope(context)
    _CPU_LATER.clear()
    return None


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
        _CONTEXT = context
        if _CPU_WANTED:
            _connect_cpu(context)
        else:
            _hold_cpu_connect(context)
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


_CPU_ROUTES = None


def _cpu_routes():
    # glue: the manifest's route table (binding name -> its CPU identical
    # module), built once per process. `_backend._HOST_MODULES` is the same
    # table, read only by the CPU-only selector
    # (tools/test_lane_select_narrowing.py).
    global _CPU_ROUTES
    if _CPU_ROUTES is None:
        from . import host_surface
        _CPU_ROUTES = host_surface.routed_modules()
    return _CPU_ROUTES


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
    basename = _cpu_routes().get(name) or _CPU_DEVICE_IMAGES.get(name)
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
