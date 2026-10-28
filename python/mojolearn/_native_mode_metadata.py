# SPDX-License-Identifier: Apache-2.0
"""Import-time connections for shared native fitted-state and binary metadata."""
from types import BuiltinFunctionType, FunctionType, ModuleType


def connect():
    from . import _backend
    from .tokenizer import _binding
    _backend.default_mode()
    entry = _binding()._module.mode_metadata_native_call
    if type(entry) is not BuiltinFunctionType:
        raise TypeError('mode metadata requires a compiled native entry')
    return entry, dict(backend_fields=vars(_backend), mode_codes=dict(_backend._CODE_MODE),
                       plans={})


def connect_generic_attributes():
    # glue: the generic attribute slot's entry; its admission context is the
    # public attributes context, linked by _native_public_attributes.connect_classes.
    from .tokenizer import _binding
    entry = _binding()._module.public_attribute_generic_native_call
    if type(entry) is not BuiltinFunctionType:
        raise TypeError('generic public attributes require a compiled native entry')
    return entry, dict(attributes=None)


def connect_generic_params():
    # glue: the generic parameter setter shares the full-reset owner's entry
    # and context; _native_full_reset_params fills the context at package import.
    from .tokenizer import _binding
    entry = _binding()._module.full_reset_params_native_call
    if type(entry) is not BuiltinFunctionType:
        raise TypeError('generic parameter setters require a compiled native entry')
    return entry, dict(plans={}, missing=object(), ready=False, originals={}, generic_setter=None)


def connect_classes(context, package):
    # glue: called once after package classes and their native descriptors exist
    import sys
    from . import _backend, _mode, _classical_host
    from . import _forest_protocol, _hierarchy_impl, _tsa_impl
    known = {
        _mode.NumericModeMixin._bind: 'standard',
        _forest_protocol.ForestProtocol._bind: 'forest',
        _tsa_impl.ExponentialSmoothing._bind: 'standard',
        _classical_host._HostBound._bind: 'host',
        _classical_host.HostGaussianProcessClassifier._bind: 'host_gp',
    }
    classes = set()
    for name, module in tuple(sys.modules.items()):  # glue: loaded package class metadata
        if name == package.__name__ or name.startswith(package.__name__ + '.'):
            if type(module) is ModuleType:
                classes.update(value for value in vars(module).values()  # glue: exact ordinary class identities
                               if type(value) is type and issubclass(value, _mode.NumericModeMixin))
    connected = {}
    for cls in classes:  # glue: fixed built-in readback routes, never a runtime resolver
        mro = type.__dict__['__mro__'].__get__(cls)
        namespaces = tuple(type.__dict__['__dict__'].__get__(base) for base in mro)  # glue: native class dictionary metadata
        bind = next((row['_bind'] for row in namespaces if '_bind' in row), None)  # glue: import-time inherited binding method identity
        basename = next((row['_BINDING'] for row in namespaces if '_BINDING' in row), None)  # glue: import-time inherited binary name
        if type(bind) is not FunctionType or bind not in known or type(basename) is not str:
            # A class without the protocol does not receive a guessed route.
            continue
        route = known[bind]
        actual = _classical_host._HOST_BASENAMES.get(basename, basename) if route == 'host' else basename
        if route == 'host_gp':
            import os
            actual = (_classical_host._GPC_HOST_INFERENCE_BASENAME
                      if os.path.exists(_backend.host_module_path(_classical_host._GPC_HOST_INFERENCE_BASENAME))  # cpu-route: import-time explicit HostGaussianProcessClassifier metadata, not a GPU fallback
                      else _classical_host._GPC_HOST_BASENAME)
        modes = {}
        for mode in ('identical', 'fast'):  # glue: connect existing supported mode binaries, no new fast campaign
            key = (route in ('host', 'host_gp'), actual, mode)
            if key not in connected:
                try:
                    module = (_backend.load_host_module(actual) if key[0] else _backend.binding(actual, mode))  # cpu-route: import-time explicit Host* readback; GPU classes retain their GPU binding
                    leaf = module.__name__.rsplit('.', 1)[-1]
                    vendor_name = _backend._vendor_fn(leaf)
                    mode_name = vendor_name.removesuffix('_vendor') + '_numeric_mode'
                    mode_entry = getattr(module, mode_name, None)
                    if mode_entry is None and leaf == '_mojolearn_metrics':
                        mode_entry = getattr(module, 'umap_numeric_mode', None)
                    try:
                        vendor_entry = getattr(module, vendor_name, None)
                    except (ImportError, AttributeError):
                        vendor_entry = None
                    for entry in (mode_entry, vendor_entry):  # glue: exact compiled read-back ABI identities
                        if entry is not None and type(entry) is not BuiltinFunctionType:
                            raise TypeError('mode metadata cannot call a Python binary read-back function')
                    pieces = module.__name__.split('.')
                    connected[key] = dict(mode=mode_entry, vendor=vendor_entry,
                                          legacy_mode=pieces[-2] if len(pieces) > 1 else None, error=None)
                except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                    # Retain the actual import-time refusal; native execution
                    # restores it without constructing/calling its Python class.
                    connected[key] = dict(error=error)
            modes[mode] = connected[key]
        context['plans'][cls] = dict(binding=basename, bind=bind, route=route, modes=modes)
