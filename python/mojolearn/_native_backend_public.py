# SPDX-License-Identifier: Apache-2.0
"""Import-only provider connection for the process's native backend metadata."""

from .tokenizer import _binding

# The package imports this module after every provider loads, so the compiled
# entry is bound here once; connect() supplies the context and installs the
# public names (mojolearn.gpu_arch and friends) in __init__.py.
_NATIVE = _binding()._module.backend_public_native_call
_CONTEXT = None


def gpu_arch():
    return _NATIVE("gpu_arch", (), _CONTEXT)


def gpu_arch_how():
    return _NATIVE("gpu_arch_how", (), _CONTEXT)


def numeric_mode():
    return _NATIVE("numeric_mode", (), _CONTEXT)


def set_default_mode(mode):
    return _NATIVE("set_default_mode", (mode,), _CONTEXT)


def vendor():
    return _NATIVE("vendor", (), _CONTEXT)


def connect(package):
    # glue: discovery/imports finish before public operations are installed;
    # native execution retains entries and never invokes the Python loader.
    global _CONTEXT
    from types import BuiltinFunctionType
    from . import _backend
    entry = _NATIVE
    if type(entry) is not BuiltinFunctionType:
        raise TypeError("backend metadata requires its compiled native owner")
    _backend.default_mode()
    if _backend._CPU_ONLY is None:
        _backend._layout()
    def native_entry(module, name):
        # glue: resolve optional compiled metadata entries at import only
        try:
            fn = getattr(module, name, None) if module is not None else None
        except (ImportError, AttributeError):
            return None
        if fn is not None and type(fn) is not BuiltinFunctionType:
            raise TypeError("backend metadata readbacks must be compiled native entries")
        return fn
    plans = {}
    for mode in ("identical", "fast"):  # glue: connect the two existing provider tiers once
        try:
            tier = _backend.load_set(mode)
        except Exception as error:
            plans[mode] = dict(error=error, mode=None, vendors=())
            continue
        mode_entry = native_entry(tier._modules.get("_mojolearn_gbdt"), "gbdt_numeric_mode")
        vendors = []
        for name in _backend._MODULES:  # glue: fixed compiled provider metadata, preserving selector order
            if name in tier.missing:
                continue
            module = tier._modules.get(name)
            # A deferred Mojo type owner must not be initialized merely to
            # inspect a later vendor entry: the original reader stops at the
            # first compiled vendor, and dual-tier type registration is not
            # supported by Mojo. Resolve a deferred owner only if needed.
            if module is None and not vendors:
                try:
                    module = getattr(tier, name)
                except (ImportError, AttributeError):
                    continue
            if module is None:
                continue
            fn = native_entry(module, _backend._vendor_fn(name))
            if fn is not None:
                vendors.append(fn)
        plans[mode] = dict(error=None, mode=mode_entry, vendors=tuple(vendors))
    selected_mode = native_entry(vars(package).get("_mojolearn_gbdt"), "gbdt_numeric_mode")
    _CONTEXT = dict(backend_fields=vars(_backend), plans=plans,
                    selected_mode=selected_mode, mode_codes=dict(_backend._CODE_MODE),
                    native_function_type=BuiltinFunctionType,
                    readback_skip_errors=(ImportError, AttributeError))
    for name in ("gpu_arch", "gpu_arch_how", "numeric_mode", "vendor", "set_default_mode"):  # glue: install one-call public metadata wrappers after provider setup
        fn = globals()[name]
        fn.__doc__ = getattr(_backend, name).__doc__
        setattr(_backend, name, fn)
        setattr(package, "set_numeric_mode" if name == "set_default_mode" else name, fn)
