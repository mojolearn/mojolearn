# SPDX-License-Identifier: Apache-2.0
"""Import-only provider connection for the process's native backend metadata."""

from .tokenizer import _binding

# The package imports this module at import (it needs only the core host
# unit), so the compiled entry is bound here once; connect() supplies the
# context and installs the public names (mojolearn.gpu_arch and friends) in
# __init__.py, and refresh() re-reads the plans when a unit loads later.
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


def _native_entry(module, name):
    # glue: resolve optional compiled metadata entries at connection only
    from types import BuiltinFunctionType
    try:
        fn = getattr(module, name, None) if module is not None else None
    except (ImportError, AttributeError):
        return None
    if fn is not None and type(fn) is not BuiltinFunctionType:
        raise TypeError("backend metadata readbacks must be compiled native entries")
    return fn


def _plans():
    # glue: the two existing provider tiers, read from their LOADED members
    # only (lane lazy-family-import, 2026-11-12): a unit loads when its family
    # is first used, and refresh() re-runs this when one does.
    from . import _backend
    plans = {}
    for mode in ("identical", "fast"):  # glue: connect the two existing provider tiers
        try:
            tier = _backend.load_set(mode)
        except Exception as error:
            plans[mode] = dict(error=error, mode=None, vendors=())
            continue
        members = tier._modules
        mode_entry = _native_entry(members.get("_mojolearn_gbdt"), "gbdt_numeric_mode")
        vendors = []
        for name in _backend._MODULES:  # glue: fixed compiled provider metadata, preserving selector order
            if name in tier.missing:
                continue
            module = members.get(name)
            # A deferred member is not loaded merely to inspect it: the
            # original reader stops at the first compiled vendor, so only
            # when no member of the tier is loaded yet is the first one
            # resolved (dual-tier type registration is not supported by Mojo).
            if module is None and not vendors:
                try:
                    module = getattr(tier, name)
                except (ImportError, AttributeError):
                    continue
            if module is None:
                continue
            fn = _native_entry(module, _backend._vendor_fn(name))
            if fn is not None:
                vendors.append(fn)
        plans[mode] = dict(error=None, mode=mode_entry, vendors=tuple(vendors))
    return plans


def refresh():
    """Re-read the loaded units' metadata entries after a unit loads."""
    from . import _backend
    if _CONTEXT is None:
        return
    _CONTEXT.update(plans=_plans(),
                    selected_mode=_native_entry(_backend.canonical_loaded("_mojolearn_gbdt"), "gbdt_numeric_mode"))


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
    _CONTEXT = dict(backend_fields=vars(_backend), plans=_plans(),
                    selected_mode=_native_entry(_backend.canonical_loaded("_mojolearn_gbdt"), "gbdt_numeric_mode"),
                    mode_codes=dict(_backend._CODE_MODE),
                    native_function_type=BuiltinFunctionType,
                    readback_skip_errors=(ImportError, AttributeError))
    for name in ("gpu_arch", "gpu_arch_how", "numeric_mode", "vendor", "set_default_mode"):  # glue: install one-call public metadata wrappers after provider setup
        fn = globals()[name]
        fn.__doc__ = getattr(_backend, name).__doc__
        setattr(_backend, name, fn)
        setattr(package, "set_numeric_mode" if name == "set_default_mode" else name, fn)
