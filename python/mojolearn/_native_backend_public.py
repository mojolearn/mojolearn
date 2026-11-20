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


def runtime_report():
    """What this process is running, as a plain dictionary.

    Returns a new ``dict`` on every call: the package ``version``; the
    ``vendor`` (``'cuda'``, ``'hip'``, ``'metal'`` or ``'cpu'``); the loaded
    ``numeric_mode`` and the process ``default_mode``; ``code_mode``, the mode
    the loaded gbdt binary says it was compiled as; ``code_kind``,
    ``'ptx'`` when the PTX GPU set is loaded, else ``'native'``;
    ``cpu_only``; ``gpu_arch`` and ``gpu_arch_how``; ``host_dir``, the
    ``host_families_built`` there and ``host_paths`` (each built CPU binding
    and the file it loads from); ``bindings``, every native binding module
    of the package keyed by module name, each with its ``path``,
    ``expected_path`` (for a CPU binding, the file it should load from),
    ``loaded``, ``identity`` (the SHA-256 of the file), ``vendor``, ``mode``
    and, for a loaded CPU binding, its ``sabotage``, ``column`` and
    ``detected_column`` read-backs;
    ``fallbacks``; and the ``harness_source``, ``source_commit``,
    ``checkout_root`` and ``package_dir`` the verification commands use.

    The report only reads: it loads no binding and changes nothing. A binding
    appears once something has loaded it or its tier.

    Example::

        >>> import mojolearn
        >>> report = mojolearn.runtime_report()
        >>> report["vendor"] in ("cuda", "hip", "metal", "cpu")
        True
    """
    return _NATIVE("runtime_report", (), _CONTEXT)


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


def _report_context(package):
    # glue: import-only connection of what runtime_report() reads: the live
    # module table, C posix entries, the merged-index cache, the frozen
    # fallback names, the package directory and its parents. Nothing here
    # runs per call; runtime_report() itself is one native call.
    import posix
    import sys
    import types
    from . import _merged, _runtime_fallbacks
    import os
    package_dir = os.path.dirname(os.path.abspath(package.__file__))
    parents, base = [], package_dir
    for _ in range(6):  # glue: the checkout-root candidates, walked as _identity._checkout_root walked them
        parent = base.rsplit("/", 1)[0] or "/"
        if parent == base or parent == "/":
            break
        parents.append(parent)
        base = parent
    return dict(version=package.__version__, package_dir=package_dir, package_parents=tuple(parents),
                fallbacks=tuple(_runtime_fallbacks.FALLBACKS), modules=sys.modules,
                module_type=types.ModuleType, list_type=list, listdir=posix.listdir, getcwd=posix.getcwd,
                merged_cache=_merged._INDEX_CACHE, merged_index_name=_merged.INDEX_NAME,
                identity_cache=dict(stat=posix.stat, digests={}))


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
    # Decide, once and here at import, which set answers the name `fast`
    # (`_backend._FAST_ANSWER`); set_numeric_mode() reads the stored answer.
    _backend.fast_answer()
    _CONTEXT = dict(backend_fields=vars(_backend), plans=_plans(),
                    selected_mode=_native_entry(_backend.canonical_loaded("_mojolearn_gbdt"), "gbdt_numeric_mode"),
                    mode_codes=dict(_backend._CODE_MODE),
                    native_function_type=BuiltinFunctionType,
                    readback_skip_errors=(ImportError, AttributeError),
                    **_report_context(package))
    # Once, at import: every door's mode admission reads this stored answer.
    _NATIVE("connect_mode_answer", (), _CONTEXT)
    package.runtime_report = runtime_report
    for name in ("gpu_arch", "gpu_arch_how", "numeric_mode", "vendor", "set_default_mode"):  # glue: install one-call public metadata wrappers after provider setup
        fn = globals()[name]
        fn.__doc__ = getattr(_backend, name).__doc__
        setattr(_backend, name, fn)
        setattr(package, "set_numeric_mode" if name == "set_default_mode" else name, fn)
