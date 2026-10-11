# SPDX-License-Identifier: Apache-2.0
"""The `_mojolearn_forest_host` binary's path and loader, alone (lane/lazy-imports,
2026-10-10).

Split out of `_forest_host.py` so the buffer helpers (`_buffer._host_native`,
which every lane loads) can resolve their seven native helpers from this
binary without importing the forest inference door, whose `host_model` names
`_classical_host` and through it every classical door. `_forest_host`
re-exports `binary_path` and `_load`; the rules (the
`MOJOLEARN_FOREST_HOST_BINARY` override, the IDENTICAL, cpu and sabotage
read-backs) are documented there.
"""
import importlib.machinery
import importlib.util
import os
import sys

from . import _backend

_EXTENSION = '_mojolearn_forest_host'
_MODULE_NAME = 'mojolearn._host.' + _EXTENSION
_IDENTICAL_CODE = 1
_MODULE = None


def binary_path():
    """The binary this process loads, or would load: the file
    `MOJOLEARN_FOREST_HOST_BINARY` names, else this install's host binding
    of that name, wherever `_backend.host_module_path` resolves it."""
    override = os.environ.get('MOJOLEARN_FOREST_HOST_BINARY', '').strip()
    if override:
        return os.path.abspath(override)
    return _backend.host_module_path(_EXTENSION)


def _load():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = binary_path()
    if not os.path.exists(path):
        raise ImportError(
            f"mojolearn: {path} is not built. Build it with "
            "bindings/build_forest_host.sh")
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        loader = importlib.machinery.ExtensionFileLoader(_MODULE_NAME, path)
        spec = importlib.util.spec_from_loader(_MODULE_NAME, loader, origin=path)
        module = importlib.util.module_from_spec(spec)
        _backend._exec_binding(loader, module)
        sys.modules[_MODULE_NAME] = module
    if int(module.forest_host_numeric_mode()) != _IDENTICAL_CODE:
        raise RuntimeError(f"mojolearn: {path} was not compiled IDENTICAL; rebuild it")
    if str(module.forest_host_vendor()) != 'cpu':
        raise RuntimeError(f"mojolearn: {path} does not read back as the CPU binding")
    # the CTR-path arm (lane/inference-gbdt-ctr-tables, 2026-09-15) and the
    # grove association arm (DEVIATION 2961, 2026-09-17) are sabotage builds
    # too and take the same switch
    ctr_sabotage = getattr(module, 'forest_host_gbdt_ctr_sabotage', None)
    groves_sabotage = getattr(module, 'forest_host_groves_sabotage', None)
    sabotaged = (bool(module.forest_host_sabotage())
                 or bool(ctr_sabotage is not None and ctr_sabotage())
                 or bool(groves_sabotage is not None and groves_sabotage()))
    if sabotaged and os.environ.get('MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE') != '1':
        raise RuntimeError(
            f"mojolearn: {path} is the gate's SABOTAGE build and computes "
            "wrong answers on purpose; it is refused outside the gate")
    _MODULE = module
    return module
