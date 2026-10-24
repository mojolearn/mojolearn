# SPDX-License-Identifier: Apache-2.0
"""Import-time module connections admit only CPython's own module __getattr__.

Python 3.13's `types` module defines a module-level `__getattr__`; refusing
every such hook made `import mojolearn` fail on 3.13. A standard-library hook
(defined by that module's own source inside this interpreter's stdlib) is
admitted and pinned like an export; a hook from anywhere else is refused.
"""
import os
import sys
import sysconfig
import types

import pytest

from mojolearn import _native_loaded_modules as loaded


def _stdlib_source(name):
    path = os.path.join(sysconfig.get_paths()["stdlib"], name + ".py")
    if not os.path.isfile(path):
        pytest.skip(f"{name}.py is not a source file in this interpreter's stdlib")
    return path


def _module(name, origin, hook_file):
    """A module named `name` whose spec origin is `origin` and whose
    `__getattr__` is compiled as if it lived in `hook_file`."""
    module = types.ModuleType(name)
    module.__spec__ = types.SimpleNamespace(origin=origin)
    namespace = {"__name__": name}
    exec(compile("def __getattr__(attribute):\n    raise AttributeError(attribute)\n",
                 hook_file, "exec"), namespace)
    module.__getattr__ = namespace["__getattr__"]
    module.value = 1
    return module


@pytest.fixture
def connections(monkeypatch):
    monkeypatch.setattr(loaded, "CONNECTIONS", loaded.CONNECTIONS)
    return loaded


def test_stdlib_style_module_hook_is_admitted_and_pinned(connections):
    name = "colorsys"  # a stdlib module the package never connects
    assert name in sys.stdlib_module_names
    source = _stdlib_source(name)
    module = _module(name, source, source)
    connections.connect_module("test-stdlib-hook", module, ("value",))
    record = connections.CONNECTIONS[-1]
    assert record[0] == "test-stdlib-hook" and record[1] is module
    assert dict(record[2]) == {"value": 1, "__getattr__": module.__getattr__}


@pytest.mark.parametrize("case", ["foreign-file", "non-stdlib-name", "outside-stdlib-origin", "not-a-function"])
def test_foreign_module_hook_is_refused(connections, case, tmp_path):
    name = "colorsys"
    source = _stdlib_source(name)
    if case == "foreign-file":
        module = _module(name, source, __file__)  # defined outside the module's source
    elif case == "non-stdlib-name":
        module = _module("mojolearn_test_user_module", source, source)
    elif case == "outside-stdlib-origin":
        shadow = str(tmp_path / (name + ".py"))
        module = _module(name, shadow, shadow)  # a user file shadowing a stdlib name
    else:
        module = _module(name, source, source)
        module.__getattr__ = lambda attribute: None  # a lambda is not the module's own hook
    before = connections.CONNECTIONS
    with pytest.raises(TypeError, match="cannot retain module attribute hooks"):
        connections.connect_module("test-foreign-hook-" + case, module, ("value",))
    assert connections.CONNECTIONS is before


def test_connected_stdlib_modules_import_on_this_interpreter():
    # `import mojolearn` connected every fixed module, including `types`
    # (which has a module __getattr__ on Python 3.13).
    names = {record[0] for record in loaded.CONNECTIONS}
    assert {"builtins", "sys", "types", "os", "warnings"} <= names
    for record in loaded.CONNECTIONS:
        fields = vars(record[1])
        if "__getattr__" in fields:
            assert dict(record[2])["__getattr__"] is fields["__getattr__"]
            assert loaded.stdlib_module_hook(record[0], fields)
