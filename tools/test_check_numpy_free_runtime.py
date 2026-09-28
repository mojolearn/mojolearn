"""Installed metadata qualification remains independent of third-party packages."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("numpy_free_check", Path(__file__).with_name("check_numpy_free_runtime.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize("requires,extras,plugins", [
    ([], [], []),
    (['numpy>=1.26.4; extra == "verify"'], ["verify"], []),
    (["mojolearn-amd==1.2.3", "mojolearn-nvidia==1.2.3", "numpy>=1.26.4; extra == 'verify'"],
     ["verify"], ["mojolearn-amd==1.2.3", "mojolearn-nvidia==1.2.3"]),
])
def test_only_optional_verifier_and_exact_split_plugins_are_allowed(requires, extras, plugins):
    assert check.metadata_errors(requires, extras, plugins) == []


@pytest.mark.parametrize("requires,extras,plugins", [
    (["numpy>=1.26.4"], ["verify"], []),
    (['numpy>=1.26.4; extra == "verify"'], [], []),
    (['numpy>=1.26.4; extra == "verify" or python_version >= "3"'], ["verify"], []),
    (['scipy; extra == "verify"'], ["verify"], []),
    (['numpy>=1.26.4; extra == "other"'], ["verify"], []),
    (["mojolearn-amd==1.2.3"], [], []),
    (["mojolearn-amd>=1.2.3"], [], ["mojolearn-amd==1.2.3"]),
    (["mojolearn-amd==1.2.2"], [], ["mojolearn-amd==1.2.3"]),
    (["mojolearn-amd==1.2.3; extra == 'amd'"], [], ["mojolearn-amd==1.2.3"]),
    (["mojolearn-amd==1.2.3", "mojolearn-amd==1.2.3"], [], ["mojolearn-amd==1.2.3"]),
    ([], [], ["mojolearn-amd==1.2.3"]),
])
def test_unconditional_numpy_other_packages_and_invalid_plugin_pins_fail(requires, extras, plugins):
    assert check.metadata_errors(requires, extras, plugins)
