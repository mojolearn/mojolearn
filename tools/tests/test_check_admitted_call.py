"""tools/check_admitted_call.py on small synthetic sources.

    .pixi/envs/test/bin/python -m pytest tools/tests/test_check_admitted_call.py -q

Each case feeds `violations` a dict of in-memory files, so nothing is
compiled; the last test runs the real check over the tree and asserts it is
clean against its own DEBT table.
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_admitted_call", REPO / "tools" / "check_admitted_call.py")
chk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chk)

HELPER = '''
def helper(model: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    ref cpy = Python().cpython()
    var np = Python.import_module("numpy")
    return b
'''


def test_counts_each_form_and_skips_comments():
    text = HELPER + '\n    # native_loaded_module("sys") in a comment\n'
    assert chk.count_forms(text) == {"native_loaded_module": 1, "cpython": 1, "import_module": 1}


def test_qualified_names_are_not_the_loader():
    # A method or another module's function of the same suffix is not counted.
    assert chk.count_forms("x = self.native_loaded_module(name)\ny = my_native_loaded_module(n)\n") == {}


def test_new_file_is_refused():
    found = chk.violations({"bindings/new_helper.mojo": HELPER}, doors=frozenset(), debt={})
    assert sorted(f for _, f, _, _ in found) == ["cpython", "import_module", "native_loaded_module"]


def test_doors_are_unlimited():
    assert chk.violations({"bindings/door.mojo": HELPER * 5}, doors=frozenset({"bindings/door.mojo"}), debt={}) == []


def test_debt_may_shrink_never_grow():
    debt = {"bindings/old.mojo": {"native_loaded_module": 2, "cpython": 1, "import_module": 1}}
    assert chk.violations({"bindings/old.mojo": HELPER}, doors=frozenset(), debt=debt) == []
    grown = chk.violations({"bindings/old.mojo": HELPER * 3}, doors=frozenset(), debt=debt)
    assert ("bindings/old.mojo", "native_loaded_module", 3, 2) in grown


def test_admitted_call_module_is_a_door():
    assert "bindings/admitted_call.mojo" in chk.DOORS
    assert "bindings/classical_native.mojo" in chk.DOORS


def test_tree_is_within_allowance():
    sources = chk._read(list(chk.runtime_files()))
    assert len(sources) > 100
    assert chk.violations(sources) == []
