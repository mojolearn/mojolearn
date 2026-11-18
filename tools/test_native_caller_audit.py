# SPDX-License-Identifier: Apache-2.0
"""tools/native_caller_audit.py: a registered native name with no caller fails; a literal caller, a constructed
family, or a KEEP entry passes; comments, messages and Mojo functions of the same name call nothing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_caller_audit as nca  # noqa: E402

DOOR = '''
def door(operation: String) raises -> PythonObject:
    if operation == "family_x_fit":
        return fit()
    if operation == "family_x_dead":
        return dead()
    raise Error("unknown")
'''
BINDING = '''
def register(mut m: PythonModuleBuilder) raises:
    m.def_function[a_binding]("x_alive")
    m.def_function[b_binding]("x_dead_export")
    m.def_py_c_function(c_entry, "x_dead_c")
'''


def run(files, families=(), keep=None):
    regs, words = nca._scan_many(files)
    built, strict, void = nca.family_names(files, families)
    dead, unbound, stale = nca.verdict(regs, words, built, strict, keep or {})
    return {d[0] for d in dead}, unbound, stale, void


def test_literal_callers_and_dead_names():
    files = [("bindings/door.mojo", DOOR), ("bindings/_mojolearn_x.mojo", BINDING),
             ("python/mojolearn/x.py", "def fit(self):\n    return _NATIVE(self, 'family_x_fit', ())\n"
                                       "def go(m):\n    return m.x_alive()\n")]
    dead, unbound, stale, void = run(files)
    assert dead == {"family_x_dead", "x_dead_export", "x_dead_c"}
    assert not unbound and not stale and not void


def test_comments_messages_and_mojo_functions_do_not_call():
    other = '''
# x_dead_export is documented here
def x_dead_export() raises:
    raise Error("x_dead_export: refused")
def use():
    x_dead_export()
'''
    files = [("bindings/_mojolearn_x.mojo", BINDING), ("core/impl.mojo", other),
             ("python/mojolearn/x.py", "# m.x_dead_export()\nraise ValueError('x_dead_c: no')\nm.x_alive()\n")]
    dead, *_ = run(files)
    assert {"x_dead_export", "x_dead_c"} <= dead
    assert "x_alive" not in dead


def test_manifests_do_not_call():
    files = [("bindings/_mojolearn_x.mojo", BINDING),
             ("python/mojolearn/_surface_x.py", 'exports = ("x_alive", "x_dead_export")\n')]
    dead, *_ = run(files)
    assert "x_alive" in dead


def test_constructed_family_reaches_its_names_and_reports_unbound():
    door = DOOR + '''
def public(method: String) raises:
    return door("family_x_" + method)
'''
    fam = dict(site="bindings/door.mojo", literal='"family_x_" + method', strict=True, prefixes=["family_x_"],
               suffixes=nca._src(r"\.py$", r"""["']family_x_public_(fit|dead|gone)["']"""), note="test")
    files = [("bindings/door.mojo", door),
             ("python/mojolearn/x.py", "a = 'family_x_public_dead'\nb = 'family_x_public_gone'\n")]
    dead, unbound, _, void = run(files, [fam])
    assert "family_x_dead" not in dead
    assert [n for n, _ in unbound] == ["family_x_gone"]
    assert not void


def test_void_family_when_site_loses_its_literal():
    fam = dict(site="bindings/door.mojo", literal='"family_x_" + method', strict=True, prefixes=["family_x_"],
               suffixes=["dead"], note="test")
    dead, _, _, void = run([("bindings/door.mojo", DOOR)], [fam])
    assert void and "family_x_dead" in dead


def test_module_prefix_family():
    binding = 'def r(mut m: PythonModuleBuilder) raises:\n    m.def_function[v]("x_vendor")\n'
    site = 'def read(module, prefix, fn):\n    return getattr(module, prefix + "_" + fn, None)\n'
    fam = dict(site="python/mojolearn/_backend.py", literal='getattr(module, prefix + "_" + fn, None)',
               strict=False, prefixes="MODULE", suffixes=["_vendor"], note="test")
    dead, unbound, _, _ = run([("bindings/_mojolearn_x.mojo", binding), ("python/mojolearn/_backend.py", site)],
                              [fam])
    assert "x_vendor" not in dead and not unbound


def test_keep_and_stale_keep():
    files = [("bindings/_mojolearn_x.mojo", BINDING), ("python/mojolearn/x.py", "m.x_alive()\n")]
    dead, _, stale, _ = run(files, keep={"x_dead_export": "paused", "x_alive": "reason", "gone": "reason"})
    assert "x_dead_export" not in dead
    assert stale == ["gone", "x_alive"]


def test_registrations_in_tests_and_checks_are_not_entries():
    files = [("tests/boundary/x_compile.mojo", BINDING), ("checks/native_tests/x.mojo", DOOR)]
    dead, *_ = run(files)
    assert not dead


def test_repository_passes():
    """The work tree itself: every native name has a caller, a family or a KEEP reason."""
    lines = []
    assert nca.audit(out=lines.append) == 0, "\n".join(lines[-40:])
