#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Import-time code of a module `import mojolearn` runs selects every lane
(`lane_select.package_import_time_closure`, `import_time_code_changed`,
2026-11-05). Fast: synthetic old/new texts are seeded into the selector's
caches; the closure walk parses the package once. The select()-level checks
on a real file live in tools/test_lane_select.py (they need the lane map).

    python3 -m pytest tools/test_lane_select_import_time.py -q
"""
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select as ls                                        # noqa: E402

REF = "0000000000import-time-test"
PATH = os.path.join(ls.PKG, "_synthetic_import_time.py")

OLD = """
    import os
    _TOL = 1e-6

    def helper(x, scale=2):
        \"\"\"Doc.\"\"\"
        return x * scale

    class Model:
        kind = "a"

        def fit(self, X):
            return helper(X)
"""


def _changed(new, old=OLD):
    key = f"{REF}:{PATH}"
    ls._GIT_SHOW[key] = textwrap.dedent(old) if old is not None else None
    ls._read.cache[PATH] = textwrap.dedent(new)
    try:
        return ls.import_time_code_changed(REF, PATH)
    finally:
        ls._GIT_SHOW.pop(key, None)
        ls._read.cache.pop(PATH, None)


def test_an_unchanged_file_and_a_docstring_edit_run_nothing_new():
    assert not _changed(OLD)
    assert not _changed(OLD.replace('"""Doc."""', '"""Another doc."""'))


def test_a_function_body_edit_is_not_import_time():
    assert not _changed(OLD.replace("return x * scale", "return x * scale + 0"))


def test_a_method_body_edit_is_not_import_time():
    assert not _changed(OLD.replace("return helper(X)", "return helper(X, 3)"))


def test_a_module_level_statement_is_import_time():
    assert _changed(OLD.replace("_TOL = 1e-6", "_TOL = 1e-7"))
    assert _changed(OLD + "\n    _ENTRY = helper(1)\n")
    assert _changed(OLD.replace("    import os\n", "    import os\n    import sys\n"))


def test_a_header_or_class_body_statement_is_import_time():
    assert _changed(OLD.replace("scale=2", "scale=3"))                 # a default runs at def time
    assert _changed(OLD.replace('kind = "a"', 'kind = "b"'))            # class body runs at import
    assert _changed(OLD.replace("class Model:", "class Model(object):"))


def test_an_unreadable_diff_is_import_time():
    assert _changed(OLD, old=None)                                      # no old revision
    assert _changed("def broken(:\n")                                   # does not parse


def test_the_closure_holds_the_chain_every_lane_imports():
    closure = ls.package_import_time_closure()
    for name in ("__init__.py", "_samba_impl.py", "_native_samba_bridge.py", "neural_inference.py"):
        assert os.path.join(ls.PKG, name) in closure, name
    # `python -m mojolearn` is not `import mojolearn`
    assert os.path.join(ls.PKG, "__main__.py") not in closure
    print(f"package_import_time_closure: {len(closure)} files")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
