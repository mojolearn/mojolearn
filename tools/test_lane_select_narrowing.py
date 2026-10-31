#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The narrowing rules of lane release-auto-reuse-2 (2026-10-31) still refuse
where they must.

Each rule in tools/lane_select.py that lets a change select fewer lanes is
held here to BOTH sides: the shape it was written for narrows, and every
nearby shape that could move a bit falls back to the wider rule (None, False,
or the lane kept). Synthetic texts are seeded into the selector's own caches,
so nothing here derives the lane map.

    python3 -m pytest tools/test_lane_select_narrowing.py -q
    python3 tools/test_lane_select_narrowing.py
"""
import ast
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select as ls                                        # noqa: E402

REF = "0000000000narrowing-test"


class _Seeded:
    """Seed `path`'s text at HEAD and at REF into the selector's caches and
    restore them afterwards."""

    def __init__(self, files):
        self.files = files

    def __enter__(self):
        self.saved_read = dict(ls._read.cache)
        self.saved_git = dict(ls._GIT_SHOW)
        self.saved_parse = dict(getattr(ls._parse, "cache", {}))
        for path, (old, new) in self.files.items():
            ls._GIT_SHOW[f"{REF}:{path}"] = old
            ls._read.cache[path] = new
            getattr(ls._parse, "cache", {}).pop(path, None)
        return self

    def __exit__(self, *exc):
        ls._read.cache.clear()
        ls._read.cache.update(self.saved_read)
        ls._GIT_SHOW.clear()
        ls._GIT_SHOW.update(self.saved_git)
        if hasattr(ls._parse, "cache"):
            ls._parse.cache.clear()
            ls._parse.cache.update(self.saved_parse)


def _d(text):
    return textwrap.dedent(text).lstrip("\n")


# ------------------------------------------------------------ Mojo prose edges
def test_mojo_code_text_drops_docstrings_and_comments_only():
    text = _d('''
        def f(x: Int):
            """Calls `named_in_prose` in its docstring."""
            var s = "kept_in_a_string"  # named_in_comment
            return g_called(x)
        ''')
    code = ls._mojo_code_text(text)
    idents = set(ls._MOJO_IDENT_RE.findall(code))
    assert "g_called" in idents and "kept_in_a_string" in idents
    assert "named_in_prose" not in idents and "named_in_comment" not in idents


def test_mojo_code_text_keeps_an_assigned_triple_string_and_unscannable_text():
    text = 'def f():\n    var t = """value_name"""\n    return t\n'
    assert "value_name" in ls._mojo_code_text(text)
    broken = 'def f():\n    var t = "unterminated\n    return g(t)\n'
    assert ls._mojo_code_text(broken) == broken          # cannot be read with certainty: every word kept


def test_mojo_def_graph_no_longer_follows_a_docstring_name():
    a = _d('''
        def checker():
            """`fitter` names the files."""
            return 0

        def fitter():
            return 1
        ''')
    with _Seeded({"ns_test/a.mojo": (a, a)}):
        graph = ls._MojoDefGraph()
        node = ("t", "ns_test/a.mojo", "checker")
        assert ("t", "ns_test/a.mojo", "fitter") not in graph.targets(node)
        called = a.replace("return 0", "return fitter()")
        ls._read.cache["ns_test/a.mojo"] = called
        graph = ls._MojoDefGraph()
        assert ("t", "ns_test/a.mojo", "fitter") in graph.targets(node)


# ------------------------------------------------------ CPU-only device routes
def test_device_column_lanes_drops_only_cpu_route_files_on_device_backends():
    saved = ls._LANE_CPU_ROUTE_ONLY
    try:
        ls._LANE_CPU_ROUTE_ONLY = {"a": frozenset({"x/host.mojo"}), "b": frozenset()}
        lanes = {"a", "b", "c"}
        for backend in ("cuda", "hip", "metal"):
            assert ls.device_column_lanes("x/host.mojo", lanes, backend) == {"b", "c"}
        assert ls.device_column_lanes("x/host.mojo", lanes, "cpu") == lanes
        assert ls.device_column_lanes("x/host.mojo", lanes, None) == lanes
        assert ls.device_column_lanes("x/host.py", lanes, "cuda") == lanes
        ls._LANE_CPU_ROUTE_ONLY = None                  # no recorded map: nothing dropped
        assert ls.device_column_lanes("x/host.mojo", lanes, "cuda") == lanes
    finally:
        ls._LANE_CPU_ROUTE_ONLY = saved


def test_cpu_only_route_tables_have_no_reader_but_the_cpu_only_selector():
    """`cpu_only_routes` rests on `_backend` reading the routing tables only
    while it installs the CPU-only set; a new reader breaks the rule."""
    tree = ast.parse(open(os.path.join(ls.ROOT, ls.BACKEND)).read())
    readers = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and n.id in ("_HOST_MODULES", "_HOST_INFERENCE_MODULES"):
                    readers.setdefault(n.id, set()).add(fn.name)
                if isinstance(n, ast.Attribute) and n.attr == "ADAPTED_MODULES":
                    readers.setdefault("ADAPTED_MODULES", set()).add(fn.name)
    assert readers.get("_HOST_MODULES") == {"_select_cpu_only"}, readers
    assert readers.get("_HOST_INFERENCE_MODULES") == {"_select_cpu_only"}, readers
    assert readers.get("ADAPTED_MODULES") == {"_cpu_only_binding"}, readers
    body = open(os.path.join(ls.ROOT, ls.BACKEND)).read()
    assert "_CPU_ONLY = reason" in body
    pkg = os.path.join(ls.ROOT, "python", "mojolearn")
    for rel in os.listdir(pkg):
        if rel.endswith(".py") and rel != "_backend.py":
            other = ast.parse(open(os.path.join(pkg, rel)).read())
            assert not any(isinstance(n, ast.Attribute) and n.attr in ("_HOST_MODULES", "_HOST_INFERENCE_MODULES")
                           for n in ast.walk(other)), rel


def test_device_loaded_bindings_follows_import_aliases_not_cpu_routes():
    class HS:
        HOST_IMPORT_ALIASES = {"_mojolearn_tok_host": [dict(binding="_mojolearn_alias_host")]}
    got = ls.device_loaded_bindings({"_mojolearn_gpu", "_mojolearn_tok_host"}, HS())
    assert got == {"_mojolearn_gpu", "_mojolearn_tok_host", "_mojolearn_alias_host"}


# ------------------------------------------------------- the Linux set builder
_BUILDER = "packaging/linux/build_sets.sh"
_OLD_BUILDER = _d('''
    #!/bin/bash
    cat <<EOF > notes.txt
    # data line inside a heredoc
    EOF
    case "$s" in
      merged_build.sh) args="$([[ "$tier" = host ]] && echo host || echo gpu) --group $group" ;;
    esac
    ''')


def test_rootdir_only_builder_edit_narrows_a_fixed_root_and_comments():
    new = _OLD_BUILDER.replace(
        '--group $group"', '--group $group --rootdir /tmp/mojolearn-merged-roots/$tier"').replace(
        "case", "# a new comment\ncase")
    with _Seeded({_BUILDER: (_OLD_BUILDER, new)}):
        assert ls.rootdir_only_builder_edit(REF, _BUILDER)


def test_rootdir_only_builder_edit_refuses_any_other_change():
    for new in (_OLD_BUILDER.replace("--group $group", "--group $group --jobs 8"),
                _OLD_BUILDER.replace("# data line inside a heredoc", "a changed data line"),
                _OLD_BUILDER.replace('--group $group"', '--group $group --rootdir /x" ; rm -rf /y')):
        with _Seeded({_BUILDER: (_OLD_BUILDER, new)}):
            assert ls.rootdir_only_builder_edit(REF, _BUILDER) is None, new
    other = "packaging/linux/audit.sh"
    with _Seeded({other: (_OLD_BUILDER, _OLD_BUILDER.replace('$group"', '$group --rootdir /x"'))}):
        assert ls.rootdir_only_builder_edit(REF, other) is None   # not a set builder


def test_rootdir_rule_refuses_when_merged_build_reads_the_root_otherwise():
    real = open(os.path.join(ls.ROOT, ls.MERGED_BUILD)).read()
    new = _OLD_BUILDER.replace('--group $group"', '--group $group --rootdir /tmp/r/$tier"')
    leaked = real + '\necho "$rootdir" > "$outdir/where.txt"\n'
    with _Seeded({_BUILDER: (_OLD_BUILDER, new), ls.MERGED_BUILD: (real, leaked)}):
        assert ls.rootdir_only_builder_edit(REF, _BUILDER) is None
    used = real + '\ncp "$root" "$outdir/"\n'
    with _Seeded({_BUILDER: (_OLD_BUILDER, new), ls.MERGED_BUILD: (real, used)}):
        assert ls.rootdir_only_builder_edit(REF, _BUILDER) is None


# --------------------------------------------------- a native input's audit
_WHEEL = "packaging/portable_math/wheel.py"
_OLD_WHEEL = _d('''
    import json
    from stage import stage
    ORACLES = {"a.py", "b.py"}
    HELPER = "libMath.so"

    def audit_tree(root):
        errors = [p for p in root if p not in ORACLES]
        return {"oracles": sorted(ORACLES), "errors": errors}

    def finalize(wheel, dist):
        stage(wheel, HELPER)
        report = audit_tree(wheel)
        report["runtime_changes"] = []
        (dist / "portable-math.json").write_text(json.dumps(report, indent=2) + "\\n")
        return report

    def main():
        return finalize([], None)
    ''')


def test_audit_only_native_edit_narrows_an_audit_list():
    new = _OLD_WHEEL.replace('{"a.py", "b.py"}', '{"a.py"}')
    with _Seeded({_WHEEL: (_OLD_WHEEL, new)}):
        assert ls.audit_only_native_edit(REF, _WHEEL)


def test_audit_only_native_edit_refuses_bytes_side_and_leaked_results():
    cases = (
        _OLD_WHEEL.replace('HELPER = "libMath.so"', 'HELPER = "libOther.so"'),     # stage reads it
        _OLD_WHEEL.replace('{"a.py", "b.py"}', '{"a.py"}').replace(
            "stage(wheel, HELPER)", "stage(wheel, HELPER, sorted(ORACLES))"),     # bytes side reads it
        _OLD_WHEEL.replace('{"a.py", "b.py"}', '{"a.py"}').replace(
            '    return report\n', '    stage(report, HELPER)\n    return report\n'),  # result flows on
        _OLD_WHEEL.replace('{"a.py", "b.py"}', '{"a.py"}') + "\nprint(1)\n",      # module-level code
    )
    for new in cases:
        assert new != _OLD_WHEEL.replace('{"a.py", "b.py"}', '{"a.py"}')
        with _Seeded({_WHEEL: (_OLD_WHEEL, new)}):
            assert ls.audit_only_native_edit(REF, _WHEEL) is None, new


# ---------------------------------------------------- harness runtime imports
_LA = "tools/lane_applicability.py"
_OLD_LA = _d('''
    TABLE = {"x": 1}

    def use_harness(m):
        return m

    def degenerate(column):
        return {k: v for k, v in TABLE.items()}

    def _selfcheck():
        return [degenerate("c")]

    def main():
        return _selfcheck()

    if __name__ == "__main__":
        main()
    ''')
_HARNESS_USES = _d('''
    import lane_applicability as _la
    _la.use_harness(None)
    print(_la.degenerate("c"))
    ''')


def test_runtime_import_unreached_narrows_a_selfcheck_edit():
    new = _OLD_LA.replace('return [degenerate("c")]', 'return []')
    with _Seeded({_LA: (_OLD_LA, new), ls.HARNESS: (_HARNESS_USES, _HARNESS_USES)}):
        assert ls.runtime_import_unreached(REF, _LA)


def test_runtime_import_unreached_refuses_what_the_harness_runs():
    reached = _OLD_LA.replace('TABLE = {"x": 1}', 'TABLE = {"x": 2}')
    with _Seeded({_LA: (_OLD_LA, reached), ls.HARNESS: (_HARNESS_USES, _HARNESS_USES)}):
        assert ls.runtime_import_unreached(REF, _LA) is None
    body = _OLD_LA.replace("return {k: v for k, v in TABLE.items()}", "return {}")
    with _Seeded({_LA: (_OLD_LA, body), ls.HARNESS: (_HARNESS_USES, _HARNESS_USES)}):
        assert ls.runtime_import_unreached(REF, _LA) is None
    selfcheck = _OLD_LA.replace('return [degenerate("c")]', 'return []')
    handed = _HARNESS_USES + "keep = _la\n"                # the module handed on as a value
    with _Seeded({_LA: (_OLD_LA, selfcheck), ls.HARNESS: (handed, handed)}):
        assert ls.runtime_import_unreached(REF, _LA) is None
    calls_it = _HARNESS_USES + "_la._selfcheck()\n"
    with _Seeded({_LA: (_OLD_LA, selfcheck), ls.HARNESS: (calls_it, calls_it)}):
        assert ls.runtime_import_unreached(REF, _LA) is None


# ------------------------------------------------ a removed defaulted switch
def _fn(text):
    return ast.parse(_d(text)).body[0]


_OLD_LOAD = '''
    def load(path=None, axis=False):
        """doc"""
        if os.environ.get("X"):
            if axis:
                raise CannotRun("axis")
            raise CannotRun("x")
        return path
    '''
_NEW_LOAD = '''
    def load(path=None):
        """new doc"""
        if os.environ.get("X"):
            raise CannotRun("x")
        return path
    '''


def _calls(*texts):
    return [n for t in texts for n in ast.walk(ast.parse(t)) if isinstance(n, ast.Call)]


def test_default_only_specialization_accepts_the_retired_switch():
    assert ls.default_only_specialization(_fn(_OLD_LOAD), _fn(_NEW_LOAD), _calls("load()", "m.load('p')"))


def test_default_only_specialization_refuses_every_other_shape():
    old, new = _fn(_OLD_LOAD), _fn(_NEW_LOAD)
    assert not ls.default_only_specialization(old, new, _calls("load(axis=True)"))
    assert not ls.default_only_specialization(old, new, _calls("load(p, True)"))
    assert not ls.default_only_specialization(old, new, _calls("load(**kw)"))
    assert not ls.default_only_specialization(_fn(_OLD_LOAD.replace("axis=False", "axis=True")), new, [])
    assert not ls.default_only_specialization(
        _fn(_OLD_LOAD.replace('raise CannotRun("axis")', 'raise CannotRun("axis")\n            else:\n'
                              '                path = 1')), new, [])
    assert not ls.default_only_specialization(_fn(_OLD_LOAD.replace("return path", "return path, axis")),
                                              new, [])
    assert not ls.default_only_specialization(old, _fn(_NEW_LOAD.replace("return path", "return None")), [])


# --------------------------------------------------------------- retired lanes
_OLD_IB = _d('''
    import numpy as np
    LANES = {}
    BATCH = {}

    def lane(name):
        def deco(fn):
            LANES[name] = fn
            return fn
        return deco

    def _batch_decl(spec, *names, revision=None):
        for n in names:
            if n in BATCH:
                raise RuntimeError(n)
            BATCH[n] = spec

    def _rows(*methods, sl=slice(None)):
        def spec(ml, e, X):
            return [getattr(e, m)(X[sl]) for m in methods]
        return spec

    def _par_helper(x):
        return x

    PAR_ROWS = 16
    REVISIONS = {"keep": "r1", "par-x": "r2"}
    NEEDS = {name: ("model",) for name in ("par-x",)}

    @lane("keep")
    def _(ml):
        return 1

    @lane("par-x")
    def _(ml):
        return _par_helper(PAR_ROWS)

    _batch_decl(_rows("predict"), "keep", "par-x")
    _batch_decl(_rows("predict", sl=slice(0, 4)), "par-x-only")
    ''')
_NEW_IB = _d('''
    import numpy as np
    LANES = {}
    BATCH = {}

    def lane(name):
        def deco(fn):
            LANES[name] = fn
            return fn
        return deco

    def _batch_decl(spec, *names, revision=None):
        for n in names:
            if n in BATCH:
                raise RuntimeError(n)
            BATCH[n] = spec

    def _rows(*methods, sl=slice(None)):
        def spec(ml, e, X):
            return [getattr(e, m)(X[sl]) for m in methods]
        return spec

    REVISIONS = {"keep": "r1"}
    NEEDS = {}

    @lane("keep")
    def _(ml):
        return 1

    _batch_decl(_rows("predict"), "keep")
    ''')


def _retired(old, new):
    return ls.retired_lane_subtraction(old, new, path="tools/identity_break.py")


def test_retiring_a_lane_selects_no_surviving_lane():
    # "par-x-only" is not a decorated lane at the old revision, so it must not
    # count as retired: its declaration's removal is refused below; here it stays.
    old = _OLD_IB
    new = _NEW_IB + '_batch_decl(_rows("predict", sl=slice(0, 4)), "par-x-only")\n'
    assert _retired(old, new) == [], ls.RETIRED_WHY
    edited = new.replace("return 1", "return 2")
    assert _retired(old, edited) == ["keep"]


def test_retiring_refuses_what_could_move_a_surviving_lane():
    tail = '_batch_decl(_rows("predict", sl=slice(0, 4)), "par-x-only")\n'
    base = _NEW_IB + tail
    cases = {
        "a surviving lane's table entry removed": base.replace('{"keep": "r1"}', "{}"),
        "a surviving lane dropped from a declaration": base.replace('_batch_decl(_rows("predict"), "keep")',
                                                                   '_batch_decl(_rows("predict"))'),
        "a declaration's spec changed": base.replace('_batch_decl(_rows("predict"), "keep")',
                                                     '_batch_decl(_rows("predict_proba"), "keep")'),
        "a removed helper still named": base + "X = _par_helper\n",
        "a removed constant still named by string": base + 'Y = globals()["PAR_ROWS"]\n',
        "a statement added": base + "Z = 3\n",
        "a non-retired declaration removed": _NEW_IB,
    }
    for why, new in cases.items():
        assert _retired(_OLD_IB, new) is None, why


def test_retiring_refuses_a_removed_declaration_with_side_effects():
    old = _OLD_IB.replace('_batch_decl(_rows("predict"), "keep", "par-x")',
                          '_batch_decl(_rows("predict"), "keep")\n_batch_decl(_mutate(), "par-x")')
    old = old.replace("PAR_ROWS = 16", "PAR_ROWS = 16\n\ndef _mutate():\n    BATCH.clear()\n    return None")
    new = _NEW_IB + '_batch_decl(_rows("predict", sl=slice(0, 4)), "par-x-only")\n'
    assert _retired(old, new) is None


def test_per_name_declarer_shape():
    assert ls._per_name_declarer(_fn('''
        def d(spec, *names, revision=None):
            for n in names:
                if n in T:
                    raise RuntimeError(n)
                T[n] = spec
                if revision is not None:
                    R[n] = revision
        '''))
    assert not ls._per_name_declarer(_fn('''
        def d(spec, *names):
            for n in names:
                T[n] = len(names)
        '''))
    assert not ls._per_name_declarer(_fn('''
        def d(spec, *names):
            T.clear()
            for n in names:
                T[n] = spec
        '''))


def test_entries_minus_and_row_membership_take_retired_lanes_only():
    old = ast.parse('{"keep": "a", "par-x": "b"}', mode="eval").body
    assert ls._entries_minus(old, ast.parse('{"keep": "a"}', mode="eval").body, {"par-x"})
    assert not ls._entries_minus(old, ast.parse('{"par-x": "b"}', mode="eval").body, {"par-x"})
    assert not ls._entries_minus(old, ast.parse('{"keep": "z"}', mode="eval").body, {"par-x"})
    called = ast.parse('{"keep": "a", "par-x": f()}', mode="eval").body
    assert not ls._entries_minus(called, ast.parse('{"keep": "a"}', mode="eval").body, {"par-x"})
    comp = ast.parse('{n: ("model",) for n in ("par-x", "par-y")}', mode="eval").body
    assert ls._entries_minus(comp, ast.parse("{}", mode="eval").body, {"par-x", "par-y"})
    assert not ls._entries_minus(comp, ast.parse("{}", mode="eval").body, {"par-x"})
    row = lambda lanes: {"training_lanes": ast.parse(repr(tuple(lanes)), mode="eval").body}  # noqa: E731
    every = ["keep", "other"]
    assert ls._changed_row_lanes(row(["keep", "par-x"]), row(["keep"]), every, {}, {"par-x"}) == set()
    assert ls._changed_row_lanes(row(["keep", "par-x"]), row(["par-x"]), every, {}, {"par-x"}) == {"keep"}
    assert ls._changed_row_lanes(row(["keep", "nolane"]), row(["keep"]), every, {}, {"par-x"}) != set()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(exc).__name__, str(exc)[:200])
    sys.exit(1 if failed else 0)
