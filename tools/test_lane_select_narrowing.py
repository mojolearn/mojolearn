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



# ------------------------------------------------ operation-string dispatchers
_DISPATCHER = _d(r"""
    def entry_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) -> PyObjectPtr:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(call[1])
        if operation == "alpha_fit":
            return alpha_fit_impl(call)
        if operation.startswith("fam_"):
            return fam_dispatch(call, operation)
        elif operation == "beta_fit" and len(call) == 3:
            return beta_fit_impl(call)
        if operation != "gamma_fit":
            prelude_impl(call)
        else:
            gamma_impl(call)
        return tail_impl(call)
    """)


class _Vocab:
    """Seed the dispatch vocabulary and the Mojo call-site index, so nothing
    here scans the tree."""

    def __init__(self, eq, sw, callers=None):
        prefixes = {lit[:k] for lit in set(eq) | set(sw) for k in range(len(lit) + 1)}
        self.vocab = (frozenset(eq), frozenset(sw), frozenset(prefixes))
        self.callers = callers or {}

    def __enter__(self):
        self.saved = (ls._DISPATCH_VOCAB, ls._MOJO_CALLERS, dict(ls._PARAM_ADMISSIBLE), dict(ls._DISPATCH_PARSE))
        ls._DISPATCH_VOCAB, ls._MOJO_CALLERS = self.vocab, dict(self.callers)
        ls._PARAM_ADMISSIBLE.clear()
        ls._DISPATCH_PARSE.clear()
        return self

    def __exit__(self, *exc):
        ls._DISPATCH_VOCAB, ls._MOJO_CALLERS = self.saved[0], self.saved[1]
        ls._PARAM_ADMISSIBLE.clear()
        ls._PARAM_ADMISSIBLE.update(self.saved[2])
        ls._DISPATCH_PARSE.clear()
        ls._DISPATCH_PARSE.update(self.saved[3])


def _idents(text):
    return set(ls._MOJO_IDENT_RE.findall(text))


def test_a_dispatcher_follows_only_the_branches_the_lane_operations_can_enter():
    with _Vocab({"alpha_fit", "beta_fit", "gamma_fit"}, {"fam_"}):
        text, kept = ls.dispatch_view(_DISPATCHER, ls.OperationSet({"alpha_fit", "unrelated"}))
        got = _idents(text)
        assert "alpha_fit_impl" in got and "fam_dispatch" not in got
        # a condition with anything but the operation is live, and so is a `!=`, an
        # `else` and everything after the chain
        assert {"beta_fit_impl", "prelude_impl", "gamma_impl", "tail_impl"} <= got
        # the literals a condition compares with are not read as names
        assert "alpha_fit" not in _idents(text.replace("alpha_fit_impl", ""))
        # three dispatch conditions: the `and len(call) == 3` one is not one, so live
        assert len(kept) == 3 and kept.count(False) == 1
        text, _ = ls.dispatch_view(_DISPATCHER, ls.OperationSet({"fam_svc_fit"}))
        assert "fam_dispatch" in _idents(text) and "alpha_fit_impl" not in _idents(text)


def test_a_stem_opens_every_branch_a_longer_string_could_enter():
    """`f"fam_{op}"` leaves the constant `fam_`: it can become any `fam_*`."""
    with _Vocab({"alpha_fit", "fam_one"}, {"fam_"}):
        text, _ = ls.dispatch_view(_DISPATCHER, ls.OperationSet({"fam_"}))
        assert "fam_dispatch" in _idents(text)
        text, _ = ls.dispatch_view(_DISPATCHER, ls.OperationSet({"al_"}))
        assert "alpha_fit_impl" not in _idents(text)        # `al_` cannot become `alpha_fit`
        text, _ = ls.dispatch_view(_DISPATCHER, ls.OperationSet({"alpha_"}))
        assert "alpha_fit_impl" in _idents(text)


def test_a_reassigned_or_computed_operation_is_not_a_dispatcher():
    with _Vocab({"alpha_fit", "beta_fit"}, {"fam_"}):
        reassigned = _DISPATCHER.replace("    return tail_impl(call)",
                                         "    operation = other(call)\n    return tail_impl(call)")
        assert reassigned != _DISPATCHER
        assert ls.dispatch_view(reassigned, ls.OperationSet({"x"})) == (reassigned, ())
        computed = _DISPATCHER.replace("String(call[1])", "String(call[1]) + suffix")
        assert ls.dispatch_view(computed, ls.OperationSet({"x"})) == (computed, ())


def test_a_string_parameter_is_a_dispatcher_only_when_every_call_site_is_admissible():
    callee = _d("""
        def fam_dispatch(model: PythonObject, operation: String) raises -> PythonObject:
            if operation == "fam_one":
                return one_impl(model)
            if operation == "fam_two":
                return two_impl(model)
            return model
        """)
    good = _d("""
        def caller_a(call: PythonObject) raises -> PythonObject:
            var operation = String(call[1])
            return fam_dispatch(call, operation)

        def caller_b(call: PythonObject) raises -> PythonObject:
            var route = "fam_one" if Bool(call) else "fam_two"
            return fam_dispatch(call, route) + fam_dispatch(call, "fam_two")
        """)
    with _Vocab({"fam_one", "fam_two"}, set(), {"x/good.mojo": good, "x/callee.mojo": callee}):
        text, kept = ls.dispatch_view(callee, ls.OperationSet({"fam_two"}))
        assert kept == (False, True) and "one_impl" not in _idents(text)
    bad = good + "\n" + _d("""
        def caller_c(call: PythonObject, name: String) raises -> PythonObject:
            return fam_dispatch(call, "fam_" + name)
        """)
    with _Vocab({"fam_one", "fam_two"}, set(), {"x/good.mojo": bad, "x/callee.mojo": callee}):
        assert ls.dispatch_view(callee, ls.OperationSet({"fam_two"})) == (callee, ())


def test_an_operation_set_is_keyed_by_what_a_condition_can_tell_apart():
    with _Vocab({"alpha_fit"}, {"fam_"}):
        a = ls.OperationSet({"alpha_fit", "noise", "fam_x", "fam_"})
        b = ls.OperationSet({"alpha_fit", "other", "fam_x", "fam_"})
        assert a == b and hash(a) == hash(b)
        assert a.names == {"alpha_fit", "fam_x", "fam_"} and a.stems == {"fam_"}
        assert ls.OperationSet({"alpha_fit"}) != a


# ------------------------------------------------ host_model routes on devices
def test_a_device_column_reads_a_host_model_holder_for_its_route_calls_only():
    """`mojo_edit_lanes` on a device column uses `_LANE_DEVICE_CALLS`: the
    holder `KMeans` routes to (`HostKMeans`) runs `kmeans_predict` there, the
    k-means fit runs on the GPU set. Every other column keeps the full map."""
    path = "ns_test/oracle.mojo"
    old = "def host_fit(x: Int) -> Int:\n    return x\n\n\ndef host_predict(x: Int) -> Int:\n    return x\n"
    new = old.replace("    return x\n\n\ndef host_predict", "    return x + 1\n\n\ndef host_predict")

    class Graph:
        def closure(self, src, export, ops=None):
            return {("t", path, "host_fit" if export == "fit" else "host_predict"): None}
    saved = ls._LANE_SEED_CALLS, ls._LANE_DEVICE_CALLS
    try:
        with _Seeded({path: (old, new)}):
            ls._LANE_SEED_CALLS = {"lane": [("bindings/h.mojo", True, ("fit", "predict"), frozenset({path}))]}
            ls._LANE_DEVICE_CALLS = {"lane": [("bindings/h.mojo", "narrow", ("predict",), frozenset({path}))]}
            got = ls.mojo_edit_lanes(REF, path, {"lane"}, Graph(), None)
            assert got is not None and got[1] == {"lane"}
            for backend in ("cpu", None):
                assert ls.mojo_edit_lanes(REF, path, {"lane"}, Graph(), backend)[1] == {"lane"}
            for backend in ls.DEVICE_BACKENDS:
                assert ls.mojo_edit_lanes(REF, path, {"lane"}, Graph(), backend)[1] == set()
            # the route itself calling the changed definition keeps the lane on a device
            ls._LANE_DEVICE_CALLS = {"lane": [("bindings/h.mojo", "narrow", ("fit",), frozenset({path}))]}
            assert ls.mojo_edit_lanes(REF, path, {"lane"}, Graph(), "cuda")[1] == {"lane"}
            ls._LANE_DEVICE_CALLS = None                 # no device map recorded: the full map
            assert ls.mojo_edit_lanes(REF, path, {"lane"}, Graph(), "cuda")[1] == {"lane"}
    finally:
        ls._LANE_SEED_CALLS, ls._LANE_DEVICE_CALLS = saved


def test_host_route_names_come_from_the_route_files_and_bind_calling_bases():
    """The route of `_mojolearn_core_host` names the k-means inference entries
    and never the fit; a routed public base whose body calls `_bind` would
    hand its own names over too."""
    files = ls.host_route_files("_mojolearn_core_host")
    assert os.path.join(ls.PKG, "_classical_host.py") in files
    names = ls.host_route_names("_mojolearn_core_host")
    assert {"kmeans_predict", "kmeans_transform"} <= names and "kmeans_fit" not in names
    saved_routes = {k: set(v) for k, v in ls._HOST_ROUTE_CLASSES.items()}
    saved_symbols, saved_parse = ls._python_symbols, dict(ls._parse.cache)
    try:
        ls._HOST_ROUTE_CLASSES["_mojolearn_ns_host"] = {("ns_test/route.py", "HostThing", "Thing")}
        ls._parse.cache["ns_test/thing.py"] = ast.parse(
            "class Thing:\n    def fit(self):\n        return self._bind().thing_fit()\n")
        ls._parse.cache["ns_test/route.py"] = ast.parse("class HostThing(Thing):\n    _BINDING = 'x'\n")
        ls._python_symbols = lambda files: {"Thing": {"ns_test/thing.py"}}
        assert "thing_fit" in ls.host_route_names("_mojolearn_ns_host")
        ls._parse.cache["ns_test/thing.py"] = ast.parse(
            "class Thing:\n    def fit(self):\n        return _NATIVE(self, 'thing_fit')\n")
        assert "thing_fit" not in ls.host_route_names("_mojolearn_ns_host")
    finally:
        ls._HOST_ROUTE_CLASSES.clear()
        ls._HOST_ROUTE_CLASSES.update(saved_routes)
        ls._python_symbols = saved_symbols
        ls._parse.cache.clear()
        ls._parse.cache.update(saved_parse)


# ---------------------------------------------------- precompiled Mojo packages
def test_the_max_ops_package_is_attributed_to_no_lane_and_refuses_otherwise():
    path = "adapters/max_ops/__init__.mojo"
    assert ("packaging/adapters/build.sh", "adapters/max_ops", "mojolearn_ops.mojoc") in ls._precompiled_packages()
    saved = ls._named_by_the_corpus, ls._mojo_importers
    try:
        ls._named_by_the_corpus = lambda token, skip=(), whole=False, name_end=False: []
        ls._mojo_importers = lambda rel: set()
        assert "mojolearn_ops.mojoc" in (ls._precompiled_package_only(path) or "")
        assert ls._precompiled_package_only("adapters/native/c_api.mojo") is None   # not in the package
        ls._mojo_importers = lambda rel: {"bindings/_mojolearn_x.mojo"} if rel == path else set()
        assert ls._precompiled_package_only(path) is None          # a binding imports it
        ls._mojo_importers = lambda rel: set()
        ls._named_by_the_corpus = lambda token, skip=(), whole=False, name_end=False: (
            ["python/mojolearn/adapters/max.py"] if token == "mojolearn_ops.mojoc" else [])
        assert ls._precompiled_package_only(path) is None          # a lane's file loads the package
    finally:
        ls._named_by_the_corpus, ls._mojo_importers = saved


def test_a_docstring_edit_is_docstring_only_and_a_deleted_or_added_one_is_not():
    """bindings/classical_exceptions.mojo admits NotFittedError's doc only as a
    str, so a docstring's presence is read at run time; its text is not."""
    path = "ns_test/doc_only.py"
    old = 'class A:\n    """one"""\n    def f(self):\n        return 1\n'
    saved = ls._SOURCE_HASHED
    try:
        ls._SOURCE_HASHED = set()
        with _Seeded({path: (old, old.replace('"""one"""', '"""two"""'))}):
            assert ls.docstring_only(REF, path)
        with _Seeded({path: (old, old.replace('    """one"""\n', ''))}):
            assert not ls.docstring_only(REF, path)
        with _Seeded({path: (old, old.replace('return 1', '"""new"""\n        return 1'))}):
            assert not ls.docstring_only(REF, path)
    finally:
        ls._SOURCE_HASHED = saved


def test_only_an_operation_passed_to_a_dispatcher_opens_its_branch():
    """A table that merely SPELLS `family_kmeans_fit` passes nothing; a call
    of the dispatcher with it (or with a literal-only variable holding it)
    does, and only then is the k-means branch of `family_operation` live."""
    family = _d("""
        def family_operation(model: PythonObject, operation: String) raises -> PythonObject:
            if operation.startswith("family_kmeans_"):
                return kmeans_operation(model, operation)
            if operation.startswith("family_gp_"):
                return gp_operation(model, operation)
            return model
        """)
    table = _d("""
        def par_spec_families() -> List[String]:
            return List[String]("family_kmeans_fit", "family_gp_fit")

        def caller(m: PythonObject, call: PythonObject) raises -> PythonObject:
            raise Error("family_kmeans_fit requires fitted data")
        """)
    direct = _d("""
        def caller(m: PythonObject) raises -> PythonObject:
            return family_operation(m, "family_kmeans_fit")
        """)
    routed = _d("""
        def caller(m: PythonObject, b: Bool) raises -> PythonObject:
            var route = "family_kmeans_fit" if b else "family_kmeans_predict"
            return family_operation(m, route)
        """)
    saved = set(ls._DISPATCHER_NAMES), dict(ls._PASSED_CACHE), ls._dispatch_vocabulary
    try:
        ls._dispatch_vocabulary = lambda: (frozenset(), frozenset({"family_kmeans_", "family_gp_"}),
                                           frozenset({"family_", "family_kmeans_", "family_gp_"}))
        ls._DISPATCHER_NAMES.clear()
        ls._DISPATCHER_NAMES.add("family_operation")
        ls._PASSED_CACHE.clear()
        assert ls.passed_operations(table) == set()
        assert ls.passed_operations(direct) == {"family_kmeans_fit"}
        assert ls.passed_operations(routed) == {"family_kmeans_fit", "family_kmeans_predict"}
        with _Vocab(set(), {"family_kmeans_", "family_gp_"},
                    {"x/caller.mojo": direct, "x/family.mojo": family}):
            for text, live in ((table, False), (direct, True), (routed, True)):
                ops = ls.OperationSet(ls.passed_operations(text))
                view, _kept = ls.dispatch_view(family, ops)
                assert ("kmeans_operation" in _idents(view)) is live, (text, ops)
                assert "gp_operation" not in _idents(view)
    finally:
        ls._DISPATCHER_NAMES.clear()
        ls._DISPATCHER_NAMES.update(saved[0])
        ls._PASSED_CACHE.clear()
        ls._PASSED_CACHE.update(saved[1])
        ls._dispatch_vocabulary = saved[2]


def test_an_unused_registry_edit_is_inert_only_when_no_reached_file_names_it():
    # and `import_bound_definitions` closes a changed def over the globals its
    # import-time calls bind and the defs that read them, refusing any other
    # import-time evaluation (tested below).
    """`unused_registry_definitions`: a registry's changed definition named
    only by files no lane reaches (the retired multi-GPU drivers) selects no
    lane; one a lane-reached file, or the harness, names still refuses."""
    reg = os.path.join(ls.PKG, "_ns_registry.py")
    driver = os.path.join(ls.PKG, "_ns_parallel_driver.py")
    public = os.path.join(ls.PKG, "_ns_public.py")
    files = {reg: "def _connect_binding(x):\n    return x\n",
             driver: "from ._ns_registry import _connect_binding\n_connect_binding(1)\n",
             public: "def fit(self):\n    return 2\n"}
    saved_tracked, saved_search = ls._TRACKED, dict(ls._searchable.cache)
    saved_read = dict(ls._read.cache)
    try:
        ls._TRACKED = set(files)
        for rel, text in files.items():
            ls._read.cache[rel] = text
            ls._searchable.cache[rel] = text
        ls._read.cache[ls.HARNESS] = "def lane():\n    return 0\n"
        rev = {reg: {"lane-a"}, public: {"lane-a"}}
        why = ls.unused_registry_definitions(reg, ["_connect_binding"], rev)
        assert why and driver in why
        assert ls.registry_definition_lanes(reg, ["_connect_binding"], rev)[0] == set()
        rev[driver] = {"lane-b"}                       # a lane reaches the namer: not inert, it is selected
        assert ls.unused_registry_definitions(reg, ["_connect_binding"], rev) is None
        assert ls.registry_definition_lanes(reg, ["_connect_binding"], rev)[0] == {"lane-b"}
        del rev[driver]
        ls._TRACKED = set(files) | {ls.HARNESS}        # the harness names it: refused
        ls._searchable.cache[ls.HARNESS] = "def lane():\n    return _connect_binding(0)\n"
        assert ls.unused_registry_definitions(reg, ["_connect_binding"], rev) is None
        assert ls.registry_definition_lanes(reg, ["_connect_binding"], rev) is None
        assert ls.unused_registry_definitions(reg, [], rev) is None
    finally:
        ls._TRACKED = saved_tracked
        ls._searchable.cache.clear()
        ls._searchable.cache.update(saved_search)
        ls._read.cache.clear()
        ls._read.cache.update(saved_read)

def test_import_bound_definitions_follow_bound_globals_and_refuse_bare_calls():
    path = "ns_test/shard.py"
    old = "def _c(n):\n    return n\n\n\n_W = {'a': _c(1)}\n\n\ndef register(ctx):\n    ctx['w'] = _W\n"
    new = old.replace("return n", "return n + 1")
    with _Seeded({path: (old, new)}):
        assert ls.import_bound_definitions(REF, path) == ["_W", "_c", "register"]
    bare = old + "\n_c(2)\n"
    with _Seeded({path: (bare, bare.replace("return n", "return n + 1"))}):
        assert ls.import_bound_definitions(REF, path) is None



def test_the_definition_graph_follows_only_the_lane_operations_branches():
    """The edit rule's graph reads a dispatcher through the lane's operations:
    a lane whose operations cannot enter `fam_fit_` does not reach the fit
    owner (the RF owner reached 418 lanes through `trees_public_operation`)."""
    a = "ns_test/dispatch.mojo"
    text = _d("""
        from ns_test.owners import fit_owner, predict_owner

        def dispatch(model: PythonObject, operation: String) raises -> PythonObject:
            if operation.startswith("fam_fit_"):
                return fit_owner(model)
            if operation.startswith("fam_predict_"):
                return predict_owner(model)
            return model
        """)
    owners = "def fit_owner(m: Int) -> Int:\n    return m\n\n\ndef predict_owner(m: Int) -> Int:\n    return m\n"
    saved = (dict(ls._FILE_VOCAB), ls._dispatch_vocabulary)
    try:
        with _Seeded({a: (text, text), "ns_test/owners.mojo": (owners, owners)}), \
                _Vocab(set(), {"fam_fit_", "fam_predict_"}, {a: text}):
            ls._FILE_VOCAB[a] = (set(), {"fam_fit_", "fam_predict_"})
            ls._FILE_VOCAB["ns_test/owners.mojo"] = (set(), set())
            orig = ls._mojo_module_files
            ls._mojo_module_files = lambda module, rel, names=(): (
                {"ns_test/owners.mojo"} if module == "ns_test.owners" else orig(module, rel, names))
            try:
                g = ls._MojoDefGraph()
                node = ("t", a, "dispatch")
                fit, predict = ("t", "ns_test/owners.mojo", "fit_owner"), ("t", "ns_test/owners.mojo", "predict_owner")
                assert {fit, predict} <= set(g.targets(node))                    # no operations: both
                only_predict = set(g.targets(node, ls.OperationSet({"fam_predict_x"})))
                assert predict in only_predict and fit not in only_predict
                assert fit in set(g.targets(node, ls.OperationSet({"fam_fit_x"})))
            finally:
                ls._mojo_module_files = orig
    finally:
        ls._FILE_VOCAB.clear()
        ls._FILE_VOCAB.update(saved[0])


def test_a_release_label_bump_is_inert_by_the_ledger_rule_only():
    """`release_label_only` calls the ledger's `version_literal_unconsumed`:
    a `__version__` literal edit with unchanged reviewed consumers is inert;
    any other edit, or a consumer whose bytes moved, is not."""
    import hashlib
    import release_requirement_dependencies as deps
    path = deps.VERSION_PATH
    old = '# Kept in one place.\n__version__ = "0.8.63"\n'
    consumer = "ns_test/consumer.py"
    saved = dict(deps.VERSION_LABEL_ANCHORS)
    try:
        deps.VERSION_LABEL_ANCHORS.clear()
        deps.VERSION_LABEL_ANCHORS[consumer] = hashlib.sha256(b"reviewed\n").hexdigest()
        with _Seeded({path: (old, old.replace("0.8.63", "0.8.64")), consumer: ("reviewed\n", "reviewed\n")}):
            assert ls.release_label_only(REF, path)
        with _Seeded({path: (old, old.replace("0.8.63", "0.8.64")), consumer: ("reviewed\n", "edited\n")}):
            assert ls.release_label_only(REF, path) is None          # a consumer moved
        with _Seeded({path: (old, old + "OTHER = 1\n"), consumer: ("reviewed\n", "reviewed\n")}):
            assert ls.release_label_only(REF, path) is None          # not only the label
        assert ls.release_label_only(REF, "ns_test/other.py") is None
    finally:
        deps.VERSION_LABEL_ANCHORS.clear()
        deps.VERSION_LABEL_ANCHORS.update(saved)


def test_a_call_site_no_binding_compiles_does_not_block_narrowing():
    """A compile probe outside every binding (checks/*_compile.mojo) passing a
    computed operation runs in no lane; the same call inside a binding still
    leaves the dispatcher unnarrowed."""
    callee = _d("""
        def fam_dispatch(model: PythonObject, operation: String) raises -> PythonObject:
            if operation == "fam_one":
                return one_impl(model)
            if operation == "fam_two":
                return two_impl(model)
            return model
        """)
    probe = _d("""
        def probe(call: PythonObject, name: PythonObject) raises -> PythonObject:
            return fam_dispatch(call, String(name) + "x")
        """)
    saved = ls._in_some_binding
    try:
        for held, narrowed in ((False, True), (True, False)):
            ls._in_some_binding = lambda rel, held=held: held or rel != "checks/probe_compile.mojo"
            with _Vocab({"fam_one", "fam_two"}, set(), {"checks/probe_compile.mojo": probe, "x/callee.mojo": callee}):
                got = ls.dispatch_view(callee, ls.OperationSet({"fam_two"}))
                assert (got != (callee, ())) is narrowed
    finally:
        ls._in_some_binding = saved


# ------------------------------------------- pandas-gated glue (2026-11-05)
_GATE_HEAD = _d("""
    from std.python import Python, PythonObject
    from bindings.py_balanced import py_call, py_type
    from bindings.classical_pandas import _pd_class
    """)
_GATED = _d("""
    def names_of(source: PythonObject, context: PythonObject) raises -> PythonObject:
        # only a DataFrame has column names
        if py_type(source) is not _pd_class("pandas.core.frame", "DataFrame"):
            return PythonObject(None)
        var columns = source.columns
        return columns
    """)
_AFTER = _d("""
    def after_fit(fields: PythonObject, source: PythonObject, context: PythonObject) raises:
        var names = names_of(source, context)
        fields["names"] = names
    """)
_GATE_PATH = "ns_gate/lifecycle.mojo"
#: `_AFTER` also calling the helper (bodies are dedented to 4 spaces).
_CALLING_AFTER = _AFTER.replace("    fields[", "    _ = gate_helper(source)\n    fields[")
assert _CALLING_AFTER != _AFTER and "gate_helper" in _CALLING_AFTER, "fixture edit did not apply"


def _gate_file(*defs):
    return _GATE_HEAD + "\n\n" + "\n\n".join(defs)


class _PandasFree:
    """Hold `pandas_free_identity_process`'s answer for a synthetic case."""

    def __init__(self, free):
        self.free = free

    def __enter__(self):
        self.saved = dict(ls._PANDAS_FREE)
        ls._PANDAS_FREE.clear()
        ls._PANDAS_FREE["answer"] = (self.free, "seeded for the test")

    def __exit__(self, *exc):
        ls._PANDAS_FREE.clear()
        ls._PANDAS_FREE.update(self.saved)


def _gated_lanes(old, new, free=True, calls=None):
    """`mojo_edit_lanes` for one seeded edit of _GATE_PATH, with one lane that
    reaches every unit through export `fit` and one kept by the whole-file
    rule (no recorded export)."""
    class Graph:
        def closure(self, src, export, ops=None):
            units = ls._mojo_units(ls._read(_GATE_PATH)) or dict(top={})
            return {("t", _GATE_PATH, name): None for name in units["top"]}
    saved = ls._LANE_SEED_CALLS, ls._LANE_DEVICE_CALLS
    try:
        with _Seeded({_GATE_PATH: (old, new)}), _PandasFree(free):
            ls._LANE_SEED_CALLS = calls or {
                "through": [("bindings/h.mojo", True, ("fit",), frozenset({_GATE_PATH}))],
                "kept": [("bindings/h.mojo", True, (), frozenset({_GATE_PATH}))]}
            ls._LANE_DEVICE_CALLS = None
            return ls.mojo_edit_lanes(REF, _GATE_PATH, {"through", "kept"}, Graph(), None)
    finally:
        ls._LANE_SEED_CALLS, ls._LANE_DEVICE_CALLS = saved


def test_pandas_gate_shape_is_exact():
    assert ls._pandas_gate(_GATED) is not None
    refused = (
        ("a statement before the gate", _GATED.replace("    # only", "    var k = 1\n    # only")),
        ("an else after the gate", _GATED.replace("    var columns", "    else:\n        pass\n    var columns")),
        ("two statements in the gate", _GATED.replace("        return PythonObject(None)\n",
                                                      "        touch(source)\n        return PythonObject(None)\n")),
        ("a gate that returns a call", _GATED.replace("return PythonObject(None)", "return touch(source)")),
        ("a test that is not `is not`", _GATED.replace("is not _pd_class", "is _pd_class")),
        ("a class that is not pandas", _GATED.replace('"pandas.core.frame"', '"numpy"')),
        ("a decorated def", "@always_inline\n" + _GATED),
        ("a one-line gate", _GATED.replace(":\n        return PythonObject(None)", ": return PythonObject(None)")),
    )
    for label, text in refused:
        assert text != _GATED, label
        assert ls._pandas_gate(text) is None, label


def test_a_post_gate_edit_selects_no_lane_and_its_new_helper_neither():
    """SELECTED NOWHERE: the edit is past an unchanged gate and the helper it
    adds is named only there. The kept lane is dropped too: the proof is over
    every caller."""
    old = _gate_file(_GATED, _AFTER)
    new_gated = _GATED.replace("    var columns = source.columns\n",
                               "    var columns = gate_helper(source)\n")
    assert new_gated != _GATED, "fixture edit did not apply"
    helper = "def gate_helper(source: PythonObject) raises -> PythonObject:\n    return source.columns\n"
    got = _gated_lanes(old, _gate_file(helper, new_gated, _AFTER))
    assert got is not None and got[1] == set() and got[3] == set(), got
    assert any("names_of" in label and "pandas-gated" in label for label in got[0]), got[0]
    assert any("gate_helper" in label and "pandas-gated" in label for label in got[0]), got[0]


def test_a_pandas_gate_selects_as_before_when_any_half_of_the_proof_fails():
    """SELECTED: each shape that could run the changed code in a lane keeps the
    ordinary rules (the lane reaching it, and the kept lane)."""
    old = _gate_file(_GATED, _AFTER)
    post = _GATED.replace("var columns = source.columns", "var columns = source.columns.copy()")
    cases = (
        ("no pandas-free proof", _gate_file(post, _AFTER), False),
        ("the gate line changed", _gate_file(post.replace('"DataFrame"', '"Series"'), _AFTER), True),
        ("an ungated definition changed", _gate_file(_GATED, _AFTER.replace('fields["names"]', 'fields["n"]')), True),
        ("an ungated def that calls the new helper changed too", _gate_file(
            "def gate_helper(source: PythonObject) raises -> PythonObject:\n    return source.columns\n",
            _GATED.replace("var columns = source.columns", "var columns = gate_helper(source)"),
            _CALLING_AFTER), True),
        ("_pd_class defined locally", _gate_file(
            "def _pd_class(m: String, n: String) raises -> PythonObject:\n    return PythonObject(None)\n",
            post, _AFTER), True),
    )
    for label, new, free in cases:
        got = _gated_lanes(old, new, free)
        assert got is not None and {"through", "kept"} <= got[1], f"{label}: {got}"
    # An existing helper the gated body calls, edited together with that body,
    # while an UNCHANGED ungated definition also calls it: the gate is counted
    # out, the helper is not.
    helper_v1 = "def gate_helper(source: PythonObject) raises -> PythonObject:\n    return source.columns\n"
    helper_v2 = helper_v1.replace("source.columns", "source.columns.copy()")
    gated = _GATED.replace("var columns = source.columns", "var columns = gate_helper(source)")
    after = _CALLING_AFTER
    old = _gate_file(helper_v1, gated, after)
    new = _gate_file(helper_v2, gated.replace("return columns", "return columns.copy()"), after)
    got = _gated_lanes(old, new)
    assert got is not None and {"through", "kept"} <= got[1], got
    assert any("names_of" in label and "pandas-gated" in label for label in got[0]), got[0]
    assert not any("gate_helper" in label and "pandas-gated" in label for label in got[0]), got[0]


def test_a_helper_another_file_names_is_not_counted_out():
    other = "ns_gate/other.mojo"
    text = "from ns_gate.lifecycle import gate_helper\n\ndef use(x: PythonObject) raises:\n    _ = gate_helper(x)\n"
    old = _gate_file(_GATED, _AFTER)
    new = _gate_file("def gate_helper(source: PythonObject) raises -> PythonObject:\n    return source.columns\n",
                     _GATED.replace("var columns = source.columns", "var columns = gate_helper(source)"), _AFTER)
    with _Seeded({other: (text, text)}):
        got = _gated_lanes(old, new)
    assert got is not None and "through" in got[1], got


def test_the_real_feature_names_gate_keeps_lanes_without_a_process_proof():
    """The real gate is recognized, but computed package imports (including
    _backend.py's f-string import_module calls) prevent a process-wide absence
    proof. Keep both gated-body and ordinary lifecycle changes selected."""
    path = "bindings/classifier_public_lifecycle.mojo"
    text = ls._read(path)
    units = ls._mojo_units(text)
    assert units is not None and ls._pandas_gate(units["top"]["_feature_names"][1][0]) is not None
    saved = dict(ls._PANDAS_FREE)
    ls._PANDAS_FREE.clear()
    try:
        free, why = ls.pandas_free_identity_process()
        assert not free and why, "computed package imports must not prove pandas absence"
        msg = '"TypeError: native pandas column axes are unavailable"'
        assert text.count(msg) == 1
        post = text.replace(msg, '"TypeError: native pandas column axes are missing"')
        changed, _ = ls.mojo_changed_definitions(text, post)
        assert changed == {("t", "_feature_names")}, changed
        gated, why = ls.pandas_gated_units(path, text, post, changed)
        assert gated == set(), why
        anchor = '    py_setitem(fields, "_mojolearn_fitted", True)\n'
        assert text.count(anchor) == 1
        after = text.replace(anchor, anchor + '    py_setitem(fields, "_probe", True)\n')
        changed, _ = ls.mojo_changed_definitions(text, after)
        assert changed == {("t", "classifier_after_fit")}, changed
        assert ls.pandas_gated_units(path, text, after, changed)[0] == set()
    finally:
        ls._PANDAS_FREE.clear()
        ls._PANDAS_FREE.update(saved)


def test_pandas_free_process_refuses_a_harness_or_mojo_pandas_name():
    assert ls._PANDAS_TOKENS_RE.search("import pandas as pd")
    assert not ls._PANDAS_TOKENS_RE.search("import numpy as np")
    probe = "ns_gate/importer.mojo"
    for text, bad in (('var m = Python.import_module("pandas")\n', True),
                      ('var m = Python.import_module(name)\n', True),
                      ('var m = Python.import_module("builtins")  # pandas in a comment\n', False)):
        with _Seeded({probe: (text, text)}):
            assert (ls._pandas_free_mojo(probe) is not None) is bad, text


def test_pandas_mojo_import_proof_checks_the_argument_not_other_literals():
    probe = "ns_gate/computed_import.mojo"
    refused = (
        'var m = Python.import_module("pan" + "das")\n',
        'var m = Python.import_module(names["selected"])\n',
        'var m = Python.import_module(name); print("builtins")\n',
        'var m = Python.import_module(\n    names["selected"]\n)\n',
        'var m = cpy.PyImport_ImportModule(name + "")\n',
        'var loader = Python.import_module\n',
        'var loader = table["PyImport_ImportModule"]\n',
        'var m = Python.import_module("pan\\u0064as")\n',
        'var m = Python.import_module(".child")\n',
    )
    for text in refused:
        with _Seeded({probe: (text, text)}):
            assert ls._pandas_free_mojo(probe) is not None, text
    for text in ('var m = Python.import_module("builtins")\n',
                 'var m = Python.import_module(\n    "numpy"\n)\n'):
        with _Seeded({probe: (text, text)}):
            assert ls._pandas_free_mojo(probe) is None, text


def test_pandas_python_import_proof_refuses_computed_or_indirect_names():
    probe = "ns_gate/computed_import.py"
    refused = (
        'import importlib\nm = importlib.import_module("pan" + "das")\n',
        'import importlib\nm = importlib.import_module(names["selected"])\n',
        'm = __import__(name)\n',
        'from importlib import import_module as load\nm = load(name)\n',
        'from importlib import import_module as load\nloader = load\n',
        'import importlib\nloader = importlib.import_module\n',
        'import importlib\nloader = getattr(importlib, "import_module")\n',
        'import importlib as il\nloader = getattr(il, method)\n',
        'import builtins as bi\nloader = bi.__dict__[method]\n',
        'import importlib\nm = importlib.import_module(".child", package)\n',
    )
    for text in refused:
        with _Seeded({probe: (text, text)}):
            assert ls._pandas_free_python(probe) is not None, text
    for text in ('import importlib\nm = importlib.import_module("numpy")\n',
                 'from importlib import import_module as load\nm = load("numpy")\n'):
        with _Seeded({probe: (text, text)}):
            assert ls._pandas_free_python(probe) is None, text


def test_pandas_gate_refuses_parameter_and_local_shadowing():
    bodies = []
    for name in ("_pd_class", "py_type"):
        bodies.extend((
            _GATED.replace("context: PythonObject)", "context: PythonObject, " + name + ": PythonObject)"),
            _GATED.replace("    var columns", "    var " + name + ' = context["factory"]\n    var columns'),
            _GATED.replace("    var columns", "    alias " + name + " = other_factory\n    var columns"),
            _GATED.replace("    var columns", "    (temporary, " + name + ") = factories\n    var columns"),
        ))
    bodies.append(_GATED.replace("def names_of(", "def names_of[py_type: AnyType]("))
    for before in bodies:
        assert ls._pandas_gate(before) is None, before
        after = before.replace("var columns = source.columns", "var columns = source.columns.copy()")
        got = _gated_lanes(_gate_file(before, _AFTER), _gate_file(after, _AFTER))
        assert got[1] == {"through", "kept"}, (before, got)
        assert not any("pandas-gated" in label for label in got[0]), got

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

