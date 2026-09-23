#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The harness call graph and the data-file rule of `tools/lane_select.py`
(lane/release-gpu-columns, 2026-09-22).

The 0.8.14 release-check printed FALLING BACK TO EVERY LANE because
`tools/identity_break.py` changed outside a lane body and three
`training/corpus/*/manifest.json` files changed. Both are attributed now.
Every narrowing below has an arm that MUST WIDEN beside it, run first, so a
selector that quietly narrows too far fails here rather than in a release.

    .pixi/envs/test/bin/python -m pytest tools/test_lane_select_harness.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select                                              # noqa: E402

#: A miniature harness with the shapes the real one has: a column recorder
#: (`run`) and the helper it shares, lane-only helpers, a constant one lane
#: reads, a lane that shadows the CLI's `diff` with a local, a mutator, a
#: lane-keyed declaration, a CLI with a --diff branch, and the main guard.
MINI = '''import json

TOL_A = 2
_CLOSERS = []


def shared(x):
    return x


def helper_a(x):
    return x + TOL_A


def helper_b(x):
    diff = 3
    return x + diff


def by_name(x):
    return x * 5


def diff(a, b):
    return a == b


def register(fn):
    _CLOSERS.append(fn)
    return fn


def _batch_alpha(ml, e, Xh):
    return [e]


def run(args):
    for name, fn in LANES.items():
        shared(fn(None))


def main():
    args = parse()
    if args.diff:
        return diff(1, 2)
    return run(args)


LANES = {}


@lane("alpha")
def _(ml):
    return helper_a(1)


@lane("beta")
def _(ml):
    return helper_b(2)


@lane("gamma")
def _(ml):
    return register(lambda: 1)


@lane("delta")
def _(ml):
    return getattr(sys.modules[__name__], "by_name")(1)


_batch_decl(_batch_alpha, "alpha")

if __name__ == "__main__":
    main()
'''


def answer(new, old=MINI):
    lanes, why = lane_select.harness_closure_lanes(old, new, path="<mini harness>")
    return lanes, why


# ------------------------------------------------------ must not narrow
EVERY = lane_select.EVERY_LANE
#: (label, text, expected): EVERY_LANE is an ATTRIBUTED rule (a change every
#: column runs, with its reason); None is a REFUSAL (the change cannot be
#: placed, and the release stops until it is). Neither is a narrow answer.
MUST_NOT_NARROW = (
    ("the helper the column recorder shares",
     MINI.replace("    return x\n\n\ndef helper_a", "    return -x\n\n\ndef helper_a"), EVERY),
    ("the column recorder itself", MINI.replace("shared(fn(None))", "shared(fn(1))"), EVERY),
    ("a mutator (it writes module-level state any lane may read)",
     MINI.replace("_CLOSERS.append(fn)", "_CLOSERS.insert(0, fn)"), EVERY),
    ("new import-time code", MINI.replace("LANES = {}", "LANES = {}\nprint('side effect')"), EVERY),
    ("an import", MINI.replace("import json", "import json\nimport os"), EVERY),
    ("main outside its --diff branch", MINI.replace("return run(args)", "return run(args) or 0"), EVERY),
    ("the main guard", MINI.replace("    main()\n", "    main()\n    main()\n"), EVERY),
    ("a new helper nothing references (unknown reach: refused)",
     MINI.replace("LANES = {}", "LANES = {}\n\n\ndef orphan():\n    return 1"), None),
    ("a name defined twice (ambiguous: refused)", MINI.replace("TOL_A = 2", "TOL_A = 2\nTOL_A = 3"), None),
)


def test_shared_harness_changes_select_every_lane_by_rule_or_are_refused():
    """THE ARMS THAT MUST NOT NARROW, run first."""
    for label, text, want in MUST_NOT_NARROW:
        lanes, why = answer(text)
        assert lanes == want, f"{label}: answered {lanes!r}, want {want!r} ({why})"
        assert why, f"{label}: no reason given"


def test_select_turns_the_two_answers_into_a_rule_and_a_refusal():
    """EVERY_LANE is attributed (every lane, `every_rules` says why); None is
    UNATTRIBUTED (no lane selected, the path named, the caller refuses)."""
    ref = "<armprobe ref>"
    key = f"{ref}:{lane_select.HARNESS}"
    new = lane_select._read(lane_select.HARNESS)
    for old, want_every in ((new.replace("\ndef merge(", "\ndef merge(\n    *_unused_args,", 1), True),
                            (new + "\n\nLANES = dict(LANES)\nLANES = dict(LANES)\n", False)):
        lane_select._GIT_SHOW[key] = old if want_every else new
        lane_select._read.cache[lane_select.HARNESS] = new if want_every else old
        try:
            lane_select.HARNESS_WHY.pop(key, None)
            sel = lane_select.select([lane_select.HARNESS], ref=ref)
        finally:
            lane_select._GIT_SHOW.pop(key, None)
            lane_select._read.cache.pop(lane_select.HARNESS, None)
        if want_every:
            assert lane_select.HARNESS in sel["every_rules"], sel["reasons"]
            assert len(sel["lanes"]) == sel["total"] and not sel["unattributed"]
        else:
            assert sel["unattributed"] == [lane_select.HARNESS], sel["reasons"]
            assert sel["lanes"] == []


# ------------------------------------------------------------- must narrow
def test_a_helper_one_lane_calls_selects_that_lane():
    lanes, why = answer(MINI.replace("return x + TOL_A", "return x - TOL_A"))
    assert lanes == ["alpha"], (lanes, why)
    assert "helper_a" in why


def test_a_constant_one_lane_reads_selects_that_lane():
    assert answer(MINI.replace("TOL_A = 2", "TOL_A = 7"))[0] == ["alpha"]


def test_a_lane_keyed_declaration_selects_the_lanes_it_names():
    lanes, why = answer(MINI.replace("    return [e]", "    return [e, e]"))
    assert lanes == ["alpha"], (lanes, why)
    lanes, _ = answer(MINI.replace('_batch_decl(_batch_alpha, "alpha")',
                                   '_batch_decl(_batch_alpha, "alpha", "beta")'))
    assert lanes == ["alpha", "beta"]


def test_a_local_that_shadows_a_module_name_is_not_a_reference():
    """`helper_b` binds a local `diff`; the module-level `diff` is the CLI's
    comparison. Reading the local as the module name joined the radius lanes
    of the real harness to the comparison code."""
    lanes, why = answer(MINI.replace("return a == b", "return a != b"))
    assert lanes == [], (lanes, why)
    assert "--diff" in why


def test_main_changed_only_in_its_diff_branch_selects_nothing():
    lanes, why = answer(MINI.replace("return diff(1, 2)", "return diff(2, 1)"))
    assert lanes == [], (lanes, why)


def test_a_name_looked_up_by_string_is_still_a_reference():
    lanes, why = answer(MINI.replace("return x * 5", "return x * 6"))
    assert lanes == ["delta"], (lanes, why)


def test_a_lane_body_selects_itself():
    assert answer(MINI.replace("return helper_b(2)", "return helper_b(3)"))[0] == ["beta"]


# ---------------------------------------------------- the real harness
def _real_edit(fn):
    """The real harness with one line added to `fn`'s body, as the OLD side,
    so the working tree (which the imported harness matches) is the new."""
    text = lane_select._read(lane_select.HARNESS)
    i = text.index(f"\ndef {fn}(")
    j = text.index("\n", i + 1)
    return text[:j + 1] + "    _unused = 0\n" + text[j + 1:], text


def test_the_real_harness_places_loop_registered_lanes():
    """`_kde_lane` builds the kde variants inside a registration loop over
    computed names. The imported harness places them, so a change to the
    factory selects exactly those lanes and not every lane."""
    old, new = _real_edit("_kde_lane")
    lanes, why = lane_select.harness_closure_lanes(old, new, path=lane_select.HARNESS)
    assert lanes is not None, why
    assert lanes and all(n.startswith("kde-") for n in lanes), lanes
    assert "kmeans" not in lanes


def test_the_real_harness_widens_on_what_every_column_runs():
    old, new = _real_edit("merge")
    lanes, why = lane_select.harness_closure_lanes(old, new, path=lane_select.HARNESS)
    assert lanes == lane_select.EVERY_LANE, (lanes, why)


def test_select_prints_the_call_graph_reason():
    old, new = _real_edit("_radius_for")
    ref = "<armprobe ref>"
    key = f"{ref}:{lane_select.HARNESS}"
    lane_select._GIT_SHOW[key] = old
    try:
        sel = lane_select.select([lane_select.HARNESS], ref=ref)
    finally:
        lane_select._GIT_SHOW.pop(key, None)
    assert not sel["unattributed"] and not sel["every_rules"], sel["reasons"]
    assert sel["lanes"] and all("radius" in n for n in sel["lanes"]), sel["lanes"]
    assert "_radius_for" in sel["reasons"][lane_select.HARNESS]


# ------------------------------------------------------------ data files
def test_a_data_file_whose_directory_a_lane_names_still_widens():
    """THE ARM THAT MUST WIDEN: a directory named by code a lane
    reaches (mamba/corpus/mamba2/ is), so a manifest there is not dismissed for its generic name."""
    path = next(p for p in sorted(lane_select.tracked_files())
                if p.startswith("mamba/corpus/mamba2/") and p.endswith("/manifest.json"))
    assert lane_select._generic_data_name(path)
    assert lane_select.unreachable(path) is None, path


def test_a_training_corpus_manifest_selects_nothing_and_says_why():
    paths = sorted(p for p in lane_select.tracked_files()
                   if p.startswith("training/corpus/") and p.endswith("/manifest.json"))
    assert paths
    sel = lane_select.select(paths)
    assert not sel["unattributed"] and sel["lanes"] == [], sel["reasons"]
    assert all("data file whose name" in sel["reasons"][p] for p in paths)


def test_a_def_named_like_a_tool_is_not_an_import_of_it():
    """`_buffer.py` defines `release()`; that is not a load of tools/release.py."""
    sel = lane_select.select(["tools/release.py"])
    assert not sel["unattributed"] and sel["lanes"] == [], sel["reasons"]


def test_the_linux_set_builder_is_every_lane_for_linux_and_inert_on_the_mac():
    path = "packaging/linux/build_sets.sh"
    for backend in (None, "cuda", "hip"):
        sel = lane_select.select([path], backend=backend)
        assert path in sel["every_rules"] and len(sel["lanes"]) == sel["total"], (backend, sel["reasons"])
    if sys.platform == "darwin":
        sel = lane_select.select([path], backend="metal")
        assert not sel["unattributed"] and sel["lanes"] == [], sel["reasons"]
    # the rest of packaging/linux/ packs a wheel no column runs from
    sel = lane_select.select(["packaging/linux/pack_wheel.py"], backend="cuda")
    assert not sel["unattributed"], sel["reasons"]


def test_a_one_kernel_change_selects_only_its_lanes():
    sel = lane_select.select(["cluster/host/kmeans_oracle.mojo"])
    assert not sel["unattributed"] and not sel["every_rules"]
    assert "kmeans" in sel["lanes"] and "ols" not in sel["lanes"] and len(sel["lanes"]) < 40, sel["lanes"]
    assert sel["by_path"]["cluster/host/kmeans_oracle.mojo"] == sel["lanes"]


def test_an_inert_doc_selects_nothing():
    sel = lane_select.select(["docs/RELEASE_CHECKLIST.md"])
    assert sel["lanes"] == [] and not sel["unattributed"] and not sel["every_rules"]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)


# ------------------------------------------------- a binding that only grew
BINDING_BASE = '''from core.x import helper


def a_binding(x: PythonObject) raises -> PythonObject:
    return helper(x)


def PyInit__mojolearn_armprobe() abi("C") -> PythonObject:
    module.def_function[a_binding]("a")
    return module
'''


def _additions(new, old=BINDING_BASE):
    ref, path = "<armprobe ref>", "bindings/_mojolearn_armprobe.mojo"
    lane_select._GIT_SHOW[f"{ref}:{path}"] = old
    lane_select._read.cache[path] = new
    try:
        return lane_select.binding_additions(ref, path)
    finally:
        lane_select._GIT_SHOW.pop(f"{ref}:{path}", None)
        lane_select._read.cache.pop(path, None)


GROWN = BINDING_BASE.replace(
    '\n\ndef PyInit_', '\n\ndef b_binding(x: PythonObject) raises -> PythonObject:\n    return x\n\n\ndef PyInit_').replace(
    '    module.def_function[a_binding]("a")\n', '    module.def_function[a_binding]("a")\n    module.def_function[b_binding]("b")\n')


def test_a_binding_that_only_gains_exports_is_attributed_to_its_users():
    """MUST NOT NARROW, first: an edited export, an edited import, a new
    block named like something old, and a registration of an OLD impl."""
    assert _additions(GROWN.replace("return helper(x)", "return helper(x + 1)")) is None
    assert _additions(GROWN.replace("from core.x import helper", "from core.y import helper")) is None
    assert _additions(GROWN.replace("b_binding", "helper")) is None, "a new block shadowing an old name"
    assert _additions(BINDING_BASE.replace('("a")\n', '("a")\n    module.def_function[a_binding]("a2")\n')) is None
    assert _additions(GROWN) == {"b"}


def test_the_real_byte_lm_growth_selects_its_users_not_every_lane():
    """v0.8.14 -> 0.8.15 added five parallel-training exports to the byte LM
    binding, which every lane LOADS through the buffer helpers; the lanes
    that really use it are the byte-LM ones."""
    base = lane_select._read("bindings/_mojolearn_byte_lm.mojo")
    blocks = lane_select._mojo_blocks(base)
    name = next(n for n in blocks if n.startswith("byte_lm_") and n.endswith("_binding"))
    grown = base.replace(f"\ndef {name}(", f"\ndef armprobe_new_binding(x: PythonObject) raises -> PythonObject:\n"
                         f"    return x\n\n\ndef {name}(", 1)
    ref, path = "<armprobe ref>", "bindings/_mojolearn_byte_lm.mojo"
    lane_select._GIT_SHOW[f"{ref}:{path}"] = base
    lane_select._read.cache[path] = grown
    try:
        sel = lane_select.select([path], ref=ref)
    finally:
        lane_select._GIT_SHOW.pop(f"{ref}:{path}", None)
        lane_select._read.cache[path] = base
    assert not sel["unattributed"] and "only GAINED exports" in sel["reasons"][path], sel["reasons"]
    assert 0 < len(sel["lanes"]) < 40 and "byte-lm" in sel["lanes"] and "ols" not in sel["lanes"], sel["lanes"]
