#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What `tools/lane_select.py` must never do (lane/lane-selector, 2026-09-16).

A selector is only worth having if it fails LOUDLY. The failure that would
matter here is the quiet one: an empty or narrow selection for a change that
really does move a lane, which makes every change look verified while
checking nothing. Every test below is written so that it FAILS if the
selector goes quiet, not merely if it errors.

    .pixi/envs/test/bin/python -m pytest tools/test_lane_select.py -q
    python3 tools/test_lane_select.py          # same checks, no pytest
"""
import ast
import os
import re
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select                                              # noqa: E402

def _refuse_if_manifest_dirty():
    """REFUSE TO RUN OVER SOMEBODY ELSE'S UNCOMMITTED WORK (2026-09-16).

    Three tests in this file WRITE the manifest in the tree and restore it in
    a `finally`. That is safe in a private worktree and unsafe in the shared
    checkout, where several lanes edited this exact file on one afternoon: a
    crash between the write and the restore, or a concurrent editor, loses
    work that was never committed. This repository has already lost a lane's
    uncommitted change that way once. A dirty manifest means somebody is
    mid-edit, so stop rather than race.
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", lane_select.MANIFEST],
        cwd=lane_select.ROOT, capture_output=True, text=True,
    ).stdout.strip()
    if dirty and not os.environ.get("MOJOLEARN_LANE_SELECT_TEST_FORCE"):
        raise AssertionError(
            "REFUSING: " + lane_select.MANIFEST + " has uncommitted changes (" + dirty + ").\n"
            "  These tests rewrite that file and would race whoever is editing it.\n"
            "  Run them in your own worktree, or commit first. "
            "MOJOLEARN_LANE_SELECT_TEST_FORCE=1 overrides."
        )


#: A path every lane in the family depends on, and a lane that must NOT be
#: selected by it. The second half is the half that catches a selector which
#: has quietly widened to "everything" and so looks right by accident.
#:
#: EVERY ENTRY MUST BE A FILE THAT EXISTS. The third case read `mamba/impl`
#: until 2026-09-16, which is a DIRECTORY: the map could not attribute it, the
#: selection fell back to every lane, `must in lanes` passed for that reason
#: alone and the narrowing half was skipped. The case asserted nothing. Both
#: guards below exist because of it: the paths are checked against the tree,
#: and a fallback now FAILS the case instead of satisfying it.
FAMILY_CASES = (
    ("glm/host/qn_oracle.mojo", "logistic", ("mamba1", "transformer", "tokenizer")),
    ("cluster/host/kmeans_oracle.mojo", "kmeans", ("mamba1", "transformer", "arima")),
    ("mamba/impl/modeling/modeling_mamba.mojo", "mamba1", ("ols", "kmeans", "tokenizer")),
)


def test_every_family_case_names_a_file_that_exists():
    """A case pointed at a directory or a deleted file falls back to every
    lane and then passes vacuously, which is the shape of a check that cannot
    fail."""
    for path, _, _ in FAMILY_CASES:
        full = os.path.join(lane_select.ROOT, path)
        assert os.path.isfile(full), f"FAMILY_CASES names {path}, which is not a file"


#: The families whose lanes are built by a registration LOOP rather than by a
#: `@lane(...)` decorator. They are why `grep -c '@lane('` undercounts, and
#: naming them here is what lets the next test fail when the undercount stops
#: being the one we think it is.
REGISTERED_BY_CALL = ("kde-", "knn-", "radius-", "gp-", "gmm-")


def test_registry_count_comes_from_the_import():
    """THE COUNT IS NOT A NUMBER TO MAINTAIN. One afternoon produced four lane
    totals (176, 192, 199, 210) that each claimed to be it, so this test pins
    the PROPERTY instead: the registry read by import is strictly larger than
    the decorator grep, and every lane in the gap belongs to a family that
    registers by call.

    A hard-coded total would rot the first time a lane landed. It already did:
    this file was written when the answer was 199 against 176, and four
    commits later it is 211 against 188, with the SAME 23 lanes in the gap."""
    lanes = lane_select.all_lanes()
    assert len(lanes) == len(set(lanes)), "a lane name is registered twice"

    harness = os.path.join(lane_select.ROOT, lane_select.HARNESS)
    text = open(harness, encoding="utf-8").read()
    by_decorator = set(re.findall(r'@lane\(\s*"([A-Za-z0-9_.\-]+)"', text))
    assert by_decorator, "the decorator regex matched nothing: it, not the registry, is broken"

    gap = [n for n in lanes if n not in by_decorator]
    assert gap, ("no lane registers by call any more. If the kde/knn/radius/gp/gmm loops were "
                 "rewritten as decorators that is fine, but this test and the docs that quote "
                 "it must be rewritten with them.")
    assert len(lanes) > len(by_decorator), (
        f"the import found {len(lanes)} lanes and the grep {len(by_decorator)}: "
        "the registry was read by grep, not by import")
    stray = [n for n in gap if not n.startswith(REGISTERED_BY_CALL)]
    assert not stray, (f"{len(stray)} lane(s) are missing from the decorator grep and are not in a "
                       f"family known to register by call: {stray[:10]}")


def test_every_lane_maps_to_real_source():
    """A lane with an empty map selects nothing for a change to its own code."""
    sources, _ = lane_select.lane_sources()
    empty = [lane for lane, files in sources.items()
             if not any(f.endswith(".mojo") for f in files)
             or not any(f.endswith(".py") for f in files)]
    assert not empty, f"lanes with an empty side of the map: {empty[:10]}"


def test_a_lane_is_selected_by_its_own_dependencies():
    """THE ROUND TRIP. Every file the map gives a lane must select that lane
    back. This is the property that would break first if the map drifted."""
    sources, _ = lane_select.lane_sources()
    rev = lane_select.reverse_map(sources)
    for lane in sorted(sources)[::17]:                  # a spread across the registry
        for rel in sorted(sources[lane])[:40]:
            assert lane in rev.get(rel, ()), f"{rel} does not select {lane} back"


def test_family_path_selects_its_lane_and_not_the_others():
    for path, must, must_not in FAMILY_CASES:
        sel = lane_select.select([path])
        assert not sel["unattributed"] and not sel["every_rules"], \
            f"{path} was not attributed to its lanes, so this case proves nothing: {sel['reasons'][path]}"
        assert must in sel["lanes"], f"{path} did not select {must}"
        for other in must_not:
            assert other not in sel["lanes"], \
                f"{path} selected the unrelated lane {other}; the map is not narrowing"


def test_inert_paths_select_nothing_and_do_not_pretend_otherwise():
    sel = lane_select.select(["docs/START_HERE.md", "CHANGELOG.md"])
    assert sel["lanes"] == [], "a prose change selected lanes"
    assert not sel["unattributed"], "a prose change was refused"


def test_several_paths_in_one_argument_are_never_inert():
    """THE SILENT ZERO, caught on 2026-09-16. zsh does not word-split an
    unquoted variable, so a whole commit's file list arrived as ONE argument;
    it began with CHANGELOG.md, matched the inert prefix rule, and twelve
    changed files selected no lanes at all and said nothing. An argument that
    is not a single path is REFUSED by name (2026-09-22: never widened)."""
    blob = "CHANGELOG.md cluster/host/kmeans_oracle.mojo tools/identity_break.py"
    sel = lane_select.select([blob])
    assert sel["unattributed"] == [blob], "a multi-path argument was not refused"
    assert sel["lanes"] == [], "a refused path selected lanes"
    assert "NOT A SINGLE PATH" in sel["reasons"][blob]


#: A file that MENTIONS a binding without calling into it, in the two ways
#: this codebase mentions things: a docstring and a comment. It is fed to
#: `_binding_names` through the read cache, so the check does not depend on
#: which file in the tree happens to be written this way today.
PROSE_ONLY_SOURCE = '''"""This door explains itself at length.

It talks about _mojolearn_forest_host, because the forest host binding is
what the reader will ask about next. It does not load it.
"""
# _mojolearn_byte_lm_host lives next door and is not called here either.
from ._array import Array


def fit(x):
    return Array(x)
'''

#: The same names, RESOLVED, in each of the four shapes the selector reads.
RESOLVING_SOURCE = '''from . import _mojolearn_tokenizer_host
_BINDING = "_mojolearn_forest_host"


def go(be):
    be.binding("_mojolearn_byte_lm_host")
    return _backend.load_host_module("_mojolearn_core_host")
'''


def test_binding_edges_do_not_come_from_prose():
    """A SENTENCE ABOUT A BINDING IS NOT A CALL INTO IT.

    Harvesting `_mojolearn_*` by text search matched docstrings and comments,
    every lane picked up the forest and byte LM host bindings, hit their
    whole-closure fallback, and one oracle selected every lane. Edges come
    from syntax now.

    Both arms run. The text search is asked the same question FIRST and must
    FIND the names, because a check whose fixture no longer contains the thing
    it is looking for passes for the wrong reason and is indistinguishable
    from a real pass."""
    lane_select._read.cache["<prose only>"] = PROSE_ONLY_SOURCE
    lane_select._read.cache["<resolving>"] = RESOLVING_SOURCE
    try:
        lane_select._binding_names.cache.pop("<prose only>", None)
        lane_select._binding_names.cache.pop("<resolving>", None)
        lane_select._parse.cache.pop("<prose only>", None)
        lane_select._parse.cache.pop("<resolving>", None)

        # THE ARM THAT MUST FIND THEM. If this is empty the fixture is broken,
        # not the selector.
        by_text = sorted(set(lane_select._BINDING_RE.findall(PROSE_ONLY_SOURCE)))
        assert by_text == ["_mojolearn_byte_lm_host", "_mojolearn_forest_host"], (
            f"the prose fixture no longer mentions the bindings by text ({by_text}), "
            "so the next assertion would pass for the wrong reason")

        from_prose = sorted(lane_select._binding_names("<prose only>"))
        assert from_prose == [], f"a docstring and a comment produced binding edges: {from_prose}"

        resolved = sorted(lane_select._binding_names("<resolving>"))
        assert resolved == ["_mojolearn_byte_lm_host", "_mojolearn_core_host",
                            "_mojolearn_forest_host", "_mojolearn_tokenizer_host"], (
            f"a binding that IS resolved was missed: {resolved}. An empty answer for every "
            "file would satisfy the assertion above while making the whole map blind.")
    finally:
        for cache in (lane_select._read.cache, lane_select._parse.cache,
                      lane_select._binding_names.cache):
            cache.pop("<prose only>", None)
            cache.pop("<resolving>", None)


def test_a_binding_every_lane_reaches_does_not_hand_over_its_tree():
    """THE SECOND ROAD INTO THE SAME DEFECT, found 2026-09-16 on the merged
    tree. Making binding edges come from syntax stopped PROSE giving every
    lane the forest and byte LM bindings. It did not stop the IMPORT CLOSURE
    doing it: `_bufcheck.py` -> `_buffer.py` is in every lane's closure and
    reaches `_forest_host.py`, `_byte_lm_impl.py` and `_byte_lm_host.py`, so
    all three bindings landed on all 211 lanes and each handed over its whole
    Mojo tree. `core/forest_host_predict.mojo` and the mamba modeling file
    still selected every lane, which is the symptom the earlier fix was
    written to remove.

    A binding everything reaches is not evidence about one lane, so it now
    contributes its SOURCE to a lane the CPU manifest does not declare for its
    family, and the whole binding to one the manifest does. This test pins the
    consequence, deriving who is declared from the manifest rather than
    listing lanes here."""
    sources, why = lane_select.lane_sources()
    hs = lane_select.host_surface()
    declared = {f["family"]: set(f["training_lanes"]) | set(f["inference_lanes"])
                for f in hs.FAMILIES}
    rel, family = "training/byte_lm.mojo", "byte_lm"
    assert family in declared, "the manifest no longer has a byte_lm family"
    carriers = {lane for lane, files in sources.items() if rel in files}
    assert carriers, f"{rel} is carried by no lane at all, which is the opposite failure"
    strays = sorted(carriers - declared[family])
    assert not strays, (f"{rel} belongs to the {family} family but {len(strays)} lane(s) the "
                        f"manifest does not declare for it carry it: {strays[:10]}")

    # The two files that still selected every lane after the syntax fix. They
    # are shared across families (`core/forest_host_predict.mojo` serves the
    # extratrees lanes as well as the forest ones), so the check is the one
    # FAMILY_CASES uses: a lane with no relationship to them at all.
    for path in ("core/forest_host_predict.mojo", "mamba/impl/modeling/modeling_mamba.mojo",
                 "core/gbdt_host_predict.mojo"):
        sel = lane_select.select([path])
        assert not sel["unattributed"], f"{path} is no longer attributable: {sel['reasons'][path]}"
        for other in ("ols", "kmeans", "arima"):
            assert other not in sel["lanes"], \
                f"{path} still selects {other}; the whole-closure fallback is back"


def test_a_bindings_edge_always_has_a_door_that_resolves_it():
    """THE ROUND TRIP FOR BINDINGS. Every binding the map gives a lane must be
    resolved, by syntax, by one of that lane's own non-registry doors. A
    binding that appears without such a door came from somewhere this file
    does not know about, which is how the text search got in.

    This replaces an assertion that `tokenizer` never sees the forest host
    binding. It does see it, and legitimately: `_bufcheck.py` -> `_buffer.py`,
    whose line 807 is `from . import _forest_host`. That widens the tokenizer
    lane, which is the safe direction, and the narrowing this file has to
    protect is covered by FAMILY_CASES."""
    sources, why = lane_select.lane_sources()
    sinks = lane_select.enumerator_files()
    for lane in sorted(sources)[::13]:
        doors = [f for f in sources[lane] if f.endswith(".py") and f not in sinks]
        resolved = set()
        for rel in doors:
            resolved |= lane_select._binding_names(rel)
        for binding in why[lane]["bindings"]:
            assert binding in resolved, \
                f"{lane} carries {binding} but no door of its own resolves it"


def test_an_unattributable_path_is_refused_by_name_and_selects_nothing():
    """NO FALLBACK (Andrew, 2026-09-22). A path the map cannot place selects
    NOTHING, is named in `unattributed`, and every caller refuses to run: the
    CLI exits non-zero printing UNATTRIBUTED PATH. It used to widen to every
    lane, which made a narrow run and a full sweep look alike."""
    # The harness with no ref to diff against cannot be placed by its call
    # graph: the plainest unattributable path the tree has.
    sel = lane_select.select(["tools/identity_break.py"], ref=None)
    assert sel["unattributed"] == ["tools/identity_break.py"], sel["reasons"]
    assert sel["lanes"] == [] and not sel["fallback"]
    lines = []
    assert lane_select.refuse_unattributed(sel, out=lines.append)
    assert lines[0].startswith("UNATTRIBUTED PATH: tools/identity_break.py")
    assert any(ln.startswith("NOTHING IS VERIFIED") for ln in lines)
    rc = subprocess.run([sys.executable, os.path.join(lane_select.ROOT, "tools", "lane_select.py"),
                         "--lanes-for-paths", "tools/identity_break.py"], capture_output=True, text=True)
    assert rc.returncode != 0 and "UNATTRIBUTED PATH: tools/identity_break.py" in rc.stdout, rc.stdout[-800:]


def test_the_pinned_toolchain_selects_every_lane_by_rule_not_by_fallback():
    sel = lane_select.select(["pixi.lock"])
    assert not sel["unattributed"]
    assert len(sel["lanes"]) == len(lane_select.all_lanes())
    assert "pixi.lock" in sel["every_rules"] and "toolchain" in sel["every_rules"]["pixi.lock"]


def test_a_registry_change_selects_every_lane():
    """`_backend.py` and `host_surface.py` name the whole binding surface, so
    the map drops their per-lane edges. That debt is paid here: changing one
    selects everything."""
    for path in ("python/mojolearn/_backend.py", "python/mojolearn/host_surface.py"):
        sel = lane_select.select([path])
        assert path in sel["every_rules"], f"{path} did not select every lane by rule"
        assert len(sel["lanes"]) == len(lane_select.all_lanes())


def test_the_selector_is_not_imported_by_what_it_selects():
    """SELECTION_MACHINERY is only safe while nothing under test imports it.
    The moment `identity_break.py` or the package imports one of these files,
    a change to it CAN move a lane's bits and calling it inert would be the
    silent pass this whole file is about."""
    watched = [os.path.join(lane_select.ROOT, lane_select.HARNESS)]
    pkg = os.path.join(lane_select.ROOT, lane_select.PKG)
    watched += [os.path.join(pkg, n) for n in sorted(os.listdir(pkg)) if n.endswith(".py")]
    stems = [os.path.basename(p)[:-3] for p in lane_select.SELECTION_MACHINERY]
    for path in watched:
        text = open(path, encoding="utf-8", errors="replace").read()
        for stem in stems:
            hits = [line.strip() for line in text.splitlines()
                    if re.search(r"^\s*(import|from)\s+%s\b" % re.escape(stem), line)]
            assert not hits, f"{path} imports {stem}: {hits[:3]}"


def test_the_runner_refuses_an_empty_selection():
    """THE REFUSAL, RUN. An empty selection must not exit 0. This calls
    `verify_lanes.main` rather than reading it, and the second arm runs the
    SAME command on a path that does select lanes, so a refusal that fired for
    an unrelated reason (a bad argument, a missing file) would show up as both
    arms refusing."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import verify_lanes

    empty = verify_lanes.main(["--lanes-for-paths", "CHANGELOG.md", "--plan"])
    assert empty == 2, f"an empty selection exited {empty}, not the refusal"

    narrow = verify_lanes.main(["--lanes-for-paths", "cluster/host/kmeans_oracle.mojo", "--plan"])
    assert narrow == 0, (f"a non-empty selection also exited {narrow}: the refusal above did not "
                         "come from the selection being empty")


def test_shards_are_the_lane_set_and_are_deterministic():
    lanes = lane_select.all_lanes()
    first, _ = lane_select.shard(lanes, 8)
    second, _ = lane_select.shard(lanes, 8)
    assert first == second, "the same lane set produced different shards"
    assert sorted(sum(first, [])) == sorted(lanes), "the shards are not the lane set"
    assert len(first) == 8


def test_shard_count_never_exceeds_the_lane_count():
    groups, _ = lane_select.shard(["ols", "ridge"], 16)
    assert sorted(sum(groups, [])) == ["ols", "ridge"]


#: Pairs that differ in DOCSTRINGS AND COMMENTS ONLY. `code_dump` must call
#: these equal.
SAME_CODE = (
    ("a comment", '# one\nx = 1\n', '# two\nx = 1\n'),
    ("a module docstring", '"""old."""\nx = 1\n', '"""new, and longer."""\nx = 1\n'),
    ("a docstring added", 'def f():\n    return 1\n', 'def f():\n    """added."""\n    return 1\n'),
    ("a class docstring", 'class C:\n    """a."""\n    x = 1\n', 'class C:\n    """b."""\n    x = 1\n'),
    ("blank lines", 'x = 1\ny = 2\n', 'x = 1\n\n\ny = 2\n'),
    ("a docstring that was the whole body",
     'def f():\n    """only this."""\n', 'def f():\n    pass\n'),
)

#: Pairs that differ in CODE. `code_dump` must call these different. This is
#: the half that matters: a stripper that is subtly too greedy turns a real
#: change into "nothing affected", which is far worse than an extra sweep.
DIFFERENT_CODE = (
    ("a string that is not a docstring",
     'def f():\n    """d."""\n    raise ValueError("old message")\n',
     'def f():\n    """d."""\n    raise ValueError("new message")\n'),
    ("a constant", 'TOL = 1\n', 'TOL = 2\n'),
    ("an operator", 'def f(a, b):\n    return a + b\n', 'def f(a, b):\n    return a - b\n'),
    ("a docstring AND an operator",
     'def f(a, b):\n    """old."""\n    return a + b\n',
     'def f(a, b):\n    """new."""\n    return a - b\n'),
    ("a string moved out of the docstring position",
     'def f():\n    """d."""\n    x = 1\n', 'def f():\n    x = 1\n    """d."""\n'),
    ("a bare string used as a value",
     'MESSAGE = "old"\n', 'MESSAGE = "new"\n'),
    ("an f-string opening a function, which is not a docstring",
     'def f(n):\n    f"{n}"\n    return n\n', 'def f(n):\n    f"{n}!"\n    return n\n'),
)


def test_docstring_stripping_keeps_every_real_change():
    """BOTH ARMS, and the different-code arm first. A comparison that called
    everything equal would satisfy the same-code arm on its own."""
    for label, old, new in DIFFERENT_CODE:
        a, b = lane_select.code_dump(old), lane_select.code_dump(new)
        assert a is not None and b is not None, f"{label}: a fixture does not parse"
        assert a != b, f"{label}: a CODE change was called docstring-only"
    for label, old, new in SAME_CODE:
        a, b = lane_select.code_dump(old), lane_select.code_dump(new)
        assert a is not None and b is not None, f"{label}: a fixture does not parse"
        assert a == b, f"{label}: a docstring or comment change was called a code change"
    assert lane_select.code_dump("def f(:\n") is None, "a file that does not parse must not compare"


def test_a_source_that_is_hashed_at_runtime_is_never_docstring_only():
    """`_byte_lm_impl.py` publishes `source_sha256` over six named sources and
    over itself, so a COMMENT in one of them moves a recorded value. Those
    files are derived from the modules that hash a file, not listed."""
    hashed = lane_select.source_hashed_files()
    assert "python/mojolearn/_byte_lm_impl.py" in hashed, "the module that hashes was not found"
    for rel in ("python/mojolearn/language_model.py", "bindings/_mojolearn_byte_lm.mojo",
                "training/byte_lm.mojo", "python/mojolearn/_byte_lm_config.py"):
        assert rel in hashed, f"{rel} is hashed by _binding_metadata but is not in the derived set"
    for rel in hashed:
        assert not lane_select.docstring_only("HEAD", rel), \
            f"{rel} is hashed at run time and must never be exempt"
    # and a module that hashes ARRAYS, not files, is not swept in
    assert "python/mojolearn/model_selection.py" not in hashed


#: A miniature `identity_break.py`: a constant, a helper, two lanes, and a
#: registration loop, in that order.
HARNESS_BASE = '''TOL = 1


def helper(x):
    return x + TOL


@lane("alpha")
def _alpha():
    return helper(1)


@lane("beta")
def _beta():
    return helper(2)


for name in ("gamma-1", "gamma-2"):
    lane(name)(lambda: helper(3))
'''

NEW_LANE_IN_THE_MIDDLE = HARNESS_BASE.replace(
    '@lane("beta")', '@lane("delta")\ndef _delta():\n    return helper(9)\n\n\n@lane("beta")')

#: Each must return None, meaning every lane. Ordered failure first.
HARNESS_MUST_FALL_BACK = (
    ("the helper changed", HARNESS_BASE.replace("return x + TOL", "return x - TOL")),
    ("a constant changed", HARNESS_BASE.replace("TOL = 1", "TOL = 2")),
    ("the registration loop changed", HARNESS_BASE.replace('"gamma-2"', '"gamma-3"')),
    ("a helper was deleted", HARNESS_BASE.replace("def helper(x):\n    return x + TOL", "pass")),
    ("a new lane AND a new module-level assignment",
     NEW_LANE_IN_THE_MIDDLE.replace("TOL = 1", "TOL = 1\nEXTRA = 2")),
    ("a new lane AND a new helper that shadows a name an existing lane calls",
     NEW_LANE_IN_THE_MIDDLE.replace("TOL = 1", "TOL = 1\n\n\ndef helper(x):\n    return 0")),
    ("a new lane AND a new class, whose body runs when it is defined",
     NEW_LANE_IN_THE_MIDDLE.replace("TOL = 1", "TOL = 1\n\n\nclass Extra:\n    v = 1")),
    ("a new lane AND a new helper with a non-constant default",
     NEW_LANE_IN_THE_MIDDLE.replace(
         "TOL = 1", "TOL = 1\n\n\ndef fresh(x=helper(1)):\n    return x")),
)


def _harness_answer(new_text, old_text=HARNESS_BASE):
    """`harness_lanes` over two texts, with no file and no commit involved."""
    ref, path = "<fake ref>", "<fake harness>"
    lane_select._GIT_SHOW[f"{ref}:{path}"] = old_text
    lane_select._read.cache[path] = new_text
    try:
        return lane_select.harness_lanes(ref, path)
    finally:
        lane_select._GIT_SHOW.pop(f"{ref}:{path}", None)
        lane_select._read.cache.pop(path, None)


def test_the_harness_falls_back_on_anything_that_is_not_a_lane_body():
    """THE ARM THAT MUST FAIL, run first. Every one of these reaches beyond
    the lane it looks like it edits, so none may be narrowed: each answers
    every lane BY RULE (EVERY_LANE, a change every column runs) or is REFUSED
    (None: the miniature's registration loop cannot be placed without the
    imported harness)."""
    for label, text in HARNESS_MUST_FALL_BACK:
        assert _harness_answer(text) in (None, lane_select.EVERY_LANE), \
            f"{label}: the harness narrowed a change that can reach any lane"


def test_an_added_lane_selects_only_that_lane():
    """The defect lane/data-ordering-determinism hit on 2026-09-16: ONE
    additive hunk, no existing lane body touched, and the selector answered
    212 of 212. The old spelling keyed a bare top-level statement by its LINE
    NUMBER, so inserting a lane renumbered everything below it and every key
    changed. The keys are AST dumps now, which have no positions."""
    assert _harness_answer(NEW_LANE_IN_THE_MIDDLE) == ["delta"]

    # the same insertion with a helper that nothing existing names
    with_helper = NEW_LANE_IN_THE_MIDDLE.replace(
        "TOL = 1", "TOL = 1\n\n\ndef fresh(x):\n    return x")
    assert _harness_answer(with_helper) == ["delta"]

    # one existing body edited
    assert _harness_answer(HARNESS_BASE.replace("return helper(1)", "return helper(11)")) == ["alpha"]

    # a comment and a docstring in the harness reach no lane at all
    commented = HARNESS_BASE.replace("def _alpha():", 'def _alpha():\n    """what alpha does."""')
    assert _harness_answer("# a new comment\n" + commented) == []


def test_the_new_narrow_answers_are_narrow_for_the_right_reason():
    """A narrow answer has to name what made it narrow. These two rules are
    the only ones that may call a non-prose path inert, and each prints its
    own sentence."""
    ref = "HEAD"
    sel = lane_select.select(["tools/identity_break.py"], ref=ref)
    assert "harness diff touches only these lane bodies" in sel["reasons"]["tools/identity_break.py"] \
        or "docstrings and comments only" in sel["reasons"]["tools/identity_break.py"], \
        f"unexpected reason: {sel['reasons']['tools/identity_break.py']}"
    assert not sel["unattributed"], "the harness against its own HEAD was refused"


#: A path whose blast radius is empty, and the one thing that would change
#: that. `unreachable` is the rule that saved lane/umap-batch-determinism and
#: lane/discarded-atomic-audit from a 212-lane sweep.
def test_a_path_nothing_reaches_selects_nothing():
    """A file no lane's map contains and that nothing a lane reaches NAMES
    cannot move a cell. Written in a temporary directory inside the tree so
    both halves are exercised for real."""
    probe_dir = os.path.join(lane_select.ROOT, "armprobedir")
    src = "armprobedir/armprobesource.mojo"
    fixture = "armprobedir/armprobefixture.bin"
    _refuse_if_manifest_dirty()
    manifest = os.path.join(lane_select.ROOT, lane_select.MANIFEST)
    before = open(manifest, encoding="utf-8").read()
    os.makedirs(probe_dir, exist_ok=True)
    try:
        for rel in (src, fixture):
            with open(os.path.join(lane_select.ROOT, rel), "w") as fh:
                fh.write("# probe\n")
        lane_select.reset_caches()
        for rel in (src, fixture):
            assert lane_select.unreachable(rel), f"{rel} should be unreachable and is not"

        # THE ARM THAT MUST FALL BACK. A file the map DOES contain now names the
        # source by path and the DIRECTORY by a glob. The fixture is named by
        # nothing but that glob, which is the case a data file a lane reads
        # sits in.
        with open(manifest, "w") as fh:
            fh.write(before + f"\n_ARM_PROBE_SOURCE = {src!r}\n_ARM_PROBE_GLOB = 'armprobedir/' + '*'\n")
        lane_select.reset_caches()
        for rel in (src, fixture):
            assert not lane_select.unreachable(rel), \
                f"{rel} is named by a file every lane reaches and was still called unreachable"
    finally:
        with open(manifest, "w") as fh:
            fh.write(before)
        for rel in (src, fixture):
            path = os.path.join(lane_select.ROOT, rel)
            if os.path.exists(path):
                os.unlink(path)
        if os.path.isdir(probe_dir) and not os.listdir(probe_dir):
            os.rmdir(probe_dir)
        lane_select.reset_caches()


def test_a_test_module_is_inert_unless_something_a_lane_reaches_names_it():
    """A test cannot change what a lane computes, which is why
    `_python_files()` leaves `python/mojolearn/tests/` out of the map. The
    selector never acted on that, so every test file read NOT ATTRIBUTABLE and
    sent its lane to the full sweep.

    ONLY FILES A LANE REACHES VOTE (2026-09-21). A pixi task or a tool under
    tools/ that imports a test module does not put it into any lane's process,
    so `tools/transformer_transfer_check.py` importing
    `test_transformer_surface._weights` no longer keeps that module out. THE
    ARM THAT MUST DECLINE is synthesised in a file every lane reaches (the
    manifest), both as a plain name and as a dynamic import string, and in the
    tree `test_tokenizer_surface` is named by the manifest's own gate text."""
    tests = os.path.join(lane_select.ROOT, "python", "mojolearn", "tests")
    names = sorted(n for n in os.listdir(tests) if n.endswith(".py"))
    assert len(names) > 50, "the tests directory is not where this test thinks it is"

    assert lane_select.test_module_inert("python/mojolearn/tests/test_tokenizer_surface.py") is None, \
        "host_surface.py names test_tokenizer_surface in code and it must not be called inert"
    probe = "python/mojolearn/tests/test_host_model_kmeans.py"
    assert lane_select.test_module_inert(probe), f"{probe} should be inert to begin with"
    _refuse_if_manifest_dirty()
    manifest = os.path.join(lane_select.ROOT, lane_select.MANIFEST)
    before = open(manifest, encoding="utf-8").read()
    try:
        # Strings only: the manifest is imported by the selector, and a real
        # import_module call here would run the test module.
        for line in ("_ARM = ('import_module', 'mojolearn.tests.test_host_model_kmeans')",
                     "_ARM = 'test_host_model_kmeans'"):
            with open(manifest, "w") as fh:
                fh.write(before + "\n" + line + "\n")
            lane_select.reset_caches()
            assert lane_select.test_module_inert(probe) is None, \
                f"a file every lane reaches names {probe} ({line}) and it was still called inert"
    finally:
        with open(manifest, "w") as fh:
            fh.write(before)
        lane_select.reset_caches()

    inert = [n for n in names if lane_select.test_module_inert("python/mojolearn/tests/" + n)]
    assert len(inert) > len(names) // 2, \
        f"only {len(inert)} of {len(names)} test modules read inert; the rule stopped firing"

    for rel in ("python/mojolearn/cluster.py", "python/mojolearn/host_surface.py",
                "tools/identity_break.py", "pixi.toml"):
        assert lane_select.test_module_inert(rel) is None, \
            f"{rel} is not a test module and this rule must not touch it"


def test_a_registry_with_no_ref_to_diff_against_selects_every_lane_by_rule():
    """The narrow answer for a registry is read out of a DIFF. With no ref
    there is no diff, so `--lanes-for-paths` on a registry must say every lane
    and say why, rather than looking like the rule failed."""
    sel = lane_select.select(["python/mojolearn/host_surface.py"])
    assert len(sel["lanes"]) == len(lane_select.all_lanes())
    assert "with no ref an addition cannot be read" in sel["reasons"]["python/mojolearn/host_surface.py"]


def test_the_things_a_lane_does_reach_are_never_called_unreachable():
    """The controls. Every one of these is in some lane's closure or is
    resolved by name at load, and calling any of them unreachable would turn a
    real change into nothing affected."""
    for rel in ("cluster/host/kmeans_oracle.mojo", "python/mojolearn/cluster.py",
                "bindings/_mojolearn.mojo", "gemm/host/gemm_oracle.mojo",
                "tools/identity_break.py", "python/mojolearn/host_surface.py",
                "mamba/checks/mamba_fixture.mojo", "embedding/checks/embedding_oracle.mojo",
                "umap/impl/umap.mojo", "core/step_glue.mojo", "bench/oracle.txt",
                "ensemble/decisiontree/batched_levelalgo/split.mojo", "training/byte_lm.mojo"):
        assert not lane_select.unreachable(rel), f"{rel} was called unreachable"


#: A miniature whole-surface registry, and the changes to it that must and
#: must not narrow.
REGISTRY_BASE = '''from .cluster import KMeans
from .density import DBSCAN


class Base:
    pass


class HostDBSCAN(Base):
    _ARRAYS = ("labels_",)


def binary_path():
    return "x"


FORMATS = {
    "mojolearn-dbscan-1": {"DBSCAN": HostDBSCAN},
}
'''

REGISTRY_ADDITIVE = '''from .cluster import KMeans
from .density import DBSCAN


class Base:
    pass


class HostDBSCAN(Base):
    _ARRAYS = ("labels_",)


class HostKMeans(Base, KMeans):
    _ARRAYS = ("cluster_centers_",)


def binary_path():
    return "x"


FORMATS = {
    "mojolearn-dbscan-1": {"DBSCAN": HostDBSCAN},
    "mojolearn-kmeans-1": {"KMeans": HostKMeans},
}
'''

#: Each must answer every lane. Ordered so the failing arm runs first.
REGISTRY_MUST_FALL_BACK = (
    ("an entry removed",
     REGISTRY_ADDITIVE.replace('    "mojolearn-dbscan-1": {"DBSCAN": HostDBSCAN},\n', "")),
    ("an existing function body edited",
     REGISTRY_ADDITIVE.replace('    return "x"', '    return "y"')),
    ("an existing class edited",
     REGISTRY_ADDITIVE.replace('    _ARRAYS = ("labels_",)', '    _ARRAYS = ("labels_", "core_")')),
    ("a new bare module-level statement beside the addition",
     REGISTRY_ADDITIVE.replace("FORMATS = {", "SIDE_EFFECT = [n for n in ()]\n\nFORMATS = {")),
    ("a decorated new class",
     REGISTRY_ADDITIVE.replace("class HostKMeans(", "@staticmethod\nclass HostKMeans(")),
    ("a new class whose body runs something",
     REGISTRY_ADDITIVE.replace('    _ARRAYS = ("cluster_centers_",)',
                               "    for _ in range(1):\n        pass")),
    ("an existing key pointed at something else",
     REGISTRY_ADDITIVE.replace('"mojolearn-dbscan-1": {"DBSCAN": HostDBSCAN}',
                               '"mojolearn-dbscan-2": {"DBSCAN": HostDBSCAN}')),
)


def _registry_answer(new_text, old_text=REGISTRY_BASE):
    ref, path = "<fake ref>", "<fake registry>"
    lane_select._GIT_SHOW[f"{ref}:{path}"] = old_text
    lane_select._read.cache[path] = new_text
    try:
        return lane_select.registry_lanes(ref, path)
    finally:
        lane_select._GIT_SHOW.pop(f"{ref}:{path}", None)
        lane_select._read.cache.pop(path, None)


def test_a_registry_change_that_is_not_purely_additive_still_selects_every_lane():
    """THE ARM THAT MUST FAIL, first. A registry names the whole binding
    surface, so anything but a verified addition has to stay wide."""
    for label, text in REGISTRY_MUST_FALL_BACK:
        assert _registry_answer(text) is None, \
            f"{label}: a registry change that is not an addition was narrowed"


def test_an_additive_registry_entry_selects_what_it_names():
    """lane/kmeans-save added an import, a Host class and one FORMATS entry to
    two whole-surface registries and got 212 of 212. An addition is attributed
    through the files that define the names it mentions."""
    answer = _registry_answer(REGISTRY_ADDITIVE)
    assert answer, "a purely additive registry entry was not narrowed at all"
    assert "kmeans" in answer, f"the addition names KMeans and did not select kmeans: {answer[:8]}"
    for unrelated in ("mamba1", "transformer", "tokenizer", "gbdt-rmse"):
        assert unrelated not in answer, \
            f"an addition naming KMeans selected {unrelated}; it is not narrowing"


def test_the_source_hygiene_patterns_still_fire():
    """The discarded-atomic grep lives next door and its own self-test is what
    makes it worth running; a pattern that matches nothing reads exactly like a
    clean tree. `git grep -E` is POSIX ERE and has no `\\s`, which is how the
    first spelling of it passed while checking nothing."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import source_hygiene_check

    assert source_hygiene_check.self_test() == 0, "a source hygiene pattern no longer fires"


# --------------------------------------------------------------------------
# THE INVERSION CHECKS. Everything else here asks whether the map NARROWS
# correctly. These ask whether it can narrow WRONGLY, which is the failure
# that reads as a pass: a lane whose map is missing a file gets a green run
# for a change that moved its bits. They are run over the WHOLE TREE rather
# than against a case list, because the case list is what missed
# `neural_inference.py` for eight months.
# --------------------------------------------------------------------------

def _lane_sets():
    sources, _ = lane_select.lane_sources()
    return sources, lane_select.reverse_map(sources)


def test_every_lane_that_reaches_an_importer_reaches_what_it_imports():
    """INVERSION 1. If F imports B then a lane reaching F executes B, so
    lanes(F) must be a subset of lanes(B). A break here means the import walk
    stopped somewhere it should not have."""
    sources, rev = _lane_sets()
    sinks = lane_select.enumerator_files()
    bad = []
    for f in lane_select._python_files():
        if f in sinks:
            continue
        for b in lane_select._python_imports(f):
            if b in sinks:
                continue
            missing = rev.get(f, set()) - rev.get(b, set())
            if missing:
                bad.append(f"{f} imports {b}, but {len(missing)} lane(s) reach the importer and "
                           f"not the import, e.g. {sorted(missing)[:4]}")
    assert not bad, "\n  ".join([""] + bad[:6])


def test_every_lane_that_reaches_a_base_reaches_what_extends_it():
    """INVERSION 2, and the one that was broken. If F subclasses or patches a
    class in B then a lane reaching B can run F's override, so lanes(B) must
    be a subset of lanes(F).

    `python/mojolearn/neural_inference.py` defines
    `Mamba1BlockInference(_RecurrentBlockInference, Mamba1Block)`. It IMPORTS
    `_mamba_impl`, so following imports forward never reached it from a mamba
    lane, and the map credited it with `mlp`, `transformer` and
    `transformer-window` while it also serves four mamba and two samba lanes.
    Deleting those wrappers' `forward` overrides is about as load-bearing as
    an edit to that file gets. Reported by lane/stateful-cpu-decoding."""
    sources, rev = _lane_sets()
    bad = []
    for f, targets in lane_select.extenders().items():
        for b in targets:
            missing = rev.get(b, set()) - rev.get(f, set())
            if missing:
                bad.append(f"{f} extends {b}, but {len(missing)} lane(s) reach the base and not "
                           f"the extender, e.g. {sorted(missing)[:4]}")
    assert not bad, "\n  ".join([""] + bad[:6])


def test_the_inversion_check_fails_when_the_backward_edge_is_removed():
    """THE ARM THAT MUST FAIL. An inversion that holds no matter what the map
    does is not a check. With the backward edge disabled, inversion 2 must
    report `neural_inference.py` and the six lanes it was missing."""
    real = lane_select._extending_files
    lane_select._extending_files = lambda closure: set()
    lane_select.reset_caches()
    try:
        sources, rev = _lane_sets()
        broken = []
        for f, targets in lane_select.extenders().items():
            for b in targets:
                if rev.get(b, set()) - rev.get(f, set()):
                    broken.append(f)
        assert "python/mojolearn/neural_inference.py" in broken, (
            "with the backward edge removed the inversion did NOT flag neural_inference.py, so it "
            f"is not what is holding the map together. flagged: {sorted(set(broken))[:5]}")
        missing = rev.get("python/mojolearn/_mamba_impl.py", set()) \
            - rev.get("python/mojolearn/neural_inference.py", set())
        for lane in ("mamba1", "mamba2", "mamba2-dtlimit", "mamba3"):
            assert lane in missing, f"{lane} was expected to be missing without the edge"
    finally:
        lane_select._extending_files = real
        lane_select.reset_caches()


def test_the_six_lanes_the_report_named_are_attributed():
    """The worked example, kept as itself. `neural_inference.py` serves the
    mamba and samba lanes and the map must say so."""
    sources, rev = _lane_sets()
    got = rev.get("python/mojolearn/neural_inference.py", set())
    for lane in ("mamba1", "mamba2", "mamba2-dtlimit", "mamba3",
                 "samba", "samba-untied-dropout-accum",
                 "mlp", "transformer", "transformer-window"):
        assert lane in got, f"neural_inference.py serves {lane} and the map does not say so"
    assert len(got) < len(sources) // 2, \
        f"neural_inference.py now claims {len(got)} of {len(sources)} lanes; the edge is too wide"


def test_a_base_everything_inherits_is_not_evidence():
    """The backward edge needs the same sink rule the forward one has.
    `python/mojolearn/_mode.py` is extended by 23 package files against 2 for
    the next most extended, and without excluding it every lane pulled in all
    23: the median file went from 32 lanes to 197."""
    # Counted per EXTENDING FILE, not per base NAME. `_classical_host.py`
    # subclasses four different neighbors classes and is still one file;
    # counting names made this test call `neighbors.py` universal while the
    # implementation, correctly, did not.
    counts = {}
    symbols = lane_select._python_symbols(lane_select._python_files())
    for rel in lane_select._python_files():
        targets = {other for name in lane_select._extends(rel)
                   for other in symbols.get(name, ()) if other != rel}
        for other in targets:
            counts[other] = counts.get(other, 0) + 1
    universal = sorted(r for r, n in counts.items() if n > lane_select.UNIVERSAL_BASE_MAX_EXTENDERS)
    assert universal, "no universal base was found at all; the threshold or the walk is broken"
    edges = lane_select.extenders()
    for rel, targets in edges.items():
        for u in universal:
            assert u not in targets, f"{rel} kept an edge to the universal base {u}"


def test_the_mojo_conformance_inversion_holds_over_the_whole_tree():
    """INVERSION 3, the Mojo side, and the answer is that there is no backward
    edge to follow.

    A Mojo under-attribution is worse than a Python one: a Python miss means a
    lane's door was missed, a Mojo miss means the ARITHMETIC changed and no
    cell was asked about it. The feared shape was a struct conforming to a
    trait that a kernel dispatches on. Measured on this tree: 10 repo traits,
    9 real conformance edges, and lanes(B) == lanes(F) on every one of them,
    23 against 23 and 35 against 35. Mojo conformance is not Python
    subclassing: a conforming struct is reached only when something
    parametrises on the trait AND IS HANDED THAT STRUCT BY NAME, and naming a
    symbol from another file requires importing it, so the forward walk
    already has it."""
    sources, rev = _lane_sets()
    edges = lane_select.mojo_conformance_edges()
    assert len(edges) >= 5, f"only {len(edges)} conformance edges found; the scan is broken"
    bad = []
    for f, trait, b in edges:
        missing = rev.get(b, set()) - rev.get(f, set())
        if missing:
            bad.append(f"{f} conforms to {trait} declared in {b}, but {len(missing)} lane(s) "
                       f"reach the declaration and not the implementation, e.g. {sorted(missing)[:4]}")
    assert not bad, "\n  ".join([""] + bad[:6])


def test_the_mojo_inversion_fires_when_an_implementation_is_cut_loose():
    """THE ARM THAT MUST FAIL. An inversion with nothing to catch is not a
    check, and this one holds today only because the forward walk reaches the
    implementations. Cut ONE incoming import and it must fire.

    The first attempt at this control was invalid and was caught by measuring
    both sides: dropping every path matching `pointwise_hist2` also removed
    the template that carries the lanes, so lanes(B) and lanes(F) fell
    together and the inversion stayed silent at zero. A negative control whose
    two arms move together proves nothing. This one removes a single edge and
    asserts the dispatcher KEEPS its lanes while the implementation loses
    them."""
    victim = "gbdt/methods/kernel/pointwise_hist2_one_byte_7bit.mojo"
    declarer = "gbdt/methods/kernel/compute_point_hist2_loop.mojo"
    _, rev = _lane_sets()
    before_b, before_f = len(rev.get(declarer, set())), len(rev.get(victim, set()))
    assert before_b and before_f, "the baseline is already empty; this control cannot fire"

    real = lane_select._mojo_imports
    lane_select._mojo_imports = lambda rel: ({f for f in real(rel) if f != victim}
                                             if rel != victim else real(rel))
    lane_select.reset_caches()
    try:
        _, broken = _lane_sets()
        assert len(broken.get(declarer, set())) == before_b, \
            "the sabotage moved the DECLARER too, so the two arms move together and prove nothing"
        assert not broken.get(victim, set()), "the sabotage did not detach the implementation"
        fired = [f for f, _t, b in lane_select.mojo_conformance_edges()
                 if broken.get(b, set()) - broken.get(f, set())]
        assert victim in fired, f"the inversion did not name {victim}; it caught {fired[:4]}"
    finally:
        lane_select._mojo_imports = real
        lane_select.reset_caches()


def test_the_two_mojo_exemptions_are_the_ones_that_were_measured():
    """Both exemptions on the conformance edge are derived, and both split this
    tree exactly. A file with its own `main` is a standalone program; all the
    files that conform to a repo trait purely to TEST it have one and none of
    the shipped implementations does. And the conforming file must actually
    IMPORT the declaration: `core/philox.mojo` and `mamba/host/gen/philox.mojo`
    each declare their OWN `U32Stream` and neither imports the other, so
    matching by trait name alone invented an edge and claimed 80 missing
    lanes."""
    for rel in ("checks/feature_tensor_check.mojo", "checks/newton_walker_check.mojo",
                "checks/pointwise_loop_check.mojo", "ensemble/checks/core_primitives_check.mojo",
                "ensemble/checks/philox_check.mojo"):
        assert lane_select.is_standalone_program(rel), \
            f"{rel} was a standalone program and is not any more; the exemption needs re-deriving"
    for rel in ("gbdt/methods/kernel/pointwise_hist2_one_byte_7bit.mojo",
                "gbdt/methods/leaves_estimation/pointwise_oracle.mojo",
                "core/philox.mojo", "mamba/host/gen/philox.mojo"):
        assert not lane_select.is_standalone_program(rel), \
            f"{rel} became a standalone program; it would now be exempted wrongly"
    a, b = "core/philox.mojo", "mamba/host/gen/philox.mojo"
    assert b not in lane_select._mojo_imports(a) and a not in lane_select._mojo_imports(b), \
        "the two philox files now import each other, so the name collision is a real edge"
    pairs = {(f, b) for f, _t, b in lane_select.mojo_conformance_edges()}
    assert (a, b) not in pairs and (b, a) not in pairs, \
        "the philox name collision is back in the conformance graph"


def test_a_mojo_file_something_imports_is_never_unreachable():
    """A Mojo import is a compile-time fact and does not need the corpus to be
    believed. `core/forest_inference_model.mojo` is imported by
    `bindings/forest_inference_binding.mojo`, which is ITSELF outside the map,
    so nothing in the corpus named either and the model file read "nothing
    reaches it" while being part of a shipped binding's tree. An importer that
    is a standalone program still does not count, which is what keeps three
    new check files that import each other unreachable."""
    for rel in ("core/forest_inference_model.mojo", "core/forest_inference_pool.mojo"):
        assert lane_select._mojo_importers(rel), f"{rel} is imported by nothing; pick a new case"
        assert lane_select.unreachable(rel) is None, \
            f"{rel} is imported by a non-program and was still called unreachable"


# --------------------------------------------------------------------------
# THE ADVERSARIAL PASS, 2026-09-16. Everything above asks whether a rule
# narrows when it should. These ask the question that costs a defect rather
# than an afternoon: can a rule be made to report a NARROW answer for a change
# that genuinely moves a cell. Four could. Each attack is kept beside the
# control it must not break, because a fix that refuses everything passes the
# attack and destroys the rule.
# --------------------------------------------------------------------------

def _harness_answer_pair(old, new):
    ref, path = "<fake ref>", "<fake harness>"
    lane_select._GIT_SHOW[f"{ref}:{path}"] = old
    lane_select._read.cache[path] = new
    try:
        return lane_select.harness_lanes(ref, path)
    finally:
        lane_select._GIT_SHOW.pop(f"{ref}:{path}", None)
        lane_select._read.cache.pop(path, None)


def test_a_lane_carrying_a_second_decorator_is_not_a_lane_body():
    """ATTACK. `@mutate_everything` above `@lane("beta")` was admitted as one
    new lane. The other decorator RUNS at import and can touch a fixture
    table, a registry or another lane's defaults."""
    base = '@lane("alpha")\ndef _alpha():\n    return 1\n'
    attacked = base + '\n\n@mutate_everything\n@lane("beta")\ndef _beta():\n    return 2\n'
    assert _harness_answer_pair(base, attacked) is None, \
        "a new lane carrying a second decorator was narrowed"
    plain = base + '\n\n@lane("beta")\ndef _beta():\n    return 2\n'
    assert _harness_answer_pair(base, plain) == ["beta"], \
        "the fix broke the ordinary added lane, which is the whole point of the rule"


def test_two_registry_keys_that_are_the_same_value_are_not_an_addition():
    """ATTACK. `"mojolearn-dbscan-" + "1"` beside `"mojolearn-dbscan-1"` is a
    different EXPRESSION and the same VALUE, so the later entry overrides the
    earlier one and an existing lane's dispatch moves. It was admitted as an
    addition."""
    base = 'FORMATS = {\n    "mojolearn-dbscan-1": {"DBSCAN": HostDBSCAN},\n}\n'
    attacked = base.replace("}\n", '    "mojolearn-dbscan-" + "1": {"DBSCAN": HostOther},\n}\n')
    assert _registry_answer(attacked, base) is None, \
        "a key equal in value to an existing one was admitted as an addition"
    genuine = base.replace("}\n", '    "mojolearn-kmeans-1": {"KMeans": HostKMeans},\n}\n')
    answer = _registry_answer(genuine, base)
    assert answer and "kmeans" in answer, \
        "the fix broke the ordinary additive entry, which is the whole point of the rule"


def test_a_test_module_named_any_way_at_all_is_not_inert():
    """ATTACK. `importlib.import_module('mojolearn.tests.test_host_model_kmeans')`
    is invisible to an import-statement regex, and the module was called
    inert. The NAME is searched now, anywhere outside the tests directory, so
    a dynamic import, a `python -m` line and a bare mention all count. That
    cost 33 modules of narrowing (124 of 125 down to 91 of 125) and is the
    right trade: over-firing here costs a sweep, under-firing costs a defect."""
    _refuse_if_manifest_dirty()
    manifest = os.path.join(lane_select.ROOT, lane_select.MANIFEST)
    before = open(manifest, encoding="utf-8").read()
    target = "python/mojolearn/tests/test_host_model_kmeans.py"
    try:
        lane_select.reset_caches()
        assert lane_select.test_module_inert(target), f"{target} should be inert to begin with"
        with open(manifest, "w") as fh:
            fh.write(before + "\n_DYN = __import__('importlib').import_module("
                     "'mojolearn.tests.test_host_model_kmeans')\n")
        lane_select.reset_caches()
        assert lane_select.test_module_inert(target) is None, \
            "a test module imported dynamically by string was still called inert"
    finally:
        with open(manifest, "w") as fh:
            fh.write(before)
        lane_select.reset_caches()


def test_a_directory_named_without_its_slash_still_counts():
    """ATTACK. `os.path.join('armprobedir', name + '.bin')` names the
    directory without ever writing a slash, so the path tokens saw nothing and
    a file in it was called unreachable. Path-building calls are read
    STRUCTURALLY, not as text, because the bare word `umap` is also a package
    module and a family-table entry and matching those would send every new
    file under `umap/` to a sweep."""
    _refuse_if_manifest_dirty()
    manifest = os.path.join(lane_select.ROOT, lane_select.MANIFEST)
    before = open(manifest, encoding="utf-8").read()
    probe_dir = os.path.join(lane_select.ROOT, "armprobedir")
    rel = "armprobedir/armprobefixture.bin"
    os.makedirs(probe_dir, exist_ok=True)
    try:
        with open(os.path.join(lane_select.ROOT, rel), "w") as fh:
            fh.write("x\n")
        lane_select.reset_caches()
        assert lane_select.unreachable(rel), f"{rel} should be unreachable to begin with"
        with open(manifest, "w") as fh:
            fh.write(before + "\nimport os as _o\n_ARM = _o.path.join('armprobedir', 'x' + '.bin')\n")
        lane_select.reset_caches()
        assert lane_select.unreachable(rel) is None, \
            "a directory handed to os.path.join without a slash did not count as naming it"
    finally:
        with open(manifest, "w") as fh:
            fh.write(before)
        probe = os.path.join(lane_select.ROOT, rel)
        if os.path.exists(probe):
            os.unlink(probe)
        if os.path.isdir(probe_dir) and not os.listdir(probe_dir):
            os.rmdir(probe_dir)
        lane_select.reset_caches()


def test_no_docstring_is_read_back_at_run_time():
    """The docstring rule rests on nothing consuming a docstring as a VALUE.
    That is an absence claim, so it is checked rather than assumed: the only
    `__doc__` in the package and the harness are ASSIGNMENTS in the kde, knn,
    radius, gp and gmm registration loops, which are code and survive the
    strip, plus argparse help."""
    bad = []
    pkg = os.path.join(lane_select.ROOT, lane_select.PKG)
    paths = [os.path.join(pkg, n) for n in sorted(os.listdir(pkg)) if n.endswith(".py")]
    paths.append(os.path.join(lane_select.ROOT, lane_select.HARNESS))
    for path in paths:
        tree = lane_select._parse(os.path.relpath(path, lane_select.ROOT))
        for node in ast.walk(tree or ast.Module(body=[], type_ignores=[])):
            if isinstance(node, ast.Attribute) and node.attr == "__doc__" \
                    and isinstance(node.ctx, ast.Load):
                bad.append(f"{os.path.relpath(path, lane_select.ROOT)}:{node.lineno}")
    assert not bad, ("a docstring is READ as a value, so a docstring-only change can reach it: "
                     f"{bad[:5]}")


# ---------------------------------------------------------------- 2026-09-21
# The rules that took the 0.8.12 Apple pass's 148 NOT ATTRIBUTABLE paths down
# to a handful. Each is tested twice: once where it narrows, and once where a
# path that CAN reach lane arithmetic must still widen.

def _with_fake(path, new_text, old_text=None, ref="<fake ref>"):
    lane_select._read.cache[path] = new_text
    if old_text is not None:
        lane_select._GIT_SHOW[f"{ref}:{path}"] = old_text
    return ref


def _drop_fake(path, ref="<fake ref>"):
    lane_select._read.cache.pop(path, None)
    lane_select._GIT_SHOW.pop(f"{ref}:{path}", None)


def test_a_build_script_selects_the_lanes_of_the_binding_it_compiles():
    """bindings/build_svm.sh compiles bindings/_mojolearn_svm.mojo, so it
    selects exactly the lanes that reach that source, not every lane."""
    sources, _ = lane_select.lane_sources()
    rev = lane_select.reverse_map(sources)
    sel = lane_select.select(["bindings/build_svm.sh"], ref="HEAD")
    assert not sel["unattributed"], sel["reasons"]
    want = rev["bindings/_mojolearn_svm.mojo"]
    assert want and set(sel["lanes"]) == want, "the build script did not select its binding's lanes"
    assert len(want) < len(sources), "the svm binding is reached by every lane; the case proves nothing"


def test_a_build_script_whose_source_is_not_literal_is_refused():
    """THE ARM THAT MUST NOT NARROW. A `mojo build` line whose source is a
    variable cannot be attributed, and a build script is lane arithmetic: the
    path is refused by name (never widened, never dropped)."""
    path = "bindings/build_armprobe.sh"
    _with_fake(path, 'src=bindings/_mojolearn_svm.mojo\npixi run mojo build "$src" -o x.so\n')
    try:
        assert lane_select.build_script_roots(path) is None
        sel = lane_select.select([path], ref="HEAD")
        assert path in sel["unattributed"] and not sel["lanes"], sel["reasons"]
    finally:
        _drop_fake(path)


def test_pixi_tasks_are_inert_and_a_dependency_is_every_lane():
    base = '[workspace]\nname = "x"\n\n[tasks]\na = "echo a"\n\n[dependencies]\nmax = "==26.5.0"\n'
    tasks = base.replace('a = "echo a"', 'a = "echo a"\nb = "echo b"  # a new task')
    dep = base.replace("26.5.0", "26.6.0")
    try:
        ref = _with_fake("pixi.toml", tasks, base)
        sel = lane_select.select(["pixi.toml"], ref=ref)
        assert not sel["unattributed"] and not sel["lanes"], sel["reasons"]
        assert "task tables" in sel["reasons"]["pixi.toml"]
        ref = _with_fake("pixi.toml", dep, base)
        sel = lane_select.select(["pixi.toml"], ref=ref)
        assert "pixi.toml" in sel["every_rules"], "a dependency change in pixi.toml must select every lane"
        assert len(sel["lanes"]) == len(lane_select.all_lanes())
    finally:
        _drop_fake("pixi.toml")


def test_native_code_the_package_loads_selects_the_loaders_lanes():
    """stage.py is a Python file outside the package that no lane imports, and
    it still compiles the library `_portable_math.py` loads. It must never be
    called unreachable."""
    sources, _ = lane_select.lane_sources()
    rev = lane_select.reverse_map(sources)
    for path in ("packaging/portable_math/stage.py", "packaging/portable_math/portable_math.c"):
        # No ref: against HEAD an unchanged file is "docstrings only" first.
        sel = lane_select.select([path])
        assert not sel["unattributed"] and set(sel["lanes"]) == rev["python/mojolearn/_portable_math.py"], \
            f"{path}: {sel['reasons']}"
        assert "native code" in sel["reasons"][path]


def test_a_tool_named_only_by_a_literal_join_is_unreachable():
    """`join(base, "tools", "identity_break.py")` names one file; it does not
    walk tools/. A leg script nothing names is unreachable, a helper a build
    script runs is not, and a tool a lane's file imports is not."""
    assert lane_select.unreachable("tools/gemm_remote_leg.sh"), "a rental leg script should be unreachable"
    assert lane_select.unreachable("tools/with_build_lock.sh") is None, \
        "bindings/build_preprocessing.sh execs tools/with_build_lock.sh; it is a build input"
    assert lane_select.unreachable("tools/identity_trace_diff.py") is None, \
        "the package names identity_trace_diff.py and it must stay reachable"


def test_a_directory_walk_under_tools_still_widens():
    """THE ARM THAT MUST WIDEN: a join whose tail is a variable walks the
    directory, so everything in it stays reachable."""
    _refuse_if_manifest_dirty()
    manifest = os.path.join(lane_select.ROOT, lane_select.MANIFEST)
    before = open(manifest, encoding="utf-8").read()
    try:
        with open(manifest, "w") as fh:
            fh.write(before + "\nimport os as _o\n_ARM = lambda n: _o.path.join('tools', n)\n")
        lane_select.reset_caches()
        assert lane_select.unreachable("tools/gemm_remote_leg.sh") is None, \
            "a variable join under tools/ did not count as walking it"
    finally:
        with open(manifest, "w") as fh:
            fh.write(before)
        lane_select.reset_caches()


def test_a_mojo_program_nothing_imports_is_unreachable_and_an_imported_one_is_not():
    assert lane_select.unreachable("transformer/checks/attention_v2_forward_bench.mojo"), \
        "a benchmark program no Mojo file imports should be unreachable"
    assert lane_select.unreachable("mamba/checks/mamba_fixture.mojo") is None, \
        "mamba_fixture.mojo is in lanes' closures and must never be unreachable"
    assert lane_select.unreachable("glm/host/qn_oracle.mojo") is None


def test_package_modules_the_verifier_alone_uses_are_unreachable():
    """`_verify_par.py` runs from `python -m mojolearn verify`, never in a lane.
    The controls: `_mode.py`, which every lane imports, and a package module
    that only `__init__.py` imports, whose top level runs in every lane."""
    assert lane_select.unreachable("python/mojolearn/_verify_par.py")
    assert lane_select.unreachable("python/mojolearn/verify_reference/table.json")
    assert lane_select.unreachable("python/mojolearn/_mode.py") is None
    closure = lane_select._package_import_closure()
    only_init = sorted(f for f in lane_select._python_imports(os.path.join(lane_select.PKG, "__init__.py"))
                       if f in closure)
    assert only_init, "__init__.py imports nothing the closure holds; the control is gone"
    for rel in only_init:
        assert lane_select.unreachable(rel) is None, f"{rel} runs at `import mojolearn` and was called unreachable"


def test_a_package_module_named_by_a_lane_file_stays_reachable():
    """THE ARM THAT MUST WIDEN for the package rule: a dynamic import string in
    a file every lane reaches."""
    probe = "python/mojolearn/_verify_par.py"
    assert lane_select.unreachable(probe), "the probe must start unreachable"
    _refuse_if_manifest_dirty()
    manifest = os.path.join(lane_select.ROOT, lane_select.MANIFEST)
    before = open(manifest, encoding="utf-8").read()
    try:
        with open(manifest, "w") as fh:
            fh.write(before + "\n_ARM = 'mojolearn._verify_par'\n")
        lane_select.reset_caches()
        assert lane_select.unreachable(probe) is None
    finally:
        with open(manifest, "w") as fh:
            fh.write(before)
        lane_select.reset_caches()


LANE_PROSE_BASE = HARNESS_BASE.replace(
    "TOL = 1", 'TOL = 1\nNOTES = {\n    "alpha": "why alpha changed",\n    "beta": "why beta changed",\n}')


def test_a_lane_keyed_prose_table_selects_the_lanes_it_names():
    reworded = LANE_PROSE_BASE.replace("why alpha changed", "why alpha changed, said better")
    assert _harness_answer(reworded, LANE_PROSE_BASE) == ["alpha"]
    only_beta = LANE_PROSE_BASE.replace('    "alpha": "why alpha changed",\n', "")
    assert _harness_answer(LANE_PROSE_BASE, only_beta) == ["alpha"], "an added entry names its lane"
    # A lane registered by a loop (gamma-1) is not a lane body the harness
    # reader can see, so a key naming it widens rather than guessing.
    looped = LANE_PROSE_BASE.replace('    "beta": "why beta changed",\n',
                                     '    "beta": "why beta changed",\n    "gamma-1": "new",\n')
    assert _harness_answer(looped, LANE_PROSE_BASE) is None


def test_a_lane_keyed_table_that_is_not_prose_still_widens():
    """THE ARMS THAT MUST WIDEN: a value that computes, a key that is not a
    lane, and a removed entry."""
    computed = LANE_PROSE_BASE.replace('"why alpha changed"', "helper(1)")
    assert _harness_answer(computed, LANE_PROSE_BASE) is None
    stray = LANE_PROSE_BASE.replace('    "beta": "why beta changed",\n',
                                    '    "beta": "why beta changed",\n    "not-a-lane": "x",\n')
    assert _harness_answer(stray, LANE_PROSE_BASE) is None
    removed = LANE_PROSE_BASE.replace('    "beta": "why beta changed",\n', "")
    assert _harness_answer(removed, LANE_PROSE_BASE) is None
    both = LANE_PROSE_BASE.replace("why alpha changed", "x").replace("return x + TOL", "return x - TOL")
    assert _harness_answer(both, LANE_PROSE_BASE) is None


def test_a_failed_diff_refuses_instead_of_selecting_nothing():
    try:
        lane_select.changed_paths("no-such-ref-armprobe")
    except SystemExit as exc:
        assert "REFUSING" in str(exc)
    else:
        raise AssertionError("a diff against a missing ref returned a path list")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"{'OK' if not failures else 'FAILED'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())


# --------------------------------------------------------------------------
# THE FORWARD WALK LOST A SHIPPED SURFACE, 2026-09-16 (lane/lane-map-census).
# `core/forest_inference.mojo` is imported by `bindings/_mojolearn_rf.mojo`
# and `bindings/_mojolearn_trees.mojo` and was in NO lane's map. It is not a
# backward edge and it is not correct as is: the shipped `_mojolearn_rf.so`
# carries that file's Metal kernels as the AIR blobs
# `core_forest_inference_forest_g...` and `core_forest_inference_forest_v...`
# (over python/mojolearn/identical/, where all 17 bindings exist, they are in
# _mojolearn_rf.so and _mojolearn_trees.so and in NEITHER of the other 15; the
# first control named three .so that do not exist, so `strings` failed and its
# empty output read exactly like a clean probe),
# and a `RandomForestClassifier(..., inference_engine="parallel_groves")`,
# which is exactly what `@lane("rf-clf-balanced-parallel")` builds, was
# measured calling `forest_prepare_gpu`, `forest_predict_resident_reuse_gpu`
# and `forest_release_gpu` on Metal while the sequential control called
# `rf_predict_proba` and none of the three. Three separate rules each hid it,
# and each is pinned below beside the case that must STAY narrow.
# --------------------------------------------------------------------------

FOREST_INFERENCE_LANES = ("rf-clf", "rf-clf-balanced-parallel", "rf-reg",
                          "et-clf", "et-reg", "et-reg-bootstrap-parallel",
                          "par-forest", "par-forest-pool")


def test_a_shipped_inference_file_is_in_the_map_of_the_lanes_that_run_it():
    """`core/forest_inference.mojo` carries the forest prediction kernels the
    rf and trees bindings launch. A change to it can move those lanes' bits,
    so they must be named. THE FAILING SIDE: before the three fixes below the
    reverse map held this file under NO lane at all."""
    rev = lane_select.reverse_map()
    for rel in ("core/forest_inference.mojo", "core/forest_inference_model.mojo",
                "bindings/forest_inference_binding.mojo"):
        lanes = rev.get(rel, set())
        assert lanes, f"{rel} is a shipped forest inference source and is in no lane's map"
        missing = [lane for lane in FOREST_INFERENCE_LANES if lane not in lanes]
        assert not missing, f"{rel} does not name the lanes that run it: {missing}"


def test_a_mojo_import_resolves_against_the_importing_files_own_directory():
    """Every binding is built `-I . -I bindings` (bindings/build_rf.sh:123),
    so `bindings/_mojolearn_rf.mojo` writing `from forest_inference_binding
    import ...` means `bindings/forest_inference_binding.mojo`.

    THE FAILING SIDE, run here rather than asserted: resolving against the
    repository root ALONE finds no such file, and the import is dropped as one
    of the toolchain's own."""
    rel = "bindings/_mojolearn_rf.mojo"
    assert "bindings/forest_inference_binding.mojo" in lane_select._mojo_imports(rel), \
        "the sibling import the build's -I bindings resolves is not in the map"

    def root_only(dotted, _rel):
        parts = dotted.split(".")
        return {c for c in (os.path.join(*parts) + ".mojo",
                            os.path.join(*parts, "__init__.mojo"))
                if os.path.exists(os.path.join(lane_select.ROOT, c))}

    keep = lane_select._mojo_module_files
    lane_select._mojo_module_files = root_only
    try:
        lane_select.reset_caches()
        blind = lane_select._mojo_imports(rel)
        assert "bindings/forest_inference_binding.mojo" not in blind, \
            ("root-only resolution still found the sibling import, so this test "
             "cannot fail and is not a check")
        assert "core/forest_inference.mojo" in blind, \
            "root-only resolution lost an ordinary root import too; the arm is not clean"
    finally:
        lane_select._mojo_module_files = keep
        lane_select.reset_caches()

    # ... and it must not invent an edge out of the toolchain's own modules.
    assert not [c for c in lane_select._mojo_module_files("max.gpu.host", rel)], \
        "a toolchain module resolved to a file in this tree"


def test_a_parametrized_def_function_is_still_an_export():
    """`def_function[forest_prepare_gpu_binding[True]]("forest_prepare_gpu")`
    is the ordinary spelling in the forest bindings. Requiring a bare
    identifier dropped 35 exports across five bindings, among them the FIT
    entry points `rf_classifier_fit` and `et_classifier_fit`.

    A dropped export is not a wide answer. The lane still HITS other exports,
    so the per-export branch runs and the dropped export's whole tree is
    simply absent."""
    rf = lane_select._binding_exports("bindings/_mojolearn_rf.mojo")
    trees = lane_select._binding_exports("bindings/_mojolearn_trees.mojo")
    for name in ("rf_classifier_fit", "rf_regressor_fit", "forest_prepare_gpu",
                 "forest_predict_resident_reuse_gpu", "forest_release_gpu"):
        assert name in rf, f"{name} is registered in _mojolearn_rf.mojo and is not an export"
    for name in ("et_classifier_fit", "et_regressor_fit", "forest_prepare_gpu"):
        assert name in trees, f"{name} is registered in _mojolearn_trees.mojo and is not an export"
    # THE FAILING SIDE: the old pattern, run here, must miss them.
    text = lane_select._read("bindings/_mojolearn_rf.mojo")
    old = {m.group(2) for m in re.finditer(
        r"def_function\[\s*([A-Za-z0-9_]+)\s*\]\s*\(\s*\"([A-Za-z0-9_]+)\"", text)}
    assert "rf_classifier_fit" not in old, \
        "the old pattern already found the parametrized export; this test cannot fail"
    # It must still be an export NAME, never the impl, that lands in the map.
    assert "forest_prepare_gpu_binding" not in rf, "the impl name leaked in as an export"


def test_an_export_reaches_what_its_impl_reaches():
    """An export reaches a Mojo file two ways the body scan used to miss, and
    both are the ordinary spelling here:

    * the impl is IMPORTED, not defined in the binding. `forest_prepare_gpu`
      is `forest_prepare_gpu_binding` from bindings/forest_inference_binding.mojo,
      so `blocks.get` returned "" and the export contributed nothing;
    * the impl calls a file-local helper which is where the imported symbol
      appears. `rf_predict_proba_gpu_parallel_binding` calls
      `_rf_predict_gpu_parallel`, and only that helper names
      `forest_predict_gpu`.
    """
    src = "bindings/_mojolearn_rf.mojo"
    blocks = lane_select._mojo_blocks_for(src)
    syms = lane_select._mojo_import_symbols(src)
    exports = lane_select._binding_exports(src)

    # the imported impl
    assert exports["forest_prepare_gpu"] not in blocks, \
        "forest_prepare_gpu's impl is defined in the binding now; pick a new case"
    assert "bindings/forest_inference_binding.mojo" in syms.get(exports["forest_prepare_gpu"], set()), \
        "the imported impl does not resolve to the file that defines it"

    # the file-local helper
    impl = exports["rf_predict_proba_gpu_parallel"]
    assert "forest_predict_gpu" not in blocks[impl], \
        "the export's own body names it now; pick a new case"
    assert "_rf_predict_gpu_parallel" in blocks[impl], "the helper call moved; pick a new case"
    assert "forest_predict_gpu" in blocks["_rf_predict_gpu_parallel"], "the helper moved"
    assert "core/forest_inference.mojo" in syms["forest_predict_gpu"]


def test_the_wider_mojo_walk_did_not_widen_the_narrow_answers():
    """THE CONTROL THE THREE FIXES ABOVE MUST NOT BREAK. Widening a walk is
    how a map goes back to answering every lane, so the files whose narrow
    answers were measured when the per-export rule landed are pinned here.
    `core/forest_host_predict.mojo` is 15 rather than its old 7 on purpose:
    `rf_predict_proba` routes to it and the rf lanes were missing.

    REMEASURED 2026-09-21 (212 lanes then, 273 now), and every move was
    attributed before a number changed; none is the walk widening. The first
    pin failing (kmeans_oracle) had hidden the other four since 2026-09-17.
      kmeans_oracle        20 -> 21  `kmeans-cosine` removed; `par-ivf` and
                                     `spectral-embedding` added through doors
                                     their siblings already walked
      gbdt_host_predict    23 -> 34  eleven NEW lanes, no old lane moved
      forest_host_predict  15 -> 59  thirteen new lanes, and 31 old gbdt lanes
                                     through REAL imports added 2026-09-17:
                                     `core/gbdt_host_predict.mojo` and
                                     `gbdt/resident_model.mojo` now import
                                     `core.forest_host_predict`
      forest_inference     23 -> 26  three NEW lanes
      neural_inference.py  21 -> 40  nineteen NEW lanes (low-bit, decode
                                     session, causal LM), no old lane moved

    REMEASURED 2026-09-22 (274 lanes). Found red on main by
    lane/release-gpu-columns, with main's own selector reading the same
    numbers, so no selector change moved them:
      gbdt_host_predict    34 -> 49  every one a boosting, forest-driver or
                                     cross-validation lane (gbdt-binary-columns,
                                     gbdt-stochastic-arms, the *-defaults lanes
                                     and their par- twins were added since); no
                                     other family entered
      forest_host_predict  59 -> 60  the same additions"""
    rev = lane_select.reverse_map()
    for rel, want in (("cluster/host/kmeans_oracle.mojo", 21),
                      ("core/gbdt_host_predict.mojo", 49),
                      ("core/forest_host_predict.mojo", 60),
                      ("core/forest_inference.mojo", 26),
                      ("python/mojolearn/neural_inference.py", 40)):
        got = len(rev.get(rel, set()))
        assert got == want, f"{rel} answers {got} lanes, not {want}"
    lanes = len(lane_select.all_lanes())
    # Tracked files only: a release build drops ignored generated copies into
    # the package (python/mojolearn/_identity_break.py, a copy of
    # tools/identity_break.py) that no diff can ever name, and counting them
    # made this read 56 in a built tree and 55 in a fresh checkout.
    import subprocess
    tracked = set(subprocess.run(["git", "-C", lane_select.ROOT,
                                  "ls-files"], capture_output=True, text=True).stdout.split())
    every = [rel for rel, seen in rev.items() if len(seen) == lanes and rel in tracked]
    # 41 when the per-export rule landed, 55 on 2026-09-21. The fourteen were
    # traced commit by commit and each is a REAL import into a closure every
    # lane already had, never a wider walk: `_byte_lm_host.py` importing
    # `lowbit.py` (2026-09-17) brought lowbit, linalg, _linalg_impl,
    # _cholesky_impl, _mode and the linalg and gp bindings; `_byte_lm_impl.py`
    # brought _byte_lm_checkpoint and _portable_math; the byte LM host build
    # brought block_options, rtf_seam and identical_gemm.
    assert len(every) <= 55, \
        f"{len(every)} files now select every lane, against 55 measured on 2026-09-21"


# --------------------------------------------------------------------------
# THE CENSUS AT 3, READ BY A PERSON (lane/lane-map-census, 2026-09-16). Two of
# the 60 entries were not narrow files. They were a case-insensitive
# filesystem and a public door that no lane walked through.
# --------------------------------------------------------------------------

def test_a_package_module_is_resolved_from_the_listing_not_the_filesystem():
    """`_python_imports` asks whether a name a file imports is itself a
    package module, and an imported name is often a CLASS. On a
    case-insensitive checkout `os.path.exists` answers yes to
    python/mojolearn/UMAP.py for `from ._umap_impl import UMAP` and to
    python/mojolearn/HDBSCAN.py for `HDBSCAN`. Neither path is tracked; the
    map carried two files that exist only on that laptop, HDBSCAN.py holding
    24 lanes, and on a Linux box the same map is a different map."""
    tracked = set(lane_select.tracked_files())
    rev = lane_select.reverse_map()
    for phantom, real in (("python/mojolearn/UMAP.py", "python/mojolearn/umap.py"),
                          ("python/mojolearn/HDBSCAN.py", "python/mojolearn/hdbscan.py")):
        assert real in tracked, f"{real} is the tracked spelling; pick a new case"
        assert phantom not in tracked, f"{phantom} is tracked now; pick a new case"
        assert phantom not in rev, \
            f"{phantom} is in the map and is not a file this repository has"
        assert rev.get(real), f"{real} is the real public door and is in no lane's map"
    # THE FAILING SIDE, only where the filesystem can produce it.
    if os.path.exists(os.path.join(lane_select.ROOT, "python/mojolearn/UMAP.py")):
        assert os.path.join(lane_select.PKG, "UMAP.py") not in lane_select._python_files(), \
            "the listing itself is case-folding, so this test cannot fail"


def test_the_public_door_a_name_is_bound_from_is_in_the_map():
    """A lane body writes `ml.UMAP`, and that attribute is bound by
    `__init__.py`'s `from .umap import UMAP`, which executes
    python/mojolearn/umap.py. Seeding only the file that DEFINES the class
    walked straight past that door: umap.py, neural_network.py and
    language_model.py were in no lane's map.

    THE CASE THAT MUST STAY NARROW is in the same assert. Treating every
    `from .X import N` inside the package as a binding of N was measured and
    is far too wide: the median file went from 29 lanes to 60 and
    neural_inference.py from 21 to all 212.

    NARROW MEANS EVERY LANE AT THE DOOR NAMES WHAT THE DOOR BINDS. The first
    spelling was a count (`<= 12`), and a count measures the registry as much
    as the rule: the fourteen low-bit lanes (2026-09-17), `language-model-config`
    and `transformer-decode-session` all name `LanguageModelInference` through
    `_neural_inference`, so language_model.py went from 11 lanes to 25 with the
    door rule unchanged. The rule itself is what is held now: a lane reaches a
    door only when its own code closure names a name `__init__.py` binds from
    that module, which the wide rule above breaks at once."""
    rev = lane_select.reverse_map()
    ib = lane_select.identity_break()
    init = ast.parse(open(os.path.join(lane_select.ROOT, lane_select.PKG, "__init__.py"),
                          encoding="utf-8").read())
    for rel, want in (("python/mojolearn/umap.py", "umap"),
                      ("python/mojolearn/neural_network.py", "mlp"),
                      ("python/mojolearn/language_model.py", "byte-lm")):
        lanes = rev.get(rel, set())
        assert want in lanes, f"{rel} is the public door for {want} and the lane does not reach it"
        stem = os.path.basename(rel)[:-3]
        bound = {a.asname or a.name for n in ast.walk(init)
                 if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == stem
                 for a in n.names}
        assert bound, f"__init__.py binds nothing from {stem}; pick a new case"
        naming = {n for n, fn in ib.LANES.items()
                  if lane_select._code_names(fn, vars(ib)) & bound}
        assert len(naming) < len(ib.LANES) // 4, \
            f"{len(naming)} lanes name {sorted(bound)}, so the subset check below cannot fail"
        wide = sorted(lanes - naming)
        assert not wide, (f"{rel} answers {len(lanes)} lanes and {wide[:5]} name nothing it "
                          f"binds ({sorted(bound)}); the door rule has gone wide")
    assert len(rev.get("python/mojolearn/neural_inference.py", ())) == 40, \
        "the re-export rule moved neural_inference.py off its measured 40 lanes"

    # THE FAILING SIDE: with no public rebindings the lane each door is
    # checked for loses it. Held per LANE and not per file since 2026-09-21:
    # `language-model-config` (2026-09-19) names `LanguageModelConfig`, which
    # language_model.py DEFINES (`LanguageModelConfig = ByteLanguageModelConfig`),
    # so that file stays in the map by the definition rule alone. What this
    # side must show is that `want in lanes` above is carried by the door.
    keep = lane_select._public_rebindings
    lane_select._public_rebindings = lambda files: {}
    try:
        lane_select.reset_caches()
        blind = lane_select.reverse_map()
        for rel, want in (("python/mojolearn/umap.py", "umap"),
                          ("python/mojolearn/neural_network.py", "mlp"),
                          ("python/mojolearn/language_model.py", "byte-lm")):
            assert want not in blind.get(rel, set()), \
                f"{want} reaches {rel} without the public door rule, so this test cannot fail"
    finally:
        lane_select._public_rebindings = keep
        lane_select.reset_caches()


def test_an_aliased_mojo_import_is_recorded_under_the_name_the_body_uses():
    """`bindings/_mojolearn_metrics.mojo:56` is
    `from umap.estimator import fit_transform as umap_fit_transform`, and
    `umap_fit_transform_binding` calls `umap_fit_transform`. Recording the
    ORIGINAL name made the per-export body scan search for
    `\\bfit_transform\\b`, which does not match inside `umap_fit_transform`
    because the underscore before `fit` is a word character. The whole umap
    tree was therefore invisible to the scan, and reached the `umap` lane only
    because the metrics HOST family happens to list umap/graph.mojo among its
    host modules. `par-graph-umap`, which runs the same fit across devices and
    is not in that family, was credited with none of it."""
    syms = lane_select._mojo_import_symbols("bindings/_mojolearn_metrics.mojo")
    body = lane_select._mojo_blocks_for("bindings/_mojolearn_metrics.mojo")["umap_fit_transform_binding"]
    assert "umap_fit_transform(" in body, "the export stopped calling the alias; pick a new case"
    assert "umap/estimator.mojo" in syms.get("umap_fit_transform", set()), \
        "the alias the body uses is not in the import table"
    assert not re.search(r"\bfit_transform\b", body), \
        "the ORIGINAL name matches the body after all, so this test cannot fail"

    rev = lane_select.reverse_map()
    for rel in ("umap/graph.mojo", "umap/sparse_graph.mojo", "umap/estimator.mojo"):
        lanes = rev.get(rel, set())
        assert "umap" in lanes and "par-graph-umap" in lanes, \
            f"{rel} is run by both umap lanes and the map names {sorted(lanes)}"


def test_runtime_controls_do_not_trigger_numerical_sweep():
    paths = ["tools/identity_iterate.py", "tools/mac_slot.py"]
    selected = lane_select.select(paths)
    assert selected["lanes"] == []
    assert not selected["unattributed"]


def test_a_file_the_harness_imports_at_run_time_selects_every_lane():
    """`tools/lane_applicability.py` was a runtime control until 2026-09-19,
    when `identity_break.py` began importing it to write `degenerate_lanes`
    into every column it records. A change to it now changes what every
    column says, over the whole registry, so it must select every lane and
    never read as inert."""
    selected = lane_select.select(["tools/lane_applicability.py"])
    assert "tools/lane_applicability.py" in selected["every_rules"], selected["reasons"]
    assert not selected["unattributed"]
    assert len(selected["lanes"]) == selected["total"] > 0
