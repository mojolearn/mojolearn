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
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select                                              # noqa: E402

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
        assert not sel["fallback"], \
            f"{path} fell back to every lane, so this case proves nothing: {sel['reasons'][path]}"
        assert must in sel["lanes"], f"{path} did not select {must}"
        for other in must_not:
            assert other not in sel["lanes"], \
                f"{path} selected the unrelated lane {other}; the map is not narrowing"


def test_inert_paths_select_nothing_and_do_not_pretend_otherwise():
    sel = lane_select.select(["docs/START_HERE.md", "CHANGELOG.md"])
    assert sel["lanes"] == [], "a prose change selected lanes"
    assert not sel["fallback"], "a prose change should not force the fallback"


def test_several_paths_in_one_argument_are_never_inert():
    """THE SILENT ZERO, caught on 2026-09-16. zsh does not word-split an
    unquoted variable, so a whole commit's file list arrived as ONE argument;
    it began with CHANGELOG.md, matched the inert prefix rule, and twelve
    changed files selected no lanes at all and said nothing. An argument that
    is not a single path must fall back, loudly."""
    blob = "CHANGELOG.md cluster/host/kmeans_oracle.mojo tools/identity_break.py"
    sel = lane_select.select([blob])
    assert sel["fallback"] is True, "a multi-path argument did not fall back"
    assert len(sel["lanes"]) == len(lane_select.all_lanes())
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
        assert not sel["fallback"], f"{path} is no longer attributable: {sel['reasons'][path]}"
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


def test_an_unattributable_path_falls_back_to_every_lane_and_says_so():
    """The fallback is the safety property. A path the map cannot place must
    select EVERY lane and must be named in `unattributed`, so the run cannot
    be read as a narrow one."""
    sel = lane_select.select(["pixi.toml"])
    assert sel["fallback"] is True, "an unattributable path did not fall back"
    assert len(sel["lanes"]) == len(lane_select.all_lanes())
    assert "pixi.toml" in sel["unattributed"]
    assert "NOT ATTRIBUTABLE" in sel["reasons"]["pixi.toml"]


def test_a_registry_change_selects_every_lane():
    """`_backend.py` and `host_surface.py` name the whole binding surface, so
    the map drops their per-lane edges. That debt is paid here: changing one
    selects everything."""
    for path in ("python/mojolearn/_backend.py", "python/mojolearn/host_surface.py"):
        sel = lane_select.select([path])
        assert sel["fallback"] is True, f"{path} did not select every lane"
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
    the lane it looks like it edits, so every one must answer every lane."""
    for label, text in HARNESS_MUST_FALL_BACK:
        assert _harness_answer(text) is None, \
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
    assert not sel["fallback"], "the harness against its own HEAD should not fall back"


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
