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
