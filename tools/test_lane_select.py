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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select                                              # noqa: E402

#: A path every lane in the family depends on, and a lane that must NOT be
#: selected by it. The second half is the half that catches a selector which
#: has quietly widened to "everything" and so looks right by accident.
FAMILY_CASES = (
    ("glm/host/qn_oracle.mojo", "logistic", ("mamba1", "transformer", "tokenizer")),
    ("cluster/host/kmeans_oracle.mojo", "kmeans", ("mamba1", "transformer", "arima")),
    ("mamba/impl", "mamba1", ("ols", "kmeans", "tokenizer")),
)


def test_registry_count_comes_from_the_import():
    """199 lanes, read by importing the registry. `grep -c '@lane('` answers
    176 because 23 lanes register by call; a selector built on the grep would
    silently skip them."""
    lanes = lane_select.all_lanes()
    assert len(lanes) == len(set(lanes)), "a lane name is registered twice"
    assert len(lanes) > 176, f"only {len(lanes)} lanes: the registry was read by grep, not by import"


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
        assert must in sel["lanes"], f"{path} did not select {must}"
        if not sel["fallback"]:
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


def test_binding_edges_do_not_come_from_prose():
    """A binding named in a docstring is not a call into it. If prose counted,
    the shared doors would hand every lane the forest and byte LM host
    bindings, and a change to one oracle would select all 199 lanes."""
    sources, why = lane_select.lane_sources()
    for lane in ("tokenizer", "mamba1", "ols"):
        assert "_mojolearn_forest_host" not in why[lane]["bindings"], \
            f"{lane} picked up the forest host binding from prose"


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
