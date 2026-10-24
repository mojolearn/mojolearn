#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane selector checks that re-derive the whole lane map (2026-10-24).

Each test here replaces one of the selector's own edges (an extender walk, the
Mojo import graph, the public-door resolution) and so must re-derive the map
from scratch: about a minute per test on an Apple M-series laptop, longer on a
pod CPU. They were split out of tools/test_lane_select.py, whose other checks
share one derived map; run this file on its own time budget.

    .pixi/envs/test/bin/python -m pytest tools/test_lane_select_rebuild.py -q
    python3 tools/test_lane_select_rebuild.py          # same checks, no pytest
"""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select                                              # noqa: E402
from test_lane_select import _conformance_misses, _lane_sets  # noqa: E402


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
    real_targets = lane_select._mojo_import_targets
    lane_select._mojo_imports = lambda rel: ({f for f in real(rel) if f != victim}
                                             if rel != victim else real(rel))
    # The symbol walk has its own import view. Cut the same edge there:
    # changing only the whole-file view no longer detaches the implementation.
    def cut_targets(rel):
        names, whole = real_targets(rel)
        if rel == victim:
            return names, whole
        return ({name: {(file, symbol) for file, symbol in targets if file != victim}
                 for name, targets in names.items()}, whole - {victim})
    lane_select._mojo_import_targets = cut_targets
    lane_select.reset_caches()
    try:
        broken_sources, broken = _lane_sets()
        assert len(broken.get(declarer, set())) == before_b, \
            "the sabotage moved the DECLARER too, so the two arms move together and prove nothing"
        assert not broken.get(victim, set()), "the sabotage did not detach the implementation"
        # The importers are the tree's REAL imports: the sabotage hid the
        # victim from the walk, not from the source.
        fired = [f for f, _t, b in lane_select.mojo_conformance_edges()
                 if _conformance_misses(broken_sources, broken, [(f, _t, b)], real)]
        assert victim in fired, f"the inversion did not name {victim}; it caught {fired[:4]}"
    finally:
        lane_select._mojo_imports = real
        lane_select._mojo_import_targets = real_targets
        lane_select.reset_caches()


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
    # STALE CASE DROPPED 2026-10-10: language_model.py is no longer a door.
    # 76c025466 (2026-10-02) made __init__.py bind the language-model names on
    # first touch (`_CPU_INFERENCE`, name -> module) instead of an eager
    # `from .language_model import ...`, and binds the names `byte-lm` uses
    # (SmallByteLanguageModelTrainer, ByteLanguageModelConfig) from
    # `._byte_lm_impl` / `._byte_lm_config`, which define them. So `byte-lm`
    # no longer executes language_model.py, and __init__.py binds nothing from
    # it by import (this test's own "pick a new case" guard below). The file
    # stays in `language-model-config`'s map by the definition rule
    # (`LanguageModelConfig = ByteLanguageModelConfig`), not by a door. The
    # door rule is still held on the two eager doors.
    _DOORS = (("python/mojolearn/umap.py", "umap"),
              ("python/mojolearn/neural_network.py", "mlp"))
    for rel, want in _DOORS:
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
    assert len(rev.get("python/mojolearn/neural_inference.py", ())) == 89, \
        ("the re-export rule moved neural_inference.py off its measured 89 lanes (40 + hf-checkpoint,"
         " 2026-09-23; minus gbdt-tensor-ctr-tables, which fits on the CPU since 2026-09-28;"
         " plus optim-maximize, 2026-09-28; plus four new lanes and minus gbdt-categorical-ctr-tables,"
         " 2026-10-21; plus five coverage lanes in 63022df0c2;"
         " plus nine sequence optimizer lanes, 3ce0c0b26 2026-10-21, and five sequence device lanes,"
         " e492586c6 2026-10-21, whose doors now import _training_impl -> _native_samba_bridge ->"
         " neural_inference (2430b2684);"
         " plus five new lanes in 1f1183e21 and two in 50c405ad2, 2026-10-21;"
         " plus four verify-gaps lanes in d332cd33c, 2026-10-22;"
         " plus eleven existing sequence lanes through _native_xseq_bridge -> _training_impl"
         " (58ab6ec69, d82367475, 061c9f767, bbd1ecc88), measured 2026-10-23;"
         " plus public-numeric-metadata, public-residual-dropout and public-theta-variants"
         " (8188d769f) and transformer-forward-tape (551a8cd42), 2026-10-24; see "
         "test_the_wider_mojo_walk_did_not_widen_the_narrow_answers)")

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
        for rel, want in _DOORS:
            assert want not in blind.get(rel, set()), \
                f"{want} reaches {rel} without the public door rule, so this test cannot fail"
    finally:
        lane_select._public_rebindings = keep
        lane_select.reset_caches()


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
