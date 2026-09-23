# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The import-graph lane map is the outer bound of the selector's map, and an
unmapped file is refused (CPU only: source parsing, no compiler, no GPU).

    python3 -m pytest -q tools/tests/test_lane_map_import_graph.py
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import lane_map_import_graph as graph  # noqa: E402
import lane_select  # noqa: E402


def _identity(rel):
    return rel.endswith((".mojo", ".py"))


def test_the_graph_map_is_a_superset_of_the_selectors_map_for_identity_paths():
    """THE BOUND. For every lane, every .mojo and .py file the selector
    attributes to it is reached from that lane by an import path: a file the
    selector places by any rule of its own must be one the compiler links
    into a binding the lane loads, or one the lane's Python imports. A file
    here that the graph does not reach is a hand-kept edge or an import form
    the walk does not read, and both are defects (`--compare` names them)."""
    derived, _ = graph.derive()
    narrow, _ = lane_select.lane_sources()
    assert set(derived) == set(narrow) and len(narrow) > 200
    bad = []
    for lane in sorted(narrow):
        extra = {rel for rel in narrow[lane] - derived[lane] if _identity(rel)}
        if extra:
            bad.append(f"{lane}: {sorted(extra)[:4]}")
    assert not bad, "the selector attributes files no import path reaches:\n  " + "\n  ".join(bad[:8])
    # and the bound is not trivially the whole tree
    rev = graph.reverse(derived)
    every = sum(1 for rel, lanes in rev.items() if _identity(rel) and len(lanes) == len(derived))
    assert every < len([r for r in rev if _identity(r)]) // 2, "every file reaches every lane; the bound says nothing"


def test_a_file_only_the_graph_reaches_is_attributed_by_the_graph_never_widened(monkeypatch):
    """A file compiled into a binding some lanes load, that no export any
    lane's door calls names (the per-export narrowing credits it to no lane),
    used to read NOT ATTRIBUTABLE and stop a release. Now the graph attributes
    it to the lanes loading those bindings, by a rule that says so, and never
    to every lane unless every lane loads one. Held on a real kernel with the
    selector's own map made blind to it, so the case exists whatever the
    per-export scan reaches on this tree."""
    rel = "cluster/impl/kmeans.mojo"
    sources, _ = lane_select.lane_sources()
    blind = {lane: files - {rel} for lane, files in sources.items()}
    sel = lane_select.select([rel], sources=blind)
    assert not sel["unattributed"] and not sel["every_rules"], sel["reasons"]
    assert "by the import graph" in sel["reasons"][rel], sel["reasons"][rel]
    # the core binding compiles it, so every lane DECLARED for the core
    # binding (reaching it on its own, `_mode.py` among the doors) is in,
    # kmeans first among them, and a lane that never loads it is not
    assert "kmeans" in sel["lanes"], sel["lanes"][:5]
    assert set(sel["lanes"]) == lane_select.compiled_into(rel)
    holders = {name for name in graph.build_scripts() if rel in (graph.binding_closure(name)[0] or ())}
    assert "_mojolearn" in holders and "_mojolearn_svm" not in holders, holders
    _, why = lane_select.lane_sources()
    out = [lane for lane in sel["lanes"] if not holders & set(why[lane]["declared"])]
    assert not out, out[:5]
    assert 0 < len(sel["lanes"]) <= sel["total"]


def test_an_unmapped_file_is_refused_by_name_and_selects_nothing():
    """A binding source that no map holds and no import reaches: the selector
    prints it by name, selects no lane, and never widens to every lane."""
    path = "bindings/_mojolearn_probe_that_does_not_exist.mojo"
    sel = lane_select.select([path])
    assert sel["unattributed"] == [path] and sel["lanes"] == [] and not sel["every_rules"], sel
    assert not lane_select.compiled_into(path)
    lines = []
    assert lane_select.refuse_unattributed(sel, out=lines.append) is True
    assert any(line.startswith(f"UNATTRIBUTED PATH: {path}") for line in lines), lines
    rc = subprocess.run([sys.executable, str(ROOT / "tools" / "lane_select.py"), "--lanes-for-paths", path],
                        capture_output=True, text=True, cwd=str(ROOT))
    assert rc.returncode == 3 and f"UNATTRIBUTED PATH: {path}" in rc.stdout, rc.stdout[-500:]


def test_the_mojo_closure_is_the_builds_walk(tmp_path):
    """The graph resolves a build script's root the way the build does: the
    `-I` roots on the `mojo build` line, the importer's own directory, the
    `__init__.mojo` of each package on the way, relative imports, and
    `from pkg import module`. The toolchain's modules resolve to nothing."""
    repo = tmp_path
    (repo / "bindings").mkdir()
    (repo / "pkg" / "sub").mkdir(parents=True)
    (repo / "pkg" / "__init__.mojo").write_text("")
    (repo / "pkg" / "sub" / "__init__.mojo").write_text("")
    (repo / "pkg" / "sub" / "leaf.mojo").write_text("from .sibling import g\nfrom std.math import sqrt\n")
    (repo / "pkg" / "sub" / "sibling.mojo").write_text("fn g(): pass\n")
    (repo / "pkg" / "other.mojo").write_text("fn h(): pass\n")
    (repo / "bindings" / "helper.mojo").write_text("fn k(): pass\n")
    (repo / "bindings" / "_mojolearn_x.mojo").write_text(
        "from pkg.sub.leaf import f\nfrom pkg import other\nfrom helper import k\nfrom max.gpu.host import DeviceContext\n")
    (repo / "bindings" / "build_x.sh").write_text(
        "pixi run mojo build -I . -I bindings bindings/_mojolearn_x.mojo -o out.so\n")
    keep = graph.ROOT
    graph.ROOT = str(repo)
    try:
        files, scope = graph.mojo_closure("bindings/build_x.sh")
    finally:
        graph.ROOT = keep
    assert scope == "closure"
    assert {"bindings/_mojolearn_x.mojo", "pkg/__init__.mojo", "pkg/sub/__init__.mojo", "pkg/sub/leaf.mojo",
            "pkg/sub/sibling.mojo", "pkg/other.mojo", "bindings/helper.mojo", "bindings/build_x.sh"} <= files, files
    assert not any(f.startswith("max") for f in files)


def test_a_binding_table_read_as_code_routes_the_saved_model_classes():
    """`_classical_host.py` loads `_HOST_BASENAMES[name]`: the four
    inference-only host bindings it names reach the map through that table,
    and a public class it subclasses routes to the host binding a saved model
    of it loads."""
    names = lane_select._binding_names(os.path.join(lane_select.PKG, "_classical_host.py"))
    assert {"_mojolearn_gp_infer_host", "_mojolearn_hdbscan_infer_host", "_mojolearn_mixture_infer_host",
            "_mojolearn_estimators_host"} <= names, sorted(names)
    routes = lane_select.host_class_routes()
    assert routes.get("StandardScaler") == {"_mojolearn_estimators_host"}, routes.get("StandardScaler")
    assert routes.get("GaussianMixture") == {"_mojolearn_mixture_infer_host"}, routes.get("GaussianMixture")
    assert "NumericModeMixin" not in routes, "an estimator's own _BINDING is not a host route"
    # the scaler lanes are DECLARED for the estimators host binding by that
    # route (their saved model loads through it); a lane that only resolves
    # the binding through a shared door carries its source and no more
    _, why = lane_select.lane_sources()
    assert "_mojolearn_estimators_host" in why["standard-scaler"]["declared"], why["standard-scaler"]["declared"]
    undeclared = [lane for lane, ev in why.items() if "_mojolearn_estimators_host" not in ev["declared"]]
    assert "byte-lm" in undeclared and len(undeclared) > len(why) // 2, len(undeclared)
    sel = lane_select.select(["bindings/_mojolearn_estimators_host.mojo"])
    assert "standard-scaler" in sel["lanes"], sel["lanes"][:8]
    sel = lane_select.select(["preprocessing/host/scaler_oracle.mojo"])
    assert "standard-scaler" in sel["lanes"] and "kmeans" not in sel["lanes"], sel["lanes"][:8]


def test_check_stamps_agrees_with_a_stamp_the_build_wrote(tmp_path, monkeypatch):
    """A stamp records the closure the build digested; the graph derives the
    same set for the same script, and a stamp that recorded something else
    is reported."""
    import json
    import binding_stamps
    repo = tmp_path
    (repo / "bindings").mkdir()
    (repo / "python" / "mojolearn").mkdir(parents=True)
    (repo / "k.mojo").write_text("fn f() -> Int:\n    return 1\n")
    (repo / "bindings" / "_mojolearn_k.mojo").write_text("from k import f\n")
    (repo / "bindings" / "build_k.sh").write_text(
        "pixi run mojo build -I . -I bindings bindings/_mojolearn_k.mojo -o python/mojolearn/_mojolearn_k.so\n")
    so = repo / "python" / "mojolearn" / "_mojolearn_k.so"
    so.write_bytes(b"\0")
    subprocess.run(["git", "init", "-q", str(repo)])
    stamps = repo / "python" / ".binding-stamps"
    rec = binding_stamps.cmd_write("build_k.sh", so, stamps, repo / "python" / "mojolearn", repo)
    assert set(rec["sources"]) >= {"k.mojo", "bindings/_mojolearn_k.mojo", "bindings/build_k.sh"}, rec["sources"]
    monkeypatch.setattr(graph, "ROOT", str(repo))
    monkeypatch.setattr(binding_stamps, "STAMPS", stamps)
    out = []
    assert graph.check_stamps(out=out.append) == 0, out
    assert out[-1].startswith("# 1 stamp(s)"), out
    sp = next(stamps.glob("*.json"))
    rec["sources"].append("not/compiled.mojo")
    sp.write_text(json.dumps(rec))
    out = []
    assert graph.check_stamps(out=out.append) == 1, out
    assert "not/compiled.mojo" in out[0], out
