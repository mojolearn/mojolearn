"""The precise-coverage declarations (tools/identity_lanes/coverage_declared.tsv) stay true to the tree."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verify_coverage_audit as audit  # noqa: E402

ROOT = audit.ROOT


def _rows():
    return audit.declared()


def test_every_declared_registration_exists():
    missing = []
    regs = {}
    for (entry, src) in _rows():
        if src == "*":
            continue
        if src not in regs:
            path = os.path.join(ROOT, src)
            regs[src] = set(audit.registrations(open(path).read())) if os.path.exists(path) else set()
        if entry not in regs[src]:
            missing.append(f"{entry} {src}")
    assert not missing, "declared but not registered: " + ", ".join(missing[:20])


def test_every_declared_lane_exists():
    import identity_break as ib
    unknown = sorted({lane for status, lanes, _ in _rows().values() for lane in lanes} - set(ib.LANES))
    assert not unknown, "declared lanes not in identity_break.LANES: " + ", ".join(unknown[:20])


def test_declaration_rules(tmp_path):
    good = tmp_path / "ok.tsv"
    good.write_text("# comment\n"
                    "e1\tbindings/_mojolearn.mojo\tcovered\tlane-a,lane-b\tthe path\n"
                    "e2\t*\tdead\t\tonly tools/ names it\n"
                    "e3\tbindings/_mojolearn.mojo\tn/a\t\tfast tier only\n")
    table = audit.declared(str(good))
    assert table[("e1", "bindings/_mojolearn.mojo")] == ("covered", ["lane-a", "lane-b"], "the path")
    assert table[("e2", "*")][0] == "dead"
    for bad in ("e\tbindings/_mojolearn.mojo\tcovered\t\twhy\n",          # covered without lanes
                "e\tbindings/_mojolearn.mojo\tdead\tlane\twhy\n",          # dead with lanes
                "e\tbindings/_mojolearn.mojo\tgone\t\twhy\n",              # unknown status
                "e\tpython/x.py\tdead\t\twhy\n",                           # not a binding
                "e\tbindings/_mojolearn.mojo\tdead\t\t \n",                # no reason
                "e\tbindings/_mojolearn.mojo\tdead\t\twhy\n" * 2):         # twice
        path = tmp_path / "bad.tsv"
        path.write_text(bad)
        with pytest.raises(SystemExit):
            audit.declared(str(path))


def test_triage_and_reach_state_follow_the_declaration(monkeypatch):
    table = {("x_probe", "bindings/_mojolearn_x.mojo"): ("covered", ["lane-a"], "the loader reads it"),
             ("x_old", "*"): ("dead", [], "only a tool names it")}
    monkeypatch.setattr(audit, "_DECLARED", table)
    row = dict(entry="x_probe", src="bindings/_mojolearn_x.mojo", lanes_precise=[], lanes_wide=["lane-a", "lane-b"],
               py_files=0, constant_probe=True)
    assert audit.triage(row) == ("covered", ["lane-a"], "the loader reads it")
    assert audit.reach_state(row) == "declared"
    dead = dict(row, entry="x_old")
    status, lanes, why = audit.triage(dead)
    assert (status, lanes) == ("n/a", []) and why.startswith(audit.DEAD_PREFIX)
    assert audit.reach_state(dead) == "decided"
    plain = dict(row, entry="x_other")
    assert audit.reach_state(plain) == "wide"
    assert audit.reach_state(dict(plain, lanes_precise=["lane-c"])) == "static"


def test_public_cnn_and_optimizer_lanes_have_faults_and_published_references():
    """Track CNN exports, the two standalone optimizers and the three Session extensions under review."""
    import ast
    import json
    import re
    from pathlib import Path

    import coverage_sabotage
    import identity_break

    root = Path(ROOT)
    lanes = {
        "Conv2d": "x-cnn-conv2d", "Conv1d": "x-cnn-conv1d",
        "MaxPool2d": "x-cnn-pool", "AvgPool2d": "x-cnn-pool",
        "MaxPool1d": "x-cnn-pool", "AvgPool1d": "x-cnn-pool",
        "BatchNorm1d": "x-cnn-batchnorm", "BatchNorm2d": "x-cnn-batchnorm",
        "AdaptiveAvgPool2d": "x-cnn-globalpool", "AdaptiveMaxPool2d": "x-cnn-globalpool",
        "CNNClassifier": "x-cnn-trainer", "Dropout2d": "x-cnn-dropout2d",
        "BasicBlock": "x-cnn-resnet-block", "GCNConv": "x-cnn-gcn", "SAGEConv": "x-cnn-sage",
        "Adafactor": "sequence-adafactor", "LAMB": "sequence-lamb",
        "Session.scan": "ops-scan-cell", "Session.attention": "ops-attention-qkv",
        "Session.custom_op": "nn-cbw-custom-vjp",
    }
    path = root / "python/mojolearn/_expansion_cnn.py"
    exports, = [ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__"
                                                        for t in node.targets)]
    assert set(exports) <= lanes.keys(), f"CNN exports missing verification lanes: {set(exports) - lanes.keys()}"
    sides = {}
    for arm in coverage_sabotage.arms().values():
        for lane in arm["lanes"]:
            sides.setdefault(lane, set()).update(coverage_sabotage.arm_sides(arm))

    for lane in {*lanes.values(), "sequence-optim-adafactor", "nn-ep-cnn-fits", "nn-ep-cnn-layers", "x-cnn-gnn-disabled-options"}:
        assert lane in identity_break.LANES, f"missing installed verifier lane: {lane}"
        assert sides.get(lane) == {"cpu", "gpu"}, f"{lane} declares faults on {sides.get(lane)}"

    excluded = {line.split("\t")[0] for line in (root / "docs-site/api_exclude.txt").read_text().splitlines()}
    cells = json.loads((root / "python/mojolearn/verify_reference/table.json").read_text())["cells"]
    for name, lane in lanes.items():
        if name in excluded:
            continue
        ref = cells.get(f"{lane}/base", {}).get("train", {})
        assert re.fullmatch(r"[0-9a-f]+", ref.get("ref", "")), f"{name} lacks a recorded training digest"
        assert {"apple", "nvidia", "amd", "cpu"} <= ref["cols"].keys(), f"{name} lacks a device reference"
