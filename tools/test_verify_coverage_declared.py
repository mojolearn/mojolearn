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
