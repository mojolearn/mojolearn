# SPDX-License-Identifier: Apache-2.0
"""tools/identity_at_freeze.py: records made at the frozen source are adopted by commit equality (no
selector), and a freeze that moved after a record says so loudly with the owed lanes (Andrew 2026-11-05:
identity runs ONCE per release, on the source that ships). Temporary Git repositories only: no device,
no network, no selector over the real tree."""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import identity_at_freeze as F  # noqa: E402
import release_identity_reference as RIR  # noqa: E402


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, files, message):
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def table(records):
    """Two lanes; nvidia and ptx rows on record 0/1 (records: [(class, commit), ...])."""
    recs = [dict(**{"class": c}, commit=sha, commit_time=str(1000 + i), file="r%d.json" % i)
            for i, (c, sha) in enumerate(records)]
    cols = {c: i for i, (c, _) in enumerate(records)}
    cells = {"lane1/base": {"batch": dict(cols=cols, ref="aa")},
             "lane2/base": {"batch": dict(cols=cols, ref="bb"), "batchgrad": dict(cols=cols, ref="n/a:no-backward")}}
    return dict(records=recs, cells=cells)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "src"
    r.mkdir()
    git(r, "init", "-q")
    return r


def recorded_then_admitted(repo, column="nvidia"):
    """X: the source a refresh recorded; M: X plus the admission (table, records, README) only."""
    x = commit(repo, {"python/mojolearn/a.py": "x = 1\n", "tools/identity_break.py": "# harness\n"}, "source")
    t = table([(column, x), ("ptx", x)])
    m = commit(repo, {F.TABLE_PATH: json.dumps(t), "bench/results/identity_break/r0.json": "{}",
                      "bench/results/reference_admissions/2026-11-05-ref-x/README.md": "GAPS: none.\n"}, "admit")
    return x, m, t


def no_selector(paths, ref, backend):
    raise AssertionError("a source-equal record must be adopted without the selector")


def test_record_outputs_are_the_only_paths_dropped():
    assert F.is_record_output(F.TABLE_PATH)
    assert F.is_record_output("bench/results/identity_break/retained/x/y.json")
    assert F.is_record_output("bench/results/reference_admissions/2026-11-05-ref-x/README.md")
    for path in ("python/mojolearn/verify_reference/other.json", "bench/results/identity_breakage.json",
                 "tools/identity_break.py", "bindings/x.mojo", "docs/release/x.md"):
        assert not F.is_record_output(path), path
    assert F.source_paths(["bindings/x.mojo", F.TABLE_PATH, ""]) == ["bindings/x.mojo"]


def test_a_refresh_at_the_frozen_source_is_adopted_by_commit_equality(repo):
    x, m, t = recorded_then_admitted(repo)
    assert F.source_equal(repo, x, m) and F.source_equal(repo, m, m)
    report = F.judge(repo, m, t, select=no_selector)
    for col in ("nvidia", "ptx"):
        row = report["columns"][col]
        assert row["state"] == "adopted" and row["owed"] == 0 and row["recorded_at"] == x
        assert row["source_paths"] == 0 and "source-equal" in row["reason"]
    assert report["adopted"] is True and report["held"] == []
    assert any(line.startswith("identity nvidia: ADOPTED at freeze " + m[:12]) for line in F.lines(report))


def test_a_record_at_exactly_the_freeze_says_so(repo):
    x = commit(repo, {"python/mojolearn/a.py": "x = 1\n"}, "source")
    t = table([("amd", x)])
    report = F.judge(repo, x, t, select=no_selector)
    assert report["columns"]["amd"]["state"] == "adopted"
    assert report["columns"]["amd"]["reason"] == "recorded at the freeze"


def test_a_freeze_that_moved_after_the_record_owes_the_selected_lanes_loudly(repo):
    x, m, t = recorded_then_admitted(repo)
    y = commit(repo, {"bindings/classifier_public_lifecycle.mojo": "fn f(): pass\n"}, "product fix")
    asked = []

    def select(paths, ref, backend):
        asked.append((paths, ref, backend))
        return dict(lanes=["lane1", "not-in-table"], unattributed=[])
    report = F.judge(repo, y, t, select=select)
    row = report["columns"]["nvidia"]
    assert row["state"] == "owed" and row["owed"] == 1 and row["owed_lanes"] == ["lane1"] and row["held"]
    assert row["first_paths"] == ["bindings/classifier_public_lifecycle.mojo"]   # record outputs dropped
    assert asked[0] == (["bindings/classifier_public_lifecycle.mojo"], x, "cuda")
    assert len(asked) == 1        # nvidia and ptx share (commit, cuda): one selection
    assert report["adopted"] is False and report["held"] == ["nvidia"]
    text = "\n".join(F.lines(report))
    assert ("IDENTITY RECORDED AT %s, FREEZE IS %s: 1 lane(s) of nvidia owed (1 source path(s) differ: "
            "bindings/classifier_public_lifecycle.mojo)" % (x[:12], y[:12])) in text
    assert "HOLD BEFORE ANY BUILD" in text and "tools/reference_refresh.sh " + y[:12] in text


def test_a_change_no_lane_reaches_adopts_through_the_selector(repo):
    x, m, t = recorded_then_admitted(repo)
    y = commit(repo, {"docs/notes.md": "words\n"}, "docs")
    report = F.judge(repo, y, t, select=lambda paths, ref, backend: dict(lanes=[], unattributed=[]))
    assert report["columns"]["nvidia"]["state"] == "adopted" and report["adopted"] is True


@pytest.mark.parametrize("change,select,why", [
    ({"tools/identity_break.py": "# harness v2\n"}, no_selector, "identity harness changed"),
    ({"weird/path.bin": "x"}, lambda p, r, b: dict(lanes=[], unattributed=["weird/path.bin"]), "unattributed"),
    ({"bindings/x.mojo": "x"}, lambda p, r, b: (_ for _ in ()).throw(RuntimeError("boom")), "selector failed"),
])
def test_harness_unattributed_or_failed_selection_owes_every_lane(repo, change, select, why):
    x, m, t = recorded_then_admitted(repo)
    y = commit(repo, change, "change")
    row = F.judge(repo, y, t, select=select)["columns"]["nvidia"]
    assert row["owed"] == 2 and why in row["reason"] and row["state"] in ("owed", "undecided")


def test_no_selector_never_means_unchanged(repo):
    x, m, t = recorded_then_admitted(repo)
    y = commit(repo, {"bindings/x.mojo": "x"}, "change")
    row = F.judge(repo, y, t, select=None)["columns"]["nvidia"]
    assert row["state"] == "undecided" and row["owed"] == 2


def test_an_owed_ptx_column_is_reported_but_never_holds(repo):
    x, m, t = recorded_then_admitted(repo, column="amd")
    y = commit(repo, {"bindings/x.mojo": "x"}, "change")

    def select(paths, ref, backend):
        return dict(lanes=["lane1"] if backend == "cuda" else [], unattributed=[])
    report = F.judge(repo, y, t, select=select)
    assert report["columns"]["ptx"]["state"] == "owed" and not report["columns"]["ptx"]["held"]
    assert report["columns"]["amd"]["state"] == "adopted"
    assert report["adopted"] is True
    assert "[not held: recorded after linux-builds]" in "\n".join(F.lines(report))


def write_stub_selector(repo, lanes):
    """The source checkout's own tools/lane_select.py (main() imports it from --root)."""
    return commit(repo, {"tools/lane_select.py": "def select(paths, ref=None, backend=None, **kw):\n"
                                                  "    return dict(lanes=%r, unattributed=[])\n" % lanes}, "selector")


def test_main_exits_0_when_adopted_and_4_when_owed(repo, tmp_path, monkeypatch, capsys):
    write_stub_selector(repo, ["lane2"])          # part of the recorded source, so never the tooling copy
    x, m, t = recorded_then_admitted(repo)
    monkeypatch.setattr(sys, "path", list(sys.path))
    real = sys.modules.pop("lane_select", None)
    out = tmp_path / "report.json"
    assert F.main(["check", "--root", str(repo), "--freeze", m, "--json", str(out)]) == 0
    assert json.loads(out.read_text())["adopted"] is True
    y = commit(repo, {"bindings/x.mojo": "fn f(): pass\n"}, "product fix")
    sys.modules.pop("lane_select", None)
    assert F.main(["check", "--root", str(repo), "--freeze", y, "--json", str(out)]) == 4
    doc = json.loads(out.read_text())
    assert doc["schema"] == F.SCHEMA and doc["held"] == ["nvidia"] and doc["columns"]["nvidia"]["owed_lanes"] == ["lane2"]
    assert "IDENTITY RECORDED AT " + x[:12] in capsys.readouterr().out
    sys.modules.pop("lane_select", None)
    if real is not None:
        sys.modules["lane_select"] = real


def test_main_refuses_a_checkout_that_is_not_at_the_freeze(repo, tmp_path):
    x, m, t = recorded_then_admitted(repo)
    assert F.main(["check", "--root", str(repo), "--freeze", x, "--json", str(tmp_path / "r.json")]) == 2
    assert not (tmp_path / "r.json").exists()


def test_the_ledger_reference_adopts_a_source_equal_record_without_a_selection(repo):
    # release_identity_reference.verdicts asks the selector only for bases that are NOT source-equal
    x, m, _ = recorded_then_admitted(repo)
    equal = RIR.source_equal_to(repo, m)
    assert equal(m) and equal(x)
    y = commit(repo, {"bindings/x.mojo": "x"}, "change")
    later = RIR.source_equal_to(repo, y)
    assert not later(x) and not later(m)
    assert not later("0" * 40)          # an unknown commit is never source-equal (the selector decides)
