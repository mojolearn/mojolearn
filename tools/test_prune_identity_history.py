# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""tools/prune_identity_history.py: which dated identity_break dirs are history.

A fixture git repository stands in for the tree; nothing imports a binding or
runs a fit. Each dated dir below exercises one KEEP or PRUNE rule.
"""
import gzip
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import prune_identity_history as pih  # noqa: E402

BRK = "bench/results/identity_break"
A, B, C, D = "2026-09-01_table-cited", "2026-09-02_retained-named", "2026-09-03_nv-newest", "2026-09-04_nv-older"
E, F, G, H = "2026-09-05_code-named", "2026-09-06_docs-only", "2026-09-07_other-tree", "2026-09-08_apple-copy"
I, J, K = "2026-09-09_amd-lane-l2", "2026-09-10_amd-newest", "2026-09-11_report-only"


def git(root, *args, env=None):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def commit_at(root, when, msg):
    env = dict(os.environ, GIT_AUTHOR_DATE=f"@{when} +0000", GIT_COMMITTER_DATE=f"@{when} +0000",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    git(root, "commit", "-q", "--allow-empty", "-m", msg, env=env)
    return git(root, "rev-parse", "HEAD")


def record(vendor, commit, lanes):
    return dict(mode="identical", complete=True, vendor=vendor, commit=commit,
                cells={f"{lane}/base": {"verdict": "STABLE"} for lane in lanes})


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    t = {n: commit_at(root, 1_000_000 + 100 * n, f"c{n}") for n in (1, 2, 3, 4)}
    brk = root / BRK

    def rec(d, name, vendor, c, lanes):
        write(brk / d / name, json.dumps(record(vendor, c, lanes)))

    rec(A, "nvidia/nvidia-l40s.identical.json", "nvidia-l40s", t[1], ["l1"])
    write(brk / B / "README.txt", "notes\n")
    rec(C, "nvidia-l40s.identical.json", "nvidia-l40s", t[4], ["l1"])
    rec(D, "nvidia-l40s.identical.json", "nvidia-l40s", t[2], ["l1"])
    write(brk / E / "README.txt", "notes\n")
    write(brk / F / "README.txt", "notes\n")
    write(brk / G / "README.txt", "notes\n")
    rec(H, "apple-m2.identical.json", "apple-m2", t[3], ["l1"])
    rec(I, "amd-mi325x.identical.json", "amd-mi325x", t[1], ["l1", "l2"])
    rec(J, "amd-mi325x.identical.json", "amd-mi325x", t[3], ["l1"])
    write(brk / K / "README.txt", "notes\n")
    write(brk / "nvidia-leg-x" / "README.txt", "not dated\n")
    ret = brk / "retained" / ("a" * 64)
    # H's record, byte-copied into the live set: the retained copy wins the tie.
    write(ret / "apple-m2.identical.json", (brk / H / "apple-m2.identical.json").read_text())
    write(ret / "apple-m2.provenance.txt", f"source=bench/results/identity_break/{B}/x.json\n"
                                           f"other=bench/results/identity_columns/{G}/y.json\n")
    with gzip.open(ret / "apple-m2.identical.source-report.json.gz", "wt") as fh:
        fh.write(json.dumps({"inputs": [f"bench/results/identity_break/{K}/z.json"]}))
    table = dict(records=[dict(dir=f"{A}/nvidia", file="nvidia-l40s.identical.json", commit=t[1])], cells={})
    write(root / pih.REL_TABLE, json.dumps(table))
    write(root / "tools" / "leg.sh", f'OUT="$ROOT/{BRK}/{E}"\n')
    write(root / "docs" / "notes.md", f"see {BRK}/{F}\n")
    git(root, "add", "-A")
    commit_at(root, 1_000_900, "tree")
    return root


def verdicts(root):
    rows, other = pih.plan(root)
    return {r[0].rsplit("/", 1)[1]: (r[1], r[2]) for r in rows}, other


def test_plan_classifies_every_rule(tree):
    got, other = verdicts(tree)
    assert other == ["nvidia-leg-x", "retained"]
    assert {n for n, (v, _) in got.items() if v == "KEEP"} == {A, B, C, E, I, J}
    assert {n for n, (v, _) in got.items() if v == "PRUNE"} == {D, F, G, H, K}
    assert "table" in got[A][1].split(",")
    assert "retained" in got[B][1].split(",")
    assert "newest" in got[C][1].split(",")
    assert got[E][1] == "code"
    assert got[I][1] == "newest-lane"      # the only amd record of lane l2
    assert "newest" in got[J][1].split(",")


def test_plan_rows_carry_size_and_head(tree):
    rows, _ = pih.plan(tree)
    head = git(tree, "rev-parse", "HEAD")
    by = {r[0]: r for r in rows}
    row = by[f"{BRK}/{A}"]
    assert row[3] == 1 and row[4] == len((tree / BRK / A / "nvidia" / "nvidia-l40s.identical.json").read_bytes())
    assert all(r[5] == head for r in rows)


def test_plan_refuses_without_table(tree):
    (tree / pih.REL_TABLE).unlink()
    with pytest.raises(SystemExit, match="no admitted table"):
        pih.plan(tree)


def test_plan_refuses_lean_worktree(tmp_path):
    root = tmp_path / "lean"
    root.mkdir()
    git(root, "init", "-q")
    with pytest.raises(SystemExit, match="not on disk"):
        pih.plan(root)


def test_main_plan_writes_tsv_and_changes_nothing(tree, tmp_path):
    out = tmp_path / "plan.tsv"
    assert pih.main(["--plan", "--root", str(tree), "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == list(pih.PLAN_HEADER)
    assert len(lines) == 1 + 11
    assert git(tree, "status", "--porcelain") == ""


def test_apply_refuses_while_release_runs(tree, monkeypatch):
    monkeypatch.setattr(pih, "release_running", lambda: ["4242"])
    with pytest.raises(SystemExit, match="release.py is running"):
        pih.apply(tree, "2026-11-07")
    assert git(tree, "status", "--porcelain") == ""


def test_apply_refuses_dirty_tree(tree, monkeypatch):
    monkeypatch.setattr(pih, "release_running", lambda: [])
    write(tree / BRK / D / "new.txt", "x\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        pih.apply(tree, "2026-11-07")


def test_apply_removes_only_prune_dirs_and_documents_them(tree, monkeypatch):
    monkeypatch.setattr(pih, "release_running", lambda: [])
    head = git(tree, "rev-parse", "HEAD")
    assert pih.apply(tree, "2026-11-07") == 0
    staged = git(tree, "diff", "--cached", "--name-status").splitlines()
    deleted = {line.split("\t")[1].split("/")[3] for line in staged if line.startswith("D\t")}
    assert deleted == {D, F, G, H, K}
    assert all(not line.split("\t")[1].startswith(f"{BRK}/retained/") for line in staged)
    assert (tree / BRK / "retained").is_dir() and (tree / BRK / A).is_dir()
    docs = (tree / pih.REL_DOCS_TSV).read_text().splitlines()
    assert docs[0].split("\t") == list(pih.DOCS_HEADER)
    assert {line.split("\t")[0].rsplit("/", 1)[1] for line in docs[1:]} == {D, F, G, H, K}
    assert all(line.split("\t")[3] == head for line in docs[1:])
    assert any(line == f"A\t{pih.REL_DOCS_TSV}" for line in staged)
