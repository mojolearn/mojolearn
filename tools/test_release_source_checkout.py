"""tools/release.py source_checkout: the detached source worktree is made once,
atomically, and verified before use (0.8.66: two pipelines raced it and one read
a half checked-out tree)."""
import argparse
import pathlib
import subprocess
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402


def sh(cwd, *cmd):
    return subprocess.run(["git", "-C", str(cwd), *cmd], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    r.mkdir()
    sh(r, "init", "-q")
    sh(r, "config", "user.email", "t@example.com")
    sh(r, "config", "user.name", "t")
    for i in range(2):
        for j in range(30):
            (r / f"f{j}.txt").write_text(f"{i}-{j}\n")
        sh(r, "add", "-A")
        sh(r, "commit", "-qm", f"c{i}")
    first = sh(r, "rev-parse", "HEAD~1")
    monkeypatch.setattr(release, "ROOT", r)
    return r, first


def make(tmp_path, commit):
    a = argparse.Namespace(version="0.8.66", dry_run=False, publish=None, only="", redo="", build_backend="github",
                           amd_expect_from="", smoke_gpu="", state_dir=str(tmp_path / "state"),
                           amd_build_provider=None, amd_provider="auto", smoke_via="lq", hopper_box="",
                           smoke_branch="main", cpu_column=False, gpu_smoke=False, owner_waiver=None, status=False,
                           refreeze=False, source_checkout="", published_smoke=[], published_smoke_reason="")
    r = release.Release(a, runner=lambda cmd, env, log, detach=False: 0)
    r.state["commit"] = commit
    r.lines = []
    r.say = r.lines.append
    return r


def count_adds(monkeypatch):
    adds = []
    real = subprocess.run

    def run(cmd, *a, **kw):
        if list(cmd[3:5]) == ["worktree", "add"]:
            adds.append(cmd)
            time.sleep(0.3)  # widen the window the 0.8.66 race fell into
        return real(cmd, *a, **kw)
    monkeypatch.setattr(release.subprocess, "run", run)
    return adds


def test_two_threads_get_one_complete_worktree(tmp_path, repo, monkeypatch):
    root, first = repo
    r = make(tmp_path, first)
    adds = count_adds(monkeypatch)
    got, errs = [], []

    def ask():
        try:
            p = r.source_checkout()
            got.append((p, sorted(x.name for x in p.glob("f*.txt"))))
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)
    ts = [threading.Thread(target=ask) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    assert len(adds) == 1
    assert ".tmp-" in adds[0][-2]
    (p1, files1), (p2, files2) = got
    assert p1 == p2 == r.base / "source" / first[:12]
    assert len(files1) == len(files2) == 30
    assert sh(p1, "rev-parse", "HEAD") == first
    assert not list(p1.parent.glob("*.tmp-*"))
    assert str(p1) in sh(root, "worktree", "list")


def test_a_second_release_process_reuses_the_verified_tree(tmp_path, repo, monkeypatch):
    _, first = repo
    p = make(tmp_path, first).source_checkout()
    adds = count_adds(monkeypatch)
    assert make(tmp_path, first).source_checkout() == p
    assert adds == []


def test_an_existing_tree_at_the_wrong_commit_is_refused(tmp_path, repo):
    root, first = repo
    p = tmp_path / "state" / "source" / first[:12]
    p.parent.mkdir(parents=True)
    sh(root, "worktree", "add", "-q", "--detach", str(p), "HEAD")
    with pytest.raises(release.StepFailed, match="not a clean checkout"):
        make(tmp_path, first).source_checkout()


def test_a_partial_tree_is_refused(tmp_path, repo):
    root, first = repo
    p = tmp_path / "state" / "source" / first[:12]
    p.parent.mkdir(parents=True)
    sh(root, "worktree", "add", "-q", "--detach", str(p), first)
    (p / "f7.txt").unlink()
    with pytest.raises(release.StepFailed, match="tracked changes"):
        make(tmp_path, first).source_checkout()


def test_a_leftover_tmp_tree_is_removed(tmp_path, repo):
    root, first = repo
    base = tmp_path / "state" / "source"
    base.mkdir(parents=True)
    stale = base / f"{first[:12]}.tmp-999999"
    sh(root, "worktree", "add", "-q", "--detach", str(stale), first)
    half = base / f"{first[:12]}.tmp-999998"
    half.mkdir()
    (half / "junk").write_text("x")
    p = make(tmp_path, first).source_checkout()
    assert (p / "f0.txt").exists()
    assert not stale.exists() and not half.exists()
    assert "tmp-" not in sh(root, "worktree", "list")
