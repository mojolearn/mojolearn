"""Parallel and remote frozen dependency analysis (prefetch); small git fixtures, no box, no ssh."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import release_requirement_dependencies as deps

STUBS = {
    # The frozen tree's own analyzers, reduced to what _worker_snapshot imports for these requests.
    "tools/release_reuse.py": "from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n",
    "tools/lane_select.py": ("from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n"
                             "def select(paths, ref=None, backend=None):\n"
                             "    return dict(lanes=[], selected=[], paths=sorted(paths))\n"
                             "def _python_closure(paths):\n    return set(paths)\n"),
    "tools/bincache.py": "",
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.test")
    git(root, "config", "user.name", "Prefetch Test")
    for name, text in STUBS.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    for n in range(4):
        (root / f"data{n}.txt").write_text(f"base {n}\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "base")
    base = git(root, "rev-parse", "HEAD")
    (root / "data0.txt").write_text("changed\n")
    git(root, "commit", "-q", "-am", "head")
    deps._FAILED.clear()
    return dict(root=root, base=base, head=git(root, "rev-parse", "HEAD"), logs=tmp_path / "logs")


def requests(repo):
    return [dict(extra_paths=[f"data{n}.txt"]) for n in range(4)]


class Slow:
    """A worker that takes a while and records how many ran at once."""

    def __init__(self, real, delay=0.4, fail=()):
        self.real, self.delay, self.fail = real, delay, set(fail)
        self.calls, self.active, self.peak = [], 0, 0
        self.lock = threading.Lock()

    def __call__(self, tree, temporary, request, log):
        with self.lock:
            self.calls.append(request)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if json.dumps(request, sort_keys=True) in self.fail:
                raise ValueError("frozen dependency analysis exited 1")
            return self.real(tree, temporary, request, log)
        finally:
            with self.lock:
                self.active -= 1


def test_prefetch_runs_every_needed_snapshot_at_once_then_the_real_pass_is_all_cache(repo, monkeypatch):
    slow = Slow(deps._run_snapshot_worker)
    monkeypatch.setattr(deps, "_run_snapshot_worker", slow)

    def compute():
        # A sequential consumer like the ledger gate: one row after another, each row's analysis
        # error kept on that row (RequirementsLedger._equivalence catches ValueError per row).
        rows = []
        for r in requests(repo):
            try:
                rows.append(deps.frozen_snapshot(repo["root"], repo["head"], r, log_dir=repo["logs"])["extra_sources"])
            except ValueError as error:
                rows.append(str(error))
        return rows

    ran = deps.prefetch(compute, jobs=4, say=lambda line: None)
    assert len(ran) == 4 and all(row["ok"] for row in ran)
    assert slow.peak == 4   # all four at once (wall time on a loaded Mac is not a stable assertion)
    calls = len(slow.calls)
    result = compute()
    assert len(slow.calls) == calls == 4   # the real pass ran nothing
    assert [list(r) for r in result] == [[f"data{n}.txt"] for n in range(4)]


def test_dependent_snapshots_take_one_round_each_and_match_a_sequential_run(repo, monkeypatch, tmp_path):
    slow = Slow(deps._run_snapshot_worker, delay=0.05)
    monkeypatch.setattr(deps, "_run_snapshot_worker", slow)

    def compute(logs):
        first = deps.frozen_snapshot(repo["root"], repo["head"], dict(extra_paths=["data1.txt"]), log_dir=logs)
        # The second request depends on the first result (as the gate's requests depend on the manifest).
        key = sorted(first["extra_sources"])[0]
        second = deps.frozen_snapshot(repo["root"], repo["head"], dict(extra_paths=[key.replace("1", "2")],
                                      base_revision=repo["base"]), log_dir=logs)
        return first, second

    rounds = []
    deps.prefetch(lambda: compute(repo["logs"]), jobs=4, say=rounds.append)
    assert sum("round" in line for line in rounds) == 2
    parallel = compute(repo["logs"])
    sequential = compute(tmp_path / "sequential")
    assert parallel == sequential
    assert parallel[1]["selections"]["cpu"]["changed_paths"] == ["data0.txt"]


def test_a_failed_prefetch_is_reported_never_rerun(repo, monkeypatch):
    bad = dict(extra_paths=["data3.txt"])
    slow = Slow(deps._run_snapshot_worker, delay=0.01, fail={json.dumps(bad, sort_keys=True)})
    monkeypatch.setattr(deps, "_run_snapshot_worker", slow)

    def compute():
        return [deps.frozen_snapshot(repo["root"], repo["head"], r, log_dir=repo["logs"]) for r in requests(repo)]

    ran = deps.prefetch(compute, jobs=2, say=lambda line: None)
    assert sorted(row["ok"] for row in ran) == [False, True, True, True]
    calls = len(slow.calls)
    with pytest.raises(ValueError, match="failed in prefetch"):
        compute()
    assert len(slow.calls) == calls


def test_jobs_default_to_every_core_and_reject_nonsense(monkeypatch):
    monkeypatch.delenv("MOJOLEARN_ADMISSION_JOBS", raising=False)
    assert deps.admission_jobs() == (os.cpu_count() or 1)
    monkeypatch.setenv("MOJOLEARN_ADMISSION_JOBS", "3")
    assert deps.admission_jobs() == 3
    monkeypatch.setenv("MOJOLEARN_ADMISSION_JOBS", "0")
    with pytest.raises(ValueError):
        deps.admission_jobs()


FAKE_SSH = '''#!{python}
"""ssh stand-in: drop options and the destination, run the command in a fake box home."""
import os, subprocess, sys
args = sys.argv[1:]
while args and args[0].startswith("-"):
    option = args.pop(0)
    if option == "-G":
        sys.exit(0)
    if option in ("-o", "-p", "-i", "-l") and args:
        args.pop(0)
target, command = args[0], " ".join(args[1:])
with open(os.environ["FAKE_BOX_LOG"], "a") as log:
    log.write(target + " " + command.split()[0] + "\\n")
sys.exit(subprocess.run(["sh", "-c", command], cwd=os.environ["FAKE_BOX_HOME"]).returncode)
'''


@pytest.fixture
def box(tmp_path, monkeypatch):
    home = tmp_path / "box-home"
    home.mkdir()
    fake = tmp_path / "fake-ssh"
    fake.write_text(FAKE_SSH.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("MOJOLEARN_ADMISSION_SSH", str(fake))
    monkeypatch.setenv("MOJOLEARN_ADMISSION_REMOTE_PYTHON", sys.executable)
    monkeypatch.setenv("MOJOLEARN_ADMISSION_REMOTE_DIR", "admission")
    monkeypatch.setenv("FAKE_BOX_HOME", str(home))
    monkeypatch.setenv("FAKE_BOX_LOG", str(tmp_path / "box.log"))
    return dict(home=home, log=tmp_path / "box.log")


def test_remote_worker_runs_on_the_box_and_brings_reports_back_by_token(repo, box, monkeypatch, tmp_path):
    def compute(logs):
        return [deps.frozen_snapshot(repo["root"], repo["head"], r, log_dir=logs)
                for r in requests(repo) + [dict(extra_paths=["data1.txt"], base_revision=repo["base"])]]

    ran = deps.prefetch(lambda: compute(repo["logs"]), jobs=3, worker="builder@box", say=lambda line: None)
    assert len(ran) == 5 and all(row["ok"] and row["worker"] == "builder@box" for row in ran)
    # The Mac ran no analysis: every report came back from the box, under its token.
    reports = sorted(repo["logs"].glob("*.json"))
    assert len(reports) == 5
    for report in reports:
        body = json.loads(report.read_text())
        assert body["source_view"]["worker"] == "builder@box"
        assert body["source_view"]["method"] == "git-checkout"
        assert (box["home"] / "admission/logs" / report.name).is_file()
    assert "builder@box" in box["log"].read_text()
    # And the result equals a local sequential analysis.
    monkeypatch.setattr(deps, "_run_snapshot_worker", Slow(deps._run_snapshot_worker, delay=0))
    assert compute(repo["logs"]) == compute(tmp_path / "local")


def test_remote_reports_from_another_analyzer_are_refused(repo, box, monkeypatch):
    monkeypatch.setattr(deps, "_analyzer", lambda: "f" * 64)

    def compute():
        return deps.frozen_snapshot(repo["root"], repo["head"], dict(extra_paths=["data0.txt"]), log_dir=repo["logs"])

    ran = deps.prefetch(compute, jobs=1, worker="builder@box", say=lambda line: None)
    assert [row["ok"] for row in ran] == [False]
    assert not list(repo["logs"].glob("*.json"))
