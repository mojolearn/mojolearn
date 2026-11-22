#!/usr/bin/env python3
"""Tests for tools/hooks/check_tree_additions.sh, the pre-push check that keeps run output out of the tree.

Builds a scratch bare remote and a clone with a tracked base (a live board, an old run log), then pushes
nothing: each case makes commits in the clone and runs the check the way tools/hooks/pre-push does
(`bash check_tree_additions.sh <local sha> <remote sha> origin`). Git only, no network, a few seconds.
The cases are independent scratch repositories, so they run in parallel across the cores
(one process per case through concurrent.futures, workers capped by the core count).

  python3 tools/tests/test_check_tree_additions.py
"""
import concurrent.futures
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CHECK = os.path.join(ROOT, "tools", "hooks", "check_tree_additions.sh")
PRE_PUSH = os.path.join(ROOT, "tools", "hooks", "pre-push")
ZERO = "0" * 40


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def write(repo, path, text="x\n"):
    full = os.path.join(repo, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text)


def scratch():
    """A bare remote plus a clone whose main (base) is already on the remote."""
    top = tempfile.mkdtemp(prefix="tree-additions.")
    remote = os.path.join(top, "remote.git")
    clone = os.path.join(top, "clone")
    subprocess.run(["git", "init", "-q", "--bare", remote], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", clone], check=True)
    git(clone, "config", "user.email", "t@example.invalid")
    git(clone, "config", "user.name", "t")
    git(clone, "config", "core.hooksPath", "/dev/null")
    git(clone, "remote", "add", "origin", remote)
    write(clone, "bench/results/bench_board/board.json", "{}\n")
    write(clone, "bench/results/old_run/run.log", "old\n")
    write(clone, "docs/handoffs/README.md", "pointer\n")
    git(clone, "add", "-A")
    git(clone, "commit", "-q", "-m", "base")
    git(clone, "push", "-q", "origin", "main")
    git(clone, "checkout", "-q", "-b", "lane/x")
    return clone


def commit(clone, files, remove=()):
    for path in files:
        write(clone, path)
    for path in remove:
        git(clone, "rm", "-q", path)
    git(clone, "add", "-A", "-f")
    git(clone, "commit", "-q", "-m", "change")
    return git(clone, "rev-parse", "HEAD")


def check(clone, local_sha, remote_sha=ZERO):
    r = subprocess.run(["bash", CHECK, local_sha, remote_sha, "origin"], cwd=clone, capture_output=True, text=True)
    return r.returncode, r.stderr


# Each case: (name, files added in one lane commit, files removed, expected exit, substrings in stderr)
CASES = [
    ("allowlisted records pass", [
        "bench/results/bench_board/new.json",
        "bench/results/board_best_ever/nvidia.json",
        "bench/results/board-archive/2026-11-21/board.json",
        "bench/results/opponent-archive/x/cell.json",
        "bench/results/paper_compare/compare-nvidia.json",
        "bench/results/dataset_store/manifest.tsv",
        "bench/results/dataset_store/groups/opponents@trees-linux-x86_64-cp311.txt",
        "bench/results/dataset_store/repo-data/experiments@six_lane_integration@measurements.tsv",
        "experiments/six_lane_integration/measurements/README.md",
        "bench/results/identity_break/retained/abc/cpu.identical.json",
        "bench/results/reference_admissions/0.8.67/a.json",
        "bench/results/release_evidence/0.8.67/a.json",
        "bench/results/release_verification/0.8.67/a.json",
        "bench/results/r2-index.tsv",
        "bench/results/CELLS_INDEX.tsv",
        "bench/results/identity_break/INDEX.tsv",
        "docs/plans/NEW_PLAN.md",
        "tools/new_tool.py",
    ], [], 0, []),
    ("new run dir refused", ["bench/results/nvc-measure/2026-11-21/summary.json"], [], 1,
     ["bench/results/nvc-measure/2026-11-21/summary.json", "~/mojolearn-evidence/"]),
    ("log under identity_break refused", ["bench/results/identity_break/2026-11-21/build.log"], [], 1,
     ["identity_break/2026-11-21/build.log", "logs belong in ~/mojolearn-evidence/"]),
    ("log under a live board refused", ["bench/results/bench_board/run.log"], [], 1, ["bench_board/run.log"]),
    ("new file in a repo-data bucket refused", ["experiments/six_lane_integration/measurements/20261006/x.json"], [],
     1, ["measurements/20261006/x.json", "tools/repo_data.py"]),
    ("subdir of a dataset_store pin dir refused", ["bench/results/dataset_store/repo-data/old/a.tsv"], [], 1,
     ["dataset_store/repo-data/old/a.tsv"]),
    ("subdir of paper_compare refused", ["bench/results/paper_compare/run1/cells.json"], [], 1,
     ["paper_compare/run1/cells.json"]),
    ("subdir of board_best_ever refused", ["bench/results/board_best_ever/old/nvidia.json"], [], 1,
     ["board_best_ever/old/nvidia.json"]),
    ("new handoff refused", ["docs/handoffs/HANDOFF_2026-11-22.md"], [], 1,
     ["docs/handoffs/HANDOFF_2026-11-22.md", "handoffs live in ~/mojolearn-evidence/"]),
    ("deletions always pass", [], ["bench/results/old_run/run.log", "docs/handoffs/README.md"], 0, []),
]


def run_case(case):
    name, add, remove, want, needles = case
    clone = scratch()
    sha = commit(clone, add, remove)
    code, err = check(clone, sha)
    problems = []
    if code != want:
        problems.append(f"exit {code}, wanted {want}: {err.strip()[:400]}")
    problems += [f"stderr lacks {n!r}" for n in needles if n not in err]
    return name, problems


def run_sequence_cases():
    """Cases that need several commits or a pushed lane branch."""
    out = []
    # 1. a file added and then removed inside the same push passes
    clone = scratch()
    commit(clone, ["bench/results/tmp/run.json"])
    sha = commit(clone, [], ["bench/results/tmp/run.json"])
    code, err = check(clone, sha)
    out.append(("added then removed in one push passes", [] if code == 0 else [f"exit {code}: {err[:300]}"]))
    # 2. only commits not yet on the remote are judged: an old offending file already pushed is not re-judged
    clone = scratch()
    first = commit(clone, ["bench/results/legacy/x.json"])
    git(clone, "push", "-q", "origin", "lane/x")
    sha = commit(clone, ["tools/y.py"])
    code, err = check(clone, sha, first)
    out.append(("already-pushed additions are not re-judged", [] if code == 0 else [f"exit {code}: {err[:300]}"]))
    # 3. a rename into bench/results is an addition
    clone = scratch()
    write(clone, "notes/run.json")
    git(clone, "add", "-A")
    git(clone, "commit", "-q", "-m", "notes")
    git(clone, "mv", "notes/run.json", "bench/results/moved/run.json")
    git(clone, "commit", "-q", "-m", "move")
    code, err = check(clone, git(clone, "rev-parse", "HEAD"))
    out.append(("rename into bench/results is refused",
                [] if code == 1 and "bench/results/moved/run.json" in err else [f"exit {code}: {err[:300]}"]))
    # 4. many offenders: at most 20 named, then a count
    clone = scratch()
    sha = commit(clone, [f"bench/results/many/f{i:02d}.json" for i in range(25)])
    code, err = check(clone, sha)
    out.append(("more than 20 offenders are counted",
                [] if code == 1 and "and 5 more" in err and "25 file(s)" in err else [f"exit {code}: {err[:300]}"]))
    return out


class TreeAdditions(unittest.TestCase):
    def test_cases(self):
        workers = max(1, min(len(CASES) + 1, os.cpu_count() or 1))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_case, c) for c in CASES]
            futures.append(pool.submit(run_sequence_cases))
            results = []
            for f in futures:
                r = f.result()
                results.extend(r if isinstance(r, list) else [r])
        failed = [f"{name}: {'; '.join(p)}" for name, p in results if p]
        self.assertEqual(failed, [], "\n".join(failed))

    def test_pre_push_calls_the_pushed_copy(self):
        """The hook runs the pushed tree's script and refuses with exit 4, like its neighbours."""
        text = open(PRE_PUSH).read()
        self.assertIn('git show "$local_sha:tools/hooks/check_tree_additions.sh"', text)
        self.assertIn('bash "$additions" "$local_sha" "$remote_sha" "$1"', text)
        block = text.split("check_tree_additions.sh\" 2>/dev/null; then", 1)[1].split("fi", 1)[0]
        self.assertIn("exit 4", block)

    def test_gitignore_matches_allowlist(self):
        """The .gitignore block and the hook allow the same bench/results paths."""
        probes = {
            "bench/results/new_run/summary.json": True,          # ignored
            "bench/results/bench_board/b.json": False,
            "bench/results/board_best_ever/nvidia.json": False,
            "bench/results/paper_compare/x.md": False,
            "bench/results/paper_compare/sub/x.md": True,
            "bench/results/identity_break/a/b.json": False,
            "bench/results/identity_break/a/b.log": True,
            "bench/results/release_evidence/a/b.json": False,
            "bench/results/r2-index.tsv": False,
            "bench/results/CELLS_INDEX.tsv": False,
            "docs/handoffs/README.md": False,
            "docs/handoffs/H.md": True,
        }
        for path, ignored in probes.items():
            r = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
            self.assertEqual(r.returncode == 0, ignored, path)


if __name__ == "__main__":
    sys.exit(unittest.main())
