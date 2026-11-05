# SPDX-License-Identifier: Apache-2.0
"""tools/release_requirement_dependencies.py registers its temporary dependency worktrees, removes them on
exit, and `--gc-worktrees` removes stale ones (never a worktree outside mojolearn-dependencies-*)."""
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release_requirement_dependencies as rd  # noqa: E402


def git(*a, cwd):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout


def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git("init", "-q", cwd=r)
    (r / "f").write_text("x")
    git("add", "f", cwd=r)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=r)
    return r


def add(r, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "-q", "--detach", str(path), "HEAD", cwd=r)
    return path


def listed(r):
    return [line.split(" ", 1)[1] for line in git("worktree", "list", "--porcelain", cwd=r).splitlines()
            if line.startswith("worktree ")]


def test_stale_dependency_trees_are_removed_and_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    old = add(r, tmp_path / "tmp" / "mojolearn-dependencies-old" / "tree")
    young = add(r, tmp_path / "tmp" / "mojolearn-dependencies-young" / "tree")
    lane = add(r, tmp_path / "wt" / "mojolearn-dependencies-lane" / "tree")      # under the protected root
    other = add(r, tmp_path / "tmp" / "someone-else" / "tree")
    gone = add(r, tmp_path / "tmp" / "mojolearn-dependencies-gone" / "tree")
    import shutil
    shutil.rmtree(gone.parent)                                                     # dir deleted, still registered
    week = time.time() - 7 * 86400
    for t in (old, gone):
        rd._register_worktree(r, t)
        f = rd._registry_file(t)
        f.write_text(json.dumps(dict(json.loads(f.read_text()), pid=999999, created=week)))
    rd._LIVE_WORKTREES.clear()
    result = rd.gc_worktrees(r, protect=(tmp_path / "wt",))
    still = listed(r)
    assert str(old) in result["removed"] and not old.exists()
    assert str(gone) in result["pruned"]
    assert all(str(p) not in [str(pathlib.Path(x)) for x in still] for p in (old, gone))
    resolved = {str(pathlib.Path(x).resolve()) for x in still}
    for keep in (young, lane, other):
        assert str(keep.resolve()) in resolved and keep.exists()
    assert any(t == str(young) and "younger" in why for t, why in result["kept"])
    assert not list((tmp_path / "registry").glob("*.json"))


def test_a_tree_whose_process_is_alive_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-live" / "tree")
    rd._register_worktree(r, t)
    f = rd._registry_file(t)
    f.write_text(json.dumps(dict(json.loads(f.read_text()), pid=os.getppid(),
                                start=rd._process_start(os.getppid()), created=0)))
    rd._LIVE_WORKTREES.clear()
    result = rd.gc_worktrees(r, protect=())
    assert t.exists() and any("alive" in why for _t, why in result["kept"])


def test_this_process_removes_its_live_trees_on_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-mine" / "tree")
    rd._register_worktree(r, t)
    assert rd._registry_file(t).exists()
    rd._remove_live_worktrees()   # what the atexit hook and the SIGTERM handler run
    assert not t.parent.exists() and not rd._registry_file(t).exists()
    assert str(t.resolve()) not in {str(pathlib.Path(x).resolve()) for x in listed(r)}


def test_a_killed_process_leaves_nothing_registered_after_sigterm(tmp_path):
    r = repo(tmp_path)
    tree = tmp_path / "tmp" / "mojolearn-dependencies-killed" / "tree"
    code = ("import sys, time; sys.path.insert(0, %r); import release_requirement_dependencies as rd, subprocess, "
            "pathlib; t = pathlib.Path(%r); t.parent.mkdir(parents=True); subprocess.run(['git', '-C', %r, "
            "'worktree', 'add', '-q', '--detach', str(t), 'HEAD'], check=True); rd._register_worktree(%r, t); "
            "print('ready', flush=True); time.sleep(60)") % (str(ROOT / "tools"), str(tree), str(r), str(r))
    env = dict(os.environ, MOJOLEARN_DEPENDENCY_WORKTREES=str(tmp_path / "registry"))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    assert p.stdout.readline().strip() == "ready"
    p.terminate()
    p.wait(timeout=30)
    assert not tree.exists()
    assert str(tree.resolve()) not in {str(pathlib.Path(x).resolve()) for x in listed(r)}


def test_a_dead_owner_tree_is_removed_at_any_age_from_its_owner_file(tmp_path, monkeypatch):
    # No registry row (killed before registering, or SIGKILL): the owner file in the temporary dir decides.
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-killed" / "tree")
    (t.parent / rd.WORKTREE_OWNER).write_text(json.dumps(dict(pid=999999, start="x", created=time.time())))
    result = rd.gc_worktrees(r, protect=())
    assert str(t) in result["removed"] and not t.parent.exists()
    assert str(t.resolve()) not in {str(pathlib.Path(x).resolve()) for x in listed(r)}


def test_a_reused_pid_counts_as_dead(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-reused" / "tree")
    (t.parent / rd.WORKTREE_OWNER).write_text(json.dumps(dict(pid=os.getppid(), start="Thu Jan  1 00:00:00 1970",
                                                              created=time.time())))
    result = rd.gc_worktrees(r, protect=())
    assert str(t) in result["removed"] and not t.exists()


def test_a_live_owner_tree_is_kept_however_old(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-running" / "tree")
    (t.parent / rd.WORKTREE_OWNER).write_text(json.dumps(dict(pid=os.getppid(),
                                                              start=rd._process_start(os.getppid()), created=0)))
    result = rd.gc_worktrees(r, protect=())
    assert t.exists() and any("alive" in why for _t, why in result["kept"])


def test_this_process_trees_are_never_collected(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-self" / "tree")
    rd._register_worktree(r, t)
    try:
        result = rd.gc_worktrees(r, protect=(), now=time.time() + 30 * 86400)
        assert t.exists() and any("this process" in why for _t, why in result["kept"])
    finally:
        rd._remove_live_worktrees()


def test_unknown_owner_uses_the_two_hour_age_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    assert rd.WORKTREE_GC_AGE_S == 2 * 3600
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-unknown" / "tree")
    kept = rd.gc_worktrees(r, protect=())
    assert t.exists() and any("no recorded owner" in why for _t, why in kept["kept"])
    gone = rd.gc_worktrees(r, protect=(), now=time.time() + 3 * 3600)
    assert str(t) in gone["removed"] and not t.exists()


def test_gc_runs_once_per_checkout_before_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_DEPENDENCY_WORKTREES", str(tmp_path / "registry"))
    r = repo(tmp_path)
    t = add(r, tmp_path / "tmp" / "mojolearn-dependencies-before" / "tree")
    (t.parent / rd.WORKTREE_OWNER).write_text(json.dumps(dict(pid=999999, created=time.time())))
    monkeypatch.setattr(rd, "_GC_DONE", set())
    lines = []
    first = rd.gc_before_analysis(r, say=lines.append)
    assert str(t) in first["removed"] and lines
    assert rd.gc_before_analysis(r, say=lines.append) is None


def _killed_child(tmp_path, r, tree, thread):
    # Registers a tree (from a pool thread when THREAD, as prefetch does) and waits to be signalled.
    register = ("rd._register_worktree(%r, t)" % str(r))
    if thread:
        register = ("rd._install_worktree_cleanup(); import threading; th = threading.Thread(target=lambda: "
                    "rd._register_worktree(%r, t)); th.start(); th.join()" % str(r))
    code = ("import sys, time; sys.path.insert(0, %r); import release_requirement_dependencies as rd, subprocess, "
            "pathlib; t = pathlib.Path(%r); t.parent.mkdir(parents=True); subprocess.run(['git', '-C', %r, "
            "'worktree', 'add', '-q', '--detach', str(t), 'HEAD'], check=True); %s; "
            "print('ready', flush=True); time.sleep(60)") % (str(ROOT / "tools"), str(tree), str(r), register)
    env = dict(os.environ, MOJOLEARN_DEPENDENCY_WORKTREES=str(tmp_path / "registry"))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, env=env)
    assert p.stdout.readline().strip() == "ready"
    return p


def test_sigterm_removes_a_tree_registered_on_a_pool_thread(tmp_path):
    r = repo(tmp_path)
    tree = tmp_path / "tmp" / "mojolearn-dependencies-thread" / "tree"
    p = _killed_child(tmp_path, r, tree, thread=True)
    p.terminate()
    p.wait(timeout=30)
    assert not tree.exists()
    assert str(tree.resolve()) not in {str(pathlib.Path(x).resolve()) for x in listed(r)}


def test_sigint_removes_the_tree_and_still_raises_keyboardinterrupt(tmp_path):
    import signal
    r = repo(tmp_path)
    tree = tmp_path / "tmp" / "mojolearn-dependencies-ctrlc" / "tree"
    p = _killed_child(tmp_path, r, tree, thread=False)
    p.send_signal(signal.SIGINT)
    assert p.wait(timeout=30) != 0
    assert not tree.exists()
    assert str(tree.resolve()) not in {str(pathlib.Path(x).resolve()) for x in listed(r)}
