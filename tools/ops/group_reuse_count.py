# SPDX-License-Identifier: Apache-2.0
"""Count, over consecutive commits, how many build groups keep the same key (2026-11-26; the study behind
tools/ops/native_testsuite_cache.py's docstring).

    python3 tools/ops/group_reuse_count.py <repo> <out dir> <rev> <rev> ...   (newest first)

PARALLEL (rel-fix-1, review of 3423ae229e..8f0d9a9642: the first version checked every revision out in ONE worktree,
one after another). Each revision is planned in its own detached worktree under <out dir>/wt/, all at once, as many as
the core allocation (affinity and cgroup cpu.max) allows; a plan only reads, so the units are independent. The keys
of different worktrees compare on Linux only: since KEY_SCHEMA v2 a Linux key holds no checkout path, while a darwin
key keeps it (tools/development_build.py group_fields), so on a Mac every group would read as changed.
"""
import collections
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def _allocation():
    """python/mojolearn/_allocation.py of this tree (the one allocation function, rel-fix-3), loaded by path."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "mojolearn", "_allocation.py")
    spec = importlib.util.spec_from_file_location("mojolearn_allocation_by_path", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alloc_cores():
    """The cores of this allocation (python/mojolearn/_allocation.py `cores`); this copy read cgroup v2 only."""
    return _allocation().cores()[0]


def plan_keys(repo, out, rev):
    """{mode:stem: (key, source files)} of REV, or None when its plan fails."""
    wt = os.path.join(out, "wt", rev)
    subprocess.run(["git", "-C", repo, "worktree", "add", "-q", "--detach", wt, rev], check=True)
    try:
        j = os.path.join(out, f"plan-{rev}.json")
        env = dict(os.environ, MOJOLEARN_GPU_ARCHS="sm_89")
        p = subprocess.run([sys.executable, "tools/development_build.py", "plan", "--sets", "identical,host",
                            "--cache", os.path.join(out, "cache"), "--json", j], cwd=wt, env=env,
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(rev, "plan rc", p.returncode, p.stderr[-200:].replace("\n", " "))
            return None
        with open(j) as fh:
            d = json.load(fh)
        os.remove(j)
        return {f"{g['mode']}:{g['stem']}": (g["key"], g["fields"]["source"]["files"]) for g in d["groups"]}
    finally:
        subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", wt], capture_output=True)
        shutil.rmtree(wt, ignore_errors=True)


def main(argv):
    repo, out, revs = argv[0], argv[1], argv[2:]
    os.makedirs(os.path.join(out, "wt"), exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(alloc_cores(), max(1, len(revs)))) as pool:
        keys = dict(zip(revs, pool.map(lambda r: plan_keys(repo, out, r), revs)))
    tot = same = 0
    rows = []
    for new, old in zip(revs, revs[1:]):
        a, b = keys[new], keys[old]
        if not a or not b:
            continue
        common = set(a) & set(b)
        kept = sum(a[k][0] == b[k][0] for k in common)
        srcsame = sum(a[k][1] == b[k][1] for k in common)
        files = subprocess.run(["git", "-C", repo, "diff", "--name-only", old, new], capture_output=True,
                               text=True).stdout.split()
        mojo = [f for f in files if f.endswith(".mojo")]
        rows.append((new, len(a), kept, srcsame, len(files), len(mojo)))
        tot += len(a)
        same += kept
    print("commit groups key_same source_same files_changed mojo_changed")
    for r in rows:
        print(*r)
    print(f"TOTAL groups {tot} reused {same} = {100 * same / max(tot, 1):.0f}%")
    hist = collections.Counter(min(10, round(10 * r[2] / r[1])) for r in rows if r[1])
    print("pairs by reuse decile:", dict(sorted(hist.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
