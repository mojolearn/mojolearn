#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile the patched bindings of the GPU sabotage batches ahead of their runs (lane sab-triage-1, 2026-11-26).

WHY. tools/coverage_sabotage_batch.sh compiled a batch's bindings inside the batch, one batch after another, so the
pod's 14 cores sat at a mean of 2 to 5 busy (nvc-ab-20261126T153027Z-a3757d: 9 of 155 batches in 100 minutes): one
binding compile is one busy core whatever -j says (base build records: elapsed = cpu seconds for every group), and
the host family bindings take 500 to 870 s each. The 258 binding builds of the plan are about 1,480 core-minutes; the
runs (clean, sabotaged, restored: about 100 s a batch on the one GPU) are the serial part.

WHAT. Every batch is an independent unit: its own scratch worktree at the tree's HEAD (the job's synced commit), the
clean natives of the tree copied in, the batch's patches applied, its bindings built there. Builds of every batch
share the allocation's cores: the scheduler keeps at most one compile per core and the sum of the compiles' recorded
peak memory (tools/development_build.py cache peaks, with headroom; a group without a record takes --default-gib)
under 80 percent of the allocation's memory, starting units in plan order so the run loop never waits on a late
batch. When a batch's builds are done its changed or new native files (package extensions, versioned shared libraries, wheel dependencies
and the merged index) are copied to <out>/<batch>/out/<same relative path>, the build logs, return codes and
seconds to <out>/<batch>/, `ready` is written last, and the worktree is deleted.

The batch run (coverage_sabotage_batch.sh with SAB_PREBUILT=<out>) applies the same patches to the tree, copies
<batch>/out over the tree's natives instead of compiling, runs and restores exactly as before. Nothing here reads or
writes a development cache: fault-injected binaries exist only in <out>, outside the cache and the results.

    python3 tools/sabotage_prebuild.py --plan PLAN --tree TREE --out DIR --batches "id id ..." \
        --cores N --mem-kb KB [--cache DEV_CACHE] [--unit-seconds S] [--default-gib G]

Exit 0 when every batch was written `ready` (a failed build is recorded there, not raised), 2 on bad arguments.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import time

from build_process import stop_group
from sabotage_natives import NATIVE_DIRS, configured_dirs, natives

PEAK_HEADROOM = 1.2
_stop_signal = 0
_GIT = threading.Lock()  # git worktree add/remove share the tree's administrative files


def dev_group(build):
    """(mode, stem) of a binding build script's development group (bindings/development_groups.json)."""
    if build == "build":
        return "identical", "_mojolearn_dev_core"
    name = build[len("build_"):]
    return ("host" if build.endswith("_host") else "identical"), "_mojolearn_dev_" + name


def recorded(cache, build):
    """(peak bytes, seconds) of the build's group from the development cache's last passed compile, or (0, 0)."""
    if not cache:
        return 0, 0.0
    mode, stem = dev_group(build)
    try:
        saved = json.loads((Path(cache) / "peaks" / f"{mode}-{stem}.json").read_text())
    except (OSError, ValueError):
        return 0, 0.0
    if not isinstance(saved, dict) or saved.get("status") != "passed":
        return 0, 0.0
    return int(saved.get("peak_rss_bytes") or 0), float(saved.get("elapsed_seconds") or 0.0)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def request_stop(signum, _frame):
    global _stop_signal
    _stop_signal = signum


def run(argv, cwd, log, env=None, timeout=None):
    """Reap the whole compiler group before its workspace can be harvested or removed."""
    if _stop_signal:
        return 128 + _stop_signal
    deadline = None if timeout is None else time.monotonic() + timeout
    with open(log, "ab") as out, subprocess.Popen(
        argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env=env, start_new_session=True,
    ) as proc:
        try:
            while not _stop_signal:
                interval = 0.25 if deadline is None else min(0.25, deadline - time.monotonic())
                if interval <= 0:
                    out.write(f"\nTIMEOUT after {timeout} s\n".encode())
                    return 124
                try:
                    return proc.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    continue
            return 128 + _stop_signal
        finally:
            stop_group(proc.pid)


def retire_consumed(directory, staged_receipt):
    """The fallback has no receipt, even if the producer finishes while it runs."""
    if staged_receipt.is_file():
        if (directory / "ready").read_bytes() != staged_receipt.read_bytes():
            raise ValueError(f"prebuild receipt changed after staging {directory}")
        shutil.rmtree(directory)


class Batch:
    def __init__(self, spec, args):
        self.id, self.spec, self.args = spec["id"], spec, args
        self.dir = Path(args.out) / self.id
        self.tree = self.dir / "tree"
        self.left = len(spec["builds"])
        self.rc, self.secs = {}, {}
        self.prepared = None  # None: not yet; True / False: the worktree is (not) ready to build

    def prepare(self):
        """The scratch worktree: HEAD of the tree, its clean natives, the generated sources, the patches."""
        a = self.args
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        log = self.dir / "prepare.log"
        with _GIT:
            run(["git", "-C", a.tree, "worktree", "prune"], a.tree, log)
            rc = run(["git", "-C", a.tree, "worktree", "add", "--detach", "--force", str(self.tree), "HEAD"],
                     a.tree, log)
        if rc:
            return False
        # Every source is in the job's committed tree. Only the frozen clean
        # native artifacts are copied in alongside it.
        for rel in a.clean_hashes:
            if _stop_signal:
                return False
            dst = self.tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(a.clean) / rel, dst)
        wt = shlex.quote(str(self.tree))
        if run(a.prefix + ["bash", "-c", f"cd {wt} && exec python tools/generated_sources.py prepare --repo {wt}"],
               str(self.tree), log):
            return False
        return run(["git", "apply"] + list(self.spec["patches"]), str(self.tree), log) == 0

    def build(self, b):
        """One binding script of this batch in its worktree, with the batch shell's environment."""
        env = dict(os.environ, MOJOLEARN_COMPILE_JOBS=str(self.args.cores))
        if b.endswith("_host"):
            env.pop("MOJOLEARN_GPU_ARCHS", None)
            env["MOJOLEARN_TARGET_COLUMN"] = "cpu"
        else:
            env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        started = time.time()
        # cd inside the environment's shell: the tree's pixi environment, this worktree's sources
        cmd = self.args.prefix + ["bash", "-c", f"cd {shlex.quote(str(self.tree))} && exec bash bindings/{b}.sh"]
        rc = run(cmd, str(self.tree), self.dir / f"build_{b}.log", env=env, timeout=self.args.unit_seconds)
        return rc, round(time.time() - started)

    def finish(self):
        """Harvest the changed natives, write the records and `ready`, delete the worktree."""
        out = self.dir / "out"
        changed = {}
        if self.prepared:
            for rel in natives(self.tree, self.args.native_dirs):  # driver-loop: over the worktree's native files
                if _stop_signal:
                    return
                digest = sha(self.tree / rel)
                if self.args.clean_hashes.get(rel) != digest:
                    dst = out / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.tree / rel, dst)
                    changed[rel] = digest
        for b in self.spec["builds"]:  # driver-loop: over the batch's builds
            (self.dir / f"build_{b}.rc").write_text(f"{self.rc.get(b, 98)}\n")
            (self.dir / f"build_{b}.secs").write_text(f"{self.secs.get(b, 0)}\n")
        with _GIT:
            run(["git", "-C", self.args.tree, "worktree", "remove", "--force", str(self.tree)], self.args.tree,
                self.dir / "prepare.log")
        shutil.rmtree(self.tree, ignore_errors=True)
        receipt = dict(prepared=bool(self.prepared), rc=self.rc, secs=self.secs,
                       changed=changed, clean_sha256=self.args.clean_sha256)
        if _stop_signal:
            return
        pending = self.dir / "ready.pending"
        pending.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        pending.replace(self.dir / "ready")


def reusable(directory, spec, clean_sha256, native_dirs=NATIVE_DIRS):
    """Reuse only a completed, successful build for these exact clean native files."""
    try:
        receipt = json.loads((directory / "ready").read_bytes())
        if (receipt.get("prepared") is not True or receipt.get("clean_sha256") != clean_sha256
                or receipt.get("rc") != dict.fromkeys(spec["builds"], 0)):
            return False
        changed = receipt.get("changed")
        if not isinstance(changed, dict) or not changed or set(natives(directory / "out", native_dirs)) != set(changed):
            return False
        return all(not _stop_signal and sha(directory / "out" / rel) == digest for rel, digest in changed.items())
    except (OSError, ValueError, AttributeError):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", required=True, help="batch ids, space separated, in run order")
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--mem-kb", type=int, required=True)
    ap.add_argument("--cache", default="", help="the development cache whose peaks/ size the compiles")
    ap.add_argument("--unit-seconds", type=int, default=2400, help="one binding build's limit")
    ap.add_argument("--default-gib", type=float, default=8.0, help="memory of a build without a recorded peak")
    ap.add_argument("--prefix", default="",
                    help="the command prefix of the tree's environment (default: pixi run with the TREE's manifest, so "
                         "no worktree gets an environment of its own)")
    a = ap.parse_args(argv)
    tree = Path(a.tree).resolve()
    a.tree = str(tree)
    try:
        a.native_dirs = tuple(str((tree / path).resolve().relative_to(tree)) for path in configured_dirs())
    except ValueError:
        raise SystemExit("native directory outside the tree requires an in-place build")
    a.prefix = (shlex.split(a.prefix) if a.prefix else
                ["pixi", "run", "--manifest-path", str(Path(a.tree) / "pixi.toml"), "-e", "default"])
    if a.cores < 1:
        raise SystemExit("--cores must be at least 1")
    plan = {b["id"]: b for b in json.loads(Path(a.plan).read_text())["batches"]}
    ids = [x for x in a.batches.split() if x]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-zA-Z0-9_]+-[1-9][0-9]*", x) for x in ids):
        raise SystemExit("invalid or duplicate batch IDs")
    missing = [x for x in ids if x not in plan]
    if missing:
        raise SystemExit("no batch " + ", ".join(missing) + " in " + a.plan)
    budget = int(a.mem_kb * 1024 * 0.8)
    # The clean natives, once, now: the job starts this before its first batch applies a patch.
    a.clean = str(Path(a.out) / "_clean")
    (Path(a.out) / "_clean.done").unlink(missing_ok=True)
    shutil.rmtree(a.clean, ignore_errors=True)
    def snapshot(rel):
        if _stop_signal:
            return rel, None
        dst = Path(a.clean) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(a.tree) / rel, dst)
        return rel, sha(dst)
    with ThreadPoolExecutor(max_workers=a.cores) as pool:
        a.clean_hashes = dict(pool.map(snapshot, natives(a.tree, a.native_dirs)))
    if _stop_signal:
        return 128 + _stop_signal
    if not a.clean_hashes:
        raise SystemExit("no clean native files to snapshot")
    a.clean_sha256 = hashlib.sha256(json.dumps(a.clean_hashes, sort_keys=True).encode()).hexdigest()
    batches = []
    with ThreadPoolExecutor(max_workers=a.cores) as pool:
        cached = pool.map(lambda x: reusable(Path(a.out) / x, plan[x], a.clean_sha256, a.native_dirs), ids)
        for x, valid in zip(ids, cached):
            if _stop_signal:
                return 128 + _stop_signal
            if valid:
                print(f"PREBUILD {x} ready from an earlier submission", flush=True)
            else:
                (Path(a.out) / x / "ready").unlink(missing_ok=True)
                batches.append(Batch(plan[x], a))
    queue = []  # (batch, build, need bytes) in run order
    for bt in batches:  # driver-loop: over batches
        for b in bt.spec["builds"]:  # driver-loop: over one batch's builds
            peak, _ = recorded(a.cache, b)
            need = int(peak * PEAK_HEADROOM) if peak else int(a.default_gib * (1 << 30))
            queue.append((bt, b, min(need, budget)))
    if _stop_signal:
        return 128 + _stop_signal
    # The run loop may consume a cached marker as soon as the barrier opens.
    (Path(a.out) / "_clean.done").write_text(a.clean_sha256 + "\n")
    running = {}
    used = 0
    # Workers: one per core for the compiles, plus the worktree preparations, which run beside them.
    with ThreadPoolExecutor(max_workers=a.cores + 2) as pool:
        preps = {}

        def prep(bt):
            try:
                ok = bt.prepare()
            except Exception as error:  # noqa: BLE001 -- the batch's builds read 97; prepare.log says why
                bt.dir.mkdir(parents=True, exist_ok=True)
                with open(bt.dir / "prepare.log", "a") as log:
                    log.write(f"\nprepare error: {error}\n")
                ok = False
            bt.prepared = bool(ok)
            return bt

        while queue or running:  # driver-loop: the scheduler (starts units as cores and memory free up)
            if _stop_signal:
                return 128 + _stop_signal
            # Worktrees are prepared a little ahead of the compiles (at most two at once, so a compile never waits
            # behind a queue of preparations in the pool and the disk holds a few worktrees, not 155).
            for bt, _, _ in queue[: a.cores]:
                if sum(1 for f in preps.values() if not f.done()) >= 2:
                    break
                if bt.prepared is None and bt.id not in preps:
                    preps[bt.id] = pool.submit(prep, bt)
            started = True
            while started and queue and len(running) < a.cores:
                started = False
                bt, b, need = queue[0]
                if bt.prepared is None:
                    break  # the next unit in run order waits for its worktree
                if running and used + need > budget:
                    break
                queue.pop(0)
                if not bt.prepared:
                    bt.rc[b], bt.secs[b] = 97, 0  # the worktree could not be prepared: prepare.log
                    bt.left -= 1
                    if bt.left == 0:
                        bt.finish()
                    started = True
                    continue
                running[pool.submit(bt.build, b)] = (bt, b, need)
                used += need
                print(f"PREBUILD start {bt.id} {b} need {need >> 20} MiB, {len(running)} of {a.cores} cores",
                      flush=True)
                started = True
            pending = list(running) + [f for f in preps.values() if not f.done()]
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            if _stop_signal:
                return 128 + _stop_signal
            for f in done:  # driver-loop: over finished units
                if f not in running:
                    continue
                bt, b, need = running.pop(f)
                used -= need
                try:
                    bt.rc[b], bt.secs[b] = f.result()
                except Exception as error:  # noqa: BLE001 -- recorded as the build's failure, never raised
                    (bt.dir / f"build_{b}.log").open("a").write(f"\nprebuild error: {error}\n")
                    bt.rc[b], bt.secs[b] = 96, 0
                print(f"PREBUILD done {bt.id} {b} rc={bt.rc[b]} secs={bt.secs[b]}", flush=True)
                bt.left -= 1
                if bt.left == 0:
                    bt.finish()
                    print(f"PREBUILD ready {bt.id}", flush=True)
    return 128 + _stop_signal if _stop_signal else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "retire-consumed":
        retire_consumed(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, request_stop)
        sys.exit(main())
