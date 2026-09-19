#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Cooperative CPU/GPU admission, compatible with the legacy slot directories.

All new participants serialize metadata updates with flock. Lease directories
are published with their PID already inside; live legacy owners are respected.
A monotonic ticket counter gives FIFO order even while an older job is running.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def alive(pid, group=False):
    try:
        (os.killpg if group else os.kill)(int(pid), 0)
        return True
    except (ValueError, TypeError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def read(path, default=""):
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return default


class Scheduler:
    def __init__(self, env=os.environ):
        self.base = Path(env.get("MOJOLEARN_MAC_SLOT_BASE", "/tmp/mojolearn-mac-slot"))
        self.metal = Path(env.get("MOJOLEARN_METAL_LOCK", "/tmp/mojolearn-metal-slot"))
        self.queue = Path(env.get("MOJOLEARN_METAL_QUEUE", "/tmp/mojolearn-metal-queue"))
        self.count = int(env.get("MAC_SLOTS", "5"))
        if self.count < 1:
            raise ValueError("MAC_SLOTS must be positive")
        self.queue.mkdir(parents=True, exist_ok=True)
        self.slots = [Path(f"{self.base}.{i}") for i in range(1, self.count + 1)]
        self.token = uuid.uuid4().hex
        self.ticket = None
        self.held = []

    @contextmanager
    def guard(self):
        # Never unlink the guard: unlinking lets waiters lock different inodes.
        with (self.queue / ".guard").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def stale(self, path):
        if not path.exists():
            return False
        pid = read(path / "pid")
        if not pid:
            # An old shell creates the directory before writing its PID.
            # Give it a publication grace period, never steal a fresh lock.
            return time.time() - path.stat().st_mtime > 60
        return not alive(pid) and not alive(read(path / "pgid"), group=True)

    def reap(self):
        for path in self.slots + [self.metal] + list(self.queue.glob("t[0-9]*")):
            if self.stale(path):
                shutil.rmtree(path)

    def publish(self, path, command):
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".slot-publish-", dir=path.parent))
        try:
            for key, value in {"pid": os.getpid(), "token": self.token,
                               "since": time.strftime("%H:%M:%S"),
                               "what": f"{Path.cwd()} :: {' '.join(command)}"}.items():
                (stage / key).write_text(str(value) + "\n")
            # An existing legacy directory, including an empty unpublished
            # one, must not be replaced by rename on POSIX.
            if path.exists():
                return False
            try:
                os.rename(stage, path)
            except OSError:
                if path.exists():
                    return False
                raise
            return True
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def enqueue(self, command):
        with self.guard():
            self.reap()
            live = [int(p.name[1:]) for p in self.queue.glob("t[0-9]*") if p.name[1:].isdigit()]
            counter = self.queue / ".next"
            n = max(int(read(counter, "0")), max(live, default=-1) + 1)
            counter.write_text(str(n + 1))
            self.ticket = self.queue / f"t{n}"
            if not self.publish(self.ticket, command):
                raise RuntimeError("ticket publication raced a legacy scheduler; retry")

    def attempt(self, metal, command):
        with self.guard():
            self.reap()
            free = next((p for p in self.slots if not p.exists()), None)
            if free is None:
                return False
            if metal:
                tickets = sorted((int(p.name[1:]), p) for p in self.queue.glob("t[0-9]*")
                                 if p.name[1:].isdigit())
                if not tickets or tickets[0][1] != self.ticket or self.metal.exists():
                    return False
            if not self.publish(free, command):
                return False
            self.held.append(free)
            if metal:
                if not self.publish(self.metal, command):
                    if read(free / "token") == self.token:
                        shutil.rmtree(free)
                    self.held.remove(free)
                    return False
                self.held.append(self.metal)
                shutil.rmtree(self.ticket)
                self.ticket = None
            return True

    def mark_env(self, env):
        """Tell the child it is running UNDER the slot (2026-09-19).

        Without this a GPU run cannot tell `the lock is held by me` from `the
        lock is held by SOMEONE ELSE and I am running anyway` -- and the
        second is the hazard: concurrent Metal on one M4 returns NaN,
        constant and zero output that still hashes stably. The token is the
        slot's own, so a stale variable inherited from an unrelated shell
        cannot impersonate a held slot.
        """
        env["MOJOLEARN_SLOT_TOKEN"] = self.token
        return env


    def child_started(self, pgid):
        with self.guard():
            for path in self.held:
                (path / "pgid").write_text(str(pgid))

    def _release(self):
        for path in self.held + ([self.ticket] if self.ticket else []):
            if read(path / "token") == self.token:
                shutil.rmtree(path)
        self.held = []
        self.ticket = None

    def release(self):
        with self.guard():
            self._release()

    def status(self):
        with self.guard():
            for path in self.slots + [self.metal]:
                print(f"{path.name}: pid {read(path / 'pid')} since {read(path / 'since')} :: "
                      f"{read(path / 'what')}" if path.exists() else f"{path.name}: free")
            tickets = sorted((int(p.name[1:]), p) for p in self.queue.glob("t[0-9]*")
                             if p.name[1:].isdigit())
            for n, path in tickets:
                print(f"ticket {n}: pid {read(path / 'pid')} :: {read(path / 'what')}")


def stop_group(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Descendants may survive their direct parent: retire the whole lease.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=float, default=0, help="execution seconds, excluding queue wait; 0 is unlimited")
    ap.add_argument("--wait-timeout", type=float, default=0)
    ap.add_argument("--deadline", type=float, default=0, help="shared monotonic deadline; 0 disables it")
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--timing-json")
    ap.add_argument("mode", choices=("run", "metal", "cuda", "hip", "status"))
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if (any(not math.isfinite(v) for v in (args.timeout, args.wait_timeout, args.poll, args.deadline))
            or args.timeout < 0 or args.wait_timeout < 0 or args.deadline < 0 or args.poll <= 0):
        ap.error("timeouts must be finite and nonnegative and poll must be finite and positive")
    scheduler = Scheduler()
    if args.mode == "status":
        scheduler.status()
        return 0
    if not args.command:
        ap.error("a command is required")
    if args.command[0] == "--":
        args.command.pop(0)
    started = time.monotonic()
    admitted = None
    child = None
    code = 1
    def interrupted(signum, frame):
        raise KeyboardInterrupt(signum)
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.mode in ("metal", "cuda", "hip"):
            scheduler.enqueue(args.command)
        last_log = -60.0
        while True:
            now = time.monotonic()
            if args.deadline and now >= args.deadline:
                print("mac_slot: total run budget exhausted", file=sys.stderr, flush=True)
                code = 124
                return code
            if scheduler.attempt(args.mode in ("metal", "cuda", "hip"), args.command):
                break
            waited = now - started
            if args.wait_timeout and waited >= args.wait_timeout:
                code = 124
                return code
            if waited - last_log >= 30:
                print(f"mac_slot: waiting {waited:.1f}s for {args.mode} capacity", file=sys.stderr, flush=True)
                last_log = waited
            time.sleep(min(args.poll, max(0, args.deadline - time.monotonic()))
                       if args.deadline else args.poll)
        admitted = time.monotonic()
        print(f"mac_slot: admitted after {admitted - started:.3f}s", file=sys.stderr, flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", MOJOLEARN_CPU_THREADS="1",
                   MOJOLEARN_COMPILE_JOBS="1", MOJOLEARN_BUILD_JOBS="1",
                   MODULAR_THREAD_BUSY_WAIT_US="0")
        remaining = args.deadline - time.monotonic() if args.deadline else None
        if remaining is not None and remaining <= 0:
            code = 124
            return code
        limit = min(args.timeout, remaining) if args.timeout and remaining is not None else (remaining or args.timeout or None)
        scheduler.mark_env(env)   # the child can now tell it holds the slot
        child = subprocess.Popen(["nice", "-n", "19", *args.command], env=env, start_new_session=True)
        scheduler.child_started(child.pid)
        try:
            code = child.wait(timeout=limit)
            if code < 0:
                code = 128 - code
        except subprocess.TimeoutExpired:
            print(f"mac_slot: execution exceeded available budget ({limit}s)", file=sys.stderr, flush=True)
            code = 124
    except KeyboardInterrupt as exc:
        code = 128 + (exc.args[0] if exc.args else signal.SIGINT)
    finally:
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        stop_group(child)
        scheduler.release()
        ended = time.monotonic()
        timing = dict(wait_seconds=(admitted or ended) - started,
                      run_seconds=ended - admitted if admitted else 0, exit_code=code)
        print("mac_slot: " + json.dumps(timing), file=sys.stderr, flush=True)
        if args.timing_json:
            Path(args.timing_json).write_text(json.dumps(timing) + "\n")
        for s, handler in previous.items():
            signal.signal(s, handler)
    return code


if __name__ == "__main__":
    sys.exit(main())
