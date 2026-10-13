#!/usr/bin/env python3
"""Bound a Linux benchmark session, including children in separate sessions.

This is a process watchdog, not the provider's billing deadman. Arm the provider
deadline separately. The inherited session marker lets us find orphan workers
without inspecting or saving other environment values. Full output stays on disk.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

MARKER = "MOJOLEARN_BOARD_SESSION"


def marked_processes(marker, proc_root=Path("/proc")):
    target = (MARKER + "=" + marker).encode()
    found = []
    for p in proc_root.iterdir():
        if not p.name.isdigit():
            continue
        try:
            if target in (p / "environ").read_bytes().split(b"\0"):
                found.append(int(p.name))
        except (OSError, PermissionError):
            pass
    return found


def kill_session(marker):
    # Recheck the marker immediately before signaling: a PID may have exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in marked_processes(marker):
            try:
                if (MARKER + "=" + marker).encode() in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                    os.kill(pid, sig)
            except OSError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(1)


def run(command, out, seconds, interval=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    marker = uuid.uuid4().hex
    start, end = time.time(), time.monotonic() + seconds
    state = dict(session=marker, command=command, started_epoch=start,
                 deadline_epoch=start + seconds, limit_seconds=seconds)
    interrupted = []
    def stop(signum, frame):
        interrupted.append(signum)
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    proc = None
    try:
        with (out / "session.log").open("ab", buffering=0) as log, (out / "watcher.jsonl").open("a") as watcher:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, **{MARKER: marker}), start_new_session=True)
            state["pid"] = proc.pid
            while True:
                row = dict(time_epoch=time.time(), remaining_s=max(0, round(end-time.monotonic(), 1)),
                           returncode=proc.poll(), workers=marked_processes(marker))
                board = out / "board.json"
                if board.exists():
                    try:
                        data = json.loads(board.read_text())
                        row["board"] = dict(planned=len(data.get("plan", [])),
                            recorded=len(data.get("races", {})),
                            failed=sum(r.get("status") == "failed" for r in data.get("races", {}).values()),
                            evidence_failures=sum(bool(r.get("evidence_failure")) for r in data.get("races", {}).values()),
                            preflight=(data.get("preflight") or {}).get("verdict"))
                    except (OSError, ValueError):
                        row["board_read_pending"] = True
                watcher.write(json.dumps(row, sort_keys=True) + "\n"); watcher.flush()
                if row["returncode"] is not None or interrupted or time.monotonic() >= end:
                    break
                time.sleep(min(interval, max(0, end-time.monotonic())))
            state["stop_reason"] = "signal" if interrupted else "deadline" if proc.poll() is None else "command exited"
            state["returncode"] = proc.poll()
    finally:
        kill_session(marker)
        if proc is not None:
            proc.wait(timeout=10)
        state["finished_epoch"] = time.time()
        state["remaining_workers"] = marked_processes(marker)
        (out / "session.json").write_text(json.dumps(state, indent=2) + "\n")
        for s, handler in previous.items():
            signal.signal(s, handler)
    return 124 if state.get("stop_reason") == "deadline" else 130 if interrupted else state.get("returncode", 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.seconds <= 3600 or not Path("/proc").is_dir():
        ap.error("Linux required; supply a command and a limit between 1 and 3600 seconds")
    raise SystemExit(run(command, args.out, args.seconds))
