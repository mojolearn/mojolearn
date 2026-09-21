#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The whole local release verification, CPU and Apple GPU AT ONCE
(2026-09-21).

    pixi run -e test release-check

`release-check` ran `cpu-pass` and then `apple-pass`, one after the other:
602 s and then 591 s for 0.8.12, twenty minutes during which the GPU sat idle
for the first ten and four of the five CPU slots for the last ten. The two
passes share nothing but the Mac's slot scheduler (tools/mac_slot.py), so
they run together here: the Apple pass holds the Metal lock and ONE of the
MAC_SLOTS (default 5), and the CPU pass takes the rest (MAC_SLOTS - 1 shards,
through MOJOLEARN_CPU_PASS_SLOTS). The Apple release budget of five cores is
the scheduler's own limit, so it cannot be exceeded. The CPU slots run at
nice 19 and the Metal job at normal priority, so the CPU pass yields to the
GPU's host thread rather than starving it.

Each pass keeps its own records and resume behaviour
(~/mojolearn-evidence/release-check/<commit>/{cpu,metal}/), writes its
transcript to <that directory>/../release-check.<pass>.log, and the exit
status is non-zero if either pass is incomplete. Run the two tasks separately
to keep the old serial behaviour.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    sys.path.insert(0, str(ROOT / "tools"))
    import verify_lanes
    slots = int(os.environ.get("MAC_SLOTS", "5"))
    if slots < 2:
        raise SystemExit("REFUSING: release-check needs MAC_SLOTS >= 2 (one for Metal, one for the CPU pass)")
    logs = Path(verify_lanes.pass_out_dir("metal")).parent
    logs.mkdir(parents=True, exist_ok=True)
    runs = {}
    started = time.monotonic()
    for name, flag, env in (("apple", "--apple-pass", {}),
                            ("cpu", "--cpu-pass", {"MOJOLEARN_CPU_PASS_SLOTS": str(slots - 1)})):
        log = logs / f"release-check.{name}.log"
        fh = open(log, "w")
        proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "verify_lanes.py"), flag],
                                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, **env))
        runs[name] = (proc, fh, log)
        print(f"# {name} pass started, transcript {log}", flush=True)
    codes = {}
    try:
        for name, (proc, fh, log) in runs.items():
            codes[name] = proc.wait()
            fh.close()
            print(f"# {name} pass exited {codes[name]} after {time.monotonic() - started:.0f} s", flush=True)
    except KeyboardInterrupt:
        for proc, fh, _ in runs.values():
            if proc.poll() is None:
                proc.terminate()
        for name, (proc, fh, _) in runs.items():
            codes[name] = proc.wait()
            fh.close()
    for name, (_, _, log) in runs.items():
        tail = [l for l in log.read_text(errors="replace").splitlines() if l.startswith("# verdict")
                or l.startswith("# FAIL") or l.startswith("# nothing to check")][-6:]
        for line in tail:
            print(f"  {name}: {line}")
    ok = all(c == 0 for c in codes.values())
    print(f"# release-check {'COMPLETE' if ok else 'INCOMPLETE'} in {time.monotonic() - started:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
