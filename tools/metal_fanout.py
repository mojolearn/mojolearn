#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Run N identity shards at once under ONE Metal slot (2026-09-21).

`tools/verify_lanes.py --apple-pass --metal-shards N` calls this through
`tools/mac_slot.py metal`, so the exclusive Metal lock is held once for the
whole group and no other GPU job on the Mac can start while the shards run.
The shards share the one GPU on purpose: the Apple pass is bound by host to
device synchronisation (about 4 ms a sync, docs in
metal-cost-is-syncs-not-launches), so two or three processes can overlap each
other's waits.

THIS IS OPT-IN AND UNPROVEN. Concurrent Metal jobs on this M4 returned NaN,
constant and zero outputs on 2026-09-15. Those were unrelated jobs with no
shared lock; whether sharded passes of the same harness are clean is exactly
what the procedure in docs/RELEASE_CHECKLIST.md measures (every part hash at
N = 1, 2 and 3 on one commit, against each other and against the reference
table). Until that comparison reads zero differences over the full sweep,
the default stays 1.

    python3 tools/metal_fanout.py SPEC.json

SPEC is {"children": [{"cmd": [...], "log": path}, ...], "codes": path}.
Every child starts at once; the exit codes are written to `codes` as
{"0": rc, ...} for every child, including one that was stopped. The exit
status is 0 only when every child exited 0. A signal is forwarded to every
child and the codes are still written.
"""
import json
import os
import signal
import subprocess
import sys


def main(argv):
    if len(argv) != 1:
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    spec = json.loads(open(argv[0]).read())
    children = spec["children"]
    procs, logs, codes = [], [], {}
    stopping = []

    def stop(signum, frame):
        stopping.append(signum)
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, stop)
    try:
        for child in children:
            fh = open(child["log"], "w")
            logs.append(fh)
            procs.append(subprocess.Popen(child["cmd"], stdout=fh, stderr=subprocess.STDOUT,
                                          env=dict(os.environ, MOJOLEARN_METAL_SHARDS=str(len(children)))))
        for k, proc in enumerate(procs):
            while True:
                try:
                    codes[str(k)] = proc.wait()
                    break
                except InterruptedError:
                    continue
    finally:
        for fh in logs:
            fh.close()
    for k in range(len(children)):
        codes.setdefault(str(k), 1)
    tmp = spec["codes"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(codes, fh)
    os.replace(tmp, spec["codes"])
    return 0 if all(c == 0 for c in codes.values()) and not stopping else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
