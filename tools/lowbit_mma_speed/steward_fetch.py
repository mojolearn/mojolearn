#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lowbit-mma-speed: bring one finished steward request home.

    steward_fetch.py <steward> <request> <local dir>

Reads the request's verdict.json, speed.stdout and speed.stderr from the
steward's done/ folder through tools/apple_steward.py's own remote command
(read only; it queues nothing and moves nothing), writes them under
<local dir>/steward_done/, and cuts the job's `LOWBIT` lines out of the
stdout into <local dir>/lowbit.tsv (tools/lowbit_mma_speed/price_job.sh
prints the run of record's, so the file is that run's).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import apple_steward as st  # noqa: E402


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    steward, request, local = sys.argv[1:]
    done = os.path.join(local, "steward_done")
    os.makedirs(done, exist_ok=True)
    for name in ("verdict.json", "speed.stdout", "speed.stderr"):
        body = st._cloudmac(steward, f"cat {st.REMOTE_ROOT}/done/{request}/{name} 2>/dev/null || true",
                            timeout=300, raw=True)
        with open(os.path.join(done, name), "wb") as f:
            f.write(body)
        print(f"{name}: {len(body)} bytes")
    lines = [l for l in open(os.path.join(done, "speed.stdout"), errors="replace")
             if l.startswith("LOWBIT ") or l.startswith("LOWBIT-NOT-RUN ")]
    with open(os.path.join(local, "lowbit.tsv"), "w") as f:
        f.writelines(lines)
    print(f"lowbit.tsv: {len(lines)} lines")


if __name__ == "__main__":
    main()
