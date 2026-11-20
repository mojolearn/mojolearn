#!/usr/bin/env python3
"""Start an existing bounded deadman outside the invoking process session."""
import pathlib
import subprocess
import sys


def start(script):
    script = pathlib.Path(script).resolve(strict=True)
    # nohup alone still shares the caller's process group. Execution clients
    # may clean that group after the launcher exits, silently removing its
    # lease protection. Preserve the existing script and signal policy while
    # giving the deadline process an independent session, as the remote guard
    # already has. This creates no polling or monitoring service.
    process = subprocess.Popen(
        ["/bin/sh", "-c", 'trap "" HUP INT; exec sh "$0"', str(script)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    return process.pid


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: deadman_session.py SCRIPT")
    print(start(sys.argv[1]), flush=True)
