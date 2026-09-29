#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Bring one steward request's verdict directory home (read only).

    python3 tools/lowbit_int15/fetch_steward.py <steward> <request name> <local dir>

Lane lane/lowbit-int15. It runs one `tar` on the steward through
`tools/apple_steward.py`'s own ssh (a cloud Mac through tools/cloudmac.sh,
do-amd as root) and unpacks it under <local dir>. It queues nothing and
changes nothing on the box.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import apple_steward as st  # noqa: E402


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    steward, name, local = argv
    dest = Path(local)
    dest.mkdir(parents=True, exist_ok=True)
    remote = f"cd {st.REMOTE_ROOT}/done/{name} && tar cf - ."
    cmd = st._amd_ssh(remote) if steward in st.AMD_STEWARDS else [str(st.TOOLS / "cloudmac.sh"), "ssh", steward, remote]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode != 0:
        print(f"fetch failed (exit {r.returncode}): {r.stderr.decode(errors='replace')[-400:]}")
        return 1
    subprocess.run(["tar", "xf", "-", "-C", str(dest)], input=r.stdout, check=True)
    print(f"fetched {steward}:{name} -> {dest}: {', '.join(sorted(p.name for p in dest.iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
