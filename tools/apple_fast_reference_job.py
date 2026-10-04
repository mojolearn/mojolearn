#!/usr/bin/env python3
"""Queue-only reference analysis at a pinned source, without native builds.

Usage: SOURCE tools/REFERENCE.py [args...]. The existing serial M3 runner
invokes this from an already prepared branch. Reference code runs in its own
exact-source worktree, using the board Python. This is not a GPU job launcher.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def main():
    source, script, *args = sys.argv[1:]
    assert re.fullmatch(r"[0-9a-f]{40}", source)
    assert script in {"tools/arima_assoc_scan_oracle.py", "tools/mcd_saved_oracle.py"}
    brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
    assert "Apple M3 Ultra" in brand
    home = Path.home()
    mirror = home / "mojolearn.git"
    subprocess.run(["git", "--git-dir", str(mirror), "cat-file", "-e", source + "^{commit}"], check=True)
    base = home / "mq/reference-wt"
    base.mkdir(parents=True, exist_ok=True)
    # Own one fresh worktree. Never reuse or remove another job's source.
    directory = Path(tempfile.mkdtemp(prefix=source[:12] + "-", dir=base))
    directory.rmdir()
    subprocess.run(["git", "--git-dir", str(mirror), "worktree", "add", "--quiet", "--detach",
                    str(directory), source], check=True)
    try:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        assert actual == source
        environment = os.environ.copy()
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        environment.pop("PYTHONPATH", None)
        print("REFERENCE_SOURCE", source, script, "native_builds=0", flush=True)
        return subprocess.run([str(home / "board-0834/cache/venv/bin/python"), script, *args],
                              cwd=directory, env=environment).returncode
    finally:
        subprocess.run(["git", "--git-dir", str(mirror), "worktree", "remove", "--force",
                        str(directory)], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
