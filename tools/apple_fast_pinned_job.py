#!/usr/bin/env python3
"""Serial M3 pinned-source jobs that do not need native build preparation.

Usage: SOURCE tools/ALLOWLISTED.py [args...]. Invoke only through the existing
serial queue from an already prepared branch. Reference scripts run CPU oracle
analysis; standalone probes load and verify their exact M2-built binaries.
The matrix timing helper additionally verifies its pinned matrix-only quality
report and original compiled source. No route builds, times an opponent, or
authorizes a promotion.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from apple_fast_job_policy import policy_for


def main():
    source, script, *args = sys.argv[1:]
    assert re.fullmatch(r"[0-9a-f]{40}", source)
    policy = policy_for(source, script, args)
    brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
    assert "Apple M3 Ultra" in brand
    home = Path.home()
    mirror = home / "mojolearn.git"
    subprocess.run(["git", "--git-dir", str(mirror), "cat-file", "-e", source + "^{commit}"], check=True)
    base = home / "mq/pinned-job-wt"
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
        print("PINNED_JOB_SOURCE", source, script, "policy=" + policy, "native_builds=0", flush=True)
        return subprocess.run([str(home / "board-0834/cache/venv/bin/python"), script, *args],
                              cwd=directory, env=environment).returncode
    finally:
        subprocess.run(["git", "--git-dir", str(mirror), "worktree", "remove", "--force",
                        str(directory)], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
