#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Start .github/workflows/verify-smoke.yml on one measurement build's packed core wheel.

    python3 tools/ops/verify_smoke_dispatch.py --wheel-dir <work dir>/<commit12>/linux/final [--ref BRANCH] [--dry-run]

The wheels are packed on this Mac (tools/release.py candidate, linux-pack); release-linux-build.yml uploads no wheel.
So, as tools/release_identity_wheels.py does for cpu-identity-gate.yml, the core wheel goes to R2 (credentials stay
on this Mac) and the workflow gets a presigned GET, its name and its sha256. --ref is the pushed branch whose
workflow file and smoke script run (default: this checkout's branch); the source tested is the wheel's commit
(candidate.json source_commit). tools/release_candidate.py calls dispatch() after the pack, fail soft: the smoke
reports and gates nothing. Prints `VERIFY-SMOKE dispatched ...` with the token its run-name carries.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = "verify-smoke.yml"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def core_wheel(wheel_dir):
    found = sorted(Path(wheel_dir).glob("mojolearn_core-*.whl"))
    if len(found) != 1:
        raise SystemExit(f"verify smoke: expected one mojolearn_core-*.whl in {wheel_dir}, found {len(found)}")
    return found[0]


def current_branch():
    return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def dispatch(wheel_dir, ref="", commit="", dry=False, days=2):
    """Upload the core wheel, dispatch the workflow; returns the token its run-name carries."""
    wheel = core_wheel(wheel_dir)
    if not commit:
        commit = json.loads((Path(wheel_dir) / "candidate.json").read_text())["source_commit"]
    ref = ref or current_branch()
    digest = sha256(wheel)
    token = hashlib.sha256(f"{commit} {time.time()}".encode()).hexdigest()[:12]
    key = f"verify-smoke/{commit[:12]}/{wheel.name}"
    if dry:
        print(f"VERIFY-SMOKE would upload {wheel.name} to R2 {key} and run: gh workflow run {WORKFLOW} --ref {ref} "
              f"-f ref={commit} -f wheel_url=<presigned> -f wheel_name={wheel.name} -f wheel_sha256={digest} "
              f"-f token={token}")
        return token
    sys.path.insert(0, str(ROOT / "tools"))
    import bincache
    import box_persist
    creds = box_persist.load_creds()
    status = bincache.http_put(bincache.presign("PUT", key, 3600, creds), str(wheel))
    if status not in (200, 201):
        raise SystemExit(f"verify smoke: R2 upload of {wheel.name} failed (HTTP {status})")
    url = bincache.presign("GET", key, int(days * 86400), creds)
    argv = ["gh", "workflow", "run", WORKFLOW, "--ref", ref, "-f", f"ref={commit}", "-f", f"wheel_url={url}",
            "-f", f"wheel_name={wheel.name}", "-f", f"wheel_sha256={digest}", "-f", f"token={token}"]
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"verify smoke: gh workflow run failed: {(p.stderr or p.stdout).strip()[-400:]}")
    print(f"VERIFY-SMOKE dispatched {WORKFLOW} on --ref {ref} for {commit[:12]} ({wheel.name}); run-name "
          f"'verify-smoke {commit} {token}'; find it: gh run list --workflow {WORKFLOW} -L 5")
    return token


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel-dir", required=True, help="<work dir>/<commit12>/linux/final of a measurement build")
    ap.add_argument("--ref", default="", help="pushed branch holding the workflow (default: this checkout's branch)")
    ap.add_argument("--commit", default="", help="the wheel's source commit (default: candidate.json)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dispatch(a.wheel_dir, a.ref, a.commit, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
