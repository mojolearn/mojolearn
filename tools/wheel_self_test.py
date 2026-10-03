#!/usr/bin/env python3
"""The release gate on the verifier's own comparator: install a built wheel
into a FRESH venv and run `python -m mojolearn verify --self-test`.

  python3 tools/wheel_self_test.py WHEEL --out DIR [--python python3.12] [--cpu-threads 3]

The self-test fits `ols/base` twice through the ordinary comparison (untouched
-> must read IDENTICAL against the bundled reference table; one ULP perturbed
-> must read DIVERGENT). A wheel whose bundled table no longer matches its own
bits fails here, which is how 0.8.35 shipped a `verify --quick` that stopped
at the comparator self-test (ols/base 23eecb87d9e84cc7 vs table
3d1d7c30b12d9872). tools/release.py runs it as `macos-self-test` and refuses
to publish the macOS wheel without a PASSED receipt; the preverified flows run
this command by hand on the Mac (docs/RELEASE_CHECKLIST.md).

Writes DIR/results.json {status PASSED|FAILED, wheel, wheel_sha256, rc, ...}
and DIR/self-test.log. Exit 0 only when PASSED. Stdlib only.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

FORMAT = "mojolearn.wheel-self-test.v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def passed(results, wheel):
    """A PASSED receipt for exactly this wheel's bytes."""
    try:
        d = json.loads(Path(results).read_text())
    except (OSError, ValueError):
        return False
    return (d.get("format") == FORMAT and d.get("status") == "PASSED"
            and d.get("wheel_sha256") == sha256(wheel))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wheel")
    ap.add_argument("--out", required=True, help="a new directory for the venv, log and receipt")
    ap.add_argument("--python", default=sys.executable, help="interpreter the venv is made from")
    ap.add_argument("--cpu-threads", type=int, default=3)
    args = ap.parse_args()
    wheel = Path(args.wheel).resolve()
    if not wheel.is_file():
        raise SystemExit(f"no such wheel {wheel}")
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env_dir = out / "venv"
    if Path(args.python).resolve() == Path(sys.executable).resolve():
        venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    else:
        subprocess.run([args.python, "-m", "venv", "--clear", str(env_dir)], check=True)
    py = env_dir / "bin" / "python"
    log = out / "self-test.log"
    # A fresh process environment: nothing from a checkout, no stray mojolearn
    # switches (vendor, tier, harness overrides) from the release shell.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MOJOLEARN_", "PYTHON")) and k != "VIRTUAL_ENV"}
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    record = dict(format=FORMAT, wheel=wheel.name, wheel_sha256=sha256(wheel),
                  started=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  command=f"python -m mojolearn verify --self-test --cpu-threads {args.cpu_threads} --json")
    with log.open("w") as fh:
        rc = subprocess.run([str(py), "-m", "pip", "install", "--no-cache-dir", str(wheel)],
                            cwd=out, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        record["install_rc"] = rc
        if rc == 0:
            proc = subprocess.run([str(py), "-m", "mojolearn", "verify", "--self-test",
                                   "--cpu-threads", str(args.cpu_threads), "--json"],
                                  cwd=out, env=env, capture_output=True, text=True)
            fh.write(proc.stderr)
            fh.write(proc.stdout)
            rc = proc.returncode
            try:
                report = json.loads(proc.stdout[proc.stdout.index("{"):])
            except ValueError:
                report = None
            if isinstance(report, dict):
                record["clean"] = report.get("clean")
                record["perturbed"] = report.get("perturbed")
                record["problems"] = report.get("problems")
                record["self_test_passed"] = report.get("passed")
    record["rc"] = rc
    record["status"] = "PASSED" if rc == 0 and record.get("self_test_passed") is True else "FAILED"
    (out / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    clean = record.get("clean") or {}
    print(f"WHEEL-SELF-TEST {record['status']} {wheel.name} rc={rc} "
          f"clean={clean.get('state')} {clean.get('value')} vs {clean.get('reference')} "
          f"perturbed={(record.get('perturbed') or {}).get('state')} (log {log})")
    return 0 if record["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
