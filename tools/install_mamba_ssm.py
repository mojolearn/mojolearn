#!/usr/bin/env python3
"""Install the mamba-ssm opponent (causal-conv1d + mamba-ssm at bench_board.MAMBA_SSM_COMMIT, built from source
against the venv's torch) into one Python environment, exactly as bench_board's setup does
(bench_board.mamba_ssm_install_steps), and verify both import. Idempotent: does nothing when both already import.

    python3 tools/install_mamba_ssm.py --vendor amd --python /root/gpu-rerun-cache/venv/bin/python

AMD needs the ROCm HIP SDK first (hipcc and the HIP dev headers): bash tools/amd_box_runtime_setup.sh.
Exit 0 when both import afterwards, 1 otherwise. The full build log goes to --log.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_board as B  # noqa: E402

PROBE = "import mamba_ssm, causal_conv1d; print(getattr(mamba_ssm, '__version__', '?'))"


def imports(python):
    try:
        p = subprocess.run([python, "-c", PROBE], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, repr(exc)[:400]
    text = (p.stdout + p.stderr).strip().splitlines()
    return p.returncode == 0, (text[-1] if text else "rc %d" % p.returncode)[:400]


def hip_sdk_problem():
    """None when hipcc is present (AMD), else what is missing and how to install it."""
    rocm = os.environ.get("ROCM_HOME") or "/opt/rocm"
    if os.path.exists(os.path.join(rocm, "bin", "hipcc")) or shutil.which("hipcc"):
        return None
    return ("no hipcc under %s: the ROCm HIP SDK is not installed (run tools/amd_box_runtime_setup.sh, which "
            "installs rocm-hip-sdk7.2.3)" % rocm)


def install(python, vendor, log, opponent_wheels=None):
    """{status, ...} after installing (or finding) mamba-ssm in `python`'s environment on `vendor`."""
    t0 = time.time()
    ok, detail = imports(python)
    if ok:
        return dict(status="ok", installed=False, version=detail, seconds=round(time.time() - t0, 2))
    steps = B.mamba_ssm_install_steps(python, vendor, opponent_wheels)
    if not steps:
        return dict(status="not-planned", vendor=vendor)
    if vendor == "amd":
        why = hip_sdk_problem()
        if why:
            return dict(status="failed", problem=why, before=detail)
    rcs = []
    for argv, extra in steps:
        env = dict(os.environ, **extra)
        rcs.append(B.run_logged(argv, env, log, 4 * 3600))
    ok, detail = imports(python)
    return dict(status="ok" if ok else "failed", installed=True, returncodes=rcs, log=log,
                seconds=round(time.time() - t0, 2),
                **({"version": detail} if ok else {"problem": "mamba_ssm/causal_conv1d still do not import: " + detail}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", choices=("nvidia", "amd"), required=True)
    ap.add_argument("--python", required=True, help="the interpreter of the venv to install into")
    ap.add_argument("--log", default=os.path.abspath("mamba-ssm-install.log"))
    ap.add_argument("--opponent-wheels", help="install the two builds from prebuilt wheels in this directory")
    args = ap.parse_args(argv)
    rec = install(args.python, args.vendor, os.path.abspath(args.log), args.opponent_wheels)
    print(json.dumps(rec, sort_keys=True))
    return 0 if rec["status"] in ("ok", "not-planned") else 1


if __name__ == "__main__":
    sys.exit(main())
