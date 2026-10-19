#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE AUTOGRAD SMOKE ENTRIES (2026-10-19, lane ag-bench): does the tape start on this vendor.

Four "does it start" entries per vendor (metal, nvidia, amd), 5 s each, nothing timed:

  autograd-gradients     autograd/checks/gpu.mojo, the gradient suite on the vendor's GPU (every
                         case prints PASS and its digest; the digests equal the CPU column's).
  autograd-mlp-identity  autograd/checks/mlp_gpu.mojo, the MLP tape fits against the hand-written
                         trainer on the GPU (run_mlp_cases).
  autograd-fit-smoke     tests/test_autograd_smoke.py through the installed bindings: a tiny MLP
                         and a tiny LSTM fit on the default (tape) path, finite losses, the
                         expected-digest file present.
  autograd-public        python/mojolearn/tests/test_autograd_public.py: the docs' Sequential and
                         ops.Session examples run verbatim; PUBLIC_DIGEST lines in its log.

A check binary is built ONCE per vendor by the orchestrator (`build-commands` prints the lines; this
tool never builds), then `run` starts each entry with a 5 s timer:

    python3 tools/autograd_smoke.py build-commands --vendor nvidia --mojo MOJO --bin-dir DIR
    python3 tools/autograd_smoke.py run --vendor nvidia --bin-dir DIR --python PY --out FILE.json

A row is `ok` (exited 0 inside its timer), `STARTED` (still running at the timer: it started, no
verdict), `failed` (non-zero exit, with its last lines) or `not built` (no binary: refused by name,
never a pass). tools/apple_smoke_set.py lists these entries beside its derived lanes and
tools/gate_smoke_box.py runs them after the board smoke on NVIDIA and AMD.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORS = ("metal", "nvidia", "amd")
#: Andrew, 2026-10-14: smoke is a 5 s "does it start"
TIMEOUT_S = 5
#: the vendor column define a GPU check is built with (bindings/build_byte_lm.sh's rule)
COLUMN_DEFINE = {"metal": [], "nvidia": ["-D", "MOJOLEARN_COLUMN_NVIDIA"], "amd": ["-D", "MOJOLEARN_COLUMN_AMD"]}
ENTRIES = (
    {"name": "autograd-gradients", "source": "autograd/checks/gpu.mojo",
     "what": "the autograd gradient suite on the vendor's GPU, one PASS line and digest per case"},
    {"name": "autograd-mlp-identity", "source": "autograd/checks/mlp_gpu.mojo",
     "what": "the MLP tape fits against the hand-written trainer on the vendor's GPU"},
    {"name": "autograd-fit-smoke", "pytest": "tests/test_autograd_smoke.py",
     "what": "a tiny MLP and LSTM fit through the installed bindings' default path"},
    {"name": "autograd-public", "pytest": "python/mojolearn/tests/test_autograd_public.py",
     "what": "the public Sequential and ops.Session examples of the docs, PUBLIC_DIGEST lines"},
)


def binary(bin_dir, entry, vendor):
    return Path(bin_dir) / ("%s-%s" % (entry["name"], vendor))


def build_commands(vendor, mojo, bin_dir, target=None):
    """The `mojo build` line of every check entry for `vendor` (IDENTICAL, the vendor's column define)."""
    out = []
    for e in ENTRIES:
        if "source" not in e:
            continue
        cmd = [mojo, "build", "-j", "1", "-D", "MOJOLEARN_NUMERIC_IDENTICAL=1"] + COLUMN_DEFINE[vendor]
        if target:
            cmd += ["--target-accelerator", target]
        cmd += ["-I", str(ROOT), str(ROOT / e["source"]), "-o", str(binary(bin_dir, e, vendor))]
        out.append(cmd)
    return out


def _tail(path, n=6, width=300):
    try:
        lines = [l for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return []
    return [l[:width] for l in lines[-n:]]


def start(cmd, log, timeout, env=None, cwd=None):
    """(status, rc, seconds): ok / STARTED / failed for one bounded start; the process group is
    killed at the timer."""
    t0 = time.time()
    with open(log, "ab", buffering=0) as stream:
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                env=env, cwd=cwd)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = None
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    continue
    seconds = round(time.time() - t0, 2)
    if rc is None:
        return "STARTED", None, seconds
    return ("ok" if rc == 0 else "failed"), rc, seconds


def run(vendor, bin_dir, python, out, timeout=TIMEOUT_S):
    logs = Path(out).with_suffix("")
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in ENTRIES:
        log = logs / ("%s-%s.log" % (e["name"], vendor))
        row = {"name": e["name"], "vendor": vendor, "what": e["what"], "timeout_s": timeout, "log": str(log)}
        if "source" in e:
            exe = binary(bin_dir, e, vendor)
            if not exe.is_file():
                row.update(status="not built", detail="no %s (tools/autograd_smoke.py build-commands)" % exe)
                rows.append(row)
                continue
            cmd = [str(exe)]
            env = None
        else:
            cmd = [python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(ROOT / e["pytest"])]
            env = dict(os.environ, MOJOLEARN_AUTOGRAD_SMOKE_OUT=str(logs / ("fit-digests-%s.json" % vendor)))
        status, rc, seconds = start(cmd, log, timeout, env=env, cwd=str(ROOT))
        row.update(command=cmd, status=status, rc=rc, seconds=seconds)
        if status == "failed":
            row["tail"] = _tail(log)
        rows.append(row)
        print("AUTOGRAD-SMOKE vendor=%s entry=%s status=%s seconds=%s" % (vendor, e["name"], status, seconds),
              flush=True)
    doc = {"schema": "mojolearn-autograd-smoke/1", "vendor": vendor, "rows": rows,
           "counts": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})}}
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, out)
    bad = [r for r in rows if r["status"] in ("failed", "not built")]
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-commands", help="print the check binaries' build lines (never builds)")
    b.add_argument("--vendor", required=True, choices=VENDORS)
    b.add_argument("--mojo", default="mojo")
    b.add_argument("--bin-dir", required=True)
    b.add_argument("--target-accelerator", default=None, help="e.g. sm_89 or gfx942 for a cross build")
    r = sub.add_parser("run", help="start every entry once with its 5 s timer")
    r.add_argument("--vendor", required=True, choices=VENDORS)
    r.add_argument("--bin-dir", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=TIMEOUT_S)
    sub.add_parser("list", help="the entries, one line each")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for e in ENTRIES:
            print("%-22s %s" % (e["name"], e.get("source") or e.get("pytest")))
        return 0
    if a.cmd == "build-commands":
        for cmd in build_commands(a.vendor, a.mojo, a.bin_dir, a.target_accelerator):
            print(" ".join(cmd))
        return 0
    return run(a.vendor, a.bin_dir, a.python, a.out, a.timeout)


if __name__ == "__main__":
    sys.exit(main())
