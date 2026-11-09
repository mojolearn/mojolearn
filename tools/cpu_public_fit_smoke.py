#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Call the public `fit` of every public estimator on a CPU-only process and
print one TSV row each: OK, or the refusal's first line. CPU training is
public, so a row that is not OK is a CPU/GPU parity gap. On a GPU install the
same rows are the per-estimator "does it start" smoke on that GPU.

    python3 tools/cpu_public_fit_smoke.py [--out rows.tsv] [--jobs N]

The estimators are independent units: --jobs (default: the core count) runs
that many worker processes, each fitting every N-th public name (--shard K/N),
and merges their rows in name order. A worker that dies (an abort in native
code) is reported as one CRASH row for its shard.
"""
import argparse
import inspect
import os
import subprocess
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--shard", help="K/N: fit only the names whose sorted index is K mod N (one worker)")
    args = ap.parse_args(argv)
    if args.shard is None and args.jobs > 1:
        return _parallel(args)
    shard_k, shard_n = (int(v) for v in args.shard.split("/")) if args.shard else (0, 1)
    import mojolearn as m
    from mojolearn import _backend
    rng = np.random.default_rng(0)
    X = rng.standard_normal((240, 6)).astype(np.float32)
    y = (X @ np.arange(1, 7, dtype=np.float32)).astype(np.float32)
    yc = (y > 0).astype(np.int32)
    series = (10 + np.arange(96) * 0.1 + np.sin(np.arange(96) * 2 * np.pi / 12)).astype(np.float64)
    rows = [("vendor", _backend.vendor(), "cpu_only=" + str(_backend._CPU_ONLY is not None))] if shard_k == 0 else []
    for index, name in enumerate(sorted(getattr(m, "__all__", dir(m)))):
        if index % shard_n != shard_k:
            continue
        try:
            cls = getattr(m, name)
        except Exception as exc:
            rows.append((name, "IMPORT", str(exc).splitlines()[0][:160]))
            continue
        if not inspect.isclass(cls) or not callable(getattr(cls, "fit", None)):
            continue
        outcome = None
        for call in (lambda e: e.fit(X, yc), lambda e: e.fit(X, y), lambda e: e.fit(X),
                     lambda e: e.fit(series)):
            try:
                est = cls(series, seasonal_periods=12) if name == "ExponentialSmoothing" else cls()
                if name == "ExponentialSmoothing":
                    est.fit()
                else:
                    call(est)
                outcome = ("OK", "")
                break
            except NotImplementedError as exc:
                outcome = ("REFUSED", str(exc).splitlines()[0][:200])
                break
            except Exception as exc:
                text = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
                outcome = ("REFUSED" if "no CPU implementation" in text else "ERROR", text)
                if outcome[0] == "REFUSED":
                    break
        rows.append((name,) + outcome)
    text = "\n".join("\t".join(r) for r in rows) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


def _parallel(args):
    """Run --jobs shard workers at once and merge their rows by name."""
    n = args.jobs
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--shard", "%d/%d" % (k, n)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for k in range(n)]
    head, rows = [], []
    for k, proc in enumerate(procs):
        out, err = proc.communicate()
        got = [tuple(line.split("\t")) for line in out.splitlines() if line]
        if proc.returncode != 0:
            last = (err.strip().splitlines() or ["rc=%d" % proc.returncode])[-1][:200]
            got.append(("shard-%d/%d" % (k, n), "CRASH", "rc=%d %s" % (proc.returncode, last)))
        for row in got:
            (head if row[0] == "vendor" else rows).append(row)
    rows = head + sorted(rows)
    text = "\n".join("\t".join(r) for r in rows) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
