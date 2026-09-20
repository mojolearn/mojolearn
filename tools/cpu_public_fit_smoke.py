#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Call the public `fit` of every public estimator on a CPU-only process and
print one TSV row each: OK, or the refusal's first line. CPU training is
public, so a row that is not OK is a CPU/GPU parity gap.

    python3 tools/cpu_public_fit_smoke.py [--out rows.tsv]
"""
import argparse
import inspect
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    import mojolearn as m
    from mojolearn import _backend
    rng = np.random.default_rng(0)
    X = rng.standard_normal((240, 6)).astype(np.float32)
    y = (X @ np.arange(1, 7, dtype=np.float32)).astype(np.float32)
    yc = (y > 0).astype(np.int32)
    series = (10 + np.arange(96) * 0.1 + np.sin(np.arange(96) * 2 * np.pi / 12)).astype(np.float64)
    rows = [("vendor", _backend.vendor(), "cpu_only=" + str(_backend._CPU_ONLY is not None))]
    for name in sorted(getattr(m, "__all__", dir(m))):
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


if __name__ == "__main__":
    sys.exit(main())
