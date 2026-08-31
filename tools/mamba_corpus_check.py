# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Compare a driver's Mamba-1 stage dumps against the independent corpus.

The corpus (`mamba/corpus/`, see its README.md) carries a float64 per-stage
reference (`ref64/`) computed by somebody else's algorithm (state-spaces/mamba
`selective_scan_ref` in the HuggingFace block order). This tool reads a
directory of raw little-endian float32 stage dumps the lane's driver wrote --
same stage names, same shapes, row-major, one file `<stage>.f32` per stage --
and compares each against ref64 at a stated tolerance. It is a TOLERANCE
check; the lane's bitwise oracle is its own pinned host oracle. A dump that
passes here is consistent with the reference algorithm, certified of nothing.

Usage:
    python tools/mamba_corpus_check.py <case_dir> <stage_dump_dir> [--rtol 1e-5] [--atol 1e-6]
    python tools/mamba_corpus_check.py <case_dir> --self-test [--atol 1e-6]

<case_dir> is one corpus case (holds manifest.json and ref64/). Stages absent
from the dump directory are skipped and listed; a stage passes when every
element satisfies |dump - ref| <= atol + rtol * |ref| (numpy.isclose with
equal_nan; infs must match in sign). Exit 0 only if every stage present
passes and at least one stage was compared.

--self-test feeds the case's own ref32/ (plain torch float32 on a CPU)
through the comparison and prints, per stage, the smallest rtol from a ladder
at which torch FP32 passes with the given atol. That calibrates the default
tolerance honestly; the numbers live in mamba/corpus/README.md. ref32 is
informative only, not a target.

Only numpy is needed (the repo pixi envs have it); torch is not.
"""

import argparse
import json
import os
import sys

import numpy as np


def load_manifest(case_dir):
    p = os.path.join(case_dir, "manifest.json")
    if not os.path.isfile(p):
        sys.exit(f"error: {p} not found (pass one corpus case directory)")
    with open(p) as fh:
        return json.load(fh)


def read_raw(path, dtype, shape):
    a = np.fromfile(path, dtype=dtype)
    n = int(np.prod(shape))
    if a.size != n:
        sys.exit(f"error: {path} holds {a.size} elements, manifest says {n} (shape {shape})")
    return a.reshape(shape)


def compare(dump, ref, rtol, atol):
    """Returns (ok, max_abs, max_rel, first_bad_flat_index or None)."""
    d = dump.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    ok_elem = np.isclose(d, r, rtol=rtol, atol=atol, equal_nan=True)
    finite = np.isfinite(r) & np.isfinite(d)
    diff = np.where(finite, np.abs(d - r), 0.0)
    max_abs = float(diff.max()) if diff.size else 0.0
    nz = finite & (r != 0)
    max_rel = float((diff[nz] / np.abs(r[nz])).max()) if nz.any() else 0.0
    ok = bool(ok_elem.all())
    first_bad = None if ok else int(np.argmin(ok_elem))
    return ok, max_abs, max_rel, first_bad


def unravel(i, shape):
    return tuple(int(v) for v in np.unravel_index(i, shape))


def run_check(case_dir, dump_dir, rtol, atol):
    m = load_manifest(case_dir)
    stages = m["stage_order"]
    compared, skipped, failed = [], [], []
    print(f"case {m['name']}  B={m['B']} L={m['L']} d_model={m['d_model']}  rtol={rtol:g} atol={atol:g}")
    for s in stages:
        info = m["stages"][s]
        dump_path = os.path.join(dump_dir, f"{s}.f32")
        if not os.path.isfile(dump_path):
            skipped.append(s)
            continue
        ref = read_raw(os.path.join(case_dir, info["ref64"]), "<f8", info["shape"])
        dump = read_raw(dump_path, "<f4", info["shape"])
        ok, max_abs, max_rel, bad = compare(dump, ref, rtol, atol)
        compared.append(s)
        line = f"  {s:14s} {'PASS' if ok else 'FAIL'}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        if not ok:
            failed.append(s)
            idx = unravel(bad, info["shape"])
            line += (f"  first_fail flat={bad} index={idx} "
                     f"dump={dump.ravel()[bad]!r} ref={ref.ravel()[bad]!r}")
        print(line)
    unknown = sorted(f[:-4] for f in os.listdir(dump_dir)
                     if f.endswith(".f32") and f[:-4] not in stages)
    if skipped:
        print(f"  skipped (no dump): {', '.join(skipped)}")
    if unknown:
        print(f"  warning: dumps with no reference in this case: {', '.join(unknown)}")
    if not compared:
        print("  no stage dumps found; nothing compared")
        return 2
    if failed:
        print(f"FAIL: {len(failed)}/{len(compared)} compared stages failed")
        return 1
    print(f"PASS: {len(compared)} stages compared, all within tolerance")
    return 0


RTOL_LADDER = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def run_self_test(case_dir, atol):
    m = load_manifest(case_dir)
    print(f"self-test {m['name']}: ref32 (plain torch FP32 CPU) vs ref64, atol={atol:g}")
    print(f"  {'stage':14s} {'max_abs':>10s} {'max_rel':>10s}  smallest passing rtol")
    worst = 0.0
    rc = 0
    for s in m["stage_order"]:
        info = m["stages"][s]
        ref = read_raw(os.path.join(case_dir, info["ref64"]), "<f8", info["shape"])
        r32 = read_raw(os.path.join(case_dir, info["ref32"]), "<f4", info["shape"])
        passing = None
        for rt in RTOL_LADDER:
            ok, max_abs, max_rel, _ = compare(r32, ref, rt, atol)
            if ok:
                passing = rt
                break
        ok, max_abs, max_rel, _ = compare(r32, ref, RTOL_LADDER[-1], atol)
        worst = max(worst, max_rel)
        tag = f"{passing:g}" if passing is not None else f"NONE (> {RTOL_LADDER[-1]:g})"
        if passing is None:
            rc = 1
        print(f"  {s:14s} {max_abs:10.3e} {max_rel:10.3e}  {tag}")
    print(f"  worst max_rel across stages: {worst:.3e}")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir")
    ap.add_argument("dump_dir", nargs="?")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--self-test", action="store_true",
                    help="compare the case's own ref32 against ref64 over an rtol ladder")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(run_self_test(a.case_dir, a.atol))
    if not a.dump_dir:
        ap.error("dump_dir is required unless --self-test")
    sys.exit(run_check(a.case_dir, a.dump_dir, a.rtol, a.atol))


if __name__ == "__main__":
    main()
