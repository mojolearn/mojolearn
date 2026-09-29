#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lowbit-mma-speed: what each lever bought, from the `LOWBIT` lines of
bench/gemm_lowbit_price_main.mojo (tools/lowbit_mma_speed/price_job.sh's
lowbit.tsv). The line format and its reader are lane/lowbit-units'
(tools/lowbit_units/table.py).

    table.py --levers BOX=lowbit.tsv [--levers BOX=... ...] [--out table.md]
        per box and shape: every product arm's and every conversion arm's
        median, minimum and rate, and the arm's time over fp32.v1's at the
        same shape on the same box; then THE TARGET, per shape: four
        products of each unit plan plus the conversions, over ONE fp32.v1
        product.

    table.py --same-digests a.tsv b.tsv
        two runs of one binary on one box: every arm and shape that both
        ran must carry the same digest. Exits 0 only when every one agreed
        and at least one was compared.

EVERY RATIO HERE IS A TIME OVER A TIME at the same shape on the same box.
Above 1 the numerator took longer. An arm named `probe.*` computes a wrong
product on purpose, to time a part of a kernel alone; it has a time and no
identity, and no ratio of it is a plan's.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lowbit_units"))
from table import read_lowbit  # noqa: E402  (lane/lowbit-units' reader)

FP32 = "fp32.v1"
REFERENCE_UNIT = "int8i32.v1.mma"
QUANT_A = ("convert.int8.quantize.a", "convert.int8.quantize.a.par")
PACK_B = ("convert.int8.pack.b", "convert.int8.pack.b.par")


def same_digests(a_path, b_path):
    a, _, _ = read_lowbit(a_path)
    b, _, _ = read_lowbit(b_path)
    compared = differ = 0
    for key, row in a.items():
        if key not in b:
            print(f"NOT COMPARED {key[0]} {key[1]}: the second run has no such row")
            continue
        compared += 1
        if row["digest"] != b[key]["digest"]:
            differ += 1
            print(f"DIFFER {key[0]} {key[1]}: {row['digest']} vs {b[key]['digest']}")
    print(f"compared {compared}, differ {differ}")
    if compared == 0:
        print("VERDICT: nothing was compared, which is a failure")
        return 1
    if differ:
        print("VERDICT: two runs of one binary on one box DISAGREE")
        return 1
    print("VERDICT: the two runs agree at every arm and shape compared")
    return 0


def is_unit_plan(arm):
    """A product on the integer matrix unit that is a PLAN (not a probe)."""
    return arm == REFERENCE_UNIT or arm.startswith("int8i32.v1.mma.")


def ms(x):
    return "not timed" if x is None else f"{x:.4f}"


def ratio(num, den):
    if num is None or den is None or den <= 0:
        return "n/a"
    return f"{num / den:.3f}"


def levers(boxes, out):
    w = out.write
    w("# lane/lowbit-mma-speed: time and rate per arm\n\n")
    w("Written by `tools/lowbit_mma_speed/table.py --levers`. `over` is the arm's median\n")
    w("over fp32.v1's at the same shape on the same box; above 1 the arm took longer.\n")
    w("One run per box; the median of the timed calls, the minimum beside it.\n")
    w("An arm named `probe.*` computes a WRONG product on purpose (a part of a kernel,\n")
    w("timed alone): it has no identity and is no plan.\n")
    for box, path in boxes:
        rows, shapes, not_run = read_lowbit(path)
        w(f"\n## {box}\n")
        for shape in shapes:
            arms = [r for (s, _), r in rows.items() if s == shape]
            base = rows.get((shape, FP32))
            base_ms = base["median_ms"] if base else None
            first = arms[0]
            w(f"\n### {shape} (m={first['m']} n={first['n']} k={first['k']}, {first['extent']})\n\n")
            w("| arm | median ms | min ms | rate | unit | over fp32.v1 | digest | note |\n")
            w("|---|---:|---:|---:|---|---:|---|---|\n")
            for r in arms:
                rate = "not timed" if r["rate"] is None else f"{r['rate']:.1f}"
                w(f"| {r['arm']} | {ms(r['median_ms'])} | {ms(r['min_ms'])} | {rate} | {r['unit']} | "
                  f"{ratio(r['median_ms'], base_ms)} | {r['digest']} | {r['note']} |\n")
            # ---- the digests that must agree inside the run
            ref = rows.get((shape, "int8i32.v1.flat")) or rows.get((shape, REFERENCE_UNIT))
            if ref:
                bad = [r["arm"] for r in arms
                       if is_unit_plan(r["arm"]) and r["digest"] != ref["digest"]]
                w(f"\nUnit plans whose digest differs from `{ref['arm']}`'s: "
                  f"{', '.join(bad) if bad else 'none'}.\n")
            # ---- the target
            if base_ms:
                qa = [rows.get((shape, a)) for a in QUANT_A]
                pb = [rows.get((shape, a)) for a in PACK_B]
                units = [r for r in arms if is_unit_plan(r["arm"]) and r["median_ms"] is not None]
                if units:
                    w("\nTHE TARGET: four products of a unit plan plus the conversions, over ONE\n")
                    w("fp32.v1 product. A SUM OF MEDIANS of arms timed alone, not one measured\n")
                    w("operation; the conversions are the int8 quantizer's (the 15-bit profile's own\n")
                    w("split and recombination are lane/lowbit-int15's and are not in it).\n\n")
                    w("| unit plan | 4 products | + quantize A (reference) | + quantize A (parallel) | "
                      "+ quantize A and B (reference) | + quantize A and B (parallel) |\n")
                    w("|---|---:|---:|---:|---:|---:|\n")
                    for u in units:
                        four = 4.0 * u["median_ms"]
                        cells = [ratio(four, base_ms)]
                        for i in (0, 1):
                            cells.append(ratio(four + qa[i]["median_ms"], base_ms)
                                         if qa[i] and qa[i]["median_ms"] is not None else "not run")
                        for i in (0, 1):
                            ok = (qa[i] and pb[i] and qa[i]["median_ms"] is not None
                                  and pb[i]["median_ms"] is not None)
                            cells.append(ratio(four + qa[i]["median_ms"] + pb[i]["median_ms"], base_ms)
                                         if ok else "not run")
                        cells = [cells[0], cells[1], cells[2], cells[3], cells[4]]
                        w(f"| {u['arm']} | " + " | ".join(cells) + " |\n")
        if not_run:
            w("\nNot run on this box: " + ", ".join(sorted({a for _, a in not_run})) + ".\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levers", action="append", default=[], metavar="BOX=lowbit.tsv")
    ap.add_argument("--same-digests", nargs=2, metavar=("A", "B"))
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.same_digests:
        return same_digests(*args.same_digests)
    if not args.levers:
        ap.error("give --levers BOX=lowbit.tsv or --same-digests A B")
    boxes = []
    for spec in args.levers:
        if "=" not in spec:
            ap.error(f"--levers takes BOX=path, got {spec!r}")
        box, path = spec.split("=", 1)
        boxes.append((box, path))
    out = open(args.out, "w") if args.out else sys.stdout
    return levers(boxes, out)


if __name__ == "__main__":
    sys.exit(main())
