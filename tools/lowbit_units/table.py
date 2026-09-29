#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The low-bit timing table and the cross-box hash comparison
(lane/lowbit-units, docs/lanes/LOWBIT_UNITS_PLAN.md).

Reads the `LOWBIT` lines bench/gemm_lowbit_price_main.mojo prints (one file
per box, tools/lowbit_units/price_job.sh's lowbit.tsv) and, where a box has
one, the vendor comparison's vendor_price.json (tools/vendor_gemm_price.py).

    table.py --box h100=.../lowbit.tsv --box mi325x=... --box m2pro=... \\
             [--vendor h100=.../vendor_price.json ...] [--out table.md]
        the table: per box, per shape, per arm, the median and minimum time,
        the rate, and the arm's time over fp32.v1's at the same shape on the
        same box; the conversion costs; where a low-bit plan costs more time
        than fp32.v1; the hash verdict per arm and shape across the boxes.
        Exits 1 when any arm's digests DISAGREE across boxes.

    table.py --expect-disagree clean.tsv sabotage.tsv
        the arm that must fail: every digest of the sabotage run must differ
        from the clean run's at the same arm and shape. Exits 0 only when
        every one differed and at least one was compared.

EVERY RATIO HERE IS A TIME OVER A TIME at the same shape on the same box.
Above 1 the numerator took longer. Nothing here is compared across boxes but
the digests. The vendor rows are COMPARISON ONLY: a vendor library's
arithmetic is the vendor's, and no vendor row carries a digest.
"""
import argparse
import json
import sys

PRODUCTS = ("fp32.v1", "bf16f32.v1.fused", "bf16f32.v1.widen", "int8i32.v1.flat",
            "int8i32.v1.mma", "int8i32.v1.applechunk")
CONVERSIONS = ("convert.int8.quantize.a", "convert.int8.pack.b", "convert.bf16.pack.b",
               "convert.bf16.widen.b", "convert.int8.dequantize.b")
LOWBIT_PRODUCTS = PRODUCTS[1:]


def read_lowbit(path):
    """{(shape, arm): row} and the shapes in file order. A malformed LOWBIT
    line is refused: a silently dropped row would read as an arm not run."""
    rows, shapes, not_run = {}, [], []
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "LOWBIT-NOT-RUN":
            not_run.append((f[2], f[3]))
            continue
        if f[0] != "LOWBIT":
            continue
        if len(f) != 14:
            sys.exit(f"table.py: {path}: a LOWBIT line has {len(f)} fields, not 14: {line.strip()!r}")
        timed = f[8] != "not-timed"
        row = dict(column=f[1], shape=f[2], m=int(f[3]), n=int(f[4]), k=int(f[5]), extent=f[6], arm=f[7],
                   median_ms=float(f[8]) if timed else None, min_ms=float(f[9]) if timed else None,
                   rate=float(f[10]) if timed else None, unit=f[11], digest=f[12], note=f[13])
        if (row["shape"], row["arm"]) in rows:
            sys.exit(f"table.py: {path}: {row['shape']} {row['arm']} appears twice")
        rows[(row["shape"], row["arm"])] = row
        if row["shape"] not in shapes:
            shapes.append(row["shape"])
    if not rows:
        sys.exit(f"table.py: {path} holds no LOWBIT line")
    return rows, shapes, not_run


def expect_disagree(clean_path, sab_path):
    clean, _, _ = read_lowbit(clean_path)
    sab, _, _ = read_lowbit(sab_path)
    compared = same = missing = 0
    for key, row in sab.items():
        if key not in clean:
            print(f"NOT COMPARED {key[0]} {key[1]}: the clean run has no such row")
            missing += 1
            continue
        c = clean[key]
        if (c["m"], c["n"], c["k"]) != (row["m"], row["n"], row["k"]):
            print(f"NOT COMPARED {key[0]} {key[1]}: the extents differ between the runs")
            missing += 1
            continue
        compared += 1
        if c["digest"] == row["digest"]:
            same += 1
            print(f"SABOTAGE NOT SEEN {key[0]} {key[1]}: both digests {row['digest']}")
        else:
            print(f"DISAGREE (expected) {key[0]} {key[1]}: clean {c['digest']} sabotage {row['digest']}")
    print(f"sabotage comparison: {compared} compared, {compared - same} differed, {same} did not, {missing} not compared")
    if compared == 0:
        print("VERDICT: NOTHING COMPARED, which is a failure")
        return 1
    if same or missing:
        print("VERDICT: FAIL, the sabotage arm was not seen at every arm")
        return 1
    print("VERDICT: the sabotage arm was seen at every arm and shape compared")
    return 0


def ratio(a, b):
    return "n/a" if not b or b <= 0 else f"{a / b:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", action="append", default=[], metavar="NAME=lowbit.tsv")
    ap.add_argument("--vendor", action="append", default=[], metavar="NAME=vendor_price.json")
    ap.add_argument("--identity-only", action="append", default=[], metavar="NAME",
                    help="a box judged on bitwise identity only: its time columns read "
                         "'not timed (identity only)' whatever its file holds")
    ap.add_argument("--expect-disagree", nargs=2, metavar=("CLEAN", "SABOTAGE"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.expect_disagree:
        return expect_disagree(*a.expect_disagree)
    if not a.box:
        sys.exit("table.py: name at least one --box NAME=lowbit.tsv")
    boxes, order = {}, []
    for spec in a.box:
        name, _, path = spec.partition("=")
        boxes[name] = read_lowbit(path)
        order.append(name)
        if name in a.identity_only:
            # The decision is about the BOX: a file from a run that did time
            # it is read for its digests and its times are dropped here.
            for r in boxes[name][0].values():
                r["median_ms"] = r["min_ms"] = r["rate"] = None
    for name in a.identity_only:
        if name not in boxes:
            sys.exit(f"table.py: --identity-only {name} names no --box")
    vendors = {}
    for spec in a.vendor:
        name, _, path = spec.partition("=")
        v = json.load(open(path))
        vendors[name] = (v, {r["name"]: r for r in v["rows"] if not r.get("skipped")})

    out = []
    w = out.append
    w("# Low-bit GEMM plans: times, rates and digests")
    w("")
    w("Written by `tools/lowbit_units/table.py` from the `LOWBIT` lines of")
    w("`bench/gemm_lowbit_price_main.mojo`. Every time is one box's time on one run: one call and one")
    w("synchronize per sample, the median of the timed calls, the minimum beside it. Every ratio is a time")
    w("over a time at the same shape on the same box; above 1 the numerator took longer. Rates are")
    w("G MAC/s (multiply-accumulates, `m n k`) for a product and G elem/s for a conversion.")
    w("")

    costs_more = []
    for box in order:
        rows, shapes, not_run = boxes[box]
        column = next(iter(rows.values()))["column"]
        w(f"## {box} (column {column})")
        w("")
        if box in a.identity_only:
            if box in vendors:
                sys.exit(f"table.py: {box} is identity only; it takes no --vendor")
            w("NOT TIMED (IDENTITY ONLY). This box is judged on bitwise identity (Andrew, 2026-09-29); the speed")
            w("gate is judged on NVIDIA and on Apple. Its digests are in the last section.")
            w("")
            for shape in shapes:
                first = next(r for (s, _), r in rows.items() if s == shape)
                w(f"### {shape}: m={first['m']} n={first['n']} k={first['k']} ({first['extent']})")
                w("")
                w("| arm | median ms | min ms | rate | arm over fp32.v1 | note |")
                w("|---|---:|---:|---:|---:|---|")
                for arm in PRODUCTS + CONVERSIONS:
                    r = rows.get((shape, arm))
                    if r is None:
                        if (shape, arm) in not_run:
                            w(f"| {arm} | not run | | | | the column does not have the unit |")
                        continue
                    w(f"| {arm} | not timed (identity only) | not timed (identity only) | "
                      f"not timed (identity only) | not timed (identity only) | {r['note']} |")
                w("")
            continue
        if box in vendors:
            v = vendors[box][0]
            w(f"Vendor comparison: {v['library']} on {v['device']}, torch {v['torch']}, {v['build']}, "
              f"median of {v['repeats']}. COMPARISON ONLY, no digest, no identity claim.")
            w("")
        for shape in shapes:
            first = next(r for (s, _), r in rows.items() if s == shape)
            fp32 = rows.get((shape, "fp32.v1"))
            w(f"### {shape}: m={first['m']} n={first['n']} k={first['k']} ({first['extent']})")
            w("")
            w("| arm | median ms | min ms | rate | arm over fp32.v1 | note |")
            w("|---|---:|---:|---:|---:|---|")
            for arm in PRODUCTS + CONVERSIONS:
                r = rows.get((shape, arm))
                if r is None:
                    if (shape, arm) in not_run:
                        w(f"| {arm} | not run | | | | the column does not have the unit |")
                    continue
                over = ratio(r["median_ms"], fp32["median_ms"]) if fp32 else "n/a"
                w(f"| {arm} | {r['median_ms']:.4f} | {r['min_ms']:.4f} | {r['rate']:.4f} {r['unit']} | {over} | {r['note']} |")
                if arm in LOWBIT_PRODUCTS and fp32 and r["median_ms"] > fp32["median_ms"]:
                    costs_more.append((box, shape, arm, r["median_ms"], fp32["median_ms"], r["note"]))
            # DERIVED: what one call pays when the activations arrive float32.
            qa = rows.get((shape, "convert.int8.quantize.a"))
            for arm in ("int8i32.v1.flat", "int8i32.v1.mma", "int8i32.v1.applechunk"):
                r = rows.get((shape, arm))
                if r and qa and fp32:
                    total = r["median_ms"] + qa["median_ms"]
                    w(f"| DERIVED {arm} + quantize.a | {total:.4f} | | | {ratio(total, fp32['median_ms'])} | sum of two medians |")
                    if total > fp32["median_ms"]:
                        costs_more.append((box, shape, arm + " + quantize.a (DERIVED)", total, fp32["median_ms"], r["note"]))
            if box in vendors and shape in vendors[box][1]:
                vr = vendors[box][1][shape]
                same = (vr["m"], vr["n"], vr["k"]) == (first["m"], first["n"], first["k"])
                for arm in ("strict", "tf32", "bf16"):
                    if vr.get(arm) is None:
                        if arm + "_error" in vr:
                            w(f"| vendor {arm} (COMPARISON ONLY) | failed | | | | {vr[arm + '_error'][:60]} |")
                        continue
                    if not same:
                        w(f"| vendor {arm} (COMPARISON ONLY) | {vr[arm]:.4f} | {vr[arm + '_best']:.4f} | | n/a | "
                          f"the vendor ran the whole row (n={vr['n']}), ours ran capped: no ratio |")
                        continue
                    ours = ratio(fp32["median_ms"], vr[arm]) if fp32 else "n/a"
                    w(f"| vendor {arm} (COMPARISON ONLY) | {vr[arm]:.4f} | {vr[arm + '_best']:.4f} | "
                      f"{vr['macs'] / (vr[arm] * 1e6):.4f} GMAC/s | | fp32.v1 over this: {ours} |")
            w("")

    w("## Where a low-bit plan costs more time than fp32.v1")
    w("")
    if not costs_more:
        w("Nowhere in these runs.")
    else:
        w("| box | shape | arm | arm ms | fp32.v1 ms | arm over fp32.v1 | note |")
        w("|---|---|---|---:|---:|---:|---|")
        for box, shape, arm, ms, base, note in costs_more:
            w(f"| {box} | {shape} | {arm} | {ms:.4f} | {base:.4f} | {ratio(ms, base)} | {note} |")
    w("")

    w("## Digests across the boxes")
    w("")
    w("AGREE: every box that ran the arm at the same extents printed the same digest, and at least two did.")
    w("ONE BOX: nothing to compare, which is not a pass.")
    w("")
    w("| shape | arm | verdict | " + " | ".join(order) + " |")
    w("|---|---|---|" + "---|" * len(order))
    disagree = 0
    per_arm = {}
    all_shapes = []
    for box in order:
        for s in boxes[box][1]:
            if s not in all_shapes:
                all_shapes.append(s)
    for shape in all_shapes:
        for arm in PRODUCTS + CONVERSIONS:
            got = {b: boxes[b][0].get((shape, arm)) for b in order}
            ran = {b: r for b, r in got.items() if r}
            if not ran:
                continue
            extents = {(r["m"], r["n"], r["k"]) for r in ran.values()}
            digests = {r["digest"] for r in ran.values()}
            if len(extents) > 1:
                verdict = "NOT COMPARABLE (extents differ)"
            elif len(ran) < 2:
                verdict = "ONE BOX"
            elif len(digests) == 1:
                verdict = "AGREE"
            else:
                verdict = "DISAGREE"
                disagree += 1
            per_arm.setdefault(arm, []).append(verdict)
            w(f"| {shape} | {arm} | {verdict} | " + " | ".join(got[b]["digest"] if got[b] else "not run" for b in order) + " |")
    w("")
    w("| arm | shapes | AGREE | DISAGREE | ONE BOX | not comparable |")
    w("|---|---:|---:|---:|---:|---:|")
    for arm in PRODUCTS + CONVERSIONS:
        v = per_arm.get(arm, [])
        if v:
            w(f"| {arm} | {len(v)} | {v.count('AGREE')} | {v.count('DISAGREE')} | {v.count('ONE BOX')} | "
              f"{sum(1 for x in v if x.startswith('NOT'))} |")
    w("")
    # One profile, several plans: every plan of a profile must print the
    # profile's digest on every box, so the Apple probe (one box) is held to
    # the integer units' digests and the fused plan to the widen plan's.
    w("### One profile, every plan, every box")
    w("")
    w("AGREE: every digest any plan of the profile printed on any box at the shape is the same, and at")
    w("least two boxes ran a plan of it.")
    w("")
    w("| profile | plans | shapes | AGREE | DISAGREE | ONE BOX |")
    w("|---|---|---:|---:|---:|---:|")
    for profile, plans in (("fp32.v1", ("fp32.v1",)),
                           ("bf16f32.v1", ("bf16f32.v1.fused", "bf16f32.v1.widen")),
                           ("int8i32.v1", ("int8i32.v1.flat", "int8i32.v1.mma", "int8i32.v1.applechunk"))):
        tally = {"AGREE": 0, "DISAGREE": 0, "ONE BOX": 0}
        seen = set()
        for shape in all_shapes:
            ran = [(b, p_, boxes[b][0][(shape, p_)]) for b in order for p_ in plans if (shape, p_) in boxes[b][0]]
            if not ran:
                continue
            seen.update(p_ for _, p_, _ in ran)
            if len({(r["m"], r["n"], r["k"]) for _, _, r in ran}) > 1:
                continue  # counted as not comparable in the table above
            if len({b for b, _, _ in ran}) < 2:
                tally["ONE BOX"] += 1
            elif len({r["digest"] for _, _, r in ran}) == 1:
                tally["AGREE"] += 1
            else:
                tally["DISAGREE"] += 1
                disagree += 1
        w(f"| {profile} | {', '.join(p_ for p_ in plans if p_ in seen)} | {sum(tally.values())} | "
          f"{tally['AGREE']} | {tally['DISAGREE']} | {tally['ONE BOX']} |")
    w("")
    text = "\n".join(out) + "\n"
    if a.out:
        open(a.out, "w").write(text)
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)
    if disagree:
        print(f"table.py: {disagree} arm and shape pairs DISAGREE across boxes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
