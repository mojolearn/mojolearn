#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The timing tables of the fifteen-bit GEMM, from the boxes' own lines.

    python3 tools/lowbit_int15/table.py "<label>=<file>" ["<label>=<file>" ...]

Lane lane/lowbit-int15. `<file>` holds the `INT15` lines of one run of
`bench/gemm_int15_price_main.mojo` (an `int15.tsv`, a log, or a steward's
stdout). A label is what the run is called in the table, e.g. "H100, run 2".
A label followed by `=-` is a box with no run: every cell of its tables
reads "not run yet".

TWO TABLES PER RUN, the complete operation in each, every time beside
`fp32.v1`'s at the same row in the same run:

  INFERENCE  the weights were quantized and split once. Per call:
             activations straight to planes (the parallel quantizer), the
             product on the plan the box dispatches, the recombination and
             the dequantization (inside the product's kernel).
  TRAINING   the weights and the gradients are converted every step. Per
             product: both operands to planes, the product. The forward
             row and the two backward rows of a layer are three measured
             operations; the step line ADDS them and says so.

`over` is the operation's median time over fp32.v1's median time: above 1
it took longer than fp32.v1. Nothing here is a claim about another library.
"""
import sys

INF = ("inference.int15i64.v1.planes", "inference.int15i64.v1.codes", "inference.int15i64.v1.planes.rowquant")
TRAIN = ("training.int15i64.v1.planes", "training.int15i64.v1.codes", "training.int15i64.v1.planes.rowquant")
PRODUCTS = ("int15i64.v1.mma", "int15i64.v1.flat", "int15i64.v1.pieces")


def read(path):
    rows, not_run, order = {}, {}, []
    for line in open(path, errors="replace"):
        p = line.split()
        if len(p) >= 14 and p[0] == "INT15":
            _, col, row, m, n, k, cap, arm, med, best, rate, unit, digest, note = p[:14]
            if row not in rows:
                rows[row] = {"m": int(m), "n": int(n), "k": int(k), "cap": cap, "col": col, "arms": {}}
                order.append(row)
            if med != "not-timed":
                rows[row]["arms"][arm] = (float(med), float(best), note)
        elif len(p) >= 5 and p[0] == "INT15-NOT-RUN":
            not_run.setdefault(p[2], {})[p[3]] = " ".join(p[4:])
            if p[2] not in rows and p[2] not in order:
                pass
    return rows, not_run, order


def ms(v):
    return f"{v:.4f}" if v < 100 else f"{v:.2f}"


def over(v, base):
    return f"{v / base:.3f}" if base > 0 else "n/a"


def cell(row, arm, base, not_run):
    got = row["arms"].get(arm)
    if got is None:
        why = not_run.get(arm, "")
        if why.startswith("REFUSED"):
            return "refused", "refused"
        return "not run", ""
    return ms(got[0]), over(got[0], base)


def best_of(row, arms, not_run):
    """The quickest of `arms` that ran, as (arm, median)."""
    have = [(row["arms"][a][0], a) for a in arms if a in row["arms"]]
    if not have:
        return None
    t, a = min(have)
    return a, t


def table(label, path):
    out = [f"## {label}", ""]
    if path == "-":
        out += ["not run yet", ""]
        return out
    rows, not_run, order = read(path)
    if not rows:
        out += [f"REFUSED: {path} holds no INT15 line", ""]
        return out
    col = next(iter(rows.values()))["col"]
    unit = "int15i64.v1.mma" in next(iter(rows.values()))["arms"]
    out += [f"Column `{col}`. The product's plan on this box: "
            + ("the integer matrix unit, four products per k-tile." if unit
               else "no integer matrix unit; the flat kernel on codes and the pieces kernel on planes, "
                    "one thread per cell."), ""]
    fwd = [r for r in order if ".bwd_" not in r]
    out += ["### Inference: one call, weights packed once", "",
            "| row | m x n x k | fp32.v1 ms | product alone ms | over | activations to planes ms | "
            "complete call ms | over | complete call, row quantizer ms | over |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fwd:
        row = rows[r]
        base = row["arms"]["fp32.v1"][0]
        prod = best_of(row, PRODUCTS[:1] if unit else PRODUCTS[1:], not_run.get(r, {}))
        full = best_of(row, INF[:2], not_run.get(r, {}))
        rq = row["arms"].get(INF[2])
        conv = row["arms"].get("convert.int15.planes.a.parallel")
        shape = f"{row['m']} x {row['n']} x {row['k']}" + (" (capped)" if row["cap"] == "CAPPED" else "")
        out.append(
            f"| {r.replace('llama8b.', '')} | {shape} | {ms(base)} | "
            + (f"{ms(prod[1])} | {over(prod[1], base)}" if prod else "refused | refused") + " | "
            + (ms(conv[0]) if conv else "refused") + " | "
            + (f"{ms(full[1])} | {over(full[1], base)}" if full else "refused | refused") + " | "
            + (f"{ms(rq[0])} | {over(rq[0], base)}" if rq else "refused | refused") + " |")
    out += ["", "### Training: one product of a step, both operands converted per call", "",
            "| row | m x n x k | fp32.v1 ms | product alone ms | over | A to planes ms | B to planes ms | "
            "complete product ms | over | complete, row quantizer ms | over |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    steps = {}
    for r in order:
        if ".t512" not in r:
            continue
        row = rows[r]
        base = row["arms"]["fp32.v1"][0]
        nr = not_run.get(r, {})
        prod = best_of(row, PRODUCTS[:1] if unit else PRODUCTS[1:], nr)
        full = best_of(row, TRAIN[:2], nr)
        rq = row["arms"].get(TRAIN[2])
        ca = row["arms"].get("convert.int15.planes.a.parallel")
        cb = row["arms"].get("convert.int15.planes.b.parallel")
        shape = f"{row['m']} x {row['n']} x {row['k']}" + (" (capped)" if row["cap"] == "CAPPED" else "")
        out.append(
            f"| {r.replace('llama8b.', '')} | {shape} | {ms(base)} | "
            + (f"{ms(prod[1])} | {over(prod[1], base)}" if prod else "refused | refused") + " | "
            + (ms(ca[0]) if ca else "refused") + " | " + (ms(cb[0]) if cb else "refused") + " | "
            + (f"{ms(full[1])} | {over(full[1], base)}" if full else "refused | refused") + " | "
            + (f"{ms(rq[0])} | {over(rq[0], base)}" if rq else "refused | refused") + " |")
        layer = r.split(".bwd_")[0]
        steps.setdefault(layer, []).append((r, base, full[1] if full else None))
    out += ["", "### Training: a layer's step, the forward product and the two backward products ADDED", "",
            "Each line is the sum of three operations measured one at a time, not one measured operation.", "",
            "| layer | fp32.v1, three products ms | fifteen-bit, three complete products ms | over |",
            "|---|---:|---:|---:|"]
    for layer, parts in steps.items():
        if len(parts) != 3:
            continue
        b = sum(p[1] for p in parts)
        if any(p[2] is None for p in parts):
            refused = ", ".join(p[0].replace("llama8b.", "") for p in parts if p[2] is None)
            out.append(f"| {layer.replace('llama8b.', '')} | {ms(b)} | refused ({refused}: k above 65536) | refused |")
            continue
        t = sum(p[2] for p in parts)
        out.append(f"| {layer.replace('llama8b.', '')} | {ms(b)} | {ms(t)} | {over(t, b)} |")
    out.append("")
    return out


def main(argv):
    if not argv or any("=" not in a for a in argv):
        print(__doc__)
        return 2
    lines = ["# The fifteen-bit GEMM beside fp32.v1: the complete operation", "",
             "Profile `mojolearn.identical.gemm.int15i64.v1`. Harness `bench/gemm_int15_price_main.mojo`. "
             "Every number is one box's median over the timed calls of one run; `over` is that time over "
             "fp32.v1's at the same row in the same run, so above 1 the fifteen-bit operation took longer.", ""]
    for a in argv:
        label, path = a.rsplit("=", 1)
        lines += table(label, path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
