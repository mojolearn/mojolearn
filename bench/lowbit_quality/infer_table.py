#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The inference tables from the records `infer_eval.py` wrote. Reads JSON,
runs no model.

    python3 bench/lowbit_quality/infer_table.py --text enwik8=a.json,b.json --text pile_github=c.json,d.json

Records of one text are merged; a record whose evaluation ids or baseline
perplexity differ from the first of its text is refused.

THE PASS RULE (orchestrator, 2026-09-29): the relative perplexity change AND
the upper end of its interval are both under 1 percent, on EVERY text. An
arm measured on one text only reads OWED, never PASS.

THE WIDTH TABLE: weight width by activation width, with the int8 pieces
each operand needs (8 bits one piece, up to 15 bits two) and the matrix
unit products a GEMM needs, pieces of A times pieces of B.
"""
import argparse
import json
import sys

WIDTHS = (8, 10, 12, 15)


def pieces(bits):
    return 1 if bits <= 8 else 2


def load(paths):
    first = json.load(open(paths[0]))
    arms = dict(first["arms"])
    for path in paths[1:]:
        more = json.load(open(path))
        if more["arms"]["a"]["perplexity"] != first["arms"]["a"]["perplexity"] or \
                more["evaluation_set"]["ids_sha256"] != first["evaluation_set"]["ids_sha256"]:
            raise SystemExit("REFUSED: %s was measured on another evaluation set or baseline" % path)
        for k, v in more["arms"].items():
            arms.setdefault(k, v)
    by_name = {}
    for k, v in arms.items():
        by_name.setdefault(v["spec"]["name"], v)
    return first, by_name


def passes(c):
    return bool(c["finite_logits"] and c["rel_ppl_change"] < 0.01 and c["rel_ppl_change_hi"] < 0.01)


def cell_text(c):
    return "%+.4f%% (to %+.4f%%), top-1 %.4f" % (100 * c["rel_ppl_change"], 100 * c["rel_ppl_change_hi"],
                                                 c["top1_agreement"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", required=True, help="NAME=record.json[,record.json...]")
    ap.add_argument("--json", default=None, help="write the merged verdicts here")
    args = ap.parse_args(argv)
    texts = {}
    for spec in args.text:
        name, paths = spec.split("=", 1)
        texts[name] = load(paths.split(","))
    for name, (first, arms) in texts.items():
        e = first["evaluation_set"]
        print("text %s: %s bytes [%d, %d), sha256 %s; %d invalid UTF-8 bytes in them; %d ids in %d windows of %d, "
              "ids sha256 %s; %d scored positions; baseline fp32.v1 perplexity %.6f; numerical floor %s"
              % (name, e["corpus_key"], e["used_byte_start"], e["used_byte_end"], e["used_bytes_sha256"],
                 e.get("invalid_bytes_in_used", 0), e["ids_used"], e["windows"], e["length"], e["ids_sha256"],
                 e["scored_positions"], arms["fp32.v1"]["perplexity"],
                 ("%+.2e" % arms["fp32-acc64"]["rel_ppl_change"]) if "fp32-acc64" in arms else "not measured"))
    windows = sorted(set(first["evaluation_set"]["windows"] for first, _ in texts.values()))
    print()
    print("THE INTERVAL: within each of the %s windows of a text the per-position difference of nll (arm minus "
          "baseline) is averaged; the interval is the mean of those window means plus and minus 1.96 of their "
          "standard error, 95 percent under a normal approximation, mapped through exp(x) - 1; the table prints "
          "its upper end. It bounds the sampling error on THESE texts and says nothing about other text or tasks."
          % "/".join(str(w) for w in windows))
    print("TOP-1 AGREEMENT: the share of scored positions, each with the true context supplied, where the arm's "
          "top token equals the baseline's. It is not a rate of changed tokens in generated text: free-running "
          "generation diverges from the first changed token on.")
    print()
    names = []
    for _, (_, arms) in texts.items():
        for n in arms:
            if n not in names and n not in ("fp32.v1", "fp32-acc64"):
                names.append(n)
    merged = {}
    print("| profile | weights | activations | attention products | " +
          " | ".join("%s: change (upper end), top-1 agreement" % t for t in texts) + " | verdict |")
    print("|---|---|---|---|" + "---|" * (len(texts) + 1))
    for n in names:
        cells = [arms.get(n) for _, (_, arms) in texts.items()]
        have = [c for c in cells if c is not None]
        s = have[0]["spec"]
        if any(not passes(c) for c in have):
            verdict = "MISS"
        elif len(have) < len(texts):
            verdict = "OWED (one text only)"
        else:
            verdict = "PASS"
        if all(c.get("bit_equal_nll_to_baseline") for c in have) and len(have) == len(texts):
            verdict += ", bit-equal to the baseline"
        over = ", ".join("%s=%s" % (k, "/".join(v)) for k, v in s["overrides"].items())
        attn = "yes" if s["attention_products"] else ("no" if not over else "no; " + over)
        print("| %s | %s | %s | %s | %s | %s |" % (
            n, s["weight_kind"], s["activation_kind"], attn,
            " | ".join(cell_text(c) if c is not None else "not measured" for c in cells), verdict))
        merged[n] = dict(spec=s, verdict=verdict,
                         texts={t: (None if c is None else {k: c[k] for k in (
                             "perplexity", "rel_ppl_change", "rel_ppl_change_lo", "rel_ppl_change_hi",
                             "top1_agreement", "finite_logits")}) for t, c in zip(texts, cells)})
    print()
    print("WIDTH TABLE: rows weight width, columns activation width; each cell: int8 pieces of the weight x "
          "pieces of the activation = products, then per text the change (upper end), then the verdict")
    print()
    print("| weight \\ activation | " + " | ".join("%d bits (%d piece%s)" % (a, pieces(a), "" if pieces(a) == 1 else "s")
                                                   for a in WIDTHS) + " |")
    print("|---|" + "---|" * len(WIDTHS))
    sweep = {}
    for w in WIDTHS:
        row = []
        for a in WIDTHS:
            n = "int%dw-int%da" % (w, a)
            cells = [arms.get(n) for _, (_, arms) in texts.items()]
            have = [c for c in cells if c is not None]
            products = pieces(w) * pieces(a)
            if not have:
                row.append("%d products; not measured" % products)
                continue
            verdict = "MISS" if any(not passes(c) for c in have) else ("PASS" if len(have) == len(texts) else "OWED")
            row.append("%dx%d = %d product%s; %s; %s" % (
                pieces(w), pieces(a), products, "" if products == 1 else "s",
                "; ".join("%s %+.3f%% (%+.3f%%)" % (t, 100 * c["rel_ppl_change"], 100 * c["rel_ppl_change_hi"])
                          for t, c in zip(texts, cells) if c is not None), verdict))
            sweep[n] = dict(weight_bits=w, activation_bits=a, products=products, verdict=verdict)
        print("| %d bits (%d piece%s) | " % (w, pieces(w), "" if pieces(w) == 1 else "s") + " | ".join(row) + " |")
    passing = sorted((v["products"], v["weight_bits"] + v["activation_bits"], k) for k, v in sweep.items()
                     if v["verdict"] == "PASS")
    print()
    if passing:
        least = passing[0][0]
        print("cheapest passing cells (%d products): %s" % (least, ", ".join(k for p, _, k in passing if p == least)))
    else:
        print("no cell of the width table passes on every text")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(schema="mojolearn.lowbit_quality.inference_table.v1",
                           rule="change and upper end of its interval both under 1 percent, on every text",
                           texts={t: first["evaluation_set"] for t, (first, _) in texts.items()},
                           baseline={t: arms["fp32.v1"]["perplexity"] for t, (_, arms) in texts.items()},
                           profiles=merged, width_table=sweep), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
