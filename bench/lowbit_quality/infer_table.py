#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The inference table from the record `infer_eval.py` wrote. Reads JSON,
runs no model.

    python3 bench/lowbit_quality/infer_table.py <inference.json> [<more.json> ...]

A later record's arms are added to the first's; an arm named in two records
must carry the same baseline perplexity, or the merge is refused.
"""
import json
import sys


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    first = json.load(open(argv[0]))
    arms = dict(first["arms"])
    base = first["arms"]["a"]["perplexity"]
    for path in argv[1:]:
        more = json.load(open(path))
        if more["arms"]["a"]["perplexity"] != base or \
                more["evaluation_set"]["ids_sha256"] != first["evaluation_set"]["ids_sha256"]:
            print("REFUSED:", path, "was measured on another evaluation set or baseline")
            return 1
        for k, v in more["arms"].items():
            arms.setdefault(k, v)
    floor = arms.get("floor", {}).get("rel_ppl_change")
    e = first["evaluation_set"]
    print("evaluation set: %s bytes [%d, %d), sha256 %s; %d ids in %d windows of %d, ids sha256 %s; %d scored positions"
          % (e["corpus_key"], e["used_byte_start"], e["used_byte_end"], e["used_bytes_sha256"], e["ids_used"],
             e["windows"], e["length"], e["ids_sha256"], e["scored_positions"]))
    print("baseline fp32.v1 perplexity %.6f" % base)
    if floor is not None:
        print("numerical floor (fp32-acc64 against fp32.v1): %+.2e relative" % floor)
    print()
    print("| arm | profile | weights | activations | attention products | perplexity | relative change | 95% interval | top-1 agreement | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, c in arms.items():
        s = c["spec"]
        verdict = c.get("verdict", "baseline" if key == "a" else "floor")
        if c.get("bit_equal_nll_to_baseline"):
            verdict += " (bit-equal to the baseline)"
        elif c.get("inside_numerical_floor"):
            verdict += " (inside the numerical floor)"
        over = ", ".join("%s=%s" % (k, "/".join(v)) for k, v in s["overrides"].items())
        attn = "yes" if s["attention_products"] else ("no" if not over else over)
        print("| %s | %s | %s | %s | %s | %.4f | %+.4f%% | %+.4f%% to %+.4f%% | %.4f | %s |" % (
            c.get("letter", "-"), s["name"], s["weight_kind"], s["activation_kind"], attn, c["perplexity"],
            100 * c["rel_ppl_change"], 100 * c["rel_ppl_change_lo"], 100 * c["rel_ppl_change_hi"],
            c["top1_agreement"], verdict))
    print()
    print("| profile | relative error of each product against the float64 product of the same inputs (first batch; mean over blocks) |")
    print("|---|---|")
    for key, c in arms.items():
        pe = c["product_relative_error"]
        print("| %s | %s |" % (c["spec"]["name"], ", ".join("%s %.2e" % (k, v["mean"]) for k, v in pe.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
