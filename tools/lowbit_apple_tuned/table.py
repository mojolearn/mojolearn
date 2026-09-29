#!/usr/bin/env python3
"""tools/lowbit_apple_tuned/table.py -- the table of one price.log of
bench/gemm_int15_apple_tuned_price_main.mojo: at the four forward rows of
512 tokens, per variant, THE COMPLETE INFERENCE CALL (A converted, then
the product) in ms and over fp32.v1 of the same run, best first (by the
largest of the four ratios), and the training arm (A and B converted) beside it.

    python3 tools/lowbit_apple_tuned/table.py <price.log> [--part inference|training|tuned]
"""
import sys

ROWS = ["llama8b.qkv.t512", "llama8b.mlp_up.t512", "llama8b.mlp_down.t512", "llama8b.lm_head.t512"]


def main():
    path = sys.argv[1]
    part = "inference"
    if "--part" in sys.argv:
        part = sys.argv[sys.argv.index("--part") + 1]
    t = {}
    fp = {}
    for line in open(path):
        f = line.split()
        if len(f) < 9 or f[0] != "TUNED" or f[2] not in ROWS:
            continue
        row, arm, ms = f[2], f[7], float(f[8])
        if arm == "fp32.v1":
            fp[row] = ms
        t.setdefault(arm, {})[row] = ms
    arms = [a for a in t if a.startswith(part + ".") or (part == "tuned" and a.startswith("tuned."))]
    arms += ["int15i64.v1.apple.four"]
    out = []
    for a in arms:
        if not all(r in t[a] and r in fp for r in ROWS):
            continue
        ratios = [t[a][r] / fp[r] for r in ROWS]
        out.append((max(ratios), a, ratios))
    out.sort()
    print("| arm | " + " | ".join(r.replace("llama8b.", "") + " ms (x fp32)" for r in ROWS) + " |")
    print("|---|" + "---|" * len(ROWS))
    print("| fp32.v1 | " + " | ".join("%.2f" % fp[r] for r in ROWS) + " |")
    for worst, a, ratios in out:
        print("| %s | " % a + " | ".join("%.2f (%.2f)" % (t[a][r], x) for r, x in zip(ROWS, ratios)) + " |")


main()
