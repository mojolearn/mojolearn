# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`best_split` (GreedyLogSum) against CatBoost, on columns built to tie.

    pixi run check-greedylogsum

GreedyLogSum is the NUMERIC border default and therefore sits on the path of
every dataset this repository benchmarks, which is what makes it worth a
fast host-only gate of its own rather than only the GPU oracle run.

Its score is log(n) - log(l) - log(r) over INTEGER bin sizes, so equal-size
bins tie EXACTLY, and WHICH of a tied set gets split is decided by
std::priority_queue's heap order. Budgets landing on a complete tier
(15 = 1+2+4+8) are order-invariant: an earlier list-scan queue passed at 15
for months while getting 1392 of 1600 borders wrong at 100. So these cases
cut tiers mid-way at 37, 63, 100 and 200, and budget 15 is kept as the
control that must STAY green when the others go red.

Proven to have teeth: flipping the pop's equal-pair comparison from LEFT to
RIGHT child fails 4 of 6 cases, 93 of 100 borders on two of them, while
budget 15 stays exact exactly as the tier argument predicts.
"""
from gbdt.grid_creator.binarization import best_split

def main() raises:
    var f = open("bench/greedylogsum_oracle.txt", "r")
    var text = f.read()
    f.close()
    var lines = List[String]()
    for line in text.splitlines():
        var s = String(String(line).strip())
        if s.byte_length() > 0:
            lines.append(s^)

    var pos = 0
    var h = lines[pos].split(" ")
    var ncol = Int(String(h[1]))
    pos += 1
    var cols = List[List[Float32]]()
    for _ in range(ncol):
        var ch = lines[pos].split(" ")
        var n = Int(String(ch[2]))
        pos += 1
        var c = List[Float32]()
        for i in range(n):
            c.append(Float32(Float64(lines[pos + i])))
        pos += n
        cols.append(c^)

    var ch2 = lines[pos].split(" ")
    var ncase = Int(String(ch2[1]))
    pos += 1
    var failed = 0
    for _ in range(ncase):
        var hh = lines[pos].split(" ")
        var cid = Int(String(hh[1]))
        var budget = Int(String(hh[2]))
        var nb = Int(String(hh[3]))
        pos += 1
        var exp = List[Float32]()
        for i in range(nb):
            exp.append(Float32(Float64(lines[pos + i])))
        pos += nb
        var got = best_split(cols[cid].copy(), budget)
        if len(got) != len(exp):
            print("  col", cid, "budget", budget, "COUNT catboost", len(exp), "ours", len(got))
            failed += 1
            continue
        var wrong = 0
        for i in range(len(exp)):
            if got[i] != exp[i]:
                wrong += 1
        if wrong == 0:
            print("  col", cid, "budget", budget, ":", nb, "borders exact")
        else:
            print("  col", cid, "budget", budget, ": FAIL", wrong, "of", nb, "differ")
            failed += 1
    if failed != 0:
        raise Error("GreedyLogSum borders disagree with CatBoost in "
                    + String(failed) + " of " + String(ncase) + " cases")
    print("  all", ncase, "cases match CatBoost exactly")
