#!/usr/bin/env python3
"""The GEMM lane's ratio table: OUR IDENTICAL arm against the cuBLAS FP32
row of bench/OPPONENT_REFERENCE.md, from one leg's logs.

    python3 tools/gemm_identical_table.py <leg dir>/remote/identical \
        --reference h100 [--plans]

Reads speed_v1.log and speed_core.log (FSPEED lines, medians of the timed
rounds, the v1 plan from its FSPEED-NOTE line) and, with --plans, every
probe.plan<N>.log (the forced-plan sweep: the `dispatch=` time of each
shape under each plan, `REFUSED` where the plan does not admit the shape).
The opponent column is transcribed here from bench/OPPONENT_REFERENCE.md
and must be kept equal to it; nothing here runs an opponent.

Ratios are OUR TIME divided by cuBLAS's: a ratio above 1 is how many times
slower the identical arm is. That is an internal cost, never a speed claim.
"""
import argparse
import os
import re
import statistics
import sys

# bench/OPPONENT_REFERENCE.md, GEMM sections. Keep byte-equal to the file.
REFERENCE = {
    "h100": {  # NVIDIA H100 80GB HBM3, e1g/2026-08-25_155542, cublas-fp32
        "gram.32x32x1M": 0.240, "gram.32x32x64K": 0.039,
        "gram.128sq.x100003": 0.096, "ols.step1.16x16x64K": 0.038,
        "pca.transform.8192x4x4": 0.024,
        "pca.transform.wide.8192x64x128": 0.026,
        "kmeans.dist.4096x64x64": 0.023, "ols.predict.gemv.64Kx16": 0.020,
        "llama8b.qkv.t1": 0.042, "llama8b.qkv.t8": 0.061,
        "llama8b.qkv.t512": 0.374, "llama8b.mlp_up.t1": 0.096,
        "llama8b.mlp_up.t8": 0.160, "llama8b.mlp_up.t512": 1.379,
        "llama8b.mlp_down.t1": 0.097, "llama8b.mlp_down.t8": 0.198,
        "llama8b.mlp_down.t512": 1.185, "llama8b.lm_head.t1": 0.702,
        "llama8b.lm_head.t8": 1.154, "llama8b.lm_head.t512": 10.808,
    },
    "l40s": {  # NVIDIA L40S, e1g/2026-09-09_123601 opponents_l40s2.log
        "llama8b.qkv.t512": 0.536, "llama8b.mlp_up.t512": 1.828,
        "llama8b.mlp_down.t512": 1.686, "llama8b.lm_head.t512": 16.89,
        "pca.transform.wide.8192x64x128": 0.019,
        "kmeans.dist.4096x64x64": 0.015, "gram.128sq.x100003": 0.123,
        "gram.32x32x1M": 0.378, "ols.step1.16x16x64K": 0.024,
    },
}

FSPEED = re.compile(r"^FSPEED lane=gemm arm=(\S+) shape=(\S+) round=\d+ ms=([\d.]+)")
NOTE = re.compile(r"^FSPEED-NOTE lane=gemm arm=(\S+) shape=(\S+) (.*)$")
PROBE = re.compile(r"^\s+OK\s+(\S+) \[.*-> (.*?)\]\s+m=\d+ n=\d+ k=\d+\s+BITS MATCH\s+untuned=([\d.]+)ms dispatch=([\d.]+)ms")
REFUSED = re.compile(r"^\s+REFUSED (\S+) ")


def speed(path):
    """{shape: (median ms, plan or note)} from one gemm_speed_main log."""
    rounds, notes, order = {}, {}, []
    if not os.path.exists(path):
        return {}, order
    for line in open(path):
        m = FSPEED.match(line)
        if m:
            if m[2] not in order:
                order.append(m[2])
            rounds.setdefault(m[2], []).append(float(m[3]))
            continue
        m = NOTE.match(line)
        if m:
            pm = re.search(r"plan=(.*)$", m[3])
            notes[m[2]] = pm[1] if pm else m[3]
            if m[2] not in rounds and m[2] not in order:
                order.append(m[2])
    out = {}
    for s in order:
        if s in rounds:
            out[s] = (statistics.median(rounds[s]), notes.get(s, ""))
        else:
            out[s] = (None, notes.get(s, ""))
    return out, order


def probe(path):
    """{shape: ms or 'REFUSED'} from one forced-plan probe log."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = PROBE.match(line)
        if m:
            out[m[1]] = float(m[4])
            continue
        m = REFUSED.match(line)
        if m:
            out[m[1]] = "REFUSED"
    return out


def short(plan):
    plan = re.sub(r"\(.*\)$", "", plan).strip()
    plan = re.sub(r" fold=\d+ local", "", plan)
    plan = re.sub(r" leaves on grid\.y -> workspace -> .*?(?= tpb=)", "", plan)
    plan = plan.replace(" tpb=256", "")
    return re.sub(r" hwftz=\w+", "", plan)


def fmt(ms):
    if ms is None:
        return "skipped"
    return "%.3f" % ms if ms >= 0.01 else "%.4f" % ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--reference", default="h100", choices=sorted(REFERENCE))
    ap.add_argument("--plans", action="store_true")
    ap.add_argument("--plan-ids", default="0,1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args()
    ref = REFERENCE[a.reference]
    v1, order = speed(os.path.join(a.dir, "speed_v1.log"))
    core, _ = speed(os.path.join(a.dir, "speed_core.log"))
    print("| shape | v1 identical ms (plan) | v1 / cublas-fp32 | core identical ms | core / cublas-fp32 | cublas-fp32 ms |")
    print("|---|---|---|---|---|---|")
    for s in order:
        r = ref.get(s)
        m1, plan = v1.get(s, (None, ""))
        m2 = core.get(s, (None, ""))[0]
        r1 = "%.1fx" % (m1 / r) if (m1 is not None and r) else "-"
        r2 = "%.1fx" % (m2 / r) if (m2 is not None and r) else "-"
        print("| %s | %s (%s) | %s | %s | %s | %s |" % (
            s, fmt(m1), short(plan) if plan else "?", r1, fmt(m2), r2,
            "%.3f" % r if r else "no row"))
    if not a.plans:
        return
    ids = [int(x) for x in a.plan_ids.split(",") if x != ""]
    cols = {i: probe(os.path.join(a.dir, "probe.plan%d.log" % i)) for i in ids}
    print()
    print("Forced-plan sweep, ms (dispatch arm forced to the plan; REFUSED = the plan does not admit the shape):")
    print()
    print("| shape | " + " | ".join("plan %d" % i for i in ids) + " |")
    print("|---|" + "---|" * len(ids))
    shapes = []
    for i in ids:
        for s in cols[i]:
            if s not in shapes:
                shapes.append(s)
    for s in shapes:
        cells = []
        for i in ids:
            v = cols[i].get(s)
            cells.append("-" if v is None else (v if isinstance(v, str) else fmt(v)))
        print("| %s | %s |" % (s, " | ".join(cells)))


if __name__ == "__main__":
    sys.exit(main())
