#!/usr/bin/env python3
"""Turn the NVLink leg's rows.jsonl into the three-arm table and the checks.

  python3 tools/nccl_nvlink_summarize.py <rows.jsonl> [gate.log]

Answers, in order:
  0. did the NVLink gate pass? (if not, nothing below is a result)
  1. are the inputs order-sensitive at this rank count? (the witness)
  2. is C2 bit-identical to C1, and invariant to its chunk count?
  3. do the settled findings survive peer-to-peer being ENABLED --
     no variation within a process, none across restarts, Ring != Tree?
  4. what do the arms cost: A (NCCL auto), B (NCCL pinned), C1, C2?
"""

import collections
import json
import statistics
import sys


def human(nbytes):
    for unit, div in (("MB", 1 << 20), ("KB", 1 << 10)):
        if nbytes >= div:
            v = nbytes / div
            return f"{v:.0f} {unit}" if v == int(v) else f"{v:.1f} {unit}"
    return f"{nbytes} B"


def main():
    rows = []
    with open(sys.argv[1]) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print("=== 0. the NVLink gate ===")
    if len(sys.argv) > 2:
        try:
            txt = open(sys.argv[2], errors="replace").read()
            for k in ("GATE1", "GATE2", "GATE3"):
                hit = [l for l in txt.splitlines() if l.startswith(k + "=")]
                print("  " + (hit[0] if hit else f"{k}=MISSING"))
        except OSError as e:
            print("  gate log unreadable:", e)
    else:
        print("  (no gate log given)")

    env = next((r for r in rows if "torch" in r), {})
    print("\n=== stack ===")
    for k in ("device_name", "torch", "torch_cuda", "nccl_version", "world_size"):
        print(f"  {k}: {env.get(k)}")

    print("\n=== 1. are the inputs order-sensitive at all? ===")
    for r in rows:
        if r.get("mode") == "oracle":
            n = r["distinct_order_hashes"]
            verdict = "ORDER MATTERS" if n > 1 else "BENIGN -- verdicts void"
            print(f"  world={r['world_size']} {r['dtype']:9s} n={r['elements']:>10d} "
                  f"distinct hashes over summation orders: {n}   {verdict}")
            for name, h in sorted(r["orders"].items()):
                print(f"      {name:8s} {h}")

    print("\n=== 2. is C2 the same arithmetic as C1? ===")
    for r in rows:
        if r.get("mode") == "invariance":
            print(f"  {r['dtype']:9s} {human(r['bytes']):>7s}  "
                  f"C2==C1: {r['c2_equals_c1']}   "
                  f"C2 chunk-invariant {sorted(r['c2_hash_by_chunks'])}: {r['c2_chunk_invariant']}   "
                  f"NCCL==C1: {r['nccl_equals_c1']}")
            print(f"      C1 {r['c1_hash']}   NCCL {r['nccl_hash']}")

    print("\n=== 3. settled findings with PEER-TO-PEER ENABLED ===")
    varied = [r for r in rows if r.get("mode") == "hashes"
              and r.get("distinct_result_hashes", 1) > 1]
    nh = [r for r in rows if r.get("mode") == "hashes"]
    print(f"  {len(nh)} hash cells; {len(varied)} produced more than one result hash "
          f"across their iterations")
    for r in varied:
        print(f"      VARIED: {r['case']} {r['dtype']} n={r['elements']} -> {r['all_hashes']}")

    # restart: case names end _a / _b
    by_cell = collections.defaultdict(dict)
    for r in nh:
        c = r.get("case", "")
        if c.endswith("_a") or c.endswith("_b"):
            by_cell[(c[:-2], r["dtype"], r["elements"])][c[-1]] = r["result_hash"]
    same = sum(1 for v in by_cell.values() if len(v) == 2 and len(set(v.values())) == 1)
    diff = [k for k, v in by_cell.items() if len(v) == 2 and len(set(v.values())) > 1]
    print(f"  restart test: {same} cells identical across process restart, {len(diff)} different")
    for k in diff:
        print(f"      DIFFERED: {k}")

    # Ring vs Tree at this rank count, per dtype and size
    print("  Ring vs Tree (and protocol), one line per dtype/size:")
    algo = collections.defaultdict(dict)
    for r in nh:
        c = r.get("case", "")
        for name in ("RingSimple", "TreeSimple", "RingLL128", "auto"):
            if f"p2pon_{name}_" in c:
                algo[(r["dtype"], r["elements"])].setdefault(name, set()).add(r["result_hash"])
    for key in sorted(algo):
        d, n = key
        got = algo[key]
        parts = []
        for name in ("auto", "RingSimple", "RingLL128", "TreeSimple"):
            if name in got:
                parts.append(f"{name}={sorted(got[name])[0][:8]}")
        distinct = len({h for s in got.values() for h in s})
        ring = got.get("RingSimple", set())
        tree = got.get("TreeSimple", set())
        rt = "Ring!=Tree" if ring and tree and not (ring & tree) else (
             "Ring==Tree" if ring and tree else "n/a")
        print(f"    {d:9s} {human(n * (4 if d == 'float32' else 2)):>7s}  "
              f"{distinct} distinct  {rt}  " + "  ".join(parts))

    print("\n=== 4. the price of determinism ===")
    # Group timing rows into arms. Arm B is a separate process, by necessity.
    src = {"timing_auto": {"nccl": "A_auto", "c1": "C1_gather", "c2": "C2_rs_ag"},
           "timing_pinned_full": {"nccl": "B_pinned_full"},
           "timing_pinned_algoproto": {"nccl": "B_pinned_algoproto"}}
    med = collections.defaultdict(list)
    hsh = collections.defaultdict(set)
    for r in rows:
        if r.get("mode") != "timing":
            continue
        case = r.get("case", "")
        base = next((k for k in src if case.startswith(k)), None)
        if base is None:
            continue
        for arm, label in src[base].items():
            if arm in r.get("ms_median", {}):
                med[(r["dtype"], r["bytes"], label)].append(r["ms_median"][arm])
                hsh[(r["dtype"], r["bytes"], label)].add(r["hash"][arm])

    labels = ["A_auto", "B_pinned_full", "B_pinned_algoproto", "C1_gather", "C2_rs_ag"]
    sizes = sorted({(d, b) for (d, b, _) in med})
    hdr = f"  {'dtype':9s} {'bytes/rank':>10s} " + " ".join(f"{l:>19s}" for l in labels)
    print(hdr)
    for d, b in sizes:
        cells = []
        for l in labels:
            v = med.get((d, b, l))
            cells.append(f"{statistics.median(v):19.3f}" if v else f"{'-':>19s}")
        print(f"  {d:9s} {human(b):>10s} " + " ".join(cells))

    print("\n  ratios against A (NCCL auto), median of rounds:")
    print(f"  {'dtype':9s} {'bytes/rank':>10s} {'B_full/A':>9s} {'B_ap/A':>9s} "
          f"{'C1/A':>9s} {'C2/A':>9s} {'C2/C1':>9s}")
    for d, b in sizes:
        def g(l):
            v = med.get((d, b, l))
            return statistics.median(v) if v else None
        a, bf, bap, c1, c2 = (g("A_auto"), g("B_pinned_full"),
                              g("B_pinned_algoproto"), g("C1_gather"), g("C2_rs_ag"))
        def r(x, y):
            return f"{x / y:9.2f}" if x and y else f"{'-':>9s}"
        print(f"  {d:9s} {human(b):>10s} {r(bf, a)} {r(bap, a)} "
              f"{r(c1, a)} {r(c2, a)} {r(c2, c1)}")

    print("\n  spread across rounds (min..max of the per-round medians, ms):")
    for d, b in sizes:
        parts = []
        for l in labels:
            v = med.get((d, b, l))
            if v and len(v) > 1:
                parts.append(f"{l}={min(v):.3f}..{max(v):.3f}")
        if parts:
            print(f"    {d:9s} {human(b):>7s}  " + "  ".join(parts))

    print("\n  bit equality of the timed arms:")
    for d, b in sizes:
        got = {l: sorted(hsh[(d, b, l)])[0][:10] for l in labels if hsh.get((d, b, l))}
        c1h = hsh.get((d, b, "C1_gather"), set())
        c2h = hsh.get((d, b, "C2_rs_ag"), set())
        ah = hsh.get((d, b, "A_auto"), set())
        bh = hsh.get((d, b, "B_pinned_full"), set())
        notes = []
        if c1h and c2h:
            notes.append("C2==C1" if c1h == c2h else "C2!=C1  <-- BROKEN")
        if ah and c1h:
            notes.append("A==C1" if ah & c1h else "A!=C1")
        if ah and bh:
            notes.append("A==B" if ah & bh else "A!=B")
        print(f"    {d:9s} {human(b):>7s}  " + "  ".join(notes) + "   " +
              " ".join(f"{k}={v}" for k, v in got.items()))


if __name__ == "__main__":
    main()
