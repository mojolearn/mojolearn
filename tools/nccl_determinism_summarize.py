#!/usr/bin/env python3
"""Read the leg's hashes.jsonl and answer the question it was run to answer.

  python3 tools/nccl_determinism_summarize.py <hashes.jsonl>

Prints, in order: whether the inputs were order-sensitive at all (if not, every
verdict below is void), whether any configuration varied within one process,
whether it varied across process restarts, whether forced Ring and forced Tree
agree, which knobs change the bits, and what the fixed-order all-reduce cost.
"""

import collections
import json
import sys


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load(sys.argv[1])
    ok = [r for r in rows if not r.get("failed")]
    failed = [r for r in rows if r.get("failed")]

    print("=== stack ===")
    any_row = ok[0] if ok else {}
    for k in ("torch", "torch_cuda", "nccl_version", "device_name"):
        print(f"  {k}: {any_row.get(k)}")

    print("\n=== 1. are the inputs order-sensitive at all? ===")
    for r in [r for r in ok if r.get("mode") == "oracle"]:
        print(f"  world={r['world_size']} {r['dtype']:9s} n={r['elements']:>9} "
              f"distinct hashes over 4 summation orders: {r['distinct_order_hashes']}"
              f"   {'ORDER MATTERS' if r['distinct_order_hashes'] > 1 else 'BENIGN -- verdicts void'}")
        for name, h in sorted(r["orders"].items()):
            print(f"      {name:8s} {h}")

    hh = [r for r in ok if r.get("mode") == "hashes"]
    print("\n=== 2. does one process vary run to run? ===")
    bad = [r for r in hh if r["distinct_result_hashes"] != 1]
    print(f"  {len(hh)} configurations x their iterations; "
          f"{len(bad)} produced more than one distinct result hash")
    for r in bad:
        print(f"  VARIES: {r['case']} {r['dtype']} n={r['elements']} -> {r['all_hashes']}")

    print("\n=== 3. stable across process restarts? (same config, run a vs run b) ===")
    by = collections.defaultdict(dict)
    for r in hh:
        case = r["case"]
        if case.endswith("_a") or case.endswith("_b"):
            by[(case[:-2], r["dtype"], r["elements"])][case[-1]] = r["result_hash"]
    same = diff = 0
    for key, d in sorted(by.items()):
        if "a" in d and "b" in d:
            if d["a"] == d["b"]:
                same += 1
            else:
                diff += 1
                print(f"  DIFFERS ACROSS RESTART: {key} {d}")
    print(f"  {same} config/dtype/size cells identical across restarts, {diff} different")

    print("\n=== 4. do the knobs change the bits? (per dtype and size) ===")
    cells = collections.defaultdict(dict)
    for r in hh:
        # World size is part of the key: fewer ranks reduce fewer buffers, so a
        # different hash there is arithmetic, not an algorithm choice.
        cells[(r["dtype"], r["elements"], r["world_size"])][r["case"]] = r["result_hash"]
    for (dtype, n, world), d in sorted(cells.items(), key=lambda kv: kv[0]):
        groups = collections.defaultdict(list)
        for case, h in sorted(d.items()):
            groups[h].append(case)
        print(f"  world={world} {dtype:9s} n={n:>9} ({n * (4 if dtype == 'float32' else 2)} bytes): "
              f"{len(groups)} distinct result(s) over {len(d)} configurations")
        for h, cases in sorted(groups.items(), key=lambda kv: kv[1][0]):
            print(f"      {h}  {', '.join(cases)}")

    print("\n=== 5. the price of a fixed-order all-reduce ===")
    for r in [r for r in ok if r.get("mode") == "timing"]:
        print(f"  {r['dtype']:9s} n={r['elements']:>9} ({r['bytes']:>10} B)  "
              f"nccl {r['nccl_ms_median']:8.3f} ms   fixed-order {r['det_ms_median']:9.3f} ms   "
              f"{r['det_over_nccl']:7.2f}x   same bits as nccl: {r['nccl_equals_det']}")

    if failed:
        print("\n=== configurations that did not run ===")
        for r in failed:
            print(f"  {r['case']} rc={r['rc']}")


if __name__ == "__main__":
    main()
