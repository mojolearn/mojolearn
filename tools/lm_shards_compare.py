#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two physical-device runs of the ordered LM shard probe."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("one", type=Path)
    ap.add_argument("two", type=Path)
    args = ap.parse_args()
    one = json.loads(args.one.read_text())
    two = json.loads(args.two.read_text())
    for key in ("schema", "shape", "parameters", "steps_requested", "seed"):
        if one.get(key) != two.get(key):
            raise SystemExit("MISMATCH experiment field %s" % key)
    a = {row["logical_shards"]: row for row in one["arms"]}
    b = {row["logical_shards"]: row for row in two["arms"]}
    if a.keys() != b.keys():
        raise SystemExit("MISMATCH logical shard arms")
    summary = []
    for shards in a:
        left, right = a[shards], b[shards]
        if left.get("refused") or right.get("refused"):
            raise SystemExit("REFUSED shards=%d one=%r two=%r" %
                             (shards, left.get("refused"), right.get("refused")))
        if left["final_sha256"] != right["final_sha256"]:
            raise SystemExit("MISMATCH final hashes shards=%d" % shards)
        losses_left = [row["losses_sha256"] for row in left["steps"]]
        losses_right = [row["losses_sha256"] for row in right["steps"]]
        if losses_left != losses_right:
            raise SystemExit("MISMATCH loss hashes shards=%d" % shards)
        summary.append(dict(
            logical_shards=shards,
            one_devices=left["devices"],
            two_devices=right["devices"],
            exact=True,
            one_median_seconds=left["steady_median_seconds"],
            two_median_seconds=right["steady_median_seconds"],
            speedup=left["steady_median_seconds"] / right["steady_median_seconds"],
            hashes=left["final_sha256"],
        ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
