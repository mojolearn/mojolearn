#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Testing-only probe of the parallel drivers' per-pool start-up and close cost.

Opens and closes one ReferenceShardedNeighbors pool `--opens` times on a tiny
fitted KNeighborsClassifier and prints one line per open: the wall seconds of
the whole `with` block, and, when the installed native pool records them
(lane/parallel-pool-spawn-cost onward), each worker's spawn-to-hello seconds,
close-to-reap seconds and the signal its reap needed (0 none, 15 SIGTERM,
9 SIGKILL). Run it once per arm on the same box; never a product path.

    python tools/pool_spawn_cost_probe.py --devices 0 --opens 3
"""
import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0", help="comma-separated device indices")
    parser.add_argument("--opens", type=int, default=3)
    args = parser.parse_args()
    devices = tuple(int(d) for d in args.devices.split(","))

    import numpy as np
    from mojolearn import KNeighborsClassifier
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors

    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    model = KNeighborsClassifier(n_neighbors=4).fit(X, y)
    rows = []
    for i in range(args.opens):
        t0 = time.perf_counter()
        with ReferenceShardedNeighbors(model, devices=devices) as rs:
            t1 = time.perf_counter()
            rs.kneighbors(X[:32])
            t2 = time.perf_counter()
        t3 = time.perf_counter()
        pool = vars(rs).get("_pool") or {}
        timing = pool.get("timing") if isinstance(pool, dict) else None
        row = dict(open=i, devices=list(devices), wall_s=round(t3 - t0, 3), enter_s=round(t1 - t0, 3),
                   first_call_s=round(t2 - t1, 3), exit_s=round(t3 - t2, 3), timing=timing)
        rows.append(row)
        print("POOLCOST " + json.dumps(row), flush=True)
    walls = [r["wall_s"] for r in rows]
    print("POOLCOST_SUMMARY devices=%s opens=%d wall_s=%s" % (args.devices, len(rows), walls), flush=True)


if __name__ == "__main__":
    main()
