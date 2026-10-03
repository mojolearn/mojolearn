#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""afn_ab.sh's `custom` lane: time a neural GPU entry point that has no board lane.

    afn_custom_time.py --binding embedding|x_cnn --rounds N --out DIR --arm ARM

embedding: mojolearn.Embedding.from_pretrained (V 50257, d 256, padding_idx 0), forward + backward
           of T 65536 ids; one round = one forward + one backward.
x_cnn:     mojolearn Conv2d(32, 64, 3, padding=1) forward on N 64 x 32 x 16 x 16.
One warmup call, then N timed rounds. Prints the board-shaped line afn_ab.sh parses,
`NEURAL lane=custom arm=<arm> status=ok median_ms=<m> rounds=<n>`, and keeps the last round's
outputs as <out>/custom-<binding>-<arm>.outputs.npz under key `y` (afn_ab.py compare-outputs).
Data comes from a fixed seed, so both arms see the same inputs.
"""
import argparse
import os
import statistics
import sys
import time


def _embedding(np):
    import mojolearn
    rng = np.random.default_rng(0)
    V, d, T = 50257, 256, 65536
    e = mojolearn.Embedding.from_pretrained(rng.standard_normal((V, d)).astype("f4"), padding_idx=0)
    ids = rng.integers(0, V, T).astype("i4")
    dy = rng.standard_normal((T, d)).astype("f4")

    def once():
        y = np.asarray(e.forward(ids))
        g = np.asarray(e.backward(ids, dy))
        return np.concatenate([y.ravel(), g.ravel()])
    return once


def _x_cnn(np):
    import mojolearn
    Conv2d = getattr(mojolearn, "Conv2d", None)
    if Conv2d is None:
        from mojolearn._expansion_cnn import Conv2d
    rng = np.random.default_rng(0)
    c = Conv2d(32, 64, 3, padding=1)
    x = rng.standard_normal((64, 32, 16, 16)).astype("f4")
    return lambda: np.asarray(c.forward(x)).ravel()


def main(argv=None):
    p = argparse.ArgumentParser(prog="afn_custom_time", description=__doc__.split("\n")[0])
    p.add_argument("--binding", required=True, choices=("embedding", "x_cnn"))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", required=True)
    a = p.parse_args(argv)
    if a.rounds < 1:
        p.error("--rounds must be >= 1")
    import numpy as np
    os.makedirs(a.out, exist_ok=True)
    try:
        once = {"embedding": _embedding, "x_cnn": _x_cnn}[a.binding](np)
        once()  # warmup: pipeline compile and first upload are outside the clock
        times, y = [], None
        for _ in range(a.rounds):
            t = time.perf_counter()
            y = once()
            times.append(1e3 * (time.perf_counter() - t))
    except Exception as exc:  # the line afn_ab.sh greps, then the traceback
        print("NEURAL lane=custom arm=%s status=error median_ms=None error=%s" % (a.arm, exc), flush=True)
        raise
    np.savez(os.path.join(a.out, "custom-%s-%s.outputs.npz" % (a.binding, a.arm)), y=y)
    print("NEURAL lane=custom arm=%s status=ok median_ms=%.3f rounds=%d times_ms=%s"
          % (a.arm, statistics.median(times), a.rounds, ",".join("%.3f" % t for t in times)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
