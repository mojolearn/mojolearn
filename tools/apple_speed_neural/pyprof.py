"""lane/neural-apple2 (2026-09-28): where a Python-orchestrated neural step
spends its time. The Samba train step at bench_board_neural's `full` shape
(B2 L512 d384, mamba3/attention x 2, V256), generator-initialized, one warm-up
step, then PYPROF_STEPS steps under cProfile: wall per step, the loss per step
(the bit check across commits), and the top functions by own and cumulative
time. Measurement only; changes nothing in the library.

    PYTHONPATH=python python tools/apple_speed_neural/pyprof.py
"""
import cProfile
import hashlib
import io
import os
import pstats
import struct
import time

import numpy as np

import mojolearn as ml
from mojolearn import _training_impl as T


def main():
    steps = int(os.environ.get("PYPROF_STEPS", "3"))
    cfg = ml.SambaConfig(256, 384, ("mamba3", "attention", "mamba3", "attention"),
                         n_heads=6, intermediate=1024)
    stack = ml.SambaStack(cfg, generator=T.Generator(7))
    rng = np.random.default_rng(0)
    batches = [rng.integers(0, 256, size=(2, 513), dtype=np.int32) for _ in range(steps + 1)]
    losses = []
    t0 = time.perf_counter()
    b = batches[0]
    losses.append(stack.train_step(np.ascontiguousarray(b[:, :-1]), np.ascontiguousarray(b[:, 1:]))["loss"])
    print("PYPROF samba warm step %.1f ms" % ((time.perf_counter() - t0) * 1e3))
    pr = cProfile.Profile()
    walls = []
    for k in range(1, steps + 1):
        b = batches[k]
        t0 = time.perf_counter()
        pr.enable()
        losses.append(stack.train_step(np.ascontiguousarray(b[:, :-1]), np.ascontiguousarray(b[:, 1:]))["loss"])
        pr.disable()
        walls.append((time.perf_counter() - t0) * 1e3)
    dig = hashlib.sha256(b"".join(struct.pack("<d", float(x)) for x in losses)).hexdigest()[:16]
    print("PYPROF samba step ms %s losses %s loss_digest %s" % (
        [round(w, 1) for w in walls], [float(x) for x in losses], dig))
    flat = hashlib.sha256(np.ascontiguousarray(stack.flat).tobytes()).hexdigest()[:16]
    print("PYPROF samba params_digest %s" % flat)
    for key in ("tottime", "cumulative"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(key).print_stats(35)
        for line in s.getvalue().splitlines():
            if line.strip():
                print("PYPROF %s %s" % (key[:3], line))


if __name__ == "__main__":
    main()
