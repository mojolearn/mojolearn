#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where a `matmul_int8` call's time goes (lane/neural-net-experiment,
2026-09-30): the board's gemm-int8 cell at 4096^3 read 2,917 ms on an L40S
and 1,799 ms on an MI325X against torch._int_mm's 48 ms, and the GEMM
ceiling run showed the multiply kernel itself is a few milliseconds. This
prints one call's phases from both sides:

  Python   operand conversion (`_int8_operand`, the board's `(codes, 0)`
           form and the float32 form that quantizes), the output array,
           and the binding call itself;
  binding  with MOJOLEARN_LOWBIT_TIMING=1, `gemm_int8_binding` prints
           `timing int8.<phase> <ms> ms` for context, uploads + output
           allocation, kernel, download, free + drain.

    python tools/int8_profile.py --m 4096 --n 4096 --k 4096 --calls 5

Run it on the box with the linalg binding built. No GPU here: unmeasured.
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=4096)
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--k", type=int, default=4096)
ap.add_argument("--calls", type=int, default=5)
ap.add_argument("--no-binding-timing", action="store_true",
                help="leave MOJOLEARN_LOWBIT_TIMING unset (the phases are then only Python's)")
a = ap.parse_args()
if not a.no_binding_timing:
    os.environ["MOJOLEARN_LOWBIT_TIMING"] = "1"
import numpy as np
import mojolearn
from mojolearn import linalg
from mojolearn import _linalg_impl as L

rng = np.random.default_rng(7)
A = rng.standard_normal((a.m, a.k)).astype(np.float32)
B = rng.standard_normal((a.n, a.k)).astype(np.float32)
qa = rng.integers(-127, 128, size=(a.m, a.k), dtype=np.int8)
qb = rng.integers(-127, 128, size=(a.n, a.k), dtype=np.int8)
za = np.zeros((a.m,), dtype=np.int32)
zb = np.zeros((a.n,), dtype=np.int32)


def phase(label, fn):
    t0 = time.perf_counter(); r = fn(); t1 = time.perf_counter()
    print("python  %-34s %9.2f ms" % (label, (t1 - t0) * 1e3), flush=True)
    return r


print("shape m=%d n=%d k=%d; numpy %s; mojolearn %s" % (a.m, a.n, a.k, np.__version__, getattr(mojolearn, "__version__", "?")))
for call in range(a.calls):
    print("--- call %d, the board's form: (codes, exponents) already made" % call, flush=True)
    oa = phase("_int8_operand(a codes)", lambda: L._int8_operand((qa, za), "a"))
    ob = phase("_int8_operand(b codes)", lambda: L._int8_operand((qb, zb), "b"))
    out = phase("_out_or_new (64 MB output)", lambda: L._out_or_new(None, a.m, a.n, "matmul_int8"))
    bind = L._lowbit_binding()
    addr, addr_ro = L.addr, L.addr_ro
    phase("binding gemm_int8 (device phases above it)", lambda: bind.gemm_int8(
        addr(out, name="out"), addr_ro(oa[0], name="a codes"), addr_ro(oa[1], name="a exponents"),
        addr_ro(ob[0], name="b codes"), addr_ro(ob[1], name="b exponents"), [a.m, a.n, a.k]))
    phase("matmul_int8((codes,0),(codes,0)) whole", lambda: linalg.matmul_int8((qa, za), (qb, zb)))
print("--- the float32 form: quantize both operands on the device, then the product")
phase("quantize_int8(A)", lambda: linalg.quantize_int8(A))
phase("quantize_int8(B)", lambda: linalg.quantize_int8(B))
phase("matmul_int8(A, B) whole", lambda: linalg.matmul_int8(A, B))
