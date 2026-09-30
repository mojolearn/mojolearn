#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Which Mamba-3 backward stages dominate a Samba training step
(lane/neural-net-experiment, 2026-09-30). On the L40S the Samba step read
136 ms against compiled PyTorch's 40, and removing the forward recompute
moved it 1 to 3%, so the time is in the ~40 backward stages themselves. The
binding already prints a wall per stage under MOJOLEARN_MAMBA_TIMING=1
(`timing m3bwd.<stage> <ms> ms`, a synchronize around each; see
mamba/impl/modules/mamba3_prefill_backward.mojo `_mtick`). This runs one
Mamba3Block forward + backward at the Samba shape in a child process with
that env set, parses the lines, and prints the stages sorted by time with
their share, plus the session's reuse counters.

    python tools/mamba3_backward_timing.py --batch 8 --length 512 --d-model 768 --calls 3

The first call compiles and warms; the table is the median over the calls
after it. Unmeasured here (no GPU).
"""
import argparse, collections, json, os, re, statistics, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--length", type=int, default=512)
ap.add_argument("--d-model", type=int, default=768)
ap.add_argument("--calls", type=int, default=3)
ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
a = ap.parse_args()

if a.child:
    os.environ["MOJOLEARN_MAMBA_TIMING"] = "1"
    import numpy as np
    from mojolearn import Mamba3Block
    dm = a.d_model
    di = 2 * dm
    H = di // 64
    N = 128
    dip = 2 * di + 256 + 3 * H + 32
    rng = np.random.default_rng(7)
    def f(shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)
    w = {
        "block_norm.weight": np.ones((dm,), np.float32),
        "in_proj.weight": f((dip, dm)),
        "dt_bias": f((H,), 0.1),
        "B_norm.weight": np.ones((N,), np.float32),
        "C_norm.weight": np.ones((N,), np.float32),
        "B_bias": np.ones((H, N), np.float32),
        "C_bias": np.ones((H, N), np.float32),
        "D": np.ones((H,), np.float32),
        "out_proj.weight": f((dm, di)),
    }
    block = Mamba3Block(w)
    x = f((a.batch, a.length, dm), 1.0)
    dy = f((a.batch, a.length, dm), 1.0)
    for call in range(a.calls + 1):
        print("CALL %d" % call, flush=True)
        y = block.forward(x)
        g = block.backward(x, dy)
        print("SESSION %s" % json.dumps(block.session_info()), flush=True)
    sys.exit(0)

cmd = [sys.executable, os.path.abspath(__file__), "--child", "--batch", str(a.batch),
       "--length", str(a.length), "--d-model", str(a.d_model), "--calls", str(a.calls)]
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
    sys.exit(proc.returncode)
per_call = []
cur = None
session = None
for line in proc.stdout.splitlines():
    if line.startswith("CALL "):
        cur = collections.OrderedDict()
        per_call.append(cur)
        continue
    if line.startswith("SESSION "):
        session = line[8:]
        continue
    m = re.match(r"timing (m3bwd\.\S+) ([0-9.]+) ms", line)
    if m and cur is not None:
        cur[m.group(1)] = cur.get(m.group(1), 0.0) + float(m.group(2))
if len(per_call) < 2:
    print(proc.stdout[-4000:])
    sys.exit("no timing lines: is the mamba binding built with the timing print (MOJOLEARN_MAMBA_TIMING)?")
warm = per_call[1:]
names = list(warm[0].keys())
med = {n: statistics.median(c.get(n, 0.0) for c in warm) for n in names}
total = sum(med.values())
print("Mamba-3 backward, B=%d L=%d d_model=%d, median of %d calls after warm-up (ms)" % (a.batch, a.length, a.d_model, len(warm)))
print("%-36s %9s %7s" % ("stage", "ms", "share"))
for n, v in sorted(med.items(), key=lambda kv: -kv[1]):
    print("%-36s %9.3f %6.1f%%" % (n, v, 100.0 * v / total if total else 0.0))
print("%-36s %9.3f" % ("total (sum of stages)", total))
print("session counters:", session)
