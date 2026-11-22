#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A host inference forward at the board shape on ONE thread against the
thread policy's count: bits and wall (lane neural-pass9).

Every host thread split goes through `host_parallelize`
(`core/host_parallel.mojo`), and MOJOLEARN_CPU_THREADS=1 makes every split a
single task on the calling thread, so the serial walk of each stage IS the
one-thread run. A stage put over host tasks with the same per-row statements
must therefore give the same bytes at both settings. This tool runs the
named class's forward at the board shape in two child processes, one per
setting, from the same weights and input, compares the outputs byte for
byte. OUR CPU IS NEVER TIMED (Andrew, Oct 2 2026): no wall is printed and
`--timing` refuses.

    python tools/host_threads_ab_check.py --model mamba3        # mamba3-infer's shape
    python tools/host_threads_ab_check.py --model transformer   # transformer-infer's
    python tools/host_threads_ab_check.py --model mamba3 --length 256 --calls 2

Prints `HOST_THREADS_AB PASS <model>` or `HOST_THREADS_AB FAIL <what>` and
exits 1 on FAIL. A FAIL is a bug in a threaded stage, never a result.
"""
import argparse

import numpy as np
import hashlib
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: tools/bench_board_neural.py BLOCK_SHAPES[*]["full"]
SHAPES = {
    "mamba3": dict(batch=1, length=2048, d_model=384),
    "transformer": dict(batch=1, length=2048, d_model=384, n_heads=6, n_kv=6, head_dim=64, intermediate=1024),
}


def _splitmix64(z):
    """splitmix64 over a uint64 numpy array, wrapping (the fixture generator's
    hash, transformer/checks/transformer_fixture.mojo::fixture_splitmix64)."""
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class _Fixture:
    """Platform-independent tensors (lane neural-pass21, 2026-10-01): the
    fixture generator's recipe, `f32(lo + (hi - lo) * top24 * 2^-24)` of a
    splitmix64 stream, in uint64 and float64 numpy arithmetic and one
    round to float32. numpy's float32 normal sampler goes through the
    platform libm (logf in the ziggurat tail), so `standard_normal(dtype=
    float32)` gave two x86 hosts and an Apple host different bytes for one
    weight at length 2048, which made the cross-host shas of this tool
    incomparable while the arithmetic was identical (the 30-stage digests
    of tools/transformer_stage_digests.mojo agreed). Every tensor has its
    own id, so a shape change moves no other tensor."""

    def __init__(self, seed):
        self.seed = np.uint64(seed)
        self.next_id = 1

    def __call__(self, *shape, lo=-1.0, hi=1.0):
        n = int(np.prod(shape)) if shape else 1
        tid = np.uint64(self.next_id)
        self.next_id += 1
        with np.errstate(over="ignore"):
            key = _splitmix64(self.seed ^ (tid << np.uint64(32)))
            h = _splitmix64(key + np.arange(n, dtype=np.uint64))
        unit = (h >> np.uint64(40)).astype(np.float64) * 0.000000059604644775390625
        return (lo + (hi - lo) * unit).astype("<f4").reshape(shape)


def _mamba3_weights(np, rng, dm):
    import mojolearn._mamba_impl as M
    di = 2 * dm
    nh = di // M._M3_HEADDIM
    dip = 2 * di + 2 * M._M3_NGROUPS * M._M3_D_STATE + 3 * nh + M._M3_NUM_ROPE_ANGLES
    f = lambda *s: rng(*s, lo=-0.035, hi=0.035)
    return {"block_norm.weight": f(dm) + np.float32(1.0), "in_proj.weight": f(dip, dm), "dt_bias": f(nh),
            "B_norm.weight": f(M._M3_D_STATE) + np.float32(1.0), "C_norm.weight": f(M._M3_D_STATE) + np.float32(1.0),
            "B_bias": f(nh, M._M3_D_STATE), "C_bias": f(nh, M._M3_D_STATE), "D": f(nh),
            "out_proj.weight": f(dm, di)}


def _transformer_weights(np, rng, s):
    dm, nh, nkv, hd, it = s["d_model"], s["n_heads"], s["n_kv"], s["head_dim"], s["intermediate"]
    f = lambda *shape: rng(*shape, lo=-0.035, hi=0.035)
    return {"input_layernorm.weight": f(dm) + np.float32(1.0), "q_proj.weight": f(nh * hd, dm),
            "k_proj.weight": f(nkv * hd, dm), "v_proj.weight": f(nkv * hd, dm), "o_proj.weight": f(dm, nh * hd),
            "post_attention_layernorm.weight": f(dm) + np.float32(1.0), "gate_proj.weight": f(it, dm),
            "up_proj.weight": f(it, dm), "down_proj.weight": f(dm, it)}


def child(args):
    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    import mojolearn
    s = dict(SHAPES[args.model])
    if args.length:
        s["length"] = args.length
    rng = _Fixture(11)
    x = rng(s["batch"], s["length"], s["d_model"], lo=-0.9, hi=0.9)
    if args.model == "mamba3":
        block = mojolearn.Mamba3BlockInference(_mamba3_weights(np, rng, s["d_model"]))
    else:
        block = mojolearn.TransformerBlockInference(_transformer_weights(np, rng, s), n_heads=s["n_heads"], n_kv_heads=s["n_kv"], head_dim=s["head_dim"])
    walls = []
    y = None
    for _ in range(args.calls + 1):
        t0 = time.perf_counter()
        y = block.forward(x)
        walls.append(time.perf_counter() - t0)
    out = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
    Path(args.out).write_bytes(out.tobytes())
    print("WALLS " + " ".join(f"{w * 1e3:.3f}" for w in walls[1:]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SHAPES), default="mamba3")
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--timing", action="store_true",
                    help=argparse.SUPPRESS)   # refused: our CPU is never timed
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.timing and not args.child:
        raise SystemExit("host_threads_ab_check: --timing refused: our CPU is never timed (Andrew, Oct 2 2026)")
    if args.child:
        child(args)
        return 0
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, threads in (("one", "1"), ("policy", None)):
            env = dict(os.environ)
            env.pop("MOJOLEARN_CPU_THREADS", None)
            if threads:
                env["MOJOLEARN_CPU_THREADS"] = threads
            out = os.path.join(tmp, name + ".bin")
            cmd = [sys.executable, __file__, "--child", "--model", args.model, "--calls", str(args.calls), "--out", out]
            if args.length:
                cmd += ["--length", str(args.length)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                sys.stderr.write(r.stdout[-2000:] + r.stderr[-4000:])
                print(f"HOST_THREADS_AB FAIL {name} child exited {r.returncode}")
                return 1
            walls = [float(v) for line in r.stdout.splitlines() if line.startswith("WALLS ") for v in line.split()[1:]]
            data = Path(out).read_bytes()
            results[name] = (hashlib.sha256(data).hexdigest(), statistics.median(walls), len(data))
    if args.timing:
        env = dict(os.environ, MOJOLEARN_HOST_BLOCK_TIMING="1")
        env.pop("MOJOLEARN_CPU_THREADS", None)
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, __file__, "--child", "--model", args.model, "--calls", str(args.calls),
                   "--out", os.path.join(tmp, "t.bin")]
            if args.length:
                cmd += ["--length", str(args.length)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        import collections, re
        calls, cur = [], None
        for line in r.stdout.splitlines():
            m = re.match(r"timing (hblk\.\S+) ([0-9.]+) ms", line)
            if m:
                if cur is None:
                    cur = collections.OrderedDict()
                cur[m.group(1)] = cur.get(m.group(1), 0.0) + float(m.group(2))
                if m.group(1).endswith(".residual") or m.group(1).endswith("down_proj_residual2"):
                    calls.append(cur)
                    cur = None
        if calls:
            last = calls[-1]
            total = sum(last.values())
            print(f"stage walls of the last policy call ({total:.1f} ms ticked):")
            for name, ms in sorted(last.items(), key=lambda kv: -kv[1]):
                print(f"  {name:28s} {ms:8.2f} ms  {100 * ms / total:5.1f}%")
    one, policy = results["one"], results["policy"]
    print(f"{args.model}: {one[2] // 4} floats, sha256 one {one[0][:16]} policy {policy[0][:16]}")
    if one[0] != policy[0]:
        print(f"HOST_THREADS_AB FAIL {args.model} output bytes differ between one thread and the policy")
        return 1
    print(f"HOST_THREADS_AB PASS {args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
