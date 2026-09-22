#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""mamba2_step_probe: the DEVIATION 2712 probe.

In the 120-lane record at 65ae7612f the `mamba2` lane's `step` and `backward`
parts on a Hot Aisle MI300X (8core VM) differ from the Apple M4 and the H100,
which agree, while `forward` and `prefill` agree on all three; the same lane
read IDENTICAL x3 on a 13core MI300X earlier the same day, and the column
repeats within its run. So the AMD answer is STABLE PER PROCESS and DIFFERENT
PER BOX or per run. `backward` recomputes its own forward from a zero state
and never reads the carried state, so the carried state is not the common
factor; what `step` and `backward` share in the lane is running AFTER earlier
calls on the same block's working buffers.

This tool runs the lane's exact inputs (weights and slabs from
tools/identity_break.py's rules) through the Mamba-2 block in several CALL
ORDERS, repeats each in one process, saves every array of the first repeat
and a hash per repeat, and diffs two saved runs element by element.

    MOJOLEARN_NUMERIC_MODE=identical python3 tools/mamba2_step_probe.py run OUT.npz [--repeats 20] [--fixture base] [--dt-limit 0.01,0.1]
    MOJOLEARN_NUMERIC_MODE=identical python3 tools/mamba2_step_probe.py run WARM.npz --warm "lanes:kmeans,rf-clf,byte-lm;poison:8"
    python3 tools/mamba2_step_probe.py diff A.npz B.npz

--warm (2026-09-14, after the MI300X ran cold equal to Apple on all 86 arrays while
the 120-lane run on the same VM type had mamba2 diverge after a hundred lanes):
run other identity_break lanes first in THIS process (`lanes:<names>`, base
fixture, one fit each), or allocate and free device memory holding a NaN
pattern (`poison:<rounds>`, each a 2048 x 2048 NaN GEMM through linalg.matmul,
so any later read of unwritten device memory is a NaN the diff cannot miss),
or both, `;`-separated. A cold run and a warm run that differ name a read of
memory the lane did not initialize; the first DIFFER line is the part.

Orders (each on a FRESH block and a fresh state):
    lane           forward, prefill(state), step(state), backward  -- the identity_break lane
    backward-only  backward
    step-only      prefill(state), step(state)
    backward-first backward, forward, prefill(state), step(state)
    step-twice     prefill(state), step(state), step(state)        -- two decode tokens

A part that differs between `lane` and `backward-only` (or `step-only`) in
ONE process names an order dependence: a kernel reading working rows an
earlier call left behind. A part that differs between two PROCESSES on one
box in the same order names a per-process source (an unwritten buffer the
allocator hands over differently). A part that differs between two BOXES
only names a device-dependent path. `run` refuses to hash anything but
numeric arrays, like the harness.
"""
import argparse
import hashlib
import importlib.util
import os
import platform
import socket
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDERS = ("lane", "backward-only", "step-only", "backward-first", "step-twice")


def _harness():
    spec = importlib.util.spec_from_file_location("identity_break", os.path.join(HERE, "identity_break.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype == object or a.dtype.kind in "OUSV":
        raise TypeError(f"refusing dtype={a.dtype}")
    m = hashlib.sha256()
    m.update(str(a.dtype).encode()); m.update(str(a.shape).encode()); m.update(a.tobytes())
    return m.hexdigest()[:16]


def _weights(ib):
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    return dm, ib._block_weights("mamba2", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
        "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)},
        ones=("block_norm.weight", "norm.weight"))


def _state_arrays(st, tag):
    return {f"{tag}.h": np.asarray(st.h).copy(), f"{tag}.conv_window": np.asarray(st.conv_window).copy(),
            f"{tag}.buffer_xbc": np.asarray(st.buffer_xbc).copy(), f"{tag}.buffer_dtraw": np.asarray(st.buffer_dtraw).copy(),
            f"{tag}.buffered_tokens": np.asarray([st.buffered_tokens], dtype=np.int64)}


def _one(ml, ib, order, w, x, g, dt_limit):
    """Every array one order produces, keyed `<order>/<part>`."""
    blk = ml.Mamba2Block(w) if dt_limit is None else ml.Mamba2Block(w, dt_limit=dt_limit)
    out = {}
    def forward():
        out["forward"] = np.asarray(blk.forward(x)).copy()
    def prefill():
        st = blk.allocate_state(x.shape[0])
        out["prefill"] = np.asarray(blk.forward(x, st)).copy()
        out.update(_state_arrays(st, "state_after_prefill"))
        return st
    def step(st, tag="step"):
        out[tag] = np.asarray(blk.step(np.ascontiguousarray(x[:, :1, :]), st)).copy()
        out.update(_state_arrays(st, f"state_after_{tag}"))
    def backward():
        grads = blk.backward(x, g)
        for k in sorted(grads):
            out[f"backward.{k}"] = np.asarray(grads[k]).copy()
    if order == "lane":
        forward(); st = prefill(); step(st); backward()
    elif order == "backward-only":
        backward()
    elif order == "step-only":
        st = prefill(); step(st)
    elif order == "backward-first":
        backward(); forward(); st = prefill(); step(st)
    elif order == "step-twice":
        st = prefill(); step(st); step(st, "step2")
    else:
        raise ValueError(order)
    return {f"{order}/{k}": v for k, v in out.items()}


def _device(ml, ib):
    info = dict(vendor=str(ml.vendor()), gpu_arch=str(ml.gpu_arch()), gpu_arch_how=str(ml.gpu_arch_how()),
                platform=platform.platform(), host=socket.gethostname(), cpu_count=str(os.cpu_count()),
                python=platform.python_version(), version=str(getattr(ml, "__version__", "?")))
    try:
        from mojolearn._verify import describe_device, binding_artifacts
        info["device"] = str(describe_device())
        info["bindings"] = ";".join(f"{b['module']}={b['sha256'][:12]}" for b in binding_artifacts())
    except Exception as exc:
        info["device_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        info["commit"], info["commit_source"] = ib.commit_witness()
    except SystemExit as exc:
        info["commit"], info["commit_source"] = "unknown", str(exc)[:120]
    return info


def _warm(ml, ib, spec):
    """What ran in this process before the orders; returns a list of one
    line per warming step for the record."""
    done = []
    X, yc, yr = ib.fixture("base")
    Xh = ib.heldout("base")
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        kind, _, arg = item.partition(":")
        if kind == "lanes":
            for name in [n for n in arg.split(",") if n]:
                if name not in ib.LANES:
                    raise SystemExit(f"--warm lanes: no lane {name!r}")
                p = ib.LANES[name](ml, X, yc, yr, Xh.copy())
                done.append(f"lane {name} {ib._h(np.frombuffer('|'.join(f'{k}={v}' for k, v in sorted(p.items())).encode(), dtype=np.uint8))}")
                print(f"# warm: ran lane {name}")
        elif kind == "poison":
            rounds = int(arg or "4")
            a = np.full((2048, 2048), np.nan, dtype=np.float32)
            for r in range(rounds):
                c = np.asarray(ml.linalg.matmul(a, a, identical=True))
                done.append(f"poison round {r} nan={int(np.isnan(c).sum())} of {c.size}")
                del c
            print(f"# warm: {rounds} NaN GEMM rounds of 2048 x 2048 allocated and freed")
        else:
            raise SystemExit(f"--warm: unknown item {item!r}; use lanes:<names> or poison:<rounds>")
    return done


def cmd_run(args):
    import mojolearn as ml
    ib = _harness()
    mode = ml.numeric_mode()
    if mode != "identical":
        raise SystemExit(f"REFUSING: loaded {mode!r}; set MOJOLEARN_NUMERIC_MODE=identical")
    dm, w = _weights(ib)
    X, _, _ = ib.fixture(args.fixture)
    x = ib._seq(X, 2, 16, dm)
    g = ib._seq(X, 2, 16, dm, skip=1024)
    dt_limit = tuple(float(v) for v in args.dt_limit.split(",")) if args.dt_limit else None
    warmed = _warm(ml, ib, args.warm) if args.warm else []
    orders = [o for o in args.orders.split(",") if o]
    bad = [o for o in orders if o not in ORDERS]
    if bad:
        raise SystemExit(f"unknown orders {bad}; orders are {ORDERS}")
    arrays, hashes, moved = {}, {}, []
    for order in orders:
        for r in range(args.repeats):
            got = _one(ml, ib, order, w, x, g, dt_limit)
            for k, v in got.items():
                h = _sha(v)
                hashes.setdefault(k, []).append(h)
                if r == 0:
                    arrays[k] = v
                elif h != hashes[k][0]:
                    moved.append((k, r))
        print(f"# {order}: {args.repeats} repeats, parts {sorted(set(k.split('/', 1)[1] for k in got))}")
    info = _device(ml, ib)
    info.update(fixture=args.fixture, repeats=str(args.repeats), dt_limit=args.dt_limit or "default", orders=",".join(orders),
                warm=args.warm or "cold", warmed=";".join(warmed) or "none")
    print("# " + " ".join(f"{k}={v}" for k, v in info.items() if k != "bindings"))
    W = 44
    print(f"| {'part':<{W}} | repeat 0         | in-process |")
    print(f"|{'-' * (W + 2)}|------------------|------------|")
    for k in sorted(hashes):
        hs = hashes[k]
        print(f"| {k:<{W}} | {hs[0]} | {'STABLE' if len(set(hs)) == 1 else 'MOVED ' + str(len(set(hs)))} |")
    # order dependence, in this one process: the same part reached through two orders
    parts = {}
    for k in hashes:
        order, part = k.split("/", 1)
        parts.setdefault(part, {})[order] = hashes[k][0]
    dep = {p: v for p, v in parts.items() if len(set(v.values())) > 1}
    for p, v in sorted(dep.items()):
        print(f"ORDER-DEPENDENT {p}: " + " ".join(f"{o}={h}" for o, h in sorted(v.items())))
    for k, r in moved:
        print(f"MOVED in process: {k} at repeat {r}: {hashes[k][0]} vs {hashes[k][r]}")
    np.savez_compressed(args.out, **{k.replace("/", "__"): v for k, v in arrays.items()},
             __hashes=np.asarray([f"{k}={','.join(v)}" for k, v in sorted(hashes.items())]),
             __info=np.asarray([f"{k}={v}" for k, v in sorted(info.items())]))
    print(f"# saved {args.out}: {len(arrays)} arrays; in-process moved={len(moved)} order-dependent parts={len(dep)}")
    return 1 if moved or dep else 0


def cmd_diff(args):
    a, b = np.load(args.a), np.load(args.b)
    ia = dict(s.split("=", 1) for s in a["__info"].tolist()) if "__info" in a.files else {}
    ib_ = dict(s.split("=", 1) for s in b["__info"].tolist()) if "__info" in b.files else {}
    # a harness dump (MOJOLEARN_IDENTITY_DUMP_DIR) carries only the `lane` order
    # and its inputs; the diff runs over the keys both files have and names
    # the ones only one side has
    only_a = sorted(k for k in a.files if not k.startswith("__") and k not in b.files)
    only_b = sorted(k for k in b.files if not k.startswith("__") and k not in a.files)
    if only_a or only_b:
        print(f"# keys only in A: {len(only_a)}; only in B: {len(only_b)} (compared: the common keys)")
    for k in ("vendor", "gpu_arch", "device", "host", "cpu_count", "commit", "dt_limit", "fixture", "orders", "warm", "tag", "source"):
        print(f"# {k}: {ia.get(k, '?')}  |  {ib_.get(k, '?')}")
    keys = sorted(k for k in a.files if not k.startswith("__") and k in b.files)
    same, differ = 0, []
    for k in keys:
        va, vb = a[k], b[k]
        if va.shape != vb.shape or va.dtype != vb.dtype:
            differ.append((k, f"shape/dtype {va.shape}{va.dtype} vs {vb.shape}{vb.dtype}")); continue
        if va.tobytes() == vb.tobytes():
            same += 1; continue
        fa, fb = va.reshape(-1), vb.reshape(-1)
        ne = np.flatnonzero(fa.view(np.uint8).reshape(fa.size, -1) != fb.view(np.uint8).reshape(fb.size, -1)) if fa.dtype != object else []
        idx = np.flatnonzero(fa != fb) if fa.dtype.kind == "f" else np.flatnonzero(fa != fb)
        nan_diff = int(np.sum(np.isnan(fa) != np.isnan(fb))) if fa.dtype.kind == "f" else 0
        first = int(idx[0]) if idx.size else -1
        maxabs = float(np.max(np.abs(fa.astype(np.float64) - fb.astype(np.float64)))) if idx.size and fa.dtype.kind in "fiu" else float("nan")
        differ.append((k, f"{idx.size} of {fa.size} differ, first flat index {first} "
                          f"({np.unravel_index(first, va.shape) if first >= 0 else '-'}) A={fa[first] if first >= 0 else '-'} "
                          f"B={fb[first] if first >= 0 else '-'} maxabs={maxabs:.3e} nan_mismatch={nan_diff}"))
    print(f"equal={same} differ={len(differ)} of {len(keys)}")
    for k, why in differ:
        print(f"DIFFER {k.replace('__', '/')}: {why}")
    return 1 if differ else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("out"); r.add_argument("--repeats", type=int, default=20)
    r.add_argument("--fixture", default="base"); r.add_argument("--dt-limit", default="")
    r.add_argument("--orders", default=",".join(ORDERS))
    r.add_argument("--warm", default="", help="lanes:<names> and/or poison:<rounds>, ;-separated; see the docstring")
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b")
    args = ap.parse_args()
    sys.exit(cmd_run(args) if args.cmd == "run" else cmd_diff(args))


if __name__ == "__main__":
    main()
