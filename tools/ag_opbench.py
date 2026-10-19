#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-op microbenchmark of the user-facing tape library (mojolearn.ops.Session).

A measurement tool, not the runtime: Python here only builds inputs, calls one
Session op per timed call and reads host clocks. Every number an op computes
comes from the compiled sequence binding.

Protocol per (op, variant, shape) case:
  setup (untimed: inputs that are computed tensors, a recording forward, ...),
  one warm-up call, then N repeats (default 20), each
      sync; t0 = host clock; op call; sync; t1 = host clock
  and the session's computed tensors are dropped (Session.reset) between
  repeats, outside the clock. A sync is a one-value read back (Session.scalar),
  which drains the device queue (DeviceExec.download synchronizes the context).
  The cost of a bare sync (sync; t0; sync; t1) is measured the same way and
  reported; net_us = median_us - sync_us.

Shapes: rows in {512, 4096, 65536} x cols in {1, 64, 256, 4096}, a grid that
spans launch-bound (512 x 1) to bandwidth-bound (65536 x 4096, 1 GiB float32)
tensors; --max-elems bounds the largest tensor a case may hold. Matmul adds
the five GEMMs of one MLP batch step (batch 512, 28 inputs, hidden 256, one
output: Z1, Z2, dW2, dH, dW1) and a square grid 64 .. 4096.

Launch counts: with --launches (the default) a second pass runs in a child
process with MOJOLEARN_AG_PROFILE=1 after the timing pass has finished (so the
fork never sits between timed calls), one call per case, and reads the
process-wide launch profile through the binding's ops_profile_reset /
ops_profile_report entries. A binding without them records launches as null
and the JSON names the owed entries.

Per-launch floor: 10 000 one-element adds, issued in 100 batches of 100
between syncs; the median batch time / 100 is the per-launch floor. The
synced latency of one one-element add is reported beside it.

Usage:
  python tools/ag_opbench.py --out opbench-nvidia.json
  python tools/ag_opbench_report.py opbench-nvidia.json opbench-amd.json
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np

ROWS = (512, 4096, 65536)
COLS = (1, 64, 256, 4096)
# (M, N, K): a (M, K) @ b (K, N); one MLP batch step's five GEMMs
MLP_GEMMS = (("Z1", 512, 256, 28), ("Z2", 512, 1, 256), ("dW2", 256, 1, 512),
             ("dH", 512, 256, 1), ("dW1", 28, 256, 512))
SQUARE = (64, 128, 256, 512, 1024, 2048, 4096)
LINEAR_OUT = (1, 64, 256)
LINEAR_VARIANTS = (("nobias", False, None), ("bias", True, None), ("bias_relu", True, "relu"),
                   ("bias_sigmoid", True, "sigmoid"), ("bias_tanh", True, "tanh"),
                   ("bias_softmax", True, "softmax"))
BACKWARD_INPUTS = (64, 256)
BACKWARD_HIDDEN = 256
F32 = 4
PROFILE_OWED = "ops_profile_reset() and ops_profile_report() -> str (autograd/profile.mojo " \
               "ag_profile_reset / ag_profile_report) registered in bindings/_mojolearn_x_sequence.mojo"


def _shape_str(r, c):
    return f"{r}x{c}"


class Case:
    """One (op, variant, shape) cell: setup(s) -> state, run(s, state, i)."""

    def __init__(self, op, variant, shape, setup, run, nbytes=None, flops=None, elems=0):
        self.op, self.variant, self.shape = op, variant, shape
        self.setup, self.run = setup, run
        self.nbytes, self.flops, self.elems = nbytes, flops, elems

    @property
    def key(self):
        return f"{self.op}|{self.variant}|{self.shape}"


class Bench:
    def __init__(self, args):
        from mojolearn import ops
        self.ops = ops
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.s = ops.Session(numeric_mode=args.numeric_mode)
        self.sentinel = self.s.tensor(np.zeros((1, 1), dtype=np.float32))
        self.profile_ok = hasattr(self.s._b, "ops_profile_report") and hasattr(self.s._b, "ops_profile_reset")

    # ------------------------------------------------------------ clocks
    def sync(self):
        """Drain the device queue: a one-value read back synchronizes the context."""
        self.s.scalar(self.sentinel)

    def _uniform(self, shape, low=-1.0, high=1.0):
        return self.rng.uniform(low, high, size=shape).astype(np.float32)

    # ------------------------------------------------------------ inputs per shape
    def upload_shape(self, r, c):
        """Upload the operands every case of one (r, c) shape reads."""
        s = self.s
        h = {}
        h["x"] = s.tensor(self._uniform((r, c)))
        h["y"] = s.tensor(self._uniform((r, c)))
        h["pos"] = s.tensor(self._uniform((r, c), 0.5, 1.5))   # log, sqrt, div denominators
        h["row"] = s.tensor(self._uniform((1, c), 0.5, 1.5))
        h["col"] = s.tensor(self._uniform((r, 1), 0.5, 1.5))
        h["one"] = s.tensor(self._uniform((1, 1), 0.5, 1.5))
        h["t01"] = s.tensor(self.rng.integers(0, 2, size=(r, c)).astype(np.float32))
        h["ln_w"] = s.tensor(self._uniform((1, c), 0.5, 1.5))
        h["ln_b"] = s.tensor(self._uniform((1, c)))
        h["codes"] = s.tensor(self.rng.integers(0, max(c, 2), size=(r, 1)).astype(np.float32))
        h["labels"] = s.tensor(self.rng.integers(0, 2, size=(r, 1)).astype(np.float32))
        h["prob"] = s.tensor(self._uniform((r, c), 0.05, 0.95))
        return h

    def free_shape(self, h):
        for v in h.values():  # driver-loop: over this shape's uploads
            self.s.free(v)
        self.s.reset()

    # ------------------------------------------------------------ cases
    def shape_cases(self, r, c, h):
        s = self.s
        n = r * c
        sh = _shape_str(r, c)
        out = []
        none = lambda s_: None  # noqa: E731

        def add(op, variant, run, nbytes=None, flops=None, setup=none, shape=sh, elems=n):
            out.append(Case(op, variant, shape, setup, run, nbytes, flops, elems))

        # elementwise binary: same shape and the three broadcast forms
        forms = (("same", "y", n), ("row", "row", c), ("col", "col", r), ("scalar", "one", 1))
        for op in ("add", "sub", "mul", "div"):
            for form, key, bn in forms:
                bkey = "pos" if (op == "div" and key == "y") else key
                fn = getattr(s, op)
                add(op, form, (lambda f, k: lambda s_, st, i: f(h["x"], h[k]))(fn, bkey),
                    F32 * (2 * n + bn))
        # unary
        for op in ("relu", "sigmoid", "tanh", "exp", "log", "sqrt"):
            src = "pos" if op in ("log", "sqrt") else "x"
            add(op, "-", (lambda f, k: lambda s_, st, i: f(h[k]))(getattr(s, op), src), F32 * 2 * n)
        add("scale", "-", lambda s_, st, i: s.scale(h["x"], 0.5), F32 * 2 * n)
        # shape
        add("transpose", "-", lambda s_, st, i: s.transpose(h["x"]), F32 * 2 * n)
        add("reshape", "-", lambda s_, st, i: s.reshape(h["x"], (c, r)), 0)
        # reductions
        for op in ("sum", "mean"):
            for axis, on in ((None, 1), (0, c), (1, r)):
                add(op, f"axis={axis}", (lambda f, a: lambda s_, st, i: f(h["x"], axis=a))(getattr(s, op), axis),
                    F32 * (n + on))
        add("softmax", "rows", lambda s_, st, i: s.softmax(h["x"]), F32 * 2 * n)
        add("log_softmax", "rows", lambda s_, st, i: s.log_softmax(h["x"]), F32 * 2 * n)
        add("layer_norm", "-", lambda s_, st, i: s.layer_norm(h["x"], h["ln_w"], h["ln_b"]),
            F32 * (2 * n + 2 * c))
        add("dropout", "p=0.1", lambda s_, st, i: s.dropout(h["x"], 0.1, seed=7, counter=i), F32 * 2 * n)
        # data steps: permutation of r rows, gather of all rows and of one 512-row batch
        add("permutation", "-", lambda s_, st, i: s.permutation(r, seed=7, epoch=i), F32 * r, elems=r)
        for variant, count in (("all", r), ("batch512", min(512, r))):
            add("gather_rows", variant,
                (lambda cnt: lambda s_, st, i: s.gather_rows(h["x"], st, 0, cnt))(count),
                F32 * (2 * count * c + count),
                setup=lambda s_: s.permutation(r, seed=7, epoch=0))
        # losses, mean and sum
        for red in ("mean", "sum"):
            add("mse", red, (lambda rd: lambda s_, st, i: s.mse(h["x"], h["y"], rd))(red), F32 * (2 * n + 1))
            add("mae", red, (lambda rd: lambda s_, st, i: s.mae(h["x"], h["y"], rd))(red), F32 * (2 * n + 1))
            add("huber", red, (lambda rd: lambda s_, st, i: s.huber(h["x"], h["y"], 1.0, rd))(red),
                F32 * (2 * n + 1))
            add("bce_with_logits", red,
                (lambda rd: lambda s_, st, i: s.bce_with_logits(h["x"], h["t01"], rd))(red), F32 * (2 * n + 1))
            if c >= 2:
                add("cross_entropy", red,
                    (lambda rd: lambda s_, st, i: s.cross_entropy(h["x"], h["codes"], rd))(red),
                    F32 * (n + r + 1))
        # metrics (each returns a float: its own read back is inside the clock)
        if c >= 2:
            add("accuracy", "scores", lambda s_, st, i: s.accuracy(h["codes"], h["x"]), F32 * (n + r))
            add("log_loss", "classes", lambda s_, st, i: s.log_loss(h["codes"], h["prob"]), F32 * (n + r))
        else:
            add("accuracy", "codes", lambda s_, st, i: s.accuracy(h["labels"], h["t01"]), F32 * 2 * r)
            add("log_loss", "binary", lambda s_, st, i: s.log_loss(h["labels"], h["prob"]), F32 * 2 * r)
            add("roc_auc", "-", lambda s_, st, i: s.roc_auc(h["labels"], h["x"]), F32 * 2 * r)
        return out

    def linear_cases(self, r, c, h, w):
        s = self.s
        out = []
        for o in LINEAR_OUT:
            if r * o > self.args.max_elems:
                continue
            for variant, bias, act in LINEAR_VARIANTS:
                run = (lambda o_, b_, a_: lambda s_, st, i: s.linear(
                    h["x"], w[(c, o_)], w[("b", o_)] if b_ else None, activation=a_))(o, bias, act)
                nbytes = F32 * (r * c + c * o + (o if bias else 0) + r * o)
                out.append(Case("linear", variant, f"{r}x{c}->{o}", lambda s_: None, run,
                                nbytes, 2 * r * c * o, r * c))
        return out

    def matmul_cases(self):
        s = self.s
        out = []
        shapes = [(name, m, n, k) for name, m, n, k in MLP_GEMMS] + [(f"sq{q}", q, q, q) for q in SQUARE]
        for name, m, n, k in shapes:
            if max(m * k, k * n, m * n) > self.args.max_elems:
                continue

            def setup(s_, m=m, n=n, k=k):
                a = s.tensor(self._uniform((m, k)))
                b = s.tensor(self._uniform((k, n)))
                return (a, b)

            out.append(Case("matmul", name, f"{m}x{k}@{k}x{n}", setup,
                            lambda s_, st, i: s.matmul(st[0], st[1]),
                            F32 * (m * k + k * n + m * n), 2 * m * n * k, m * n))
        return out

    # ------------------------------------------------------------ one case
    def time_case(self, case):
        s = self.s
        reps = self.args.repeats
        times = []
        st = case.setup(s)
        try:
            case.run(s, st, 0)  # warm-up
            self.sync()
            for i in range(1, reps + 1):  # driver-loop: over timed repeats
                self.sync()
                t0 = time.perf_counter_ns()
                case.run(s, st, i)
                self.sync()
                t1 = time.perf_counter_ns()
                times.append((t1 - t0) / 1e3)
                if case.setup is not None and st is not None and not isinstance(st, tuple):
                    # a computed setup tensor (a permutation) is dropped by reset: rebuild it
                    s.reset()
                    st = case.setup(s)
                else:
                    s.reset()
        finally:
            if isinstance(st, tuple):
                for hh in st:  # driver-loop: over this case's uploads
                    s.free(hh)
            s.reset()
        return times

    def run_case(self, case, launches_only=False):
        rec = {"op": case.op, "variant": case.variant, "shape": case.shape, "key": case.key,
               "bytes": case.nbytes, "flops": case.flops, "elems": case.elems}
        try:
            if launches_only:
                rec["launches"] = self.count_launches(case)
            else:
                t = sorted(self.time_case(case))
                rec["median_us"] = float(np.median(t))
                rec["p10_us"] = float(np.percentile(t, 10))
                rec["min_us"] = float(t[0])
                rec["repeats"] = len(t)
        except Exception as e:  # an op the session refuses or a shape that does not fit
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            try:
                self.s.reset()
            except Exception:
                pass
        return rec

    def count_launches(self, case):
        if not self.profile_ok:
            return None
        s = self.s
        st = case.setup(s)
        try:
            case.run(s, st, 0)  # warm-up: first-call allocations launch nothing extra later
            self.sync()
            s.reset()
            if not isinstance(st, tuple):
                st = case.setup(s)
            self.sync()
            s._b.ops_profile_reset()
            case.run(s, st, 1)
            self.sync()
            text = str(s._b.ops_profile_report())
        finally:
            if isinstance(st, tuple):
                for hh in st:  # driver-loop: over this case's uploads
                    s.free(hh)
            s.reset()
        m = re.search(r"AG_PROFILE level=\d+ launches=(\d+)", text)
        # the sync's one-value download is a copy, not a kernel launch
        return int(m.group(1)) if m else None

    # ------------------------------------------------------------ floors
    def sync_floor(self):
        t = []
        for _ in range(max(self.args.repeats, 20) * 5):  # driver-loop: over timed repeats
            self.sync()
            t0 = time.perf_counter_ns()
            self.sync()
            t.append((time.perf_counter_ns() - t0) / 1e3)
        return float(np.median(t)), float(np.percentile(t, 10))

    def launch_floor(self):
        s = self.s
        a = s.tensor(np.ones((1, 1), dtype=np.float32))
        b = s.tensor(np.ones((1, 1), dtype=np.float32))
        s.add(a, b)  # warm-up
        self.sync()
        s.reset()
        batches, per = self.args.floor_batches, self.args.floor_per_batch
        per_launch = []
        for _ in range(batches):  # driver-loop: over timed batches
            self.sync()
            t0 = time.perf_counter_ns()
            for _ in range(per):  # driver-loop: over one-element adds
                s.add(a, b)
            self.sync()
            per_launch.append((time.perf_counter_ns() - t0) / 1e3 / per)
            s.reset()
        synced = []
        for _ in range(max(self.args.repeats, 20) * 5):  # driver-loop: over timed repeats
            self.sync()
            t0 = time.perf_counter_ns()
            s.add(a, b)
            self.sync()
            synced.append((time.perf_counter_ns() - t0) / 1e3)
            s.reset()
        s.free(a)
        s.free(b)
        s.reset()
        return {"adds": batches * per, "per_batch": per,
                "per_launch_us": float(np.median(per_launch)),
                "per_launch_p10_us": float(np.percentile(per_launch, 10)),
                "synced_add_us": float(np.median(synced)),
                "synced_add_p10_us": float(np.percentile(synced, 10))}

    # ------------------------------------------------------------ backward
    def backward_cases(self):
        """A 2-layer model x (r, c) -> linear(c, 256, relu) -> linear(256, 1) -> BCE.
        `forward` times the recorded forward; `backward` times Session.backward
        after an untimed forward."""
        out = []
        hid = BACKWARD_HIDDEN
        for r in ROWS:
            for c in BACKWARD_INPUTS:
                if max(r * c, r * hid) > self.args.max_elems:
                    continue
                sh = f"{r}x{c}-{hid}-1"
                fl = 2 * r * (c * hid + hid)
                out.append(("forward", sh, r, c, 1 * fl))
                out.append(("backward", sh, r, c, 2 * fl))
        return out

    def _rec_model(self, r, c):
        ops = self.ops
        rs = ops.Session(record=True, numeric_mode=self.args.numeric_mode)
        hid = BACKWARD_HIDDEN
        m = {"x": rs.tensor(self._uniform((r, c))),
             "y": rs.tensor(self.rng.integers(0, 2, size=(r, 1)).astype(np.float32)),
             "w1": rs.tensor(self._uniform((c, hid), -0.1, 0.1), requires_grad=True),
             "b1": rs.tensor(np.zeros((1, hid), dtype=np.float32), requires_grad=True),
             "w2": rs.tensor(self._uniform((hid, 1), -0.1, 0.1), requires_grad=True),
             "b2": rs.tensor(np.zeros((1, 1), dtype=np.float32), requires_grad=True),
             "sentinel": rs.tensor(np.zeros((1, 1), dtype=np.float32))}
        return rs, m

    @staticmethod
    def _rec_forward(rs, m):
        z = rs.linear(m["x"], m["w1"], m["b1"], activation="relu")
        z = rs.linear(z, m["w2"], m["b2"])
        return rs.bce_with_logits(z, m["y"])

    def run_backward(self, phase, sh, r, c, flops, launches_only=False):
        rec = {"op": "backward_2layer", "variant": phase, "shape": sh,
               "key": f"backward_2layer|{phase}|{sh}", "bytes": None, "flops": flops, "elems": r * c}
        rs = None
        try:
            rs, m = self._rec_model(r, c)
            sync = lambda: rs.scalar(m["sentinel"])  # noqa: E731

            def one(timed):
                loss = self._rec_forward(rs, m)
                if phase == "backward":
                    sync()
                    if launches_only:
                        rs._b.ops_profile_reset()
                    t0 = time.perf_counter_ns()
                    rs.backward(loss)
                else:
                    t0 = None
                sync()
                return (time.perf_counter_ns() - t0) / 1e3 if t0 is not None else None

            if launches_only:
                if not self.profile_ok:
                    rec["launches"] = None
                else:
                    one(False)  # warm-up
                    rs.reset()
                    sync()
                    if phase == "forward":
                        rs._b.ops_profile_reset()
                        self._rec_forward(rs, m)
                    else:
                        one(False)
                    sync()
                    text = str(rs._b.ops_profile_report())
                    mm = re.search(r"AG_PROFILE level=\d+ launches=(\d+)", text)
                    rec["launches"] = int(mm.group(1)) if mm else None
                return rec
            one(False)  # warm-up
            rs.reset()
            t = []
            for _ in range(self.args.repeats):  # driver-loop: over timed repeats
                if phase == "forward":
                    sync()
                    t0 = time.perf_counter_ns()
                    self._rec_forward(rs, m)
                    sync()
                    t.append((time.perf_counter_ns() - t0) / 1e3)
                else:
                    t.append(one(True))
                rs.reset()
            t.sort()
            rec["median_us"] = float(np.median(t))
            rec["p10_us"] = float(np.percentile(t, 10))
            rec["min_us"] = float(t[0])
            rec["repeats"] = len(t)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            if rs is not None:
                try:
                    rs.close()
                except Exception:
                    pass
        return rec

    # ------------------------------------------------------------ the sweep
    def all_case_groups(self):
        """The (rows, cols) grid within --max-elems; each shape's uploads
        live only while its cases run."""
        for r in ROWS:
            for c in COLS:
                if r * c > self.args.max_elems:
                    continue
                yield (r, c)

    def sweep(self, launches_only=False, only=None):
        results = []
        log = sys.stderr

        def keep(case_op):
            return only is None or case_op in only

        for r, c in self.all_case_groups():
            h = self.upload_shape(r, c)
            w = {}
            if keep("linear"):
                for o in LINEAR_OUT:  # driver-loop: over linear output widths
                    if r * o <= self.args.max_elems:
                        w[(c, o)] = self.s.tensor(self._uniform((c, o), -0.1, 0.1))
                        w[("b", o)] = self.s.tensor(self._uniform((1, o), -0.1, 0.1))
            cases = self.shape_cases(r, c, h) + (self.linear_cases(r, c, h, w) if w else [])
            for case in cases:  # driver-loop: over benchmark cases
                if not keep(case.op):
                    continue
                rec = self.run_case(case, launches_only)
                results.append(rec)
                print(f"  {case.key} {rec.get('median_us', rec.get('launches', rec.get('error')))}", file=log)
            for v in w.values():  # driver-loop: over linear weights
                self.s.free(v)
            self.free_shape(h)
        for case in self.matmul_cases():  # driver-loop: over benchmark cases
            if keep("matmul"):
                rec = self.run_case(case, launches_only)
                results.append(rec)
                print(f"  {case.key} {rec.get('median_us', rec.get('launches', rec.get('error')))}", file=log)
        if keep("backward_2layer"):
            for phase, sh, r, c, fl in self.backward_cases():  # driver-loop: over benchmark cases
                rec = self.run_backward(phase, sh, r, c, fl, launches_only)
                results.append(rec)
                print(f"  {rec['key']} {rec.get('median_us', rec.get('launches', rec.get('error')))}", file=log)
        return results


def _device_info():
    info = {"host": platform.node(), "python": platform.python_version()}
    try:
        import mojolearn
        info["mojolearn"] = getattr(mojolearn, "__version__", None)
    except Exception:
        pass
    for cmd in (["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], ["rocm-smi", "--showproductname"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                info["gpu"] = " ".join(out.stdout.split())[:200]
                break
        except Exception:
            pass
    return info


def _table(doc):
    lines = []
    fl = doc.get("floor") or {}
    lines.append(f"vendor={doc['vendor']} repeats={doc['repeats']} sync_us={doc['sync_us']:.1f} "
                 f"launch_floor_us={fl.get('per_launch_us', float('nan')):.2f} "
                 f"synced_add_us={fl.get('synced_add_us', float('nan')):.1f}")
    lines.append(f"{'op':<16} {'variant':<14} {'shape':<22} {'median_us':>11} {'p10_us':>10} "
                 f"{'net_us':>10} {'launches':>8} {'GB/s':>8}")
    for r in doc["cases"]:
        if "error" in r:
            lines.append(f"{r['op']:<16} {r['variant']:<14} {r['shape']:<22} ERROR {r['error'][:80]}")
            continue
        med = r.get("median_us")
        net = r.get("net_us")
        gbs = (r["bytes"] / (net * 1e3)) if (r.get("bytes") and net and net > 0) else None
        lines.append(f"{r['op']:<16} {r['variant']:<14} {r['shape']:<22} "
                     f"{med:>11.1f} {r['p10_us']:>10.1f} {net:>10.1f} "
                     f"{'-' if r.get('launches') is None else r['launches']:>8} "
                     f"{'-' if gbs is None else f'{gbs:.1f}':>8}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="JSON output path (opbench-<vendor>.json)")
    ap.add_argument("--table", default=None, help="text table path (default: <out>.txt)")
    ap.add_argument("--vendor", default=os.environ.get("MOJOLEARN_VENDOR", "auto"))
    ap.add_argument("--numeric-mode", default="identical")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-elems", type=int, default=1 << 28,
                    help="largest tensor (elements) a case may hold; 2**28 float32 = 1 GiB")
    ap.add_argument("--floor-batches", type=int, default=100)
    ap.add_argument("--floor-per-batch", type=int, default=100)
    ap.add_argument("--only", default=None, help="comma-separated op names to run (default all)")
    ap.add_argument("--no-launches", action="store_true", help="skip the MOJOLEARN_AG_PROFILE=1 pass")
    ap.add_argument("--launches-only", action="store_true",
                    help="internal: the profile pass (run with MOJOLEARN_AG_PROFILE=1), writes --out")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    if args.launches_only:
        if os.environ.get("MOJOLEARN_AG_PROFILE") != "1":
            sys.exit("--launches-only needs MOJOLEARN_AG_PROFILE=1 in the environment")
        b = Bench(args)
        doc = {"profile_ok": b.profile_ok, "cases": b.sweep(launches_only=True, only=only) if b.profile_ok else []}
        b.s.close()
        with open(args.out, "w") as f:
            json.dump(doc, f)
        return

    if os.environ.get("MOJOLEARN_AG_PROFILE"):
        sys.exit("unset MOJOLEARN_AG_PROFILE for the timing pass (the launch pass sets it itself)")
    b = Bench(args)
    t_start = time.time()
    sync_med, sync_p10 = b.sync_floor()
    floor = b.launch_floor()
    print(f"sync_us={sync_med:.1f} launch_floor_us={floor['per_launch_us']:.2f}", file=sys.stderr)
    cases = b.sweep(only=only)
    b.s.close()
    for r in cases:  # driver-loop: over benchmark cases
        if "median_us" in r:
            r["net_us"] = max(r["median_us"] - sync_med, 0.0)
            r["net_p10_us"] = max(r["p10_us"] - sync_med, 0.0)
    vendor = args.vendor
    info = _device_info()
    if vendor == "auto":
        g = info.get("gpu", "").lower()
        vendor = "nvidia" if ("nvidia" in g or "l40" in g or "rtx" in g) else ("amd" if ("mi3" in g or "amd" in g or "instinct" in g) else "unknown")
    doc = {"tool": "ag_opbench", "vendor": vendor, "numeric_mode": args.numeric_mode,
           "repeats": args.repeats, "seed": args.seed, "max_elems": args.max_elems,
           "rows": list(ROWS), "cols": list(COLS), "device": info,
           "sync_us": sync_med, "sync_p10_us": sync_p10, "floor": floor,
           "protocol": "sync; t0; op; sync; t1 per repeat, one warm-up, reset between repeats; "
                       "sync = one-value read back; net_us = median_us - sync_us",
           "wall_s": None, "launches_source": None, "owed": [], "cases": cases}
    for r in cases:
        r.setdefault("launches", None)
    if not args.no_launches:
        lpath = args.out + ".launches.json"
        env = dict(os.environ, MOJOLEARN_AG_PROFILE="1")
        cmd = [sys.executable, os.path.abspath(__file__), "--launches-only", "--out", lpath,
               "--numeric-mode", args.numeric_mode, "--seed", str(args.seed),
               "--max-elems", str(args.max_elems)]
        if args.only:
            cmd += ["--only", args.only]
        rc = subprocess.run(cmd, env=env).returncode
        if rc == 0 and os.path.exists(lpath):
            with open(lpath) as f:
                ld = json.load(f)
            if ld.get("profile_ok"):
                by = {r["key"]: r.get("launches") for r in ld["cases"]}
                for r in cases:
                    r["launches"] = by.get(r["key"])
                doc["launches_source"] = "MOJOLEARN_AG_PROFILE=1 child pass, one call per case"
            else:
                doc["owed"].append(PROFILE_OWED)
        else:
            doc["owed"].append(f"launch pass failed rc={rc}")
    doc["wall_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    table = _table(doc)
    with open(args.table or (os.path.splitext(args.out)[0] + ".txt"), "w") as f:
        f.write(table)
    errs = sum(1 for r in cases if "error" in r)
    print(f"AG_OPBENCH vendor={vendor} cases={len(cases)} errors={errs} sync_us={sync_med:.1f} "
          f"launch_floor_us={floor['per_launch_us']:.2f} owed={len(doc['owed'])} out={args.out}")


if __name__ == "__main__":
    main()
