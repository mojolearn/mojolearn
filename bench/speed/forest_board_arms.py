#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Arm plumbing the benchmark board adds to bench/speed/forest_speed_arm.py:
OUR CPU ARM (`--ours-cpu`) and PER-ARM MEMORY (`--mem`). Both are off unless
asked for, so the driver's output is unchanged without them.

OUR CPU ARM, IN ITS OWN PROCESS
--------------------------------
The wheel's CPU switch is process-wide (MOJOLEARN_VENDOR=cpu before import,
tools/bench_board_probe.py), and this driver runs every arm in one process.
So `ours-cpu` is a PROXY arm here: a worker process (this file's `worker`
command, started under MOJOLEARN_VENDOR=cpu and MOJOLEARN_NUMERIC_MODE=
identical) loads the same dataset through the same loader with the same row
cap, builds the same `ours` estimator (forest_speed_arm.OUR_BUILDERS) and
refuses by name unless `mojolearn.vendor()` reads back `cpu` and the binding
reads back IDENTICAL. The proxy takes its turn in the driver's round-robin
like every arm: its `fit` sends one line and waits for the worker's fit, so
the conductor's clock covers the worker's make+fit+sync plus one pipe round
trip (tens of microseconds). Predictions, scores and the fitted shape come
back AFTER the clock through a file. The worker's own clock is printed beside
it (FSPEED-CPU-ROUND). In the inference phase the proxy's call is the same
public call ours makes (`predict_proba` column 1, `predict`, or
`score_samples`), on the same batch rows, and the vector comes back after the
clock.

PER-ARM MEMORY
--------------
Every arm here shares one process, so memory is read around each arm's fit
through the runner's fit_context (entry and exit are outside the timer):
the host peak is resettable (Linux VmHWM after clear_refs, macOS interval
max phys_footprint) and so is per arm and round; a process-level GPU figure
(nvidia-smi / rocm-smi per pid) is the WHOLE PROCESS, every arm's pools
included, and its method says so. The ours-cpu arm's memory is its worker's.
Lines: FSPEED-MEM lane arm round host_mb gpu_mb children_mb host_method
gpu_method.
"""
import json
import os
import select
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TOOLS = os.path.join(_ROOT, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import bench_board_probe as probe      # noqa: E402

CPU_ARM = probe.OURS_CPU_ARM


def _one_line(text, n=400):
    return " ".join(str(text).split())[:n]


# ---------------------------------------------------------------------------
# Memory around each arm's fit
# ---------------------------------------------------------------------------

class TreeMem(object):
    """fit_context for speed_gbdt_arm.run: one FSPEED-MEM line per arm and
    round (round 0 is the warm-up)."""

    def __init__(self, lane, proxies=()):
        self.lane = lane
        self.proxies = {p.name: p for p in proxies}
        self.probes = {}

    def _probe(self, arm_name):
        if arm_name not in self.probes:
            dev = "cpu" if (arm_name.endswith("-cpu") or "-cpu-" in arm_name) else "gpu"
            self.probes[arm_name] = probe.MemProbe(dev, shared=True)
        return self.probes[arm_name]

    def emit(self, arm_name, r, m):
        def f(v):
            return "-" if v is None else ("%.1f" % v)
        print("FSPEED-MEM lane=%s arm=%s round=%d host_mb=%s gpu_mb=%s children_mb=%s "
              "host_method=%s gpu_method=%s"
              % (self.lane, arm_name, r, f(m.get("host_mb")), f(m.get("gpu_mb")),
                 f(m.get("children_mb")), _one_line(m.get("host_method") or "-").replace("=", ":"),
                 _one_line(m.get("gpu_method") or "-").replace("=", ":")), flush=True)

    def context(self, arm_name, r):
        mem = self

        class _Ctx(object):
            def __enter__(self):
                if arm_name not in mem.proxies:
                    mem._probe(arm_name).start()
                return self

            def __exit__(self, exc_type, exc, tb):
                if exc_type is not None:
                    return False
                if arm_name in mem.proxies:
                    m = mem.proxies[arm_name].last_mem or {}
                else:
                    m = mem._probe(arm_name).stop()
                mem.emit(arm_name, r, m)
                return False
        return _Ctx()


# ---------------------------------------------------------------------------
# The proxy (conductor side)
# ---------------------------------------------------------------------------

class _Model(object):
    """What the runner holds for one ours-cpu fit: a handle on the worker's
    model. Scores, shape and predictions are asked of the worker."""

    def __init__(self, proxy):
        self.proxy = proxy
        self.fit_no = None

    def board_infer_spec(self, lane, task, data):
        """(call, post, text) for forest_speed_arm.run_inference."""
        proxy = self

        def call(x):
            batch = "test" if (x is data.X_test or x is getattr(data, "_ours_Xtest", None)) \
                else "large"
            proxy.proxy.request("predict %s %d" % (batch, int(x.shape[0])))
            return None

        def post(_raw):
            import numpy as np
            rep = proxy.proxy.request("fetch")
            return np.load(rep["path"])
        return call, post, self.proxy.infer_text + " (ours-cpu worker process, MOJOLEARN_VENDOR cpu; " \
                                                   "the vector comes back after the clock)"


class CpuProxy(object):
    def __init__(self, lane, dataset, rows, infer_large_rows, work):
        self.name = CPU_ARM
        self.library = "mojolearn-cpu"
        self.lane = lane
        self.last_mem = None
        self.infer_text = "mojolearn (the ours call)"
        self.work = work
        self.fit_shape = None
        env = probe.ours_cpu_env(dict(os.environ))
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "worker", "--lane", lane,
               "--dataset", dataset, "--work", work, "--infer-large-rows", str(infer_large_rows)]
        if rows:
            cmd += ["--rows", str(int(rows))]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     env=env, start_new_session=True)
        self.buf = b""

    # -- protocol -----------------------------------------------------------
    def _read(self, seconds):
        deadline = time.monotonic() + seconds
        fd = self.proc.stdout.fileno()
        while True:
            while b"\n" not in self.buf:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                ready, _, _ = select.select([fd], [], [], min(left, 5.0))
                if not ready:
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    return None
                self.buf += chunk
            line, self.buf = self.buf.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if text.startswith("{"):
                try:
                    return json.loads(text)
                except ValueError:
                    pass

    def request(self, text, seconds=None):
        from speed_gbdt_arm import per_arm_budget_s
        self.proc.stdin.write((text + "\n").encode())
        self.proc.stdin.flush()
        msg = self._read(seconds or per_arm_budget_s())
        if msg is None:
            self.kill()
            raise RuntimeError("ours-cpu worker did not answer %r (timeout or exit)" % text)
        if msg.get("event") == "error":
            raise RuntimeError(msg.get("error") or "ours-cpu worker error")
        return msg

    def ready(self, seconds):
        msg = self._read(seconds)
        if msg is None or msg.get("event") != "ready":
            self.kill()
            raise RuntimeError(_one_line((msg or {}).get("error") or "the ours-cpu worker did not "
                                         "come up (see the log above)", 600))
        b = msg.get("binding") or {}
        print("BENCH_BINDING arm=%s requested=%s resolved=%s compiled=%s vendor=%s path=%s"
              % (self.name, b.get("requested"), b.get("resolved"), b.get("compiled"),
                 b.get("vendor"), b.get("path")), flush=True)
        print("FSPEED-CPU-ARM lane=%s arm=%s switch=%s vendor_how=%s"
              % (self.lane, self.name, "MOJOLEARN_VENDOR:cpu",
                 _one_line(msg.get("vendor_how") or "-").replace("=", ":")), flush=True)
        self.infer_text = msg.get("infer_text") or self.infer_text
        return msg

    def kill(self):
        try:
            os.killpg(self.proc.pid, 9)
        except OSError:
            pass

    def close(self):
        try:
            self.proc.stdin.write(b"quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.kill()

    # -- the Arm interface speed_gbdt_arm.run uses --------------------------
    def make(self):
        return _Model(self)

    def fit(self, model, data):
        rep = self.request("fit")
        self.last_mem = rep.get("mem")
        model.fit_no = rep.get("n")
        print("FSPEED-CPU-ROUND lane=%s arm=%s fit=%s worker_ms=%.3f"
              % (self.lane, self.name, rep.get("n"), rep.get("ms", float("nan"))), flush=True)
        return model

    def sync(self):
        return None

    def score(self, model, data):
        import numpy as np
        rep = self.request("score")
        out = []
        for metric, value, path in rep.get("triples") or []:
            out.append((metric, value, np.load(path) if path else None))
        return out

    def shape(self, model):
        return self.request("shape").get("shape") or {}


def build_cpu_arm(lane, dataset, rows, infer_large_rows, emit_refused, ready_seconds=None):
    """[proxy] or [] after a by-name refusal."""
    from speed_gbdt_arm import MODEL_SHAPE_READERS, per_arm_budget_s
    work = tempfile.mkdtemp(prefix="ours-cpu-")
    try:
        proxy = CpuProxy(lane, dataset, rows, infer_large_rows, work)
        proxy.ready(ready_seconds or per_arm_budget_s())
    except Exception as exc:  # noqa: BLE001
        emit_refused(lane, CPU_ARM, "%s: %s" % (exc.__class__.__name__, _one_line(exc, 600)))
        return []
    # the fitted-shape reader for this arm asks the worker (FIT-EQUIVALENCE)
    MODEL_SHAPE_READERS.setdefault("mojolearn-cpu", lambda m: m.proxy.shape(m))
    return [proxy]


def infer_agree_pairs(last):
    """(ours, other) pairs the inference phase compares bit for bit."""
    return [("ours", o) for o in ("ours-ab", CPU_ARM) if "ours" in last and o in last]


# ---------------------------------------------------------------------------
# The worker (MOJOLEARN_VENDOR=cpu)
# ---------------------------------------------------------------------------

def _verify(arm, model, ml):
    """IDENTICAL and cpu, read back from the estimator and its binding."""
    info = probe.ours_cpu_check(ml)
    if not info:
        raise RuntimeError("REFUSED: the ours-cpu worker was started without its CPU switch")
    codes = {0: "fast", 1: "identical", 2: "deterministic"}
    resolved = model.numeric_mode_used()
    vendor = model.vendor_used()
    name = model._BINDING
    if type(model).__name__ == "IsolationForest":
        name = "_mojolearn_svm"     # forest_speed_arm.verify_our_arm, found 2026-09-11
    binding = model._bind(name)
    getter = getattr(binding, name.removeprefix("_mojolearn_") + "_numeric_mode", None)
    compiled = codes.get(int(getter()), "unknown") if getter is not None else "unknown"
    if resolved != "identical" or compiled != "identical" or vendor != "cpu":
        raise RuntimeError("REFUSED: ours-cpu mode/vendor readback: resolved=%s compiled=%s vendor=%s"
                           % (resolved, compiled, vendor))
    path = None
    try:
        # a host binding proxy raises ImportError (not AttributeError) for a
        # name its binding lacks, so getattr's default never applies here
        from mojolearn import _backend
        path = _backend.host_module_path(_backend.host_surface.routed_modules().get(name, name))
    except Exception:  # noqa: BLE001
        pass
    return dict(requested="identical", resolved=resolved, compiled=compiled, vendor=vendor,
                path=path), info


def _plain(obj):
    """JSON-safe copy: numpy scalars become Python numbers (never strings)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "shape", None) == ():
        return item()
    return obj


def worker(argv):
    import argparse
    p = argparse.ArgumentParser(prog="forest_board_arms worker")
    p.add_argument("--lane", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--work", required=True)
    p.add_argument("--infer-large-rows", type=int, default=1_000_000)
    args = p.parse_args(argv)
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        proto.flush()

    try:
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        import numpy as np
        import forest_speed_arm as fsa
        spec = fsa.spec
        lane = args.lane
        size = spec.size_tag()
        data = spec.load_with_fallback(args.dataset, size, args.rows)
        cfg = spec.lane_config(lane, size)
        spec.prepare_cuml_labels(data)
        spec.prepare_anomaly_labels(lane, data)
        fsa.prepare_our_inputs(data)
        arm = fsa.OUR_BUILDERS[lane](lane, cfg, data)
        arm.name = CPU_ARM
        import mojolearn as ml
        model = arm.make()
        binding, info = _verify(arm, model, ml)
        try:
            _c, _p, text = fsa.infer_spec("ours", lane, data.task, model,
                                          n_features=data.X_train.shape[1])
        except Exception as exc:  # noqa: BLE001  (a task the inference phase does not time)
            text = "no inference path: %s" % _one_line(exc, 200)
    except BaseException as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        say({"event": "error", "error": "%s: %s" % (exc.__class__.__name__, _one_line(exc, 600))})
        return 1
    say({"event": "ready", "binding": binding, "vendor_how": info.get("vendor_how"),
         "infer_text": text})
    mem = probe.MemProbe("cpu")
    model, n, raw, post = None, 0, None, None
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        try:
            if parts[0] == "fit":
                mem.start()
                t0 = time.perf_counter()
                m = arm.make()
                arm.fit(m, data)
                arm.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                mm = mem.stop()
                model, n = m, n + 1
                say({"event": "fit", "n": n, "ms": ms, "mem": mm})
            elif parts[0] == "score":
                triples = []
                for i, (metric, value, vec) in enumerate(arm.score(model, data)):
                    path = None
                    if vec is not None:
                        path = os.path.join(args.work, "score-%d.npy" % i)
                        np.save(path, np.asarray(vec))
                    triples.append([metric, value, path])
                say({"event": "score", "triples": triples})
            elif parts[0] == "shape":
                say({"event": "shape", "shape": _plain(spec.model_shape(arm, model))})
            elif parts[0] == "predict":
                batch, rows = parts[1], int(parts[2])
                call, post, _t = fsa.infer_spec("ours", args.lane, data.task, model,
                                                n_features=data.X_train.shape[1])
                x = data._ours_Xtest if batch == "test" else \
                    np.ascontiguousarray(data.X_train[:rows], dtype=np.float32)
                t0 = time.perf_counter()
                raw = call(x)
                ms = (time.perf_counter() - t0) * 1000.0
                say({"event": "predict", "ms": ms})
            elif parts[0] == "fetch":
                path = os.path.join(args.work, "predict.npy")
                np.save(path, post(raw))
                say({"event": "fetch", "path": path})
            elif parts[0] == "quit":
                say({"event": "bye"})
                return 0
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            say({"event": "error", "error": "%s: %s" % (exc.__class__.__name__, _one_line(exc, 600))})
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.exit(worker(sys.argv[2:]))
    raise SystemExit("usage: forest_board_arms.py worker --lane L --dataset D --work DIR [--rows N]")
