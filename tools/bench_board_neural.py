#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neural family of tools/bench_board.py: THE WHEEL'S PUBLIC PYTHON API
against torch on the same GPU, interleaved round by round, quality beside
every time.

    python3 tools/bench_board_neural.py race --lane lm-train-step --shape full \\
        --arms ours,torch-eager-fp32 --rounds 5 --out DIR --work DIR \\
        --ours-python PY --theirs-python PY

It speaks tools/classical_two_datasets.py's protocol (it reuses its `Worker`):
one persistent worker process per arm, a warm-up round then `--rounds` timed
rounds, the arm order rotated every round, and one race JSON whose shape
tools/bench_board.py's `classical_cells` already reads.

THE LANES (each is a public mojolearn entry point, installed from the wheel)
----------------------------------------------------------------------------
  lm-train-step  `mojolearn.LanguageModelTrainer(..., resident=True,
                 step_result='lean').train_step(ids)`: one complete byte-LM
                 training step (forward, mean next-byte cross entropy,
                 backward, AdamW lr 1e-3, betas 0.9/0.999, eps 1e-8, weight
                 decay 0.01). The trainer's defaults.
  lm-forward     `LanguageModelTrainer.logits(ids)` on a resident trainer:
                 the IDENTICAL device forward, float32 logits [B, L, V] back
                 on the host.
  gemm           `mojolearn.linalg.matmul(a, b)`: fp32 C = A @ B, host
                 arrays in, a host array out.

THE OPPONENT: torch, `torch-eager-fp32`
---------------------------------------
torch eager, float32, TF32 OFF (tools/torch_lm_step_opponent.py
`set_precision(torch, False)`, read back). That file names eager_fp32 THE ROW,
the opponent's fast setting AT OUR PRECISION (bench/OPPONENT_REFERENCE.md);
its compile, TF32 and bf16 columns are extras at another precision or labeled
nondeterministic and are not raced here. The device is the box's GPU: MPS on
Apple, CUDA on NVIDIA, ROCm (the torch.cuda API) on AMD. An arm with no GPU
refuses by name; it never falls back to the CPU.

The LM model is tools/torch_lm_step_opponent.py's `build_model`, OUR registry
shape for shape (embed, 9 tensors per block, untied lm_head; RMSNorm eps 1e-6,
RoPE theta 10000, causal SDPA at scale 1/sqrt(head_dim), SiLU MLP) and
`torch.optim.AdamW` with the same hyperparameters. On the torch.cuda API one
SDPA backend is pinned by its probe (`choose_sdpa_backend`, auto: efficient,
flash, math); on MPS torch's own dispatch runs and the record says so.

SAME INPUTS, BYTE FOR BYTE
--------------------------
The conductor writes ONE input file per race and every arm reads it:
  * LM lanes: initial parameters numpy default_rng(7).normal(0, .02) float32,
    +1 on every norm (the probe's recipe at the board's seed), and the batches
    of a byte stream, step k row b = stream[(k*B*L + b*L) % (n - L - 1) :
    + L + 1] (tools/lm_step_memory_probe.py's schedule). The stream is the
    installed mojolearn package's own .py sources, sorted by path and
    concatenated: text that ships in the wheel under test, never downloaded.
    Its sha256 is recorded.
  * gemm: A [m, k] and B [k, n] from default_rng(7).standard_normal, float32.

THE CLOCK (both sides)
----------------------
  lm-train-step  ids host -> device, the step, the loss back on the host,
                 synchronized. Parameters and AdamW state stay on the device
                 between steps on both sides (resident / an nn.Module).
                 Training continues across rounds: round r is step r + 1.
  lm-forward     ids host -> device, the forward, the logits back on the host.
  gemm           A and B host -> device, the product, C back on the host.
Every span is host in, host out, so no arm's clock excludes an upload that
another's includes.

QUALITY (the conductor, float64 NumPy, from each arm's saved outputs)
---------------------------------------------------------------------
  lm-train-step  loss_first_step, loss_last_step (the same init and the same
                 batches on every arm, so the values are comparable), and on
                 each opponent loss_last_abs_diff_vs_ours.
  lm-forward     mean_nll of the logits against the next bytes, and on each
                 opponent max_abs_diff_vs_ours (logits).
  gemm           max_rel_err_vs_fp64 (max |C - C64| / max |C64|, C64 the
                 float64 product of the same float32 inputs), and on each
                 opponent max_abs_diff_vs_ours.

SHAPES (`--shape`)
------------------
  full   LM: tools/torch_lm_step_opponent.py's `control` shape, batch 1,
         length 2048, d_model 384, 6 heads (6 KV), head_dim 64, intermediate
         1024, 8 layers, vocab 8192: 20,453,376 parameters, chosen to fit a
         16 GB Apple M4 beside torch (parameters plus both Adam moments are
         about 250 MB per side; not yet run at this size on the Mac). The
         GPT-3-small `target` shape (162 M parameters, vocab 50257) is left
         off the board. gemm: 4096 x 4096 x 4096.
  small  a plumbing smoke: LM batch 2, length 64, d_model 64, 4 heads (2 KV),
         head_dim 16, intermediate 128, 2 layers, vocab 256; gemm 256^3.

IDENTICAL ONLY. The neural surface builds `identical` only
(mojolearn._backend._IDENTICAL_ONLY); an `ours` worker refuses to start under
any other MOJOLEARN_NUMERIC_MODE.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import shlex
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SEED = 7
LANES = ("lm-train-step", "lm-forward", "gemm")
ARMS = ("ours", "torch-eager-fp32")
#: The data each lane reads (the board's `dataset` column).
DATA_OF = {"lm-train-step": "bytes", "lm-forward": "bytes", "gemm": "gaussian"}
#: [batch, length, d_model, n_heads, n_kv, head_dim, intermediate, n_layers, vocab]
LM_SHAPES = {
    "full": [1, 2048, 384, 6, 6, 64, 1024, 8, 8192],   # torch_lm_step_opponent.SHAPES['control']
    "small": [2, 64, 64, 4, 2, 16, 128, 2, 256],
}
LM_FIELDS = ("batch", "length", "d_model", "n_heads", "n_kv", "head_dim",
             "intermediate", "n_layers", "vocab_size")
GEMM_SHAPES = {"full": (4096, 4096, 4096), "small": (256, 256, 256)}   # m, n, k
TORCH_MODE = "torch eager, float32, TF32 off (tools/torch_lm_step_opponent.py eager_fp32, THE ROW)"


def _load(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location("bbn_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shape_record(lane, shape):
    """What the board shows as this race's shape, with the dimensions named."""
    if lane == "gemm":
        m, n, k = GEMM_SHAPES[shape]
        return {"name": shape, "m": m, "n": n, "k": k, "label": "%dx%dx%d" % (m, n, k)}
    dims = LM_SHAPES[shape]
    rec = dict(zip(LM_FIELDS, dims))
    rec.update(name=shape, label="B%d L%d DM%d H%d KV%d HD%d FF%d layers%d V%d" % tuple(dims))
    return rec


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Inputs (the conductor, once per race)
# ---------------------------------------------------------------------------

def byte_stream():
    """The installed mojolearn package's .py sources, sorted, concatenated.
    find_spec locates the package WITHOUT importing it (no binding loads in
    the conductor)."""
    spec = importlib.util.find_spec("mojolearn")
    roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    if not roots:
        raise SystemExit("bench_board_neural: mojolearn is not installed in %s" % sys.executable)
    root = roots[0]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    raw = b"".join(open(f, "rb").read() for f in files)
    return raw, {"source": "installed mojolearn package .py sources, sorted by path, concatenated",
                 "package_dir": root, "files": len(files), "bytes": len(raw), "sha256": _sha(raw)}


def lm_registry(dims):
    twin = _load("torch_lm_step_opponent")
    return twin.registry(dims)


def make_inputs(lane, shape, steps, path):
    """Write the race's single input file (.npz) and return its record."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    if lane == "gemm":
        m, n, k = GEMM_SHAPES[shape]
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        np.savez(path, a=a, b=b)
        return {"lane": lane, "shape": shape_record(lane, shape),
                "inputs": "A [m,k], B [k,n] = default_rng(%d).standard_normal float32" % SEED,
                "a_sha256": _sha(a.tobytes()), "b_sha256": _sha(b.tobytes())}
    dims = LM_SHAPES[shape]
    bsz, length, vocab = dims[0], dims[1], dims[8]
    shapes = lm_registry(dims)
    n_total = sum(int(np.prod(s)) for _, s in shapes)
    flat = rng.normal(0, .02, n_total).astype(np.float32)
    off = 0
    for name, s in shapes:
        size = int(np.prod(s))
        if "norm" in name:
            flat[off:off + size] += np.float32(1)
        off += size
    raw, stream = byte_stream()
    if len(raw) < length + 2:
        raise SystemExit("bench_board_neural: byte stream shorter than one row")
    if vocab < 256:
        raise SystemExit("bench_board_neural: byte ids need vocab >= 256")
    modulus = len(raw) - length - 1
    buf = np.frombuffer(raw, dtype=np.uint8)
    batches = np.empty((steps, bsz, length + 1), dtype=np.int32)
    for k in range(steps):
        for b in range(bsz):
            start = (k * bsz * length + b * length) % modulus
            batches[k, b] = buf[start:start + length + 1]
    np.savez(path, init=flat, batches=batches)
    return {"lane": lane, "shape": shape_record(lane, shape), "parameters": n_total,
            "init": "default_rng(%d).normal(0, .02) float32, +1 on norms" % SEED,
            "init_sha256": _sha(flat.tobytes()), "stream": stream,
            "batches_sha256": _sha(batches.tobytes()), "steps": steps,
            "schedule": "step k row b: stream[(k*B*L + b*L) % (n - L - 1) : +L+1]; "
                        "inputs [:, :L], targets [:, 1:]"}


# ---------------------------------------------------------------------------
# Workers (one process per arm)
# ---------------------------------------------------------------------------

def _ours_module():
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower()
    if want != "identical":
        raise RuntimeError("REFUSED: the neural surface is IDENTICAL only; this worker was started "
                           "under MOJOLEARN_NUMERIC_MODE=%s" % want)
    import mojolearn
    return mojolearn


def _ours_info(ml, module_path, mode_used):
    info = {"library": "mojolearn", "version": getattr(ml, "__version__", "unknown"),
            "numeric_mode_env": os.environ.get("MOJOLEARN_NUMERIC_MODE"),
            "numeric_mode_used": mode_used, "device": "gpu",
            "module_path": module_path, "pre_clock_fit": False,
            "input_home": "host"}
    try:
        info["vendor_used"] = ml.vendor()
    except Exception as exc:  # noqa: BLE001
        info["vendor_used"] = "unavailable (%r)" % (exc,)
    if mode_used != "identical":
        raise RuntimeError("ours is not IDENTICAL: read back %r" % (mode_used,))
    return info


class OursLM:
    """lm-train-step and lm-forward through LanguageModelTrainer."""

    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        dims = LM_SHAPES[shape]
        cfg = ml.LanguageModelConfig(*dims)
        twin = lm_registry(dims)
        ours = [(e["name"], tuple(e["shape"])) for e in ml.LanguageModelTrainer.parameter_registry(cfg)]
        if ours != [(n, tuple(s)) for n, s in twin]:
            raise RuntimeError("REFUSED: our parameter registry differs from the torch twin's")
        self.lane = lane
        self.batches = data["batches"]
        self.trainer = ml.LanguageModelTrainer(
            np.ascontiguousarray(data["init"]), shape=cfg, resident=True, step_result="lean",
            data_schedule={"fixture": "tools/bench_board_neural.py", "seed": SEED,
                           "batches": "installed mojolearn .py sources, board schedule"})
        meta = self.trainer.run_metadata()
        mode = "identical" if int(meta.get("native_numeric_mode", -1)) == 1 else \
            "native_numeric_mode=%s" % meta.get("native_numeric_mode")
        self.info = _ours_info(ml, meta.get("binding_file"), mode)
        self.info.update(profile=meta.get("native_profile"), native_vendor=meta.get("native_vendor"),
                         call=("LanguageModelTrainer(resident=True, step_result='lean').train_step"
                               if lane == "lm-train-step" else "LanguageModelTrainer(resident=True).logits"),
                         config=json.dumps(meta.get("config"), sort_keys=True))
        self.k = 0
        self.losses = []
        self.out = None
        self.ids = np.ascontiguousarray(self.batches[0][:, :-1])

    def call(self):
        if self.lane == "lm-train-step":
            res = self.trainer.train_step(self.np.ascontiguousarray(self.batches[self.k]))
            self.losses.append(float(res["loss"]))
            self.k += 1
        else:
            self.out = self.trainer.logits(self.ids)

    def sync(self):
        pass        # the binding synchronizes before it publishes (lm_step_memory_probe.py)

    def outputs(self):
        np = self.np
        if self.lane == "lm-train-step":
            return {"losses": np.array(self.losses, dtype=np.float64)}
        return {"logits": np.asarray(self.out)}

    def digest(self):
        if self.lane == "lm-train-step":
            return None                     # every step is a new state; nothing to repeat
        return _sha(self.np.ascontiguousarray(self.np.asarray(self.out)).data)[:16]


class OursGEMM:
    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        ml = _ours_module()
        import mojolearn.linalg as linalg
        self.linalg = linalg
        self.a, self.b = np.ascontiguousarray(data["a"]), np.ascontiguousarray(data["b"])
        self.info = _ours_info(ml, getattr(linalg, "__file__", None), linalg.numeric_mode())
        self.info.update(profile=linalg.PROFILE, call="mojolearn.linalg.matmul(a, b)")
        self.out = None

    def call(self):
        self.out = self.linalg.matmul(self.a, self.b)

    def sync(self):
        pass        # matmul returns a filled host array

    def outputs(self):
        return {"c": self.np.asarray(self.out)}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self.np.asarray(self.out)).data)[:16]


def _torch_device():
    import torch
    if torch.cuda.is_available():
        return torch, torch.device("cuda"), "cuda", torch.cuda.get_device_name(0), torch.cuda.synchronize
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch, torch.device("mps"), "mps", "Apple MPS", torch.mps.synchronize
    raise RuntimeError("REFUSED: torch sees no CUDA, ROCm or MPS device; no GPU arm on this box "
                       "(never a CPU fallback)")


def _torch_info(torch, kind, name, precision):
    return {"library": "torch", "version": torch.__version__,
            "torch_version_cuda": torch.version.cuda,
            "torch_version_hip": getattr(torch.version, "hip", None),
            "torch_backend": kind, "device": "gpu", "device_name": name,
            "mode": TORCH_MODE, "precision_readback": precision,
            "pre_clock_fit": False, "input_home": "host"}


class TorchLM:
    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        torch, dev, kind, name, sync = _torch_device()
        self.torch, self.dev, self._sync = torch, dev, sync
        twin = _load("torch_lm_step_opponent")
        precision = twin.set_precision(torch, False)
        dims = LM_SHAPES[shape]
        shapes = twin.registry(dims)
        flat = torch.from_numpy(np.ascontiguousarray(data["init"]))
        self.model = twin.build_model(torch, dims, shapes, flat, dev)
        n = sum(p.numel() for p in self.model.parameters())
        if n != flat.numel():
            raise RuntimeError("REFUSED: torch model has %d parameters, the input %d" % (n, flat.numel()))
        self.info = _torch_info(torch, kind, name, precision)
        if kind == "cuda":
            self.info["sdpa"] = twin.choose_sdpa_backend(torch, "auto", dev, dims, sync)
        else:
            self.info["sdpa"] = {"backend": "torch_default",
                                 "selection": "MPS: torch's own dispatch (no backend switches exist)"}
        self.lane = lane
        self.batches = data["batches"]
        if lane == "lm-train-step":
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=twin.LR, betas=twin.BETAS,
                                         eps=twin.ADAM_EPS, weight_decay=twin.WEIGHT_DECAY)
            self.info["optimizer"] = "torch.optim.AdamW lr %g betas %s eps %g wd %g (torch's default impl)" % (
                twin.LR, twin.BETAS, twin.ADAM_EPS, twin.WEIGHT_DECAY)
            self.info["call"] = "zero_grad; forward + mean CE; backward; AdamW step; loss.item()"
        else:
            self.model.eval()
            self.info["call"] = "no_grad forward to logits; logits.cpu()"
        self.ids = torch.from_numpy(np.ascontiguousarray(self.batches[0][:, :-1]).astype(np.int64))
        self.k = 0
        self.losses = []
        self.out = None
        sync()

    def call(self):
        torch = self.torch
        if self.lane == "lm-train-step":
            host = torch.from_numpy(self.batches[self.k].astype(self.np.int64))
            ids = host.to(self.dev)
            self.opt.zero_grad(set_to_none=True)
            loss = self.model(ids)
            loss.backward()
            self.opt.step()
            self.losses.append(float(loss.item()))
            self.k += 1
        else:
            with torch.no_grad():
                logits = self.model(self.ids.to(self.dev), return_logits=True)
                self.out = logits.cpu().numpy()

    def sync(self):
        self._sync()

    def outputs(self):
        np = self.np
        if self.lane == "lm-train-step":
            return {"losses": np.array(self.losses, dtype=np.float64)}
        return {"logits": self.out}

    def digest(self):
        if self.lane == "lm-train-step":
            return None
        return _sha(self.np.ascontiguousarray(self.out).data)[:16]


class TorchGEMM:
    def __init__(self, lane, shape, data):
        import numpy as np
        self.np = np
        torch, dev, kind, name, sync = _torch_device()
        self.torch, self.dev, self._sync = torch, dev, sync
        precision = _load("torch_lm_step_opponent").set_precision(torch, False)
        self.a = torch.from_numpy(np.ascontiguousarray(data["a"]))
        self.b = torch.from_numpy(np.ascontiguousarray(data["b"]))
        self.info = _torch_info(torch, kind, name, precision)
        self.info["call"] = "a.to(dev) @ b.to(dev), then .cpu()"
        self.out = None

    def call(self):
        c = self.a.to(self.dev) @ self.b.to(self.dev)
        self.out = c.cpu().numpy()

    def sync(self):
        self._sync()

    def outputs(self):
        return {"c": self.out}

    def digest(self):
        return _sha(self.np.ascontiguousarray(self.out).data)[:16]


BUILDERS = {("lm-train-step", "ours"): OursLM, ("lm-forward", "ours"): OursLM,
            ("gemm", "ours"): OursGEMM,
            ("lm-train-step", "torch-eager-fp32"): TorchLM, ("lm-forward", "torch-eager-fp32"): TorchLM,
            ("gemm", "torch-eager-fp32"): TorchGEMM}


def worker(args):
    # fd 1 is the protocol; everything a library prints goes to the log.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True) + "\n")
        proto.flush()

    import numpy as np
    try:
        with np.load(args.data) as z:
            data = {k: z[k] for k in z.files}
        runner = BUILDERS[(args.lane, args.arm)](args.lane, args.shape, data)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        say({"event": "error", "stage": "ready", "error": repr(exc)})
        return 1
    say({"event": "ready", "info": runner.info, "pid": os.getpid()})
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "round":
            r = int(parts[1])
            try:
                t0 = time.perf_counter()
                runner.call()
                runner.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                digest = runner.digest()
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                say({"event": "error", "stage": "round %d" % r, "error": repr(exc)})
                return 1
            say({"event": "round", "round": r, "ms": ms, "digest": digest})
        elif parts[0] == "save":
            try:
                path = parts[1]
                tmp = path + ".tmp.npz"
                np.savez(tmp, **runner.outputs())
                os.replace(tmp, path)
                say({"event": "saved", "path": path, "info": runner.info})
            except Exception as exc:  # noqa: BLE001
                say({"event": "error", "stage": "save", "error": repr(exc)})
                return 1
        elif parts[0] == "quit":
            say({"event": "bye"})
            return 0
    return 0


# ---------------------------------------------------------------------------
# Quality (the conductor, float64)
# ---------------------------------------------------------------------------

def quality(lane, data, outs):
    import numpy as np
    q = {}
    if lane == "lm-train-step":
        for arm, o in outs.items():
            losses = [float(x) for x in o["losses"]]
            q[arm] = {"loss_first_step": losses[0], "loss_last_step": losses[-1], "steps": len(losses)}
        ref = q.get("ours", {}).get("loss_last_step")
        for arm in q:
            if arm != "ours" and ref is not None:
                q[arm]["loss_last_abs_diff_vs_ours"] = abs(q[arm]["loss_last_step"] - ref)
    elif lane == "lm-forward":
        targets = data["batches"][0][:, 1:].astype(np.int64)
        for arm, o in outs.items():
            lg = o["logits"].astype(np.float64)
            mx = lg.max(axis=-1, keepdims=True)
            lse = (mx[..., 0] + np.log(np.exp(lg - mx).sum(axis=-1)))
            picked = np.take_along_axis(lg, targets[..., None], axis=-1)[..., 0]
            q[arm] = {"mean_nll": float((lse - picked).mean())}
        ref = outs.get("ours", {}).get("logits")
        for arm, o in outs.items():
            if arm != "ours" and ref is not None:
                q[arm]["max_abs_diff_vs_ours"] = float(
                    np.abs(o["logits"].astype(np.float64) - ref.astype(np.float64)).max())
    elif lane == "gemm":
        c64 = data["a"].astype(np.float64) @ data["b"].astype(np.float64)
        scale = float(np.abs(c64).max()) or 1.0
        for arm, o in outs.items():
            q[arm] = {"max_rel_err_vs_fp64": float(np.abs(o["c"].astype(np.float64) - c64).max()) / scale}
        ref = outs.get("ours", {}).get("c")
        for arm, o in outs.items():
            if arm != "ours" and ref is not None:
                q[arm]["max_abs_diff_vs_ours"] = float(
                    np.abs(o["c"].astype(np.float64) - ref.astype(np.float64)).max())
    return q


# ---------------------------------------------------------------------------
# race: the conductor for one lane
# ---------------------------------------------------------------------------

def _worker_env(arm):
    ctd = _load("classical_two_datasets")
    env = dict(os.environ)
    for k in ctd.THREAD_ENV:
        env.pop(k, None)
    if arm == "ours":
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        if os.environ.get("MOJOLEARN_BENCH_INSTALLED", "0").strip() in ("", "0"):
            tree = os.path.join(REPO, "python")
            env["PYTHONPATH"] = tree + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


SPAN = {"input_home": "host", "pre_clock_fit": False,
        "inside_clock": "host inputs to the device, the call, the result back on the host, synchronized"}


def race(args):
    import numpy as np
    ctd = _load("classical_two_datasets")
    lane, shape = args.lane, args.shape
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if (lane, a) not in BUILDERS:
            raise SystemExit("no arm %r for lane %r" % (a, lane))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    tag = "%s-%s" % (lane, DATA_OF[lane])
    data_path = os.path.join(args.work, "neural-%s-%s.npz" % (lane, shape))
    inputs = make_inputs(lane, shape, args.rounds + 1, data_path)
    srec = shape_record(lane, shape)
    result = {"lane": lane, "dataset": DATA_OF[lane], "shape": srec["label"], "shape_record": srec,
              "inputs": inputs, "arms": {}, "rounds_requested": args.rounds, "started": now_utc(),
              "script": "tools/bench_board_neural.py", "torch_mode": TORCH_MODE,
              "commit": os.environ.get("MOJOLEARN_REPO_COMMIT", "unknown")}
    workers = {}
    for arm in arms:
        py = args.ours_python if arm == "ours" else args.theirs_python
        cmd = shlex.split(py) + [os.path.abspath(__file__), "worker", "--arm", arm,
                                 "--lane", lane, "--shape", shape, "--data", data_path]
        workers[arm] = ctd.Worker(arm, cmd, _worker_env(arm),
                                  os.path.join(args.out, "%s-%s.log" % (tag, arm)), REPO)
        result["arms"][arm] = {"command": cmd, "warmup_ms": None, "ms": [], "digests": [],
                               "status": "ok"}
    for arm, w in workers.items():
        msg = w.read(args.ready_seconds)
        if msg is None or msg.get("event") != "ready":
            w.kill("not_ready", msg)
            result["arms"][arm].update(status="not_ready", error=msg)
            print("NEURAL-REFUSED lane=%s arm=%s stage=ready detail=%s" % (lane, arm, json.dumps(msg)),
                  flush=True)
            continue
        w.info = msg["info"]
        result["arms"][arm]["info"] = msg["info"]
    for r in range(args.rounds + 1):
        live = [a for a in arms if workers[a].alive]
        if not live:
            break
        shift = r % len(live)
        for arm in live[shift:] + live[:shift]:
            w = workers[arm]
            w.send("round %d" % r)
            msg = w.read(args.warmup_seconds if r == 0 else args.round_seconds)
            if msg is None or msg.get("event") != "round":
                status = "timeout" if msg is None else "error"
                w.kill(status, msg)
                result["arms"][arm].update(status=status, error=msg, failed_round=r)
                print("NEURAL-REFUSED lane=%s arm=%s stage=round%d detail=%s"
                      % (lane, arm, r, json.dumps(msg)), flush=True)
                continue
            if r == 0:
                result["arms"][arm]["warmup_ms"] = msg["ms"]
            else:
                result["arms"][arm]["ms"].append(msg["ms"])
            result["arms"][arm]["digests"].append(msg["digest"])
            print("NEURAL-ROUND lane=%s arm=%s round=%d ms=%.3f digest=%s"
                  % (lane, arm, r, msg["ms"], msg["digest"]), flush=True)
    outs = {}
    for arm in arms:
        w = workers[arm]
        if w.alive and len(result["arms"][arm]["ms"]) == args.rounds:
            path = os.path.join(args.work, "%s-%s.npz" % (tag, arm))
            w.send("save %s" % path)
            msg = w.read(args.round_seconds)
            if msg is not None and msg.get("event") == "saved":
                with np.load(path) as z:
                    outs[arm] = {k: z[k] for k in z.files}
                os.remove(path)
            else:
                result["arms"][arm].update(status="save_failed", error=msg)
        w.close()
    try:
        with np.load(data_path) as z:
            data = {k: z[k] for k in z.files}
        result["quality"] = quality(lane, data, outs)
    except Exception as exc:  # noqa: BLE001
        result["quality"] = {"error": repr(exc)}
    try:
        os.remove(data_path)
    except OSError:
        pass
    for arm in arms:
        a = result["arms"][arm]
        ok = a["status"] == "ok" and len(a["ms"]) == args.rounds
        a["median_ms"] = statistics.median(a["ms"]) if ok else None
        timed = [d for d in a["digests"][1:] if d is not None]
        # a training step changes the state every round (no repeat to compare),
        # and one timed round has nothing to compare against either
        a["digest_stable"] = (len(set(timed)) == 1) if ok and len(timed) >= 2 else None
        a["span"] = dict(SPAN)
        print("NEURAL lane=%s arm=%s status=%s median_ms=%s quality=%s"
              % (lane, arm, a["status"], a["median_ms"],
                 json.dumps(result["quality"].get(arm, {}), sort_keys=True)), flush=True)
    result["finished"] = now_utc()
    out_json = os.path.join(args.out, "%s.json" % tag)
    with open(out_json + ".tmp", "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    os.replace(out_json + ".tmp", out_json)
    failed = [a for a in arms if result["arms"][a]["status"] != "ok"]
    return 1 if failed and len(failed) == len(arms) else 0


def build_parser():
    p = argparse.ArgumentParser(prog="bench_board_neural", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("race")
    r.add_argument("--lane", required=True, choices=LANES)
    r.add_argument("--shape", default="full", choices=sorted(LM_SHAPES))
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--ours-python", default=sys.executable)
    r.add_argument("--theirs-python", default=sys.executable)
    r.add_argument("--ready-seconds", type=int, default=1800)
    r.add_argument("--warmup-seconds", type=int, default=1800)
    r.add_argument("--round-seconds", type=int, default=1800)
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True, choices=ARMS)
    w.add_argument("--lane", required=True, choices=LANES)
    w.add_argument("--shape", required=True, choices=sorted(LM_SHAPES))
    w.add_argument("--data", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "worker":
        return worker(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    return race(args)


if __name__ == "__main__":
    sys.exit(main())
