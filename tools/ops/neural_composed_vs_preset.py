#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Composed-from-layers versus monolithic preset, our arms only (lane neural-blocks, 2026-11-22).

Andrew asked whether the neural block classes can be rebuilt ON the Sequential building blocks so the monolithic
block code can be deleted. AGENTS.md "A measured winner is never deleted on reasoning": the composition replaces
fused kernels on a hot path, so this script measures it first. It also times the monolith's own forward and
backward, so the same script run on two wheels (before / after a shared-body kernel change, stage 4) compares a
kernel edit: pass --label to tag the rows.

Per architecture (the board recipe's `full` block shape from tools/bench_board_neural.py BLOCK_SHAPES, `--shape
small` for a smoke), with one weight set shared by both arms:
  preset    TransformerBlock / Mamba1Block / Mamba2Block / Mamba3Block: forward(x), backward(x, g), and one
            decode step (allocate_state + forward of the prompt, then step(x_t, state) timed)
  composed  mojolearn.Sequential built from rms_norm, attention, gated, linear, residual_block, loaded with the
            SAME weights through load_state_dict: forward = predict(x); train step = one fit epoch (sgd, lr 1,
            mse against a target chosen so dL/dy = g): forward + backward + the update. The composed decode
            step does not exist (no KV-cache session on Sequential): printed as "n/a".
            Mamba1/2/3: NOT composable today (the selective scan, the SSD and the SISO core exist only as
            block-tied stage kernels, no generic tape op): the composed arm prints "not composable" and only
            the preset is timed.
Quality: the composed forward against the preset forward value by value (max |diff|, count of differing
words), and the composed gradients recovered from the sgd step (p0 - p1, exact to the rounding of one update)
against the preset's backward. Timing: one excluded warmup; a row whose warmup takes under 10 ms is the median of
20 consecutive calls (AGENTS.md "Small rows"), else one call. Host probe lines first (AGENTS.md "Times are
compared only on the same box"): compare only rows of the same box and job.

    python tools/ops/neural_composed_vs_preset.py [--shape full|small] [--arch transformer,mamba1,...] [--label L]
"""
import argparse
import os
import platform
import statistics
import subprocess
import time

import numpy as np


def probe():
    print("PROBE host", platform.node(), "cpu", platform.processor() or platform.machine(), "cores", os.cpu_count())
    try:
        with open("/proc/cpuinfo") as f:
            model = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "?")
        print("PROBE cpu_model", model)
    except OSError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        print("PROBE gpu", out or "none")
    except (OSError, subprocess.SubprocessError):
        print("PROBE gpu none")
    import mojolearn
    print("PROBE mojolearn", mojolearn.__version__, "vendor", mojolearn._backend.vendor())


def clock(fn):
    """One excluded warmup, then one call, or the median of 20 consecutive calls when the warmup is under 10 ms."""
    t0 = time.perf_counter()
    fn()
    warm = time.perf_counter() - t0
    n = 20 if warm < 0.010 else 1
    times = []
    for _ in range(n):  # glue: consecutive calls of one scored run (benchmark tool, not product)
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    spread = (max(times) - min(times)) if n > 1 else 0.0
    return med * 1e3, n, spread * 1e3


def row(label, arch, arm, op, ms, n, spread, quality=""):
    print("ROW label=%s arch=%s arm=%s op=%s ms=%.4f calls=%d spread_ms=%.4f %s" % (label, arch, arm, op, ms, n, spread, quality))


SHAPES = {
    "transformer": {"full": dict(batch=1, length=2048, d_model=384, n_heads=6, n_kv=6, head_dim=64, intermediate=1024),
                    "small": dict(batch=2, length=64, d_model=64, n_heads=4, n_kv=2, head_dim=16, intermediate=128)},
    "mamba1": {"full": dict(batch=1, length=2048, d_model=384), "small": dict(batch=2, length=64, d_model=16)},
    "mamba2": {"full": dict(batch=1, length=2048, d_model=384), "small": dict(batch=2, length=64, d_model=64)},
    "mamba3": {"full": dict(batch=1, length=2048, d_model=384), "small": dict(batch=2, length=64, d_model=64)},
}


def compare(a, b):
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    diff = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    return "max_abs_diff=%.3e differing_words=%d of %d" % (float(np.max(np.abs(a.astype(np.float64) - b))), diff, a.size)


def transformer(label, s):
    import mojolearn
    rng = np.random.default_rng(0)
    B, T, D, H, K, hd, IT = (s[k] for k in ("batch", "length", "d_model", "n_heads", "n_kv", "head_dim", "intermediate"))
    shapes = {"input_layernorm.weight": (D,), "post_attention_layernorm.weight": (D,),
              "q_proj.weight": (H * hd, D), "k_proj.weight": (K * hd, D), "v_proj.weight": (K * hd, D),
              "o_proj.weight": (D, H * hd), "gate_proj.weight": (IT, D), "up_proj.weight": (IT, D),
              "down_proj.weight": (D, IT)}
    w = {k: (1.0 + 0.1 * rng.standard_normal(v) if "norm" in k else rng.standard_normal(v) / np.sqrt(v[-1])).astype(np.float32)
         for k, v in shapes.items()}
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    g = rng.standard_normal((B, T, D)).astype(np.float32) / (B * T * D)
    preset = mojolearn.TransformerBlock(w, n_heads=H, n_kv_heads=K, head_dim=hd, norm_eps=1e-6, numeric_mode="identical")
    S = mojolearn.Sequential
    m = S(input_shape=(T, D), numeric_mode="identical")
    m.residual_block(S().rms_norm(1e-6).attention(H, kv_heads=K, rotary=True, theta=10000.0))
    m.residual_block(S().rms_norm(1e-6).gated(S().linear(IT, bias=False).silu(), S().linear(IT, bias=False))
                     .linear(D, bias=False))
    m.flatten()
    kinds = [layer[0] for layer in m.layers]
    norms = [i for i, k in enumerate(kinds) if k == "rms_norm"]
    att = kinds.index("attention")
    lins = [i for i, k in enumerate(kinds) if k == "linear"]
    sd = {"%d.weight" % norms[0]: w["input_layernorm.weight"], "%d.weight" % norms[1]: w["post_attention_layernorm.weight"],
          "%d.q_weight" % att: w["q_proj.weight"], "%d.k_weight" % att: w["k_proj.weight"],
          "%d.v_weight" % att: w["v_proj.weight"], "%d.o_weight" % att: w["o_proj.weight"],
          "%d.weight" % lins[0]: w["gate_proj.weight"], "%d.weight" % lins[1]: w["up_proj.weight"],
          "%d.weight" % lins[2]: w["down_proj.weight"]}
    m.load_state_dict(sd)
    y_preset = np.asarray(preset.forward(x)).reshape(B, T * D)
    y_comp = np.asarray(m.predict(x.reshape(B, T * D)))
    row(label, "transformer", "composed", "forward-quality", 0, 0, 0, compare(y_comp, y_preset))
    grads = preset.backward(x, g)
    # one sgd step at lr 1 on mse: dL/dy = 2 (y - t) / (B * T * D) = g  =>  t = y - g (B T D) / 2
    target = y_comp - g.reshape(B, T * D) * (B * T * D) / 2
    before = {k: np.array(v) for k, v in m.state_dict().items()}
    m.compile(loss="mse", optimizer="sgd", lr=1.0)
    m.fit(x.reshape(B, T * D), target, epochs=1, batch_size=B, shuffle=False, seed=0)
    after = m.state_dict()
    for name, key in sd_names(sd, w).items():
        row(label, "transformer", "composed", "grad-quality:" + name, 0, 0, 0,
            compare(before[key] - np.asarray(after[key]), grads[name]))
    m.load_state_dict(sd)
    row(label, "transformer", "preset", "forward", *clock(lambda: preset.forward(x)))
    row(label, "transformer", "composed", "forward", *clock(lambda: m.predict(x.reshape(B, T * D))))
    row(label, "transformer", "preset", "backward", *clock(lambda: preset.backward(x, g)))
    row(label, "transformer", "composed", "train-step", *clock(
        lambda: m.fit(x.reshape(B, T * D), target, epochs=1, batch_size=B, shuffle=False, seed=0)))
    xt = x[:, T - 1 : T]

    def decode():
        st = preset.allocate_state(B, T + 2)
        preset.forward(x[:, : T - 1], st)
        return st
    # the step on a freshly prefilled cache each call (the prefill is outside the clock)
    times = []
    for _ in range(21):  # glue: warmup + 20 consecutive decode steps, each on its own prefilled state
        st = decode()
        t0 = time.perf_counter()
        preset.step(xt, st)
        times.append(time.perf_counter() - t0)
    times = times[1:]
    row(label, "transformer", "preset", "decode-step", statistics.median(times) * 1e3, 20,
        (max(times) - min(times)) * 1e3)
    print("ROW label=%s arch=transformer arm=composed op=decode-step n/a (no KV-cache session on Sequential)" % label)


def sd_names(sd, w):
    """preset weight name -> composed state_dict key, by the shared array."""
    out = {}
    for key, value in sd.items():
        for name, arr in w.items():
            if arr is value:
                out[name] = key
    return out


def mamba_shapes(arch, dm):
    """The block's weight shapes at d_model dm (expand 2, d_state 16 / 128, head dim 64, conv 4; as
    python/mojolearn/_verify_api_metadata.py block_weights, at the board's d_model)."""
    di = 2 * dm
    nh = max(1, di // 64)
    if arch == "mamba1":
        r = -(-dm // 16)
        return {"norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
                "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
                "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)}
    if arch == "mamba2":
        return {"block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + nh, dm),
                "conv1d.weight": (di + 256, 1, 4), "conv1d.bias": (di + 256,), "dt_bias": (nh,),
                "A_log": (nh,), "D": (nh,), "norm.weight": (di,), "out_proj.weight": (dm, di)}
    return {"block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + 3 * nh + 32, dm),
            "dt_bias": (nh,), "B_norm.weight": (128,), "C_norm.weight": (128,),
            "B_bias": (nh, 128), "C_bias": (nh, 128), "D": (nh,), "out_proj.weight": (dm, di)}


def mamba_preset(label, arch, s):
    import mojolearn
    rng = np.random.default_rng(0)
    B, T, D = s["batch"], s["length"], s["d_model"]
    cls = {"mamba1": mojolearn.Mamba1Block, "mamba2": mojolearn.Mamba2Block, "mamba3": mojolearn.Mamba3Block}[arch]
    shapes = mamba_shapes(arch, D)
    w = {k: (np.ones(v, np.float32) if "norm" in k else (rng.standard_normal(v) * 0.1).astype(np.float32))
         for k, v in shapes.items()}
    blk = cls(w, numeric_mode="identical")
    x = rng.standard_normal((B, T, D)).astype(np.float32)
    g = rng.standard_normal((B, T, D)).astype(np.float32)
    row(label, arch, "preset", "forward", *clock(lambda: blk.forward(x)))
    row(label, arch, "preset", "backward", *clock(lambda: blk.backward(x, g)))
    print("ROW label=%s arch=%s arm=composed op=all not composable (selective scan / SSD / SISO core is block-tied; "
          "no generic tape op)" % (label, arch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="full", choices=("full", "small"))
    ap.add_argument("--arch", default="transformer,mamba1,mamba2,mamba3")
    ap.add_argument("--label", default="current")
    a = ap.parse_args()
    probe()
    for arch in a.arch.split(","):  # glue: benchmark driver over architectures
        s = SHAPES[arch][a.shape]
        if arch == "transformer":
            transformer(a.label, s)
        else:
            mamba_preset(a.label, arch, s)


if __name__ == "__main__":
    main()
