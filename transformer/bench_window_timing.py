# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Wall-clock of the IDENTICAL transformer block's forward and
forward+backward under a sliding window, against PyTorch eager fp32 with
`scaled_dot_product_attention` and an explicit sliding-window mask, on
the same shape and the same weights, on one GPU.

    MOJOLEARN_NUMERIC_MODE=identical python3 transformer/bench_window_timing.py \\
        --dm 1024 --heads 16 --kv 4 --window 2048 --seq 4096 --batch 4 \\
        --log /tmp/window_timing.log

Ours is the Python surface (`mojolearn.TransformerBlock`): `forward` is
one block call from a zero cache; `backward` recomputes that forward and
runs the lane's backward, so its time IS forward+backward. Both include
the host copies the surface makes. The torch arm is autograd over an
eager fp32 block (TF32 off), with `torch.compile` as a second arm when it
compiles. Every timing is a median over `--rounds` after one warm-up,
with a device synchronize around each call.
"""

import argparse
import os
import sys
import time

import numpy as np


def weights(rng, dm, nh, nkv, hd, it):
    qw, kw = nh * hd, nkv * hd
    s_in = float(dm) ** -0.5
    s_it = float(it) ** -0.5

    def u(shape, lo, hi):
        return (lo + (hi - lo) * rng.random(shape)).astype(np.float32)

    return {
        "input_layernorm.weight": u((dm,), 0.5, 1.5),
        "post_attention_layernorm.weight": u((dm,), 0.5, 1.5),
        "q_proj.weight": u((qw, dm), -s_in, s_in),
        "k_proj.weight": u((kw, dm), -s_in, s_in),
        "v_proj.weight": u((kw, dm), -s_in, s_in),
        "o_proj.weight": u((dm, qw), -s_in, s_in),
        "gate_proj.weight": u((it, dm), -s_in, s_in),
        "up_proj.weight": u((it, dm), -s_in, s_in),
        "down_proj.weight": u((dm, it), -s_it, s_it),
    }


def median_ms(fn, rounds, sync):
    fn()
    sync()
    ts = []
    for _ in range(rounds):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), ts


def ours(args, w, x, dy, log):
    from mojolearn import TransformerBlock

    blk = TransformerBlock(w, n_heads=args.heads, n_kv_heads=args.kv,
                           window=args.window, numeric_mode="identical")
    log("ours: tier %s vendor %s" % (blk.numeric_mode_used(),
                                     blk.vendor_used()))
    f_ms, f_all = median_ms(lambda: blk.forward(x), args.rounds, lambda: None)
    log("ours forward            median %.1f ms  rounds %s"
        % (f_ms, ["%.1f" % t for t in f_all]))
    b_ms, b_all = median_ms(lambda: blk.backward(x, dy), args.rounds,
                            lambda: None)
    log("ours forward+backward   median %.1f ms  rounds %s"
        % (b_ms, ["%.1f" % t for t in b_all]))
    return blk.forward(x), blk.backward(x, dy)


def torch_block(torch, dev, w, args):
    F = torch.nn.functional
    dm, nh, nkv, hd = args.dm, args.heads, args.kv, args.dm // args.heads
    W = {k: torch.from_numpy(v).to(dev).requires_grad_(True)
         for k, v in w.items()}
    L = args.seq
    half = hd // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.arange(L, dtype=np.float64)[:, None] * inv[None, :]
    cos = torch.from_numpy(np.cos(ang).astype(np.float32)).to(dev)
    sin = torch.from_numpy(np.sin(ang).astype(np.float32)).to(dev)
    cos = torch.cat((cos, cos), -1)[None, None]
    sin = torch.cat((sin, sin), -1)[None, None]
    pos = torch.arange(L, device=dev)
    visible = pos[None, :] <= pos[:, None]
    if args.window > 0:
        visible &= pos[None, :] > pos[:, None] - args.window
    mask = visible[None, None]
    scale = float(hd) ** -0.5
    n_rep = nh // nkv

    def rms(t, g):
        var = (t * t).mean(-1, keepdim=True)
        return g * (t * torch.rsqrt(var + 1e-6))

    def rot(t):
        return torch.cat((-t[..., half:], t[..., :half]), -1)

    def block(x):
        B = x.shape[0]
        h = rms(x, W["input_layernorm.weight"])
        q = F.linear(h, W["q_proj.weight"]).view(B, L, nh, hd).transpose(1, 2)
        k = F.linear(h, W["k_proj.weight"]).view(B, L, nkv, hd).transpose(1, 2)
        v = F.linear(h, W["v_proj.weight"]).view(B, L, nkv, hd).transpose(1, 2)
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           scale=scale)
        o = o.transpose(1, 2).reshape(B, L, nh * hd)
        r1 = x + F.linear(o, W["o_proj.weight"])
        h2 = rms(r1, W["post_attention_layernorm.weight"])
        g = F.linear(h2, W["gate_proj.weight"])
        u = F.linear(h2, W["up_proj.weight"])
        return r1 + F.linear((g / (1.0 + torch.exp(-g))) * u,
                             W["down_proj.weight"])

    return block, W


def theirs(args, w, x, dy, log, compiled):
    import torch

    if not torch.cuda.is_available():
        log("torch: no CUDA device, torch arm not run")
        return None
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    block, W = torch_block(torch, dev, w, args)
    if compiled:
        try:
            block = torch.compile(block)
        except Exception as exc:  # noqa: BLE001
            log("torch.compile unavailable: %r" % (exc,))
            return None
    name = "torch-eager-fp32-sdpa" + ("-compiled" if compiled else "")
    xt = torch.from_numpy(x).to(dev).requires_grad_(True)
    dyt = torch.from_numpy(dy).to(dev)
    sync = torch.cuda.synchronize

    def fwd():
        with torch.no_grad():
            return block(xt)

    def fwd_bwd():
        xt.grad = None
        for p in W.values():
            p.grad = None
        y = block(xt)
        y.backward(dyt)
        return y

    try:
        f_ms, f_all = median_ms(fwd, args.rounds, sync)
        log("%s forward            median %.1f ms  rounds %s"
            % (name, f_ms, ["%.1f" % t for t in f_all]))
        b_ms, b_all = median_ms(fwd_bwd, args.rounds, sync)
        log("%s forward+backward   median %.1f ms  rounds %s"
            % (name, b_ms, ["%.1f" % t for t in b_all]))
    except Exception as exc:  # noqa: BLE001
        log("%s FAILED: %r" % (name, exc))
        return None
    with torch.no_grad():
        y = block(xt).float().cpu().numpy()
    fwd_bwd()
    gx = xt.grad.float().cpu().numpy()
    return y, gx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=4)
    ap.add_argument("--inter", type=int, default=4096)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--log", default="")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(msg + "\n")

    log("shape: d_model %d heads %d kv %d head_dim %d inter %d window %d "
        "seq %d batch %d rounds %d" % (
            args.dm, args.heads, args.kv, args.dm // args.heads, args.inter,
            args.window, args.seq, args.batch, args.rounds))
    rng = np.random.default_rng(0x53616D62)
    hd = args.dm // args.heads
    w = weights(rng, args.dm, args.heads, args.kv, hd, args.inter)
    x = (2.0 * rng.random((args.batch, args.seq, args.dm)) - 1.0).astype(
        np.float32)
    dy = (2.0 * rng.random((args.batch, args.seq, args.dm)) - 1.0).astype(
        np.float32)
    ours_out = None
    if not args.skip_ours:
        ours_out = ours(args, w, x, dy, log)
    if not args.skip_torch:
        t_e = theirs(args, w, x, dy, log, compiled=False)
        t_c = theirs(args, w, x, dy, log, compiled=True)
        if ours_out is not None and t_e is not None:
            y, g = ours_out
            dy_ = np.abs(y.astype(np.float64) - t_e[0]).max()
            dg = np.abs(g["x"].astype(np.float64) - t_e[1]).max()
            log("agreement vs torch eager: max|y diff| %.3e, "
                "max|dx diff| %.3e (fp32 vs fp32, different fold orders; "
                "a sanity check, not an identity claim)" % (dy_, dg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
