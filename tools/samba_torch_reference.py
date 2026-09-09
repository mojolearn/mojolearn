#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The external comparator: the same Samba-shaped stack in PyTorch EAGER
fp32 (no compile, no autocast, no fused kernels, TF32 off), trained for the
same steps on the same bytes, timed per step. Numbers from this file are a
TIMING reference only; nothing here claims bit agreement with mojolearn.

The Mamba-3 block is spelled from `mamba/IDENTICAL_MAMBA3_CONTRACT.md`'s
stage list (S1-S23) in the quadratic (single-chunk) SSD form over the whole
sequence with a zero initial state: block RMSNorm; in_proj split
z | x | B | C | dd_dt | dd_A | trap | angle; data-dependent A with the
heavy-tail activation and the A_floor clamp; dt = softplus(dd_dt + bias);
trapezoid scales; the angle recurrence and pair rotation of the normed,
biased B/C; the decay-masked q k^T v contraction; D skip plus the
diagonal qk term; the silu(z) gate; out_proj; residual. The attention
block is a Llama decoder layer (RMSNorm, RoPE, causal eager attention,
SwiGLU). Same shapes, same parameter count as `SambaStack`.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from samba_train_run import batch, CORPUS  # noqa: E402

D_STATE, HEADDIM, EXPAND, NUM_ANGLES, A_FLOOR = 128, 64, 2, 32, 1e-4


def rms_norm(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


class Mamba3(nn.Module):
    def __init__(self, dm):
        super().__init__()
        di = EXPAND * dm
        nh = di // HEADDIM
        dip = 2 * di + 2 * D_STATE + 3 * nh + NUM_ANGLES
        self.dm, self.di, self.nh = dm, di, nh
        self.block_norm = nn.Parameter(torch.ones(dm))
        self.in_proj = nn.Linear(dm, dip, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(nh).uniform_(-4, -2))
        self.B_norm = nn.Parameter(torch.ones(D_STATE))
        self.C_norm = nn.Parameter(torch.ones(D_STATE))
        self.B_bias = nn.Parameter(torch.ones(nh, D_STATE))
        self.C_bias = nn.Parameter(torch.ones(nh, D_STATE))
        self.D = nn.Parameter(torch.ones(nh))
        self.out_proj = nn.Linear(di, dm, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        di, nh = self.di, self.nh
        u = rms_norm(x, self.block_norm, 1e-5)
        zx = self.in_proj(u)
        z, xr, Bc, Cc, dd_dt, dd_A, trap, angle = torch.split(
            zx, [di, di, D_STATE, D_STATE, nh, nh, nh, NUM_ANGLES], dim=-1)
        heavy = torch.where(dd_A >= 0, 1.0 + dd_A, 1.0 / (1.0 - dd_A))
        A = torch.clamp(-heavy, max=-A_FLOOR)                       # (b,l,h)
        dt = F.softplus(dd_dt + self.dt_bias)                        # (b,l,h)
        adt = A * dt
        sig = torch.sigmoid(trap)
        gamma = dt * sig
        sig_next = torch.cat([sig[:, 1:], torch.zeros_like(sig[:, :1])], 1)
        dt_next = torch.cat([dt[:, 1:], torch.zeros_like(dt[:, :1])], 1)
        scale = gamma + dt_next * (1.0 - sig_next)
        rate = torch.tanh(angle) * math.pi                           # (b,l,32)
        inc = rate.unsqueeze(2) * dt.unsqueeze(-1)                   # (b,l,h,32)
        theta = torch.remainder(torch.cumsum(inc, dim=1), 2 * math.pi)
        c, s = torch.cos(theta), torch.sin(theta)
        k = rms_norm(Bc, self.B_norm, 1e-5).unsqueeze(2) + self.B_bias   # (b,l,h,N)
        q = rms_norm(Cc, self.C_norm, 1e-5).unsqueeze(2) + self.C_bias
        qk_dot = (q * k).sum(-1) * gamma                              # (b,l,h)

        def rotate(t):
            pairs = t[..., :2 * NUM_ANGLES].reshape(b, l, nh, NUM_ANGLES, 2)
            r0 = pairs[..., 0] * c - pairs[..., 1] * s
            r1 = pairs[..., 0] * s + pairs[..., 1] * c
            rot = torch.stack((r0, r1), -1).reshape(b, l, nh, 2 * NUM_ANGLES)
            return torch.cat([rot, t[..., 2 * NUM_ANGLES:]], -1)
        q_rot, k_rot = rotate(q), rotate(k)
        k_scaled = k_rot * scale.unsqueeze(-1)
        v = xr.reshape(b, l, nh, HEADDIM)
        cs = torch.cumsum(adt, dim=1)                                # (b,l,h)
        seg = cs.unsqueeze(2) - cs.unsqueeze(1)                      # (b,i,j,h)
        mask = torch.tril(torch.ones(l, l, device=x.device, dtype=torch.bool), -1)
        decay = torch.exp(seg) * mask.view(1, l, l, 1)
        sc = torch.einsum("bihn,bjhn->bijh", q_rot, k_scaled) * decay
        y = torch.einsum("bijh,bjhp->bihp", sc, v)
        y = y + (self.D + qk_dot).unsqueeze(-1) * v
        y = y.reshape(b, l, di) * F.silu(z)
        return x + self.out_proj(y)


class Attention(nn.Module):
    def __init__(self, dm, n_heads, intermediate):
        super().__init__()
        self.nh, self.hd = n_heads, dm // n_heads
        self.norm1 = nn.Parameter(torch.ones(dm))
        self.norm2 = nn.Parameter(torch.ones(dm))
        self.q = nn.Linear(dm, dm, bias=False)
        self.k = nn.Linear(dm, dm, bias=False)
        self.v = nn.Linear(dm, dm, bias=False)
        self.o = nn.Linear(dm, dm, bias=False)
        self.gate = nn.Linear(dm, intermediate, bias=False)
        self.up = nn.Linear(dm, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dm, bias=False)

    def forward(self, x):
        b, l, dm = x.shape
        h = rms_norm(x, self.norm1, 1e-6)
        q = self.q(h).view(b, l, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(b, l, self.nh, self.hd).transpose(1, 2)
        v = self.v(h).view(b, l, self.nh, self.hd).transpose(1, 2)
        pos = torch.arange(l, device=x.device, dtype=torch.float32)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2, device=x.device).float() / self.hd))
        fr = torch.outer(pos, inv)
        emb = torch.cat((fr, fr), -1)
        cos, sin = emb.cos(), emb.sin()

        def rope(t):
            t1, t2 = t[..., : self.hd // 2], t[..., self.hd // 2:]
            return t * cos + torch.cat((-t2, t1), -1) * sin
        q, k = rope(q), rope(k)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        mask = torch.triu(torch.ones(l, l, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, l, dm)
        x = x + self.o(y)
        h = rms_norm(x, self.norm2, 1e-6)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Stack(nn.Module):
    def __init__(self, vocab, dm, layers, n_heads, intermediate, tie):
        super().__init__()
        self.embed = nn.Embedding(vocab, dm)
        nn.init.normal_(self.embed.weight, 0.0, 0.02)
        self.blocks = nn.ModuleList([
            Mamba3(dm) if k == "mamba3" else Attention(dm, n_heads, intermediate)
            for k in layers])
        self.norm_f = nn.Parameter(torch.ones(dm))
        self.head = nn.Linear(dm, vocab, bias=False)
        if tie:
            self.head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(rms_norm(x, self.norm_f, 1e-5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", default="mamba3,mamba3")
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--intermediate", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    model = Stack(256, args.d_model, args.layers.split(","), args.n_heads,
                  args.intermediate, not args.untied).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.01)

    def lr_at(t):
        if t <= args.warmup:
            return args.lr * t / args.warmup
        if t >= args.steps:
            return args.min_lr
        p = (t - args.warmup) / (args.steps - args.warmup)
        return args.min_lr + (args.lr - args.min_lr) * (1 + math.cos(math.pi * p)) / 2
    corpus = CORPUS.read_bytes()
    rows = []
    for s in range(args.steps):
        inputs, targets = batch(corpus, s, args.batch, args.seq)
        x = torch.from_numpy(inputs.astype(np.int64)).to(dev)
        y = torch.from_numpy(targets.astype(np.int64)).to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(s + 1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"step": s + 1, "loss": float(loss), "lr": lr_at(s + 1),
                     "total_norm": float(norm), "seconds": dt})
        print("step %3d loss %.6f lr %.3e norm %.4f  %.1f ms"
              % (s + 1, float(loss), lr_at(s + 1), float(norm), dt * 1e3), flush=True)
    timed = [r["seconds"] for r in rows[8:]] or [r["seconds"] for r in rows]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"framework": "torch eager fp32 %s" % torch.__version__,
               "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               "n_parameters": n_params, "steps": args.steps, "batch": args.batch,
               "seq": args.seq, "layers": args.layers, "d_model": args.d_model,
               "first_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
               "ms_per_step_median": 1e3 * float(np.median(timed)),
               "ms_per_step_mean": 1e3 * float(np.mean(timed)),
               "timed_steps": "9..%d" % args.steps, "rows": rows}
    (out / "torch_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("torch parameters %d  median %.1f ms/step  loss %.4f -> %.4f"
          % (n_params, summary["ms_per_step_median"], rows[0]["loss"], rows[-1]["loss"]))


if __name__ == "__main__":
    main()
