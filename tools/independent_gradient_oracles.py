# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Generate independent gradient references for the trainable sequence lanes.

This program intentionally imports no mojolearn code.  PyTorch autograd is the
external implementation; central differences and closed-form invariants guard
against accidentally constructing a vacuous reference graph.  The output is a
stable NPZ bundle that a Mojo dump can be compared with without importing
PyTorch.

Run with the repository's ``skgpu`` environment (PyTorch is deliberately not a
core dependency):

    pixi run -e skgpu independent-gradient-oracles
    pixi run -e skgpu independent-gradient-oracles -- --out /tmp/oracles.npz

The computation is CPU float64 and deterministic.  It does not establish
cross-GPU bit identity; it establishes that independently computed derivatives
have the intended mathematical meaning.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SEED = 0x4D4F4A4F
ATOL = 2e-6
RTOL = 2e-5


def file_stem(key):
    return re.sub(r"[^A-Za-z0-9_-]+", "__", key)


def write_raw_bundle(directory, refs):
    """Write a language-neutral bundle consumable by Mojo or the comparator."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, value in sorted(refs.items()):
        a = np.asarray(value, dtype="<f8")
        filename = file_stem(key) + ".f64"
        a.tofile(directory / filename)
        arrays[key] = {"file": filename, "dtype": "<f8", "shape": list(a.shape)}
    (directory / "manifest.json").write_text(json.dumps({
        "format": "mojolearn.independent-gradient-raw.v1",
        "arrays": arrays,
    }, indent=2) + "\n")


def compare_bundle(expected, actual_dir, atol, rtol, family):
    """Compare raw Mojo exports named by the independent bundle manifest.

    Actual files may be float32 (the normal device result) or float64 and use
    the manifest filename with the corresponding suffix. Missing, malformed,
    nonfinite, and numerically different results are separate failures.
    """
    failures = []
    compared = 0
    for key, value in sorted(expected.items()):
        if family != "all" and not key.startswith(family + "."):
            continue
        stem = file_stem(key)
        path32, path64 = actual_dir / (stem + ".f32"), actual_dir / (stem + ".f64")
        if path32.is_file():
            got = np.fromfile(path32, dtype="<f4").astype(np.float64)
        elif path64.is_file():
            got = np.fromfile(path64, dtype="<f8")
        else:
            failures.append(f"MISSING {key}: expected {path32.name} or {path64.name}")
            continue
        want = np.asarray(value, dtype=np.float64)
        if got.size != want.size:
            failures.append(f"WRONG_SIZE {key}: got {got.size}, expected {want.size}")
            continue
        got = got.reshape(want.shape)
        if not np.isfinite(got).all():
            failures.append(f"NONFINITE {key}: {np.size(got) - np.isfinite(got).sum()} cells")
            continue
        close = np.isclose(got, want, atol=atol, rtol=rtol)
        if not close.all():
            flat = int(np.flatnonzero(~close)[0])
            failures.append(
                f"DIFFERENT {key}: {np.size(close)-int(close.sum())}/{np.size(close)} cells; "
                f"first {flat}: got {got.ravel()[flat]!r}, expected {want.ravel()[flat]!r}"
            )
            continue
        compared += 1
        print(f"PASS {key}: {want.size} cells")
    if failures:
        raise AssertionError("\n".join(failures))
    if not compared:
        raise AssertionError("no arrays compared")
    print(f"PASS bundle: {compared} arrays")


def tensor(values, shape, *, grad=True):
    return torch.tensor(values, dtype=torch.float64).reshape(shape).requires_grad_(grad)


def hashed(shape, key, scale=1.0, offset=0.0, *, grad=True):
    """Non-symmetric deterministic values; uniform fixtures hide transposes."""
    n = int(np.prod(shape))
    rng = np.random.default_rng(SEED + key)
    a = offset + scale * rng.uniform(-1.0, 1.0, n)
    return tensor(a, shape, grad=grad)


def finite_difference(loss_fn, x, flat_index, h=1e-5):
    with torch.no_grad():
        original = x.reshape(-1)[flat_index].item()
        x.reshape(-1)[flat_index] = original + h
        plus = loss_fn().item()
        x.reshape(-1)[flat_index] = original - h
        minus = loss_fn().item()
        x.reshape(-1)[flat_index] = original
    return (plus - minus) / (2.0 * h)


def assert_close(name, actual, expected, atol=ATOL, rtol=RTOL):
    if not np.isclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def transformer_reference(out):
    # Tiny grouped-query Llama block: B=2 prevents layout coincidences, L=3
    # exercises the causal mask, and n_heads != n_kv_heads exercises repetition.
    b, length, d_model, n_heads, n_kv, d_ff = 2, 3, 8, 2, 1, 12
    hd = d_model // n_heads
    x = hashed((b, length, d_model), 1, 0.7)
    weights = {
        "norm1": hashed((d_model,), 2, 0.25, 1.0),
        "norm2": hashed((d_model,), 3, 0.25, 1.0),
        "wq": hashed((n_heads * hd, d_model), 4, 0.35),
        "wk": hashed((n_kv * hd, d_model), 5, 0.35),
        "wv": hashed((n_kv * hd, d_model), 6, 0.35),
        "wo": hashed((d_model, n_heads * hd), 7, 0.35),
        "wg": hashed((d_ff, d_model), 8, 0.25),
        "wu": hashed((d_ff, d_model), 9, 0.25),
        "wd": hashed((d_model, d_ff), 10, 0.20),
    }
    upstream = hashed((b, length, d_model), 11, 0.8, grad=False)

    def forward():
        eps = 1e-6
        n1 = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weights["norm1"]
        q = F.linear(n1, weights["wq"]).view(b, length, n_heads, hd).transpose(1, 2)
        k = F.linear(n1, weights["wk"]).view(b, length, n_kv, hd).transpose(1, 2)
        v = F.linear(n1, weights["wv"]).view(b, length, n_kv, hd).transpose(1, 2)
        # RoPE with a nonzero absolute offset, independently expressed in torch.
        pos = torch.arange(7, 7 + length, dtype=torch.float64)
        inv = 10000.0 ** (-torch.arange(0, hd, 2, dtype=torch.float64) / hd)
        angle = pos[:, None] * inv[None, :]
        cos, sin = angle.cos()[None, None], angle.sin()[None, None]
        def rope(z):
            ze, zo = z[..., 0::2], z[..., 1::2]
            return torch.stack((ze * cos - zo * sin, zo * cos + ze * sin), -1).flatten(-2)
        q, k = rope(q), rope(k)
        k = k.repeat_interleave(n_heads // n_kv, dim=1)
        v = v.repeat_interleave(n_heads // n_kv, dim=1)
        scores = q @ k.transpose(-1, -2) / np.sqrt(hd)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        attn = scores.masked_fill(mask, -torch.inf).softmax(-1)
        context = (attn @ v).transpose(1, 2).reshape(b, length, d_model)
        residual = x + F.linear(context, weights["wo"])
        n2 = residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + eps) * weights["norm2"]
        mlp = F.silu(F.linear(n2, weights["wg"])) * F.linear(n2, weights["wu"])
        return residual + F.linear(mlp, weights["wd"])

    y = forward()
    loss = (y * upstream).sum()
    loss.backward()
    out["transformer.output"] = y.detach().numpy()
    out["transformer.input.x"] = x.detach().numpy()
    out["transformer.input.d_out"] = upstream.numpy()
    for name, w in weights.items():
        out[f"transformer.input.{name}"] = w.detach().numpy()
    out["transformer.d_x"] = x.grad.numpy()
    for name, w in weights.items():
        out[f"transformer.d_{name}"] = w.grad.numpy()

    # Independent directional finite differences exercise RMSNorm, attention,
    # GQA fan-in, SiLU and every projection in a single scalar check.
    analytic = x.grad.reshape(-1)[17].item()
    numeric = finite_difference(lambda: (forward() * upstream).sum(), x, 17)
    assert_close("transformer d_x finite difference", analytic, numeric)
    out["transformer.fd_dx17"] = np.array([analytic, numeric])


def loss_references(out):
    logits = hashed((5, 7), 20, 2.5)
    targets = torch.tensor([6, 0, -100, 3, 3], dtype=torch.int64)
    out["loss.input.logits"] = logits.detach().numpy()
    out["loss.input.targets"] = targets.numpy().astype(np.float64)
    for smoothing in (0.0, 0.17):
        logits.grad = None
        loss = F.cross_entropy(
            logits, targets, ignore_index=-100, reduction="mean",
            label_smoothing=smoothing,
        )
        loss.backward()
        tag = str(smoothing).replace(".", "p")
        out[f"loss.ce_{tag}"] = np.array(loss.item())
        out[f"loss.d_logits_{tag}"] = logits.grad.numpy().copy()
        # Ignored rows must be exactly zero; duplicated class ids must not be
        # conflated with duplicated positions.
        if not torch.equal(logits.grad[2], torch.zeros_like(logits.grad[2])):
            raise AssertionError("cross entropy ignore_index produced a gradient")
    analytic = logits.grad.reshape(-1)[24].item()
    numeric = finite_difference(
        lambda: F.cross_entropy(logits, targets, ignore_index=-100,
                                reduction="mean", label_smoothing=0.17),
        logits, 24,
    )
    assert_close("cross entropy finite difference", analytic, numeric)
    out["loss.fd_logit24"] = np.array([analytic, numeric])


def embedding_references(out):
    weight = hashed((6, 5), 30, 0.9)
    ids = torch.tensor([[4, 1, 4, 0], [2, 4, 3, 1]], dtype=torch.int64)
    upstream = hashed((2, 4, 5), 31, 1.1, grad=False)
    out["embedding.input.weight"] = weight.detach().numpy()
    out["embedding.input.ids"] = ids.numpy().astype(np.float64)
    out["embedding.input.d_out"] = upstream.numpy()
    y = F.embedding(ids, weight, padding_idx=0)
    (y * upstream).sum().backward()
    out["embedding.output"] = y.detach().numpy()
    out["embedding.d_weight"] = weight.grad.numpy()
    expected = torch.zeros_like(weight)
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            if ids[i, j] != 0:
                expected[ids[i, j]] += upstream[i, j]
    if not torch.equal(weight.grad, expected):
        raise AssertionError("embedding repeated-index accumulation disagrees with closed form")
    if not torch.equal(weight.grad[0], torch.zeros_like(weight.grad[0])):
        raise AssertionError("embedding padding row received a gradient")


def optimizer_references(out):
    initial = tensor([0.75, -1.25, 0.0, -0.0], (4,), grad=True)
    grad = torch.tensor([0.2, -0.4, 1e-12, -1e-12], dtype=torch.float64)
    out["optimizer.input.parameter"] = initial.detach().numpy()
    out["optimizer.input.gradient"] = grad.numpy()

    p = torch.nn.Parameter(initial.detach().clone())
    opt = torch.optim.SGD([p], lr=0.125, momentum=0.9, dampening=0,
                          weight_decay=0.03, nesterov=True)
    p.grad = grad.clone()
    opt.step()
    out["optimizer.sgd.parameter"] = p.detach().numpy()
    out["optimizer.sgd.momentum"] = opt.state[p]["momentum_buffer"].numpy()
    p0, g0 = initial.detach().numpy(), grad.numpy()
    sgd_buf = g0 + 0.03 * p0
    sgd_closed = p0 - 0.125 * (g0 + 0.03 * p0 + 0.9 * sgd_buf)
    if not np.allclose(p.detach().numpy(), sgd_closed, atol=1e-15, rtol=1e-15):
        raise AssertionError("torch SGD disagrees with independent closed form")
    out["optimizer.sgd.closed_form"] = sgd_closed

    p = torch.nn.Parameter(initial.detach().clone())
    opt = torch.optim.AdamW([p], lr=0.003, betas=(0.8, 0.95), eps=1e-8,
                            weight_decay=0.07, amsgrad=False)
    adam_p = p0.copy()
    adam_m = np.zeros_like(p0)
    adam_v = np.zeros_like(p0)
    for step, g in enumerate((grad, -0.3 * grad), start=1):
        p.grad = g.clone()
        opt.step()
        out[f"optimizer.adamw.step{step}.parameter"] = p.detach().numpy().copy()
        out[f"optimizer.adamw.step{step}.exp_avg"] = opt.state[p]["exp_avg"].numpy().copy()
        out[f"optimizer.adamw.step{step}.exp_avg_sq"] = opt.state[p]["exp_avg_sq"].numpy().copy()
        gn = g.numpy()
        adam_p *= 1.0 - 0.003 * 0.07
        adam_m = 0.8 * adam_m + 0.2 * gn
        adam_v = 0.95 * adam_v + 0.05 * gn * gn
        adam_p -= 0.003 * (adam_m / (1.0 - 0.8 ** step)) / (
            np.sqrt(adam_v / (1.0 - 0.95 ** step)) + 1e-8
        )
        if not np.allclose(p.detach().numpy(), adam_p, atol=1e-15, rtol=1e-15):
            raise AssertionError(f"torch AdamW step {step} disagrees with closed form")
        out[f"optimizer.adamw.step{step}.closed_form"] = adam_p.copy()
    if int(opt.state[p]["step"].item()) != 2:
        raise AssertionError("AdamW step state did not advance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("build/independent_gradient_oracles.npz"))
    parser.add_argument("--raw-dir", type=Path,
                        help="also write language-neutral little-endian float64 files")
    parser.add_argument("--compare-dir", type=Path,
                        help="compare .f32/.f64 Mojo exports against this run")
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--family", choices=("all", "transformer", "loss", "embedding", "optimizer"),
                        default="all", help="limit --compare-dir to one family")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    refs = {}
    transformer_reference(refs)
    loss_references(refs)
    embedding_references(refs)
    optimizer_references(refs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **refs)
    manifest = {
        "format": "mojolearn.independent-gradient-oracles.v1",
        "producer": "PyTorch CPU float64 autograd",
        "torch": torch.__version__,
        "seed": SEED,
        "arrays": {k: list(np.asarray(v).shape) for k, v in sorted(refs.items())},
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.raw_dir:
        write_raw_bundle(args.raw_dir, refs)
    if args.compare_dir:
        compare_bundle(refs, args.compare_dir, args.atol, args.rtol, args.family)
    print(f"PASS independent gradient oracles: {len(refs)} arrays -> {args.out}")


if __name__ == "__main__":
    main()
