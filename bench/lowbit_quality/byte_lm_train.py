#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Training the byte LM at a small shape under each candidate arithmetic.

Lane lane/lowbit-quality, 2026-09-29. Runs on the pod, never on the laptop.
A reference simulation in PyTorch: it answers what the ARITHMETIC does to
the loss a training run reaches, and says nothing about time.

THE MODEL is the byte LM of `training/byte_lm_config.mojo` and the graph of
`tools/byte_lm_gradient_oracle.py::reference`: embedding, `n_layers` decoder
blocks (RMSNorm epsilon 1e-6, RoPE theta 10000, grouped-query attention,
SwiGLU), an untied head, no final norm, no bias, vocabulary 256, a byte is
its own id. The registry order and the initializer
(`training/byte_lm_init.mojo`, identifier
u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1) are the
repo's; a seed changes the XOR constant of the draw and the order of the
training windows.

THE SHAPE (default): batch 32, length 128, width 128, 4 query heads over 2
key/value heads of width 32, SwiGLU width 256, 4 blocks.

WHAT STAYS FLOAT32 IN EVERY ARM: the master weights, the AdamW state and
update (lr 0.003, betas 0.9 and 0.999, epsilon 1e-8, weight decay 0.01 on
every tensor, as `tools/byte_lm_real_text_capture.py` sets them), the loss,
and every operation that is not a matrix product.

THE ARMS. Each product is OP_NT, `C = A @ B^T` (`arith.py`).
  forward only       the forward product runs in the candidate's arithmetic;
                     the two backward products are float32 products of the
                     saved float32 operands (a straight-through gradient).
  forward+backward   the backward products run in the candidate's
                     arithmetic too, each in the orientation its GEMM has:
                       dA = G @ B      = NT(G, B^T)    rows of G along n,
                                                       rows of B^T along n
                       dB = G^T @ A    = NT(G^T, A^T)  rows of both along m
                     so a weight gradient quantizes `G^T` (one row per output
                     feature, over every token of the batch) and `A^T` (one
                     row per input feature, over every token). The incoming
                     gradient `G` takes the activation kind.
  attention          as a separate switch, the QK and PV products and their
                     backward products follow the same rule.

DATA, from the R2 dataset store only: `corpus/enwik8/input.txt`. Training
windows start at bytes drawn uniformly from [0, 90,000,000 - length - 1).
Validation is 512 fixed windows, evenly spaced from byte 95,000,000, which
no training window touches.

WHAT A RUN RECORDS: the validation loss every `--eval-every` steps under
the ARM's own forward arithmetic (the profile a checkpoint trained under it
would be loaded with), and at the equal-step point and the end also under
the float32 forward.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import arith  # noqa: E402
from arith import Spec  # noqa: E402

SHAPE = dict(batch=32, length=128, d_model=128, n_heads=4, n_kv=2, head_dim=32,
             intermediate=256, n_layers=4, vocab_size=256)
OPTIMIZER = dict(kind="AdamW", lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
TRAIN_END = 90_000_000
VAL_START = 95_000_000
VAL_WINDOWS = 512
CLAIM_STALE_SECONDS = 3600
SEED_XOR_DEFAULT = 0x42595445
INIT_ID = "u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1"

ARMS = {
    "a": dict(name="fp32.v1", w="fp32", a="fp32"),
    "b": dict(name="bf16f32.v1", w="bf16", a="fp32"),
    "c": dict(name="bf16-both", w="bf16", a="bf16"),
    "d": dict(name="int8i32.v1", w="int8", a="int8"),
    "e": dict(name="int15-both", w="int15", a="int15"),
    "f": dict(name="int15w-int8a", w="int15", a="int8"),
    # ---- follow-up arms, their own names
    "int8w-int15a": dict(name="int8w-int15a", w="int8", a="int15"),
    "int8w-fp32a": dict(name="int8w-fp32a", w="int8", a="fp32"),
    "int8m-both": dict(name="int8m-both", w="int8m", a="int8m"),
    "int8w-int12a": dict(name="int8w-int12a", w="int8", a="int12"),
    # FINALIST F2 (orchestrator, 2026-09-29 03:35Z): 15-bit codes on every projection, int8 codes on
    # every attention product, as ONE configuration. FINALIST F1 is arm e with the attention switch on.
    "F2": dict(name="F2-int15proj-int8attn", w="int15", a="int15",
               overrides={"attn_qk": ("int8", "int8"), "attn_pv": ("int8", "int8")}),
    "int12": dict(name="int12-both", w="int12", a="int12"),
    "int10": dict(name="int10-both", w="int10", a="int10"),
}

#: The order the runs are started in (orchestrator, 2026-09-29): the noise
#: floor first, then the arms in the order their verdicts are wanted.
#: (arm, mode, attention products)
ORDER = (
    # Andrew, 2026-09-29: native bf16 on the matrix units is the first priority, so whether bf16
    # survives training comes first: bf16 on both operands, forward and backward, then forward
    # only; then 15-bit the same way. Complete configurations (attention included) before the
    # projections alone. Then finalist F2, the mid widths, and every arm that was already
    # planned: int8 and the mixes are not offered by the flag, and are still measured to the end.
    [("c", "fwdbwd", True), ("c", "fwd", True), ("e", "fwdbwd", True), ("e", "fwd", True),
     ("c", "fwdbwd", False), ("c", "fwd", False), ("e", "fwdbwd", False), ("e", "fwd", False),
     ("F2", "fwd", True), ("F2", "fwdbwd", True),
     ("int12", "fwdbwd", False), ("int12", "fwd", False), ("int10", "fwdbwd", False), ("int10", "fwd", False),
     ("d", "fwd", False), ("d", "fwdbwd", False), ("f", "fwd", False), ("f", "fwdbwd", False),
     ("int8w-int15a", "fwd", False), ("int8w-int15a", "fwdbwd", False),
     ("b", "fwdbwd", False), ("b", "fwd", False),
     ("d", "fwd", True), ("d", "fwdbwd", True)])

#: The widths a gradient operand is coded under for the zero-code record.
ZERO_CODE_WIDTHS = ("int8", "int10", "int12", "int15")

#: When a dict, `QMatmulNT.backward` records into it, per product, the
#: fraction of entries of each backward operand whose code is 0 under each
#: width. Set only around the one diagnostic pass.
COLLECT = None


def zero_code_record(x):
    """For one backward operand, rows along its last extent as its GEMM
    sees them: how many entries are exactly zero as float32, and under each
    width the fraction of ALL entries, and of the NONZERO entries, whose
    code is 0."""
    nonzero = x != 0
    n, nz = x.numel(), int(nonzero.sum().item())
    out = dict(entries=n, exactly_zero_fraction=1.0 - nz / n)
    # bf16 keeps an exponent PER VALUE, so only a value below the smallest normal (flushed by the
    # narrowing seam, L-2) rounds to zero, whatever the rest of its row holds.
    bzero = arith.round_bf16(x) == 0
    out["bf16"] = dict(zero_code_fraction=float(bzero.double().mean().item()),
                       zero_code_fraction_of_nonzero=(float((bzero & nonzero).sum().item()) / nz) if nz else 0.0)
    for kind in ZERO_CODE_WIDTHS:
        codes, _ = arith.quantize_rows(x, kind)
        zero = codes == 0
        out[kind] = dict(zero_code_fraction=float(zero.double().mean().item()),
                         zero_code_fraction_of_nonzero=(float((zero & nonzero).sum().item()) / nz) if nz else 0.0)
    return out


def source_sha256():
    """The sha256 of the source files this record was computed by, read at
    run time, so a record names its code even when the tree on the box was
    synced after the job that wrote it had started."""
    here = os.path.dirname(os.path.abspath(__file__))
    return {name: hashlib.sha256(open(os.path.join(here, name), "rb").read()).hexdigest()
            for name in ('arith.py', 'byte_lm_train.py')}


def profile(shape):
    return ("mojolearn.byte-lm.b{batch}-l{length}-d{d_model}-h{n_heads}-kv{n_kv}-hd{head_dim}"
            "-ff{intermediate}-v{vocab_size}-blocks{n_layers}.fp32.v3").format(**shape)


def registry(shape):
    dm, hd, ff = shape["d_model"], shape["head_dim"], shape["intermediate"]
    heads, kv, vocab = shape["n_heads"], shape["n_kv"], shape["vocab_size"]
    out = [("embed", (vocab, dm))]
    for b in range(shape["n_layers"]):
        out += [(f"block{b}.{n}", s) for n, s in (
            ("norm1_w", (dm,)), ("w_q", (heads * hd, dm)), ("w_k", (kv * hd, dm)), ("w_v", (kv * hd, dm)),
            ("w_o", (dm, heads * hd)), ("norm2_w", (dm,)), ("w_gate", (ff, dm)), ("w_up", (ff, dm)),
            ("w_down", (dm, ff)))]
    out.append(("lm_head", (vocab, dm)))
    return out


def fmix32_scalar(index, seed_xor):
    h = (index + 1) ^ seed_xor
    h = (h ^ (h >> 16)) & 0xffffffff
    h = (h * 0x85ebca6b) & 0xffffffff
    h = (h ^ (h >> 13)) & 0xffffffff
    h = (h * 0xc2b2ae35) & 0xffffffff
    h = (h ^ (h >> 16)) & 0xffffffff
    return ((h >> 24) - 128) / 1024.0


def initialize(shape, seed_xor):
    """`byte_init_params`: the draw over the FLAT index, then the norm
    vectors overwritten with 1.0."""
    reg = registry(shape)
    total = sum(int(np.prod(s)) for _, s in reg)
    m = np.uint64(0xffffffff)
    h = (np.arange(1, total + 1, dtype=np.uint64) ^ np.uint64(seed_xor)) & m
    h = (h ^ (h >> np.uint64(16))) & m
    h = (h * np.uint64(0x85ebca6b)) & m
    h = (h ^ (h >> np.uint64(13))) & m
    h = (h * np.uint64(0xc2b2ae35)) & m
    h = (h ^ (h >> np.uint64(16))) & m
    flat = ((h >> np.uint64(24)).astype(np.int64) - 128).astype(np.float32) / np.float32(1024.0)
    for i in list(range(64)) + [total // 2, total - 1]:
        if flat[i] != np.float32(fmix32_scalar(i, seed_xor)):
            raise AssertionError(f"vectorized draw differs from the scalar spelling at index {i}")
    out, offset = {}, 0
    for name, s in reg:
        n = int(np.prod(s))
        block = flat[offset:offset + n].reshape(s).copy()
        if name.endswith(("norm1_w", "norm2_w")):
            block[...] = 1.0
        out[name] = block
        offset += n
    return out, total, hashlib.sha256(b"".join(out[n].tobytes() for n, _ in reg)).hexdigest()


def seed_xor_of(seed):
    return (SEED_XOR_DEFAULT ^ ((seed * 0x9E3779B9) & 0xffffffff)) & 0xffffffff


class QMatmulNT(torch.autograd.Function):
    """`A @ B^T` with the forward in kinds `(ka, kb)` and the backward either
    float32 on the saved operands or in the candidate's arithmetic."""

    @staticmethod
    def forward(ctx, A, B, ka, kb, quant_backward, sabotage, name):
        ctx.save_for_backward(A, B)
        ctx.meta = (ka, kb, quant_backward, sabotage, name)
        return arith.product_nt(A, B, ka, kb)

    @staticmethod
    def backward(ctx, G):
        A, B = ctx.saved_tensors
        ka, kb, quant_backward, sabotage, name = ctx.meta
        Gt = G.transpose(-1, -2)
        if COLLECT is not None:
            COLLECT[name] = dict(
                shape_m_n_k=[int(G.shape[-2]), int(G.shape[-1]), int(A.shape[-1])],
                G_rows_along_n=zero_code_record(G.contiguous()),
                Gt_rows_along_m=zero_code_record(Gt.contiguous()),
                At_rows_along_m=zero_code_record(A.transpose(-1, -2).contiguous()),
                Bt_rows_along_n=zero_code_record(B.transpose(-1, -2).contiguous()))
        if quant_backward:
            kg = ka
            dA = arith.product_nt(G.contiguous(), B.transpose(-1, -2).contiguous(), kg, kb)
            dB = arith.product_nt(Gt.contiguous(), A.transpose(-1, -2).contiguous(), kg, ka)
        else:
            dA = G @ B
            dB = Gt @ A
        if sabotage:
            dB = -dB
        return dA, dB, None, None, None, None, None


class ByteLM:
    def __init__(self, shape, init, device):
        self.shape, self.device = shape, device
        self.p = {n: torch.nn.Parameter(torch.tensor(v, device=device)) for n, v in init.items()}
        self.eps = float(np.float32(1e-6))
        hd, length = shape["head_dim"], shape["length"]
        inv = 10000.0 ** (-torch.arange(0, hd, 2, dtype=torch.float64, device=device) / hd)
        angle = torch.arange(length, dtype=torch.float64, device=device)[:, None] * inv
        angle = torch.cat((angle, angle), dim=-1)[None, None]
        self.cos, self.sin = angle.cos().to(torch.float32), angle.sin().to(torch.float32)
        self.mask = torch.ones((length, length), dtype=torch.bool, device=device).triu(1)

    def parameters(self):
        return list(self.p.values())

    def _product(self, A, B, product, spec, mode):
        if mode == "plain":
            return A @ B.transpose(-1, -2)
        ka, kb = spec.kinds(product)
        return QMatmulNT.apply(A, B, ka, kb, mode == "fwdbwd", mode == "sabotage", product)

    def logits(self, ids, spec, mode):
        """`ids [B, L]` -> float32 logits `[B, L, vocab]`. `mode` is
        "fwd" (forward only), "fwdbwd", "plain" (torch's own matmul and
        autograd) or "sabotage" (the self-test's arm that must fail)."""
        s = self.shape
        bsz, length = ids.shape
        dm, heads, kv, hd = s["d_model"], s["n_heads"], s["n_kv"], s["head_dim"]

        def linear(x, name):
            lead = x.shape[:-1]
            out = self._product(x.reshape(-1, x.shape[-1]), self.p[name], name, spec, mode)
            return out.reshape(*lead, out.shape[-1])

        def norm(x, name):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.p[name]

        def rotate(a):
            half = torch.cat((-a[..., hd // 2:], a[..., :hd // 2]), dim=-1)
            return a * self.cos[:, :, :length] + half * self.sin[:, :, :length]

        h = self.p["embed"][ids]
        for b in range(s["n_layers"]):
            pre = f"block{b}."
            z = norm(h, pre + "norm1_w")
            q = linear(z, pre + "w_q").reshape(bsz, length, heads, hd).transpose(1, 2)
            k = linear(z, pre + "w_k").reshape(bsz, length, kv, hd).transpose(1, 2)
            v = linear(z, pre + "w_v").reshape(bsz, length, kv, hd).transpose(1, 2)
            q, k = rotate(q), rotate(k)
            k = k.repeat_interleave(heads // kv, dim=1)
            v = v.repeat_interleave(heads // kv, dim=1)
            scores = self._product(q, k, pre + "attn_qk", spec, mode) / math.sqrt(hd)
            prob = scores.masked_fill(self.mask[:length, :length], float("-inf")).softmax(-1)
            att = self._product(prob, v.transpose(-1, -2), pre + "attn_pv", spec, mode)
            att = att.transpose(1, 2).reshape(bsz, length, dm)
            res = h + linear(att, pre + "w_o")
            z = norm(res, pre + "norm2_w")
            gate = linear(z, pre + "w_gate")
            h = res + linear(gate * gate.sigmoid() * linear(z, pre + "w_up"), pre + "w_down")
        return linear(h, "lm_head")

    def loss(self, rows, spec, mode):
        """`rows [B, L+1]`: inputs the first L, targets the next L."""
        logits = self.logits(rows[:, :-1], spec, mode)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1))


def load_corpus(path, device):
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    return torch.tensor(np.frombuffer(raw, dtype=np.uint8).copy(), device=device), len(raw), sha


def windows(corpus, starts, length):
    idx = starts[:, None] + torch.arange(length + 1, device=corpus.device)[None, :]
    return corpus[idx].to(torch.int64)


def validation_starts(shape, corpus_bytes, device):
    width = shape["length"] + 1
    stride = (corpus_bytes - VAL_START - width) // VAL_WINDOWS
    return VAL_START + stride * torch.arange(VAL_WINDOWS, device=device), stride


@torch.no_grad()
def validate(model, corpus, starts, spec, mode="fwd"):
    total, count = 0.0, 0
    for lo in range(0, starts.numel(), 128):
        rows = windows(corpus, starts[lo:lo + 128], model.shape["length"])
        logits = model.logits(rows[:, :-1], spec, mode).to(torch.float64)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1), reduction="sum")
        total += nll.item()
        count += rows[:, 1:].numel()
    return total / count


def gradient_of(model, rows, spec, mode):
    loss = model.loss(rows, spec, mode)
    grads = torch.autograd.grad(loss, model.parameters())
    return torch.cat([g.reshape(-1).to(torch.float64) for g in grads]), loss.item()


def zero_codes_of(model, rows, spec, mode):
    """One diagnostic backward pass under the run's own arithmetic, with the
    collector on: the gradient operands the backward GEMMs would be handed."""
    global COLLECT
    COLLECT = {}
    try:
        gradient_of(model, rows, spec, mode)
        return COLLECT
    finally:
        COLLECT = None


def cosine(a, b):
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-300)).item()


def make_spec(arm, attn):
    a = ARMS[arm]
    if "overrides" in a:
        if not attn:
            raise ValueError(f"arm {arm} is a complete configuration; it has no projections-only form")
        return Spec(a["name"], w=a["w"], a=a["a"], attn=True, overrides=a["overrides"])
    return Spec(a["name"] + ("+attn" if attn else ""), w=a["w"], a=a["a"], attn=attn)


FP32 = Spec("fp32.v1")


def run_one(args, arm, mode, attn, seed, corpus, corpus_bytes, corpus_sha, device):
    tag = f"{arm}.{mode}.{'attn' if attn else 'proj'}.s{seed}"
    path = os.path.join(args.out, f"run_{tag}.json")
    if os.path.exists(path):
        print("have", tag, flush=True)
        return
    # Two jobs may hold the same run. The first to start it claims it; a claim older than
    # CLAIM_STALE_SECONDS with no record is a run that died, and is taken over.
    claim = os.path.join(args.out, f"run_{tag}.claim")
    try:
        if os.path.exists(claim) and time.time() - os.path.getmtime(claim) > CLAIM_STALE_SECONDS:
            os.remove(claim)
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, ("job=%s pid=%d utc=%s\n" % (os.environ.get("NVQ_JOB_ID"), os.getpid(),
                                                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))).encode())
        os.close(fd)
    except FileExistsError:
        print("claimed by another worker, not repeated:", tag, open(claim).read().strip(), flush=True)
        return
    shape = dict(SHAPE)
    spec = make_spec(arm, attn)
    init, total, init_sha = initialize(shape, seed_xor_of(seed))
    model = ByteLM(shape, init, device)
    opt = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER["lr"], betas=(OPTIMIZER["beta1"], OPTIMIZER["beta2"]),
                            eps=OPTIMIZER["eps"], weight_decay=OPTIMIZER["weight_decay"])
    gen = torch.Generator().manual_seed(1000 + seed)
    starts = torch.randint(0, TRAIN_END - shape["length"] - 1, (args.steps, shape["batch"]), generator=gen)
    schedule_sha = hashlib.sha256(starts.numpy().astype("<i8").tobytes()).hexdigest()
    starts = starts.to(device)
    vstarts, vstride = validation_starts(shape, corpus_bytes, device)
    trace, fp32_eval, grad_cos, zero_codes, nonfinite = [], {}, {}, {}, None
    t0 = time.time()

    def evaluate(step):
        v = validate(model, corpus, vstarts, spec)
        trace.append(dict(step=step, val_loss=v))
        if step in (args.equal_steps, args.steps):
            fp32_eval[str(step)] = validate(model, corpus, vstarts, FP32)
        return v

    evaluate(0)
    for step in range(args.steps):
        rows = windows(corpus, starts[step], shape["length"])
        if step in (0, args.equal_steps // 2, args.equal_steps - 1):
            zero_codes[str(step)] = zero_codes_of(model, rows, spec, mode)
            if arm != "a":
                ref, _ = gradient_of(model, rows, FP32, "fwd")
                got, _ = gradient_of(model, rows, spec, mode)
                grad_cos[str(step)] = dict(cosine=cosine(ref, got), norm_ratio=(got.norm() / ref.norm()).item())
        opt.zero_grad(set_to_none=True)
        loss = model.loss(rows, spec, mode)
        loss.backward()
        opt.step()
        if not math.isfinite(loss.item()):
            nonfinite = step
            break
        if (step + 1) % args.eval_every == 0:
            v = evaluate(step + 1)
            if (step + 1) % (args.eval_every * 10) == 0:
                print(tag, "step", step + 1, "train", round(loss.item(), 4), "val", round(v, 4), flush=True)
    record = dict(
        schema="mojolearn.lowbit_quality.training_run.v1", commit=args.commit, tag=tag,
        source_sha256=source_sha256(),
        arm=arm, profile_name=spec.name, mode=mode, attention_products=attn, seed=seed,
        seed_xor=seed_xor_of(seed), spec=spec.describe(), shape=shape, model_profile=profile(shape),
        parameters=total, initialization=INIT_ID, initial_parameters_sha256=init_sha,
        optimizer=OPTIMIZER, steps=args.steps, equal_steps=args.equal_steps, eval_every=args.eval_every,
        data=dict(corpus_key="corpus/enwik8/input.txt", corpus_bytes=corpus_bytes, corpus_sha256=corpus_sha,
                  train_window_starts="uniform in [0, %d)" % (TRAIN_END - shape["length"] - 1),
                  train_schedule_sha256=schedule_sha, validation_first_byte=VAL_START,
                  validation_windows=VAL_WINDOWS, validation_stride=int(vstride),
                  validation_targets=VAL_WINDOWS * shape["length"]),
        trace=trace, fp32_forward_val_loss=fp32_eval, gradient_against_fp32=grad_cos,
        gradient_zero_codes=zero_codes,
        nonfinite_at_step=nonfinite, seconds=time.time() - t0,
        library=dict(torch=torch.__version__, device_name=torch.cuda.get_device_name(0) if device == "cuda" else None),
    )
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=1)
    os.replace(tmp, path)
    at = {t["step"]: t["val_loss"] for t in trace}
    print("DONE", tag, "val@%d" % args.equal_steps, at.get(args.equal_steps), "val@%d" % args.steps,
          at.get(args.steps), "nonfinite", nonfinite, "seconds", round(record["seconds"], 1), flush=True)


def plan(args):
    runs = [("a", "fwd", False, s) for s in range(args.baseline_seeds)]
    if args.only:
        for item in args.only.split(","):
            arm, mode, where = item.split(":")
            runs += [(arm, mode, where == "attn", s) for s in range(args.seeds)]
        return runs
    wanted = set(a for a in args.arms.split(",") if a)
    for arm, mode, attn in ORDER:
        if arm in wanted:
            runs += [(arm, mode, attn, s) for s in range(args.seeds)]
    return runs


def self_test(args, corpus, device):
    """The custom products against torch's own matmul and autograd, with an
    arm that must fail. Float32 kinds, so the only difference a correct
    implementation can show is the order of float32 additions."""
    shape = dict(SHAPE)
    init, _, _ = initialize(shape, seed_xor_of(0))
    model = ByteLM(shape, init, device)
    starts = torch.arange(shape["batch"], device=device) * 1000
    rows = windows(corpus, starts, shape["length"])
    ref, ref_loss = gradient_of(model, rows, FP32, "plain")
    out = dict(gate=dict(min_cosine=0.999999, max_rel_norm_error=1e-4))

    def one(mode):
        got, loss = gradient_of(model, rows, FP32, mode)
        return dict(cosine=cosine(ref, got), rel_norm_error=((got - ref).norm() / ref.norm()).item(),
                    loss_delta=loss - ref_loss)

    def passes(r):
        return r["cosine"] >= out["gate"]["min_cosine"] and r["rel_norm_error"] <= out["gate"]["max_rel_norm_error"]

    out["forward_only_path"] = one("fwd")
    out["forward_backward_path"] = one("fwdbwd")
    out["sabotage_negated_weight_gradient"] = one("sabotage")
    # the backward products at 15 bits and in bf16, attention included, against float32: a
    # direction check on the orientation of the backward products, reported, not gated
    for arm in ("e", "c"):
        got, _ = gradient_of(model, rows, make_spec(arm, True), "fwdbwd")
        out[f"arm_{arm}_fwdbwd_attn_gradient"] = dict(cosine=cosine(ref, got),
                                                     norm_ratio=(got.norm() / ref.norm()).item())
    out["verdict"] = "PASS" if (passes(out["forward_only_path"]) and passes(out["forward_backward_path"])
                                and not passes(out["sabotage_negated_weight_gradient"])) else "FAIL"
    with open(os.path.join(args.out, "self_test.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("self_test", json.dumps(out), flush=True)
    return out["verdict"] == "PASS"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="c,e,F2,int12,int10,d,f,int8w-int15a,b")
    ap.add_argument("--only", default=None,
                    help="an explicit list in place of the plan: arm:mode:proj|attn, comma separated")
    ap.add_argument("--baseline-seeds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--equal-steps", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args(argv)
    runs = plan(args)
    if args.print_plan:
        for r in runs:
            print(*r)
        return 0
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    corpus, corpus_bytes, corpus_sha = load_corpus(args.corpus, args.device)
    if args.self_test:
        return 0 if self_test(args, corpus, args.device) else 1
    for arm, mode, attn, seed in runs[args.worker::args.workers]:
        run_one(args, arm, mode, attn, seed, corpus, corpus_bytes, corpus_sha, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
