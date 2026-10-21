#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Export the three products of a training step under the 15-bit profile,
as the training simulation computed them, for Lane C's host oracle.

Lane lane/lowbit-quality, 2026-09-29. Runs on a pod, never on the laptop.

    python3 bench/lowbit_quality/backward_export.py --corpus <enwik8> --out <dir> [--device cpu]

THE RULE BEING EXPORTED (brief, 2026-09-29 06:50Z): every product quantizes
its own operands from their float32 values, along that product's own
contracted extent. Codes are never carried between products and never
transposed. Per product of the model, three cases, each an OP_NT product
`C = A @ B^T` whose operands are the float32 tensors `QMatmulNT` handed to
`arith.product_nt`:

    forward          A = X        B = W        C = Y
    weight gradient  A = dY^T     B = X^T      C = dW
    input gradient   A = dY       B = W^T      C = dX

WHAT IS WRITTEN is what the simulation computed, not a recomputation: `C`
is cut from the tensor the training step used. The operands are cut to a
few rows each (a row's scale is its own, so a product over a slice of rows
is the same cells of the whole); the contracted extent is never cut, so a
weight gradient keeps all 4096 tokens.

THE FILE is Lane C's (`bench/lowbit_quality/int15_export.py` on
`lane/lowbit-int15`), so its reader reads it unchanged: little-endian
32-bit words; magic 0x35314951; version 1; the number of cases; the git
blob hash of `arith.py` in five words; per case `m, n, k`, A bits, B bits,
A codes, A exponents, B codes, B exponents, C bits. A manifest beside it
names every case.

TWO CHECKS, each with an arm that must fail:
  - the product of the cut operands, recomputed, equals the cut of what the
    training step computed, bit for bit; with the rounding sabotaged it
    must not;
  - THE RULE ITSELF, per case. THE OTHER RULE quantizes each tensor ONCE, in
    the orientation of the first product that reads it, and carries the
    codes, transposed, into the products that contract another extent:
        X   coded by token (the forward product's rows), carried into the
            weight gradient;
        W   coded by output feature (the forward product's rows), carried
            into the input gradient;
        dY  coded by token (the input gradient's rows), carried into the
            weight gradient.
    Under it the input gradient's right operand and both operands of the
    weight gradient carry a scale per CELL of the contraction, so its sum
    is not an integer sum; it is computed here in float64 and rounded once,
    as a comparison arm and nothing else. The manifest says, per backward
    case, in how many cells the two rules give different bits. A case where
    they give the same bits cannot check the rule. The forward product is
    where a carried code comes from, so the two rules are the same product
    there and the manifest says "not applicable".

THE LINK TO A MEASURED RUN. The exported step belongs to the trajectory of
training run `e.fwdbwd.attn.s<seed>`: the same initializer, schedule,
optimizer and arithmetic. `--validate` computes the validation loss at the
exported step the way the run did, so the two can be compared.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import arith  # noqa: E402
import byte_lm_train as T  # noqa: E402

MAGIC = 0x35314951
VERSION = 1
KIND = "int15"
PRODUCTS = ("block0.w_q", "block1.w_gate", "block3.w_down", "lm_head", "block1.attn_qk", "block2.attn_pv")


def blob_hash(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def words(t):
    return t.contiguous().view(torch.int32).reshape(-1).tolist()


def first2d(t):
    """The first matrix of a tensor with leading extents (batch, head)."""
    while t.dim() > 2:
        t = t[0]
    return t


def bits_differ(a, b):
    a, b = a.contiguous(), b.contiguous()
    both_nan = torch.isnan(a) & torch.isnan(b)
    return int(((a.view(torch.int32) != b.view(torch.int32)) & ~both_nan).sum().item())


def scaled_codes(x):
    """A tensor coded ONCE along its own rows, as float64 `code * 2^e`: what
    the other rule carries from one product into another."""
    codes, e = arith.quantize_rows(x.contiguous(), KIND)
    return codes * torch.ldexp(torch.ones_like(codes[..., 0]), e.to(torch.int64)).unsqueeze(-1)


def other_rule(which, X, W, dY, ra, rb):
    """The cut `[ra, rb]` of a backward product under THE OTHER RULE, float32."""
    if which == "weight gradient":      # dW = dY^T X, both carried: dY coded by token, X coded by token
        full = scaled_codes(dY).T[:ra] @ scaled_codes(X)[:, :rb]
    elif which == "input gradient":     # dX = dY W, W carried: coded by output feature
        full = scaled_codes(dY[:ra]) @ scaled_codes(W)[:, :rb]
    else:
        return None
    return arith.ftz(full.to(torch.float32))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50, help="training steps before the exported one")
    ap.add_argument("--schedule-steps", type=int, default=6000,
                    help="the length of the training schedule the run drew, so the windows are the run's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="int15_backward_vectors")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    device = args.device
    shape = dict(T.SHAPE)
    spec = T.make_spec("e", True)
    init, total, init_sha = T.initialize(shape, T.seed_xor_of(args.seed))
    model = T.ByteLM(shape, init, device)
    opt = torch.optim.AdamW(model.parameters(), lr=T.OPTIMIZER["lr"],
                            betas=(T.OPTIMIZER["beta1"], T.OPTIMIZER["beta2"]),
                            eps=T.OPTIMIZER["eps"], weight_decay=T.OPTIMIZER["weight_decay"])
    corpus, corpus_bytes, corpus_sha = T.load_corpus(args.corpus, device)
    gen = torch.Generator().manual_seed(1000 + args.seed)
    n_schedule = max(args.schedule_steps, args.steps + 1)
    starts = torch.randint(0, T.TRAIN_END - shape["length"] - 1, (n_schedule, shape["batch"]), generator=gen)
    starts = starts.to(device)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(T.windows(corpus, starts[step], shape["length"]), spec, "fwdbwd")
        loss.backward()
        opt.step()
    val_loss = None
    if args.validate:
        vstarts, _ = T.validation_starts(shape, corpus_bytes, device)
        val_loss = T.validate(model, corpus, vstarts, spec)
        print("validation loss at step", args.steps, "under the profile:", repr(val_loss), flush=True)
    T.EXPORT = []
    opt.zero_grad(set_to_none=True)
    loss = model.loss(T.windows(corpus, starts[args.steps], shape["length"]), spec, "fwdbwd")
    loss.backward()
    records = {r["name"]: r for r in T.EXPORT}
    T.EXPORT = None
    print("exported step", args.steps, "loss", loss.item(), "products recorded", len(records), flush=True)

    here = blob_hash(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arith.py"))
    out = [MAGIC, VERSION, 0] + [int(here[i:i + 8], 16) for i in range(0, 40, 8)]
    manifest, ok = [], True
    rule_differs, backward_cases, separating = 0, 0, 0
    for name in PRODUCTS:
        r = records[name]
        assert r["kinds"] == (KIND, KIND) and r["quant_backward"], (name, r["kinds"])
        X, W, dY = first2d(r["X"]), first2d(r["W"]), first2d(r["dY"])
        Y, dX, dW = first2d(r["Y"]), first2d(r["dX"]), first2d(r["dW"])
        cases = (("forward", X, W, Y, None),
                 ("weight gradient", dY.T, X.T, dW, ("dY^T", "X^T")),
                 ("input gradient", dY, W.T, dX, ("dY", "W^T")))
        for which, A, B, C, _ in cases:
            ra, rb = min(args.rows, A.shape[0]), min(args.rows, B.shape[0])
            a, b, c = A[:ra].contiguous(), B[:rb].contiguous(), C[:ra, :rb].contiguous()
            pa, pb = arith.prepare(a, KIND), arith.prepare(b, KIND)
            again = arith.product_prepared(pa, pb)
            differ = bits_differ(again, c)
            sa, ea = arith.quantize_rows(a, KIND, rounding="half_away")
            sb, eb = arith.quantize_rows(b, KIND, rounding="truncate")
            sab = arith.product_prepared(arith.Operand(codes=sa, e=ea, qmax=16383),
                                         arith.Operand(codes=sb, e=eb, qmax=16383))
            sab_differ = bits_differ(sab, c)
            qa, qb = pa.codes.to(torch.int32), pb.codes.to(torch.int32)
            assert int(qa.abs().max()) <= 16383 and int(qb.abs().max()) <= 16383
            m, n, k = ra, rb, a.shape[1]
            out += [m, n, k] + words(a) + words(b)
            out += qa.reshape(-1).tolist() + pa.e.to(torch.int32).reshape(-1).tolist()
            out += qb.reshape(-1).tolist() + pb.e.to(torch.int32).reshape(-1).tolist()
            out += words(c)
            out[2] += 1
            cell = dict(case=out[2] - 1, product=name, which=which, m=m, n=n, k=k,
                        full_shape_of_A=list(A.shape), full_shape_of_B=list(B.shape),
                        A_zero_codes=float((qa == 0).double().mean()), B_zero_codes=float((qb == 0).double().mean()),
                        A_exponents=[int(pa.e.min()), int(pa.e.max())], B_exponents=[int(pb.e.min()), int(pb.e.max())],
                        recomputed_bits_differing=differ, sabotage_rounding_bits_differing=sab_differ,
                        nonfinite_cells=int((~torch.isfinite(c)).sum()))
            other = other_rule(which, X, W, dY, ra, rb)
            if other is None:
                cell["other_rule"] = "not applicable: the forward product is where a carried code comes from"
            else:
                n_diff = bits_differ(other.contiguous(), c)
                cell["other_rule"] = dict(cells=int(c.numel()), cells_with_different_bits=n_diff,
                                          separates=n_diff > 0)
                backward_cases += 1
                separating += int(n_diff > 0)
                rule_differs += n_diff
            cell["exponent_spread_of_W_rows"] = [int(arith.prepare(W, KIND).e.min()), int(arith.prepare(W, KIND).e.max())]
            ok = ok and differ == 0 and sab_differ > 0
            manifest.append(cell)
            print(json.dumps(cell), flush=True)
    path = os.path.join(args.out, args.name + ".q15")
    with open(path, "wb") as fh:
        fh.write(struct.pack(f"<{len(out)}I", *[w & 0xFFFFFFFF for w in out]))
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    record = dict(
        schema="mojolearn.lowbit_quality.backward_vectors.v1", commit=args.commit, file=os.path.basename(path),
        sha256=digest, words=len(out), cases=out[2], arith_blob=here,
        source_blobs={f: blob_hash(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                      for f in ("arith.py", "byte_lm_train.py", "backward_export.py")},
        rule="every product quantizes its own operands from their float32 values, along that product's own "
             "contracted extent; codes are never carried between products and never transposed",
        model_profile=T.profile(shape), seed=args.seed, steps_before=args.steps, loss=loss.item(),
        trajectory_of_run="e.fwdbwd.attn.s%d" % args.seed, validation_loss_at_exported_step=val_loss,
        backward_cases=backward_cases, backward_cases_separating_the_rules=separating,
        initial_parameters_sha256=init_sha, corpus_sha256=corpus_sha, device=device, torch=torch.__version__,
        rows_cut_to=args.rows, carried_codes_rule_cells_differing=rule_differs, manifest=manifest,
        verdict="PASS" if ok and rule_differs > 0 else "FAIL")
    with open(os.path.join(args.out, args.name + ".json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print("wrote", path, "words", len(out), "cases", out[2], "sha256", digest, "verdict", record["verdict"],
          "backward cases separating the two rules", separating, "of", backward_cases,
          "cells differing under the other rule", rule_differs, flush=True)
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
