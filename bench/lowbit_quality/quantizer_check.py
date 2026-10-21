#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The simulated quantizer held to `mojolearn.lowbit` and `checks/numerics.mojo`.

Lane lane/lowbit-quality, 2026-09-29. Runs on the pod.

    MOJOLEARN_HOST_DIR=<dir holding _mojolearn_linalg_host.so> PYTHONPATH=python \\
        python3 bench/lowbit_quality/quantizer_check.py --model <dir> --corpus <enwik8> --out <dir>

WHAT IS COMPARED, bit for bit, on REAL tensors of SmolLM2-360M (weights as
stored, and activations captured where they enter a product on one window
of the evaluation text) and on one planted tensor of edge cases:

  int8 codes and exponents   `arith.quantize_rows(x, "int8")` against
        `mojolearn.lowbit.pack_one(x, "int8")`, whose backend here is the
        host binding `_mojolearn_linalg_host` (the compiled seams of
        `checks/numerics.mojo`), and against the pure-Python spelling
        `mojolearn.lowbit._quantize_int8_py`.
  bf16 bits                  `arith.round_bf16` against `pack_one(x, "bfloat16")`.
  materialized values        `arith.dequantize_rows` against `materialize_one`.
  the int8 product           `arith.product_nt(A, B, "int8", "int8")` against the
        host oracle `gemm_int8_oracle` through the binding's `gemm_int8`.
  L-5                        the float64 to float32 cast against the
        20-bit/12-bit spelling of `i32_to_f32_pinned`, value by value.

THE ARMS THAT MUST FAIL. Three sabotaged quantizers (ties away from zero,
truncation, a scale one exponent off) are run through the same comparison.
Each must disagree with `mojolearn.lowbit.pack_one` on the real tensors; one
that agrees means the comparison cannot see that defect and the check is
refused. The product comparison has its own failing arm.

Exit status 0 only when every real comparison is equal and every sabotage
arm was seen to fail.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import arith  # noqa: E402
from infer_eval import ARMS, evaluation_ids  # noqa: E402
from llama_sim import LlamaSim, load_config, load_weights  # noqa: E402


def planted():
    """Edge cases the real tensors may not hold: exact ties, a zero row, a
    row of subnormals, values at the clamp, signed zeros, a huge and a tiny
    row, one large outlier."""
    rows = []
    rows.append(np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 100.0], dtype=np.float32))
    rows.append(np.zeros(8, dtype=np.float32))
    rows.append(np.array([1e-40, -1e-40, 1e-39, 0, 0, 0, 0, 0], dtype=np.float32))
    rows.append(np.array([127.5, -127.5, 127.49, 126.5, 125.5, -126.5, 64.0, 63.5], dtype=np.float32))
    rows.append(np.array([0.0, -0.0, 1.0, -1.0, 0.25, 0.75, -0.25, -0.75], dtype=np.float32))
    rows.append(np.array([3e38, -3e38, 1e38, 1e37, 1.0, 0, 0, 0], dtype=np.float32))
    rows.append(np.array([1.2e-38, -1.3e-38, 2e-38, 5e-38, 1e-37, 0, 0, 0], dtype=np.float32))
    rows.append(np.array([1e-3, 2e-3, -3e-3, 4e-3, 5e-3, -6e-3, 7e-3, 1e3], dtype=np.float32))
    rows.append(np.array([255.99998, 128.0, 129.0, 130.0, 131.0, 2.0, 6.0, 10.0], dtype=np.float32))
    return np.stack(rows)


def real_tensors(model_dir, corpus, device, tail_from, length):
    cfg = load_config(model_dir)
    weights, _ = load_weights(model_dir, device)
    ids, _ = evaluation_ids(corpus, os.path.join(model_dir, "tokenizer.json"), tail_from, 1, length)
    sim = LlamaSim(weights, cfg, ARMS["a"], device)
    sim.capture = {}
    sim.forward(ids.to(device))
    cap = sim.capture
    out = {}

    def put(name, t):
        t = t.detach().to(torch.float32).contiguous()
        assert t.dim() == 2, (name, t.shape)
        out[name] = t

    put("weight layers.0.q_proj", weights["model.layers.0.self_attn.q_proj.weight"])
    put("weight layers.15.down_proj", weights["model.layers.15.mlp.down_proj.weight"])
    put("weight layers.31.gate_proj", weights["model.layers.31.mlp.gate_proj.weight"])
    put("weight layers.7.k_proj", weights["model.layers.7.self_attn.k_proj.weight"])
    put("weight embed rows 0..2047", weights["model.embed_tokens.weight"][:2048])
    put("activation layers.0.q_proj.input", cap["layers.0.q_proj.input"])
    put("activation layers.16.down_proj.input", cap["layers.16.down_proj.input"])
    put("activation layers.31.o_proj.input", cap["layers.31.o_proj.input"])
    put("activation lm_head.input", cap["lm_head.input"])
    put("activation layers.10.attn_qk.a head 0", cap["layers.10.attn_qk.a"][0, 0])
    put("activation layers.10.attn_qk.b head 0", cap["layers.10.attn_qk.b"][0, 0])
    put("activation layers.10.attn_pv.a head 3 (P rows)", cap["layers.10.attn_pv.a"][0, 3])
    put("activation layers.10.attn_pv.b head 3 (V^T rows)", cap["layers.10.attn_pv.b"][0, 3])
    put("activation layers.31.attn_pv.a head 14 (P rows)", cap["layers.31.attn_pv.a"][0, 14])
    out["planted edge cases"] = torch.tensor(planted(), device=device)
    return out, weights, cap


def pinned_i32_to_f32(v):
    """`i32_to_f32_pinned` over numpy float32: two exact conversions, one
    exact scaling, one IEEE addition."""
    neg = v < 0
    mag = np.abs(v.astype(np.int64)).astype(np.uint64)
    hi = (mag >> np.uint64(12)).astype(np.float32)
    lo = (mag & np.uint64(0xFFF)).astype(np.float32)
    r = hi * np.float32(4096.0) + lo
    return np.where(neg, -r, r).astype(np.float32)


def check_l5(device):
    rng = np.random.default_rng(20260929)
    v = np.concatenate([
        rng.integers(-(2 ** 31) + 1, 2 ** 31 - 1, size=1_000_000, dtype=np.int64),
        np.arange(2 ** 24 - 64, 2 ** 24 + 4096, dtype=np.int64),
        -np.arange(2 ** 24 - 64, 2 ** 24 + 4096, dtype=np.int64),
        np.arange(2 ** 25 - 64, 2 ** 25 + 4096, dtype=np.int64),
        np.array([0, 1, -1, 16129 * 131072, -16129 * 131072, 2 ** 31 - 1, -(2 ** 31) + 1,
                  16129 * 2560, 16129 * 960, 3 * 2 ** 23 + 1, 3 * 2 ** 24 + 2, 3 * 2 ** 24 + 6], dtype=np.int64),
    ])
    want = pinned_i32_to_f32(v)
    got = torch.tensor(v, dtype=torch.float64, device=device).to(torch.float32).cpu().numpy()
    differ = int((want.view(np.uint32) != got.view(np.uint32)).sum())
    # the arm that must fail: truncation toward zero in place of nearest even
    trunc = np.sign(v) * (np.abs(v) >> np.maximum(0, np.floor(np.log2(np.maximum(np.abs(v), 1))).astype(np.int64) - 23)
                          << np.maximum(0, np.floor(np.log2(np.maximum(np.abs(v), 1))).astype(np.int64) - 23))
    sab = int((want.view(np.uint32) != trunc.astype(np.float32).view(np.uint32)).sum())
    return dict(values=int(v.size), differing=differ, equal=differ == 0,
                sabotage_truncate_differing=sab, sabotage_fails=sab > 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tail-from", type=int, default=99_000_000)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False

    import mojolearn
    from mojolearn import lowbit
    from mojolearn import _backend
    from mojolearn._buffer import addr, addr_ro, empty

    backend = lowbit._conversion_backend()
    host = _backend.load_host_module("_mojolearn_linalg_host")
    record = dict(schema="mojolearn.lowbit_quality.quantizer_check.v1", commit=args.commit,
                  stamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  mojolearn=getattr(mojolearn, "__version__", None), pack_backend=backend,
                  host_binding=getattr(host, "__file__", None),
                  host_dir=os.environ.get("MOJOLEARN_HOST_DIR"), tensors={}, sabotage={})
    print("pack backend:", backend, "host binding:", record["host_binding"], flush=True)

    tensors, weights, cap = real_tensors(args.model, args.corpus, args.device, args.tail_from, args.length)
    sab_names = {"half_away": dict(rounding="half_away"), "truncate": dict(rounding="truncate"),
                 "scale_one_exponent_off": dict(target_shift=1)}
    sab_total = {k: 0 for k in sab_names}
    sab_total_real = {k: 0 for k in sab_names}
    all_equal = True
    for name, t in tensors.items():
        x = np.ascontiguousarray(t.cpu().numpy())
        packed = lowbit.pack_one(x, "int8", name)
        ref_q = np.asarray(packed.codes).astype(np.int8)
        ref_e = np.asarray(packed.exponents).astype(np.int32)
        codes, e = arith.quantize_rows(t, "int8")
        sim_q = codes.cpu().numpy().astype(np.int8)
        sim_e = e.cpu().numpy().astype(np.int32)
        cell = dict(shape=list(x.shape), elements=int(x.size),
                    codes_differing_vs_pack=int((sim_q != ref_q).sum()),
                    exponents_differing_vs_pack=int((sim_e != ref_e).sum()),
                    exponent_min=int(ref_e.min()), exponent_max=int(ref_e.max()),
                    codes_at_clamp=int((np.abs(ref_q.astype(np.int16)) == 127).sum()),
                    codes_zero_fraction=float((ref_q == 0).mean()))
        # exact ties in the scaled value: where the tie rule is visible
        s = arith.ftz(arith.ftz(t) * arith.pow2_f32(-e).unsqueeze(-1))
        cell["exact_ties"] = int(((s - torch.floor(s)) == 0.5).sum().item())
        # the pure-Python spelling (a per-element loop): the smaller tensors whole, the larger by their first rows
        rows = x.shape[0] if x.size <= 600_000 else max(1, 600_000 // x.shape[1])
        py_q, py_e = lowbit._quantize_int8_py(lowbit._f32_2d(np.ascontiguousarray(x[:rows]), name))
        cell["pure_python_rows"] = int(rows)
        cell["codes_differing_vs_pure_python"] = int((sim_q[:rows] != np.asarray(py_q).astype(np.int8)).sum())
        cell["exponents_differing_vs_pure_python"] = int((sim_e[:rows] != np.asarray(py_e).astype(np.int32)).sum())
        # bf16 bits
        ref_bits = np.asarray(lowbit.pack_one(x, "bfloat16", name).bits).astype(np.uint16)
        sim_bits = arith.round_bf16(t, return_bits=True).cpu().numpy().astype(np.uint16)
        cell["bf16_bits_differing_vs_pack"] = int((sim_bits != ref_bits).sum())
        cell["bf16_changes_value"] = int((arith.round_bf16(t).contiguous().view(torch.int32)
                                          != arith.ftz(t).contiguous().view(torch.int32)).sum().item())
        # materialized values
        ref_f = np.asarray(lowbit.materialize_one(packed, name)).astype(np.float32)
        sim_f = arith.dequantize_rows(codes, e).cpu().numpy()
        cell["materialized_bits_differing"] = int((ref_f.view(np.uint32) != sim_f.view(np.uint32)).sum())
        cell["equal"] = (cell["codes_differing_vs_pack"] == 0 and cell["exponents_differing_vs_pack"] == 0
                         and cell["codes_differing_vs_pure_python"] == 0
                         and cell["exponents_differing_vs_pure_python"] == 0
                         and cell["bf16_bits_differing_vs_pack"] == 0
                         and cell["materialized_bits_differing"] == 0)
        all_equal = all_equal and cell["equal"]
        cell["sabotage_codes_differing"] = {}
        for sab, kw in sab_names.items():
            sq, se = arith.quantize_rows(t, "int8", **kw)
            n = int((sq.cpu().numpy().astype(np.int16) != ref_q.astype(np.int16)).sum()) \
                + int((se.cpu().numpy() != ref_e).sum())
            cell["sabotage_codes_differing"][sab] = n
            sab_total[sab] += n
            if not name.startswith("planted"):
                sab_total_real[sab] += n
        record["tensors"][name] = cell
        print(("EQUAL    " if cell["equal"] else "DIFFERENT"), name, json.dumps(cell), flush=True)

    record["sabotage"] = dict(differing_on_real_tensors=sab_total_real, differing_with_planted=sab_total,
                              every_arm_fails=all(v > 0 for v in sab_total_real.values()))

    # ---- the int8 product against the host oracle
    products = {}

    def oracle(a, b):
        pa, pb = lowbit.pack_one(a, "int8", "a"), lowbit.pack_one(b, "int8", "b")
        m, k = a.shape
        n = b.shape[0]
        out = empty((m, n), "<f4")
        host.gemm_int8(addr(out, name="out"), addr_ro(pa.codes, name="a codes"),
                       addr_ro(pa.exponents, name="a exponents"), addr_ro(pb.codes, name="b codes"),
                       addr_ro(pb.exponents, name="b exponents"), [int(m), int(n), int(k)])
        return np.asarray(out).astype(np.float32)

    pairs = {
        "layers.0.q_proj (128 tokens x 96 weight rows, k=960)":
            (cap["layers.0.q_proj.input"][:128], weights["model.layers.0.self_attn.q_proj.weight"][:96]),
        "layers.16.down_proj (96 tokens x 64 weight rows, k=2560)":
            (cap["layers.16.down_proj.input"][:96], weights["model.layers.16.mlp.down_proj.weight"][:64]),
        "lm_head (64 tokens x 256 table rows, k=960)":
            (cap["lm_head.input"][:64], weights["model.embed_tokens.weight"][:256]),
        "layers.10.attn_qk head 0 (128 x 128, k=64)":
            (cap["layers.10.attn_qk.a"][0, 0][:128], cap["layers.10.attn_qk.b"][0, 0][:128]),
        "layers.10.attn_pv head 3 (128 x 64, k=512)":
            (cap["layers.10.attn_pv.a"][0, 3][:128], cap["layers.10.attn_pv.b"][0, 3]),
        "planted edge cases (9 x 9, k=8)":
            (tensors["planted edge cases"], tensors["planted edge cases"]),
    }
    product_equal, product_sab = True, 0
    for name, (a, b) in pairs.items():
        a, b = a.contiguous(), b.contiguous()
        want = oracle(np.ascontiguousarray(a.cpu().numpy()), np.ascontiguousarray(b.cpu().numpy()))
        got = arith.product_nt(a, b, "int8", "int8").cpu().numpy()
        differ = int((want.view(np.uint32) != got.view(np.uint32)).sum())
        nan_both = int((np.isnan(want) & np.isnan(got)).sum())
        ca, ea = arith.quantize_rows(a, "int8", rounding="half_away")
        cb, eb = arith.quantize_rows(b, "int8", rounding="half_away")
        sab = arith.product_prepared(arith.Operand(codes=ca, e=ea, qmax=127),
                                     arith.Operand(codes=cb, e=eb, qmax=127)).cpu().numpy()
        sab_differ = int((want.view(np.uint32) != sab.view(np.uint32)).sum())
        fp = (a.to(torch.float64) @ b.to(torch.float64).T).cpu().numpy()
        with np.errstate(all="ignore"):
            rel = float(np.sqrt(np.nansum((want.astype(np.float64) - fp) ** 2) / max(np.nansum(fp ** 2), 1e-300))) \
                if np.isfinite(fp).all() else None
        products[name] = dict(cells=int(want.size), bits_differing_vs_host_oracle=differ, nan_in_both=nan_both,
                              equal=differ == 0, sabotage_half_away_bits_differing=sab_differ,
                              relative_error_against_float64_product=rel)
        product_equal = product_equal and differ == 0
        if not name.startswith("planted"):
            product_sab += sab_differ
        print(("EQUAL    " if differ == 0 else "DIFFERENT"), "product", name, json.dumps(products[name]), flush=True)
    record["int8_product_vs_host_oracle"] = dict(products=products, all_equal=product_equal,
                                                 sabotage_fails=product_sab > 0)
    record["l5_sum_to_float"] = check_l5(args.device)
    print("L-5", json.dumps(record["l5_sum_to_float"]), flush=True)

    ok = (all_equal and record["sabotage"]["every_arm_fails"] and product_equal and product_sab > 0
          and record["l5_sum_to_float"]["equal"] and record["l5_sum_to_float"]["sabotage_fails"])
    record["verdict"] = "PASS" if ok else "FAIL"
    with open(os.path.join(args.out, "quantizer_check.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print("QUANTIZER CHECK", record["verdict"], flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
