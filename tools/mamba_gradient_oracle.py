#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent Mamba 1/2/3 whole-block gradient oracle.

The forward functions come from ``mamba/corpus/gen_corpus.py``: literal,
cited PyTorch transcriptions of the upstream reference implementations.  This
program differentiates their float64 block result with PyTorch autograd and
then audits selected cells with central finite differences.  The latter is
deliberately independent of autograd and catches a self-consistent bad
backward transcription.

The output is a portable directory of little-endian float64 files plus a
manifest.  A Mojo backward runner should emit files with the same names; use
``--compare`` to check them.  This is a tolerance oracle, not an IDENTICAL
profile card: PyTorch's contraction order is not the repository's pinned
order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = ROOT / "mamba" / "corpus" / "gen_corpus.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("mojolearn_mamba_corpus", GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


def _case(family: str, name: str | None):
    if family == "mamba1":
        cases, build, forward, default = (
            GEN.CASES, GEN.build_case, GEN.block_forward, "base_b2_l4_d8"
        )
    elif family == "mamba2":
        cases, build, forward, default = (
            GEN.M2_CASES, GEN.m2_build_case, GEN.m2_forward,
            "m2_base_b2_l4_d32",
        )
    else:
        cases, build, forward, default = (
            GEN.M3_CASES, GEN.m3_build_case, GEN.m3_forward,
            "m3_base_b2_l4_d32",
        )
    wanted = name or default
    try:
        case = next(c for c in cases if c["name"] == wanted)
    except StopIteration as exc:
        raise SystemExit(f"unknown {family} case {wanted!r}") from exc
    tensors, meta = build(case)
    return case, tensors, meta, forward


def _objective(y: torch.Tensor) -> torch.Tensor:
    """A dense, asymmetric scalar objective with exactly represented weights."""
    flat = y.reshape(-1)
    # Small signed dyadic coefficients keep every output live without making
    # finite differences ill-conditioned or hiding permutations behind a sum.
    i = torch.arange(flat.numel(), dtype=torch.int64, device=flat.device)
    numer = ((i * 37 + 11) % 31) - 15
    numer = torch.where(numer == 0, torch.ones_like(numer), numer)
    return torch.sum(flat * (numer.to(flat.dtype) / 16.0))


def _prepare(tensors, device):
    leaves = {}
    for name, value in tensors.items():
        t = torch.from_numpy(np.array(value, copy=True)).to(device, torch.float64)
        t.requires_grad_(True)
        leaves[name] = t
    x = leaves.pop("x")
    return x, leaves


def _forward_stages(family, case, forward, params, x, dtype=torch.float64):
    if family == "mamba1":
        return forward(params, x, dtype)
    if family == "mamba2":
        lim = GEN.m2_effective_dt_limit(case)
        return forward(params, x, dtype, dt_limit=lim)
    return forward(params, x, dtype)


def _forward(family, case, forward, params, x):
    stages = _forward_stages(family, case, forward, params, x)
    return stages["block.out" if family == "mamba1" else "residual.out"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_array(path: Path, tensor: torch.Tensor):
    arr = np.asarray(tensor.detach().cpu(), dtype="<f8", order="C")
    data = arr.tobytes(order="C")
    path.write_bytes(data)
    return list(arr.shape), _sha256(data)


def generate(args):
    case, tensors, case_meta, forward = _case(args.family, args.case)
    device = torch.device(args.device)
    x, params = _prepare(tensors, device)
    stages = _forward_stages(args.family, case, forward, params, x)
    y = stages["block.out" if args.family == "mamba1" else "residual.out"]
    loss = _objective(y)
    leaves = [("x", x), *sorted(params.items())]
    # Expose the activation gradient at the implemented Mamba-2 projection
    # tail. m2_forward records a view of `gout` but applies F.linear to the
    # pre-view tensor, so that recorded view is not an ancestor of the loss.
    # Rebuild only this isolated, independently specified linear seam rather
    # than asking autograd for a mathematically-related but graph-unused view.
    intermediate_gradients = []
    reference32 = {}
    if args.family == "mamba2":
        tail_input = stages["gnorm.out"].detach().requires_grad_(True)
        tail_output = torch.nn.functional.linear(
            tail_input, params["out_proj.weight"].detach()
        )
        d_tail = torch.autograd.grad(_objective(tail_output), tail_input)[0]
        intermediate_gradients.append(("stage.gnorm.out", d_tail))
        gate_input = stages["gnorm.gate"].detach().requires_grad_(True)
        gate_sumsq = gate_input.pow(2).sum(-1, keepdim=True)
        gate_rstd = 1.0 / torch.sqrt(gate_sumsq / gate_input.shape[-1] + GEN.M2_EPS)
        gate_output = gate_input * gate_rstd * params["norm.weight"]
        projected = torch.nn.functional.linear(
            gate_output, params["out_proj.weight"].detach()
        )
        d_gate = torch.autograd.grad(_objective(projected), gate_input)[0]
        intermediate_gradients.append(("stage.gnorm.gate", d_gate))
        skip_input = stages["skip.out"].detach().reshape(
            -1, stages["gnorm.gate"].shape[-1]
        ).requires_grad_(True)
        z_input = stages["in_proj.out"].detach()[:, :skip_input.shape[-1]].requires_grad_(True)
        gated = skip_input * torch.nn.functional.silu(z_input)
        gated_rstd = 1.0 / torch.sqrt(
            gated.pow(2).sum(-1, keepdim=True) / gated.shape[-1] + GEN.M2_EPS
        )
        gated_norm = gated * gated_rstd * params["norm.weight"].detach()
        gated_out = torch.nn.functional.linear(
            gated_norm, params["out_proj.weight"].detach()
        )
        d_skip, d_z = torch.autograd.grad(
            _objective(gated_out), (skip_input, z_input)
        )
        intermediate_gradients.append(("stage.skip.out", d_skip))
        intermediate_gradients.append(("stage.in_proj.z", d_z))
        d_scan = d_skip.reshape(stages["scan.y"].shape)
        d_x_from_d = d_scan * params["D"].detach()[None, :, None]
        intermediate_gradients.append(("stage.scan.y", d_scan))
        intermediate_gradients.append(("partial.silu.x.from_D", d_x_from_d))

        # Isolated S18 oracle.  Treat the recorded incoming chunk states as
        # leaves and differentiate yoff = (C @ h_in) * exp(dacs).  This is
        # independent of the Mojo reverse recurrence and deliberately stops
        # before S17 adds the next-chunk carry.
        pass_input = stages["pass.states"].detach().requires_grad_(True)
        bsz = pass_input.shape[0]
        nheads, headdim = pass_input.shape[2:4]
        length = d_scan.numel() // (bsz * nheads * headdim)
        d_scan_ssd = d_scan.reshape(bsz, length, nheads, headdim)
        nstate = pass_input.shape[-1]
        conv_dim = stages["silu.out"].shape[-1]
        c_input = stages["silu.out"].reshape(bsz, length, conv_dim)[
            ..., headdim * nheads + nstate : headdim * nheads + 2 * nstate
        ].detach()
        qsize = stages["dacs.out"].shape[-1]
        yoff_terms = []
        for token in range(length):
            chunk, inner = divmod(token, qsize)
            dot = (
                c_input[:, token, None, None, :]
                * pass_input[:, chunk]
            ).sum(-1)
            scale = torch.exp(stages["dacs.out"][:, :, chunk, inner]).unsqueeze(-1)
            yoff_terms.append(dot * scale)
        isolated_yoff = torch.stack(yoff_terms, dim=1)
        direct_d_pass = torch.autograd.grad(
            (isolated_yoff * d_scan_ssd.detach()).sum(), pass_input
        )[0]
        intermediate_gradients.append(("stage.pass.states.direct", direct_d_pass))

        # The RMSNorm subtraction is ill-conditioned on this fixture
        # (rstd ~= 222). Preserve the float64 oracle above, but also record an
        # independent PyTorch-float32 calibration for native-f32 dumps. This
        # distinguishes expected precision loss from an ordering/layout bug.
        p32 = {
            k: torch.from_numpy(np.array(v, copy=True)).to(device, torch.float32)
            for k, v in tensors.items() if k != "x"
        }
        x32 = torch.from_numpy(np.array(tensors["x"], copy=True)).to(
            device, torch.float32
        )
        stages32 = _forward_stages(
            args.family, case, forward, p32, x32, torch.float32
        )
        gate32 = stages32["gnorm.gate"].detach().requires_grad_(True)
        ss32 = gate32.pow(2).sum(-1, keepdim=True)
        rs32 = 1.0 / torch.sqrt(ss32 / gate32.shape[-1] + GEN.M2_EPS)
        gout32 = gate32 * rs32 * p32["norm.weight"]
        projected32 = torch.nn.functional.linear(gout32, p32["out_proj.weight"])
        reference32["stage.gnorm.gate"] = torch.autograd.grad(
            _objective(projected32), gate32
        )[0]
        skip32 = stages32["skip.out"].detach().reshape(
            -1, stages32["gnorm.gate"].shape[-1]
        ).requires_grad_(True)
        z32 = stages32["in_proj.out"].detach()[:, :skip32.shape[-1]].requires_grad_(True)
        gated32 = skip32 * torch.nn.functional.silu(z32)
        grstd32 = 1.0 / torch.sqrt(
            gated32.pow(2).sum(-1, keepdim=True) / gated32.shape[-1] + GEN.M2_EPS
        )
        gnorm32 = gated32 * grstd32 * p32["norm.weight"]
        gout32 = torch.nn.functional.linear(gnorm32, p32["out_proj.weight"])
        dskip32, dz32 = torch.autograd.grad(_objective(gout32), (skip32, z32))
        reference32["stage.skip.out"] = dskip32
        reference32["stage.in_proj.z"] = dz32
        dscan32 = dskip32.reshape(stages32["scan.y"].shape)
        reference32["stage.scan.y"] = dscan32
        reference32["partial.silu.x.from_D"] = (
            dscan32 * p32["D"][None, :, None]
        )
        pass32 = stages32["pass.states"].detach().requires_grad_(True)
        b32 = pass32.shape[0]
        h32, p_dim32 = pass32.shape[2:4]
        l32 = dscan32.numel() // (b32 * h32 * p_dim32)
        dscan_ssd32 = dscan32.reshape(b32, l32, h32, p_dim32)
        n32 = pass32.shape[-1]
        cd32 = stages32["silu.out"].shape[-1]
        c32 = stages32["silu.out"].reshape(b32, l32, cd32)[
            ..., h32 * p_dim32 + n32 : h32 * p_dim32 + 2 * n32
        ].detach()
        q32 = stages32["dacs.out"].shape[-1]
        yoff32 = []
        for token in range(l32):
            chunk, inner = divmod(token, q32)
            dot32 = (c32[:, token, None, None, :] * pass32[:, chunk]).sum(-1)
            scale32 = torch.exp(
                stages32["dacs.out"][:, :, chunk, inner]
            ).unsqueeze(-1)
            yoff32.append(dot32 * scale32)
        reference32["stage.pass.states.direct"] = torch.autograd.grad(
            (torch.stack(yoff32, dim=1) * dscan_ssd32.detach()).sum(), pass32
        )[0]
    leaf_gradients = torch.autograd.grad(loss, [v for _, v in leaves])
    named_gradients = [*intermediate_gradients, *zip(
        (name for name, _ in leaves), leaf_gradients
    )]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, grad in named_gradients:
        filename = "grad." + name + ".f64"
        shape, digest = _write_array(out / filename, grad)
        files[name] = {"file": filename, "shape": shape, "sha256": digest}
        if name in reference32:
            ref32_name = "grad." + name + ".ref32.f32"
            arr32 = np.asarray(
                reference32[name].detach().cpu(), dtype="<f4", order="C"
            )
            data32 = arr32.tobytes(order="C")
            (out / ref32_name).write_bytes(data32)
            files[name]["ref32_file"] = ref32_name
            files[name]["ref32_sha256"] = _sha256(data32)

    # Audit deterministic cells in x and every parameter. Central finite
    # differences use fresh forward graphs and never inspect autograd internals.
    finite = []
    for (name, leaf), grad in zip(leaves, leaf_gradients):
        if leaf.numel() == 0:
            continue
        indexes = sorted({0, leaf.numel() // 2, leaf.numel() - 1})[: args.fd_cells]
        for index in indexes:
            original = float(leaf.detach().reshape(-1)[index].cpu())
            values = []
            with torch.no_grad():
                for sign in (1.0, -1.0):
                    leaf.reshape(-1)[index] = original + sign * args.fd_step
                    values.append(float(_objective(_forward(
                        args.family, case, forward, params, x
                    )).cpu()))
                leaf.reshape(-1)[index] = original
            numeric = (values[0] - values[1]) / (2.0 * args.fd_step)
            automatic = float(grad.reshape(-1)[index].detach().cpu())
            abs_error = abs(numeric - automatic)
            scale = max(1.0, abs(numeric), abs(automatic))
            finite.append({
                "tensor": name, "flat_index": index, "autograd": automatic,
                "finite_difference": numeric, "abs_error": abs_error,
                "relative_scale_error": abs_error / scale,
            })

    worst = max((row["relative_scale_error"] for row in finite), default=0.0)
    manifest = {
        "schema": "mojolearn.mamba.gradient-oracle.v1",
        "family": args.family,
        "case": case["name"],
        "case_seed": case_meta["seed"],
        "dtype": "float64",
        "objective": "sum(flat(block_output) * signed_dyadic_weight_v1)",
        "loss": float(loss.detach().cpu()),
        "oracle": "PyTorch autograd over cited upstream forward transcription",
        "independent_audit": "central finite difference",
        "finite_difference_step": args.fd_step,
        "finite_difference_worst_relative_scale_error": worst,
        "finite_difference": finite,
        "gradients": files,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device), "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.family}/{case['name']} gradients to {out}")
    print(f"finite-difference worst relative-scale error: {worst:.3e}")
    if worst > args.fd_rtol:
        raise SystemExit(
            f"finite-difference audit failed: {worst:.3e} > {args.fd_rtol:.3e}"
        )


def compare(args):
    oracle_dir, actual_dir = Path(args.compare[0]), Path(args.compare[1])
    manifest = json.loads((oracle_dir / "manifest.json").read_text())
    failures = []
    compared = 0
    dump_manifest_path = actual_dir / "dump_manifest.json"
    if not dump_manifest_path.exists():
        raise SystemExit(f"missing dump provenance: {dump_manifest_path}")
    dump_manifest = json.loads(dump_manifest_path.read_text())
    expected_provenance = {
        "schema": "mojolearn.mamba.gradient-dump.v1",
        "family": manifest["family"],
        "case": manifest["case"],
        "objective": "signed_dyadic_weight_v1",
    }
    for key, expected_value in expected_provenance.items():
        if dump_manifest.get(key) != expected_value:
            failures.append(
                f"provenance {key}: {dump_manifest.get(key)!r}, expected {expected_value!r}"
            )
    for name, entry in manifest["gradients"].items():
        expected = np.fromfile(oracle_dir / entry["file"], dtype="<f8").reshape(entry["shape"])
        path = actual_dir / entry["file"]
        if not path.exists():
            # Native Mojo kernels produce float32. Keep the oracle files at
            # float64 and accept the same stem with an explicit .f32 suffix.
            path = actual_dir / entry["file"].replace(".f64", ".f32")
            if not path.exists():
                if not args.allow_partial:
                    failures.append(f"{name}: missing gradient dump")
                continue
        if name not in dump_manifest.get("tensors", []):
            failures.append(f"{name}: file exists but dump manifest does not name it")
            continue
        dtype = "<f4" if path.suffix == ".f32" else "<f8"
        actual = np.fromfile(path, dtype=dtype).astype(np.float64)
        if dtype == "<f4" and "ref32_file" in entry:
            expected = np.fromfile(
                oracle_dir / entry["ref32_file"], dtype="<f4"
            ).astype(np.float64).reshape(entry["shape"])
        if actual.size != expected.size:
            failures.append(f"{name}: cells {actual.size}, expected {expected.size}")
            continue
        actual = actual.reshape(expected.shape)
        compared += 1
        if not np.allclose(actual, expected, rtol=args.rtol, atol=args.atol, equal_nan=False):
            delta = np.abs(actual - expected)
            failures.append(
                f"{name}: maxabs={float(delta.max()):.3e}, bad="
                f"{int((delta > args.atol + args.rtol * np.abs(expected)).sum())}"
            )
    if compared == 0:
        failures.append("no gradient files were compared")
    if failures:
        print("Mamba gradient comparison FAILED", file=sys.stderr)
        print("\n".join("  " + f for f in failures), file=sys.stderr)
        raise SystemExit(1)
    qualifier = "partial " if args.allow_partial else ""
    print(f"Mamba gradient {qualifier}comparison passed: {compared} tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("mamba1", "mamba2", "mamba3"))
    parser.add_argument("--case")
    parser.add_argument("--device", default="cpu",
                        help="cpu, cuda, or another torch device; ROCm uses cuda")
    parser.add_argument("--out")
    parser.add_argument("--fd-step", type=float, default=1e-5)
    parser.add_argument("--fd-cells", type=int, default=2)
    parser.add_argument("--fd-rtol", type=float, default=2e-4)
    parser.add_argument("--compare", nargs=2, metavar=("ORACLE_DIR", "ACTUAL_DIR"))
    # Match the established Mamba-2/3 float32-vs-float64 corpus calibration.
    # The Mojo dump is native float32 while this oracle intentionally remains
    # float64; a sub-ulp float32 contraction gap must not be judged by a
    # float64-sized absolute floor.
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--allow-partial", action="store_true",
                        help="compare present tensors but still require at least one")
    args = parser.parse_args()
    if args.compare:
        compare(args)
    else:
        if not args.family or not args.out:
            parser.error("generation requires --family and --out")
        generate(args)


if __name__ == "__main__":
    main()
