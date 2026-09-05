#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare real upstream CUDA Mamba blocks with retained Mojo native bytes.

Main lane only; this program never builds, installs dependencies or rents GPUs.
Requires an NVIDIA GPU, torch, mamba-ssm (with CUDA extensions), causal-conv1d
(for Mamba 1/2), generated repository corpus and the IDENTICAL Mamba binding.
Mamba 3 additionally requires an upstream version containing its Triton SISO
kernel. NO reference implementation fallback is permitted.

Example (from the repository root, after the native certificate succeeds):
  PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical python \
    tools/mamba_external_compare.py --family mamba2 \
    --native /tmp/certificate/mamba2/native --out /tmp/m2-external

All shared FP32 inputs/weights and outputs are retained with SHA256 witnesses.
Forward uses the actual public Mojo binding; backward uses the supplied native
certificate's public prefill leaves and signed dyadic output cotangent. State
cotangents are explicitly outside this adapter. Native-source digest must match
this checkout, preventing stale certificate admission.

Mamba 3's official SISO wrapper casts derived activations to BF16 internally.
This is explicitly a MIXED PRECISION comparison, never an FP32/bitwise claim.
Default tolerances remain strict; a failure is preserved, never relaxed. CLI
rtol/atol overrides are recorded. Optional timings measure ONLY upstream CUDA
resident forward and forward+backward, including Python dispatch/synchronization.
They are NOT competitive ratios against Mojo's host-transfer API or certificates.
Official source entry points:
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")


class Unavailable(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_array(out, name, value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype="<f4", order="C")
    data = arr.tobytes()
    filename = name + ".f32"
    (out / filename).write_bytes(data)
    return {"file": filename, "shape": list(arr.shape), "sha256": digest(data)}


def compare(got, want, rtol, atol):
    import numpy as np
    a, b = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if a.shape != b.shape:
        return {"pass": False, "shape_mismatch": [list(a.shape), list(b.shape)]}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"pass": False, "finite": False}
    delta = np.abs(a - b)
    return {"pass": bool(np.all(delta <= atol + rtol * np.abs(b))),
            "finite": True, "max_abs": float(delta.max()),
            "relative_l2": float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30)),
            "worst_excess": float(np.max(delta-atol-rtol*np.abs(b))),
            "rtol": rtol, "atol": atol}


def load_module(family):
    suffix = {"mamba1": "mamba_simple", "mamba2": "mamba2", "mamba3": "mamba3"}[family]
    try:
        mod = importlib.import_module("mamba_ssm.modules." + suffix)
        if family in ("mamba1", "mamba2"):
            conv = importlib.import_module("causal_conv1d")
            if not callable(getattr(conv, "causal_conv1d_fn", None)):
                raise Unavailable("causal_conv1d CUDA entry point missing")
        if family == "mamba1" and mod.causal_conv1d_fn is None:
            raise Unavailable("Mamba1 fused branch would silently fall back")
        return mod
    except (ImportError, OSError) as exc:
        raise Unavailable(f"upstream fused dependency unavailable: {exc}") from exc


def run(args, result):
    import numpy as np
    import torch
    import mamba_backward_identity as identity
    # This helper only generates the exact fixture inputs, not oracle forwards.
    spec = importlib.util.spec_from_file_location("external_corpus", ROOT / "mamba/corpus/gen_corpus.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    torch.set_num_threads(min(2, max(1, int(os.environ.get("MOJOLEARN_CPU_THREADS", "2")))))
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise Unavailable("an NVIDIA CUDA device is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(), "python": sys.version,
                             "torch_matmul_cudnn_tf32": False, "cpu_threads": torch.get_num_threads()}
    result["packages"] = {}
    for name in ("mamba-ssm", "causal-conv1d", "triton"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    native = json.loads((args.native / "manifest.json").read_text())
    if (native.get("schema") != identity.SCHEMA or native.get("family") != args.family
            or native.get("mode") != "IDENTICAL"
            or native.get("objective") != "signed_dyadic_weight_v1"
            or native.get("public_prefill_leaves") != identity.PUBLIC_LEAVES[args.family]):
        raise ValueError("native manifest is not a matching strict public-prefill certificate")
    source_sha = identity.source_digest(ROOT)
    if native.get("source_sha256") != source_sha:
        raise ValueError("native certificate source differs from this checkout; rerun native gate")
    result["native_manifest_sha256"] = digest((args.native / "manifest.json").read_bytes())
    result["native_source_sha256"] = source_sha
    family = args.family
    cases, build = {"mamba1": (gen.CASES, gen.build_case),
                    "mamba2": (gen.M2_CASES, gen.m2_build_case),
                    "mamba3": (gen.M3_CASES, gen.m3_build_case)}[family]
    case = next(c for c in cases if c["name"] == native["case"])
    arrays, meta = build(case)
    result["case"] = meta["name"]
    if any(k.startswith("init_") or k == "initial_states" for k in arrays):
        raise ValueError("incoming-state fixture is outside this zero-state adapter")
    corpus = ROOT / "mamba/corpus"
    if family != "mamba1":
        corpus /= family
    corpus /= meta["name"]
    for name, array in arrays.items():
        if (corpus / (name + ".f32")).read_bytes() != np.asarray(array, dtype="<f4").tobytes():
            raise ValueError(f"corpus bytes differ from exact fixture generator: {name}")
    result["inputs"] = {n: save_array(args.out, "input." + n, v) for n, v in arrays.items()}
    mod = load_module(family)
    module_path = Path(inspect.getfile(mod))
    result["upstream_module"] = {"path": str(module_path), "sha256": digest(module_path.read_bytes())}
    common = dict(d_model=meta["d_model"], d_state=meta["d_state"], expand=meta["expand"],
                  device="cuda", dtype=torch.float32)
    if family == "mamba1":
        upstream = mod.Mamba(**common, d_conv=meta["d_conv"], dt_rank=meta["dt_rank"], use_fast_path=True)
        norm_name = "norm.weight"
        result["external_precision"] = "FP32; official fused selective_scan CUDA"
    elif family == "mamba2":
        upstream = mod.Mamba2(**common, d_conv=meta["d_conv"], headdim=meta["headdim"],
                             ngroups=meta["ngroups"], chunk_size=meta["chunk_size"],
                             norm_before_gate=False, use_mem_eff_path=True,
                             dt_limit=gen.m2_effective_dt_limit(case))
        norm_name = "block_norm.weight"
        result["external_precision"] = "FP32 operands; official fused conv/SSD Triton CUDA (Triton dot policy belongs to upstream)"
    else:
        upstream = mod.Mamba3(**common, headdim=meta["headdim"], ngroups=meta["ngroups"],
                             chunk_size=meta["chunk_size"], A_floor=meta["a_floor"],
                             rope_fraction=0.5, is_outproj_norm=False, is_mimo=False)
        norm_name = "block_norm.weight"
        kernel_path = Path(inspect.getfile(importlib.import_module(
            "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined")))
        result["upstream_kernel"] = {"path": str(kernel_path), "sha256": digest(kernel_path.read_bytes())}
        result["external_precision"] = "MIXED BF16 SISO activations; FP32 parameters/dt/decay (upstream wrapper)"
    weights = {n: v for n, v in arrays.items() if n != "x"}
    params = dict(upstream.named_parameters())
    if set(params) != set(weights) - {norm_name}:
        raise ValueError(f"upstream parameter inventory differs: {set(params) ^ (set(weights)-{norm_name})}")
    with torch.no_grad():
        for name, dest in params.items():
            src = torch.from_numpy(weights[name]).to("cuda")
            if family == "mamba3" and name in ("B_bias", "C_bias"):
                src = src.unsqueeze(1)  # singleton SISO rank, a pure reshape
            if src.shape != dest.shape:
                raise ValueError(f"parameter shape differs: {name}")
            dest.copy_(src)
    x = torch.tensor(arrays["x"], device="cuda", requires_grad=True)
    norm = torch.tensor(weights[norm_name], device="cuda", requires_grad=True)
    leaves = {"x": x, norm_name: norm, **params}
    names = identity.PUBLIC_LEAVES[family]
    eps = meta["rms_eps"]

    def forward():
        # Our public block wraps the upstream mixer with this RMSNorm/residual.
        normalized = norm * (x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps))
        return x + upstream(normalized)

    y = forward()
    idx = torch.arange(y.numel(), device="cuda", dtype=torch.int64)
    numer = ((idx * 37 + 11) % 31) - 15
    dy = torch.where(numer == 0, torch.ones_like(numer), numer).float().reshape(y.shape) / 16
    grads = torch.autograd.grad(y, [leaves[n] for n in names], grad_outputs=dy)
    torch.cuda.synchronize()
    result["outputs"] = {"external.forward": save_array(args.out, "external.forward", y),
                         "objective.dy": save_array(args.out, "objective.dy", dy)}
    # Native forward is executed only once here, outside any timings.
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    sys.path.insert(0, str(ROOT / "python"))
    import mojolearn
    cls = getattr(mojolearn, {"mamba1": "Mamba1Block", "mamba2": "Mamba2Block", "mamba3": "Mamba3Block"}[family])
    ours = cls(weights, **({"dt_limit": gen.m2_effective_dt_limit(case)} if family == "mamba2" else {}))
    ours.numeric_mode = "identical"
    if int(ours._extension().mamba_numeric_mode()) != 1:
        raise ValueError("native Mamba binary is not compiled IDENTICAL")
    actual_y = ours.forward(arrays["x"])
    result["outputs"]["native.forward"] = save_array(args.out, "native.forward", actual_y)
    result["comparisons"] = {"forward": compare(actual_y, y.detach().cpu().numpy(), args.rtol, args.atol)}
    for name, gradient in zip(names, grads):
        entry = native["tensors"][name]
        path = args.native / entry["file"]
        raw = path.read_bytes()
        if entry["dtype"] != "<f4" or digest(raw) != entry["sha256"]:
            raise ValueError(f"native gradient bytes failed witness: {name}")
        expected_shape = arrays[name].shape
        if list(expected_shape) != entry["shape"]:
            raise ValueError(f"native gradient shape differs: {name}")
        actual = np.frombuffer(raw, dtype="<f4").reshape(expected_shape)
        external = gradient.detach().cpu().numpy().reshape(expected_shape)
        result["outputs"]["external.grad." + name] = save_array(args.out, "external.grad." + name, external)
        result["outputs"]["native.grad." + name] = save_array(args.out, "native.grad." + name, actual)
        result["comparisons"]["grad." + name] = compare(actual, external, args.rtol, args.atol)
    result["excluded_native_leaves"] = sorted(set(native["tensors"]) - set(names))
    passed = all(row["pass"] for row in result["comparisons"].values())
    result["status"] = "PASS" if passed else "FAIL"
    if args.samples and passed:
        def backward():
            yy = forward()
            torch.autograd.grad(yy, [leaves[n] for n in names], grad_outputs=dy)
        def inference():
            with torch.no_grad():
                forward()
        operations = {"forward": inference, "forward_backward": backward}
        timings = {name: [] for name in operations}
        for fn in operations.values():
            fn()
            fn()
        torch.cuda.synchronize()
        for sample in range(args.samples):
            order = list(operations)
            if sample % 2:
                order.reverse()
            for name in order:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                operations[name]()
                torch.cuda.synchronize()
                timings[name].append((time.perf_counter()-begin)*1000)
        result["external_only_timings"] = {
            "scope": "resident weights/input, Python dispatch + synchronization; backward includes forward, no transfers; NOT a Mojo speed ratio",
            "samples_ms": timings,
            "median_ms": {k: statistics.median(v) for k, v in timings.items()}}
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=("mamba1", "mamba2", "mamba3"), required=True)
    parser.add_argument("--native", type=Path, required=True, help="captured native/ directory with manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="new output directory (never overwritten)")
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--samples", type=int, default=0, help="0 disables timings; otherwise at least 7")
    args = parser.parse_args()
    if (not math.isfinite(args.rtol) or not math.isfinite(args.atol)
            or args.rtol < 0 or args.atol < 0 or (args.samples != 0 and args.samples < 7)):
        parser.error("tolerances must be nonnegative and samples must be 0 or >=7")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"schema": "mojolearn.mamba.external-cuda.v1", "family": args.family,
              "rtol": args.rtol, "atol": args.atol,
              "scope": "zero-state whole-block forward and all native public-prefill backward leaves",
              "cross_library_bitwise_identity_claim": False}
    try:
        code = run(args, result)
    except (Unavailable, ImportError) as exc:
        result.update(status="UNAVAILABLE", error=str(exc))
        code = 3
    except Exception as exc:
        result.update(status="ERROR", error=str(exc), traceback=traceback.format_exc())
        code = 2
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.out / "result.json")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
