#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE INCUMBENT: the same model, prompts and greedy decode through
`transformers` + PyTorch on the same box, in the incumbent's FAST DEFAULT,
plus a second arm under the incumbent's own determinism switch.

    python3 bench/model/torch_twin.py --model <local dir> \\
        --prompts bench/model/prompts.txt --max-new 64 \\
        --column nvidia-h100-torch --out torch.nvidia-h100.json [--arms fast,deterministic]

ARMS (bench/OPPONENT_REFERENCE.md: the opponent's fast arm as shipped is the
row; nothing here is tuned):
  fast           GPU: bfloat16 weights and activations, torch's shipped
                 switches untouched and READ BACK (matmul allow_tf32, cudnn
                 allow_tf32, float32_matmul_precision, the SDPA backend
                 transformers picked, deterministic flag off). Named
                 `torch-bf16-shipped`, or `torch-bf16-mps-shipped` on Metal.
                 CPU: float32, `torch-fp32-cpu-shipped`. A device that
                 refuses bfloat16 falls back to float32 and the arm says so
                 (`torch-fp32-<device>-shipped`, `bf16_fallback` recorded).
  deterministic  the same dtype with torch.use_deterministic_algorithms(True),
                 TF32 off on both switches, float32_matmul_precision "highest",
                 CUBLAS_WORKSPACE_CONFIG=:4096:8 set BEFORE torch imports.
                 Named `torch-deterministic`. An op with no deterministic
                 kernel raises; the prompt then reads REFUSED with the
                 message, which is a finding about the incumbent.
Each arm runs in ITS OWN PROCESS (the driver re-invokes this file with
--arm), because the CUBLAS workspace variable and the deterministic flag
are process-wide and the fast arm must not inherit them.

The record has the same fields as ours (bench/model/_common.py); its
`library` pins torch and transformers versions, CUDA or HIP, the device
name and every switch read back. The first-step logits are hashed as
float32 bytes (cast from the arm's dtype; `logits_dtype` says which) so the
field has one meaning across records. The model is on disk before any
clock: this file refuses a path without config.json.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def pick_device(requested):
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def switches():
    import torch
    cuda = torch.backends.cuda
    out = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               float32_matmul_precision=torch.get_float32_matmul_precision(),
               cudnn_deterministic=torch.backends.cudnn.deterministic,
               cudnn_benchmark=torch.backends.cudnn.benchmark,
               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
               cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    try:
        out["matmul_allow_tf32"] = cuda.matmul.allow_tf32
    except Exception:  # noqa: BLE001
        out["matmul_allow_tf32"] = None
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        fn = getattr(cuda, name, None)
        if fn is not None:
            try:
                out[name] = bool(fn())
            except Exception:  # noqa: BLE001
                out[name] = None
    return out


def library_record(device, dtype, model):
    import torch
    import transformers
    lib = dict(torch=torch.__version__, transformers=transformers.__version__,
               cuda=getattr(torch.version, "cuda", None), hip=getattr(torch.version, "hip", None),
               device=device, dtype=str(dtype).replace("torch.", ""),
               device_name=(torch.cuda.get_device_name(0) if device == "cuda" else None),
               attn_implementation=getattr(model.config, "_attn_implementation", None),
               switches=switches(), threads=torch.get_num_threads())
    return lib


def run_arm(args):
    """One arm, one process."""
    if args.arm == "deterministic":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device(args.device)
    if args.arm == "deterministic":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:  # noqa: BLE001
            pass
        torch.set_float32_matmul_precision("highest")
    dtype, fallback = torch.float32, None
    if device in ("cuda", "mps"):
        dtype = torch.bfloat16
        try:
            (torch.ones(2, 2, device=device, dtype=dtype) @ torch.ones(2, 2, device=device, dtype=dtype)).sum().item()
        except Exception as e:  # noqa: BLE001
            dtype, fallback = torch.float32, str(e)[:200]
    if args.arm == "deterministic":
        arm_name = C.TORCH_DETERMINISTIC_ARM
    else:
        tag = "bf16" if dtype == torch.bfloat16 else "fp32"
        arm_name = f"torch-{tag}-shipped" if device == "cuda" else f"torch-{tag}-{device}-shipped"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()
    sync = None
    if device == "cuda":
        sync = torch.cuda.synchronize
    elif device == "mps":
        sync = torch.mps.synchronize
    clock = C.Clock(sync)
    prompts = C.read_prompts(args.prompts)
    lib = library_record(device, dtype, model)
    lib.update(arm=arm_name, bf16_fallback=fallback, logits_dtype="float32 (cast)")
    arm = dict(prompts={}, arm=arm_name, library=lib)

    def one(text, runs):
        enc = tok(text, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        n = int(ids.shape[1])
        if n == 0:
            return C.refused_cell("the tokenizer returned no ids for this prompt", 0)
        prompt_ids = [int(t) for t in ids[0].tolist()]
        ids_runs, pre, dec, first_sha, gen_text, returns = [], [], [], None, None, None
        with torch.inference_mode():
            for _ in range(runs):
                p_ms, out = clock.ms(model, input_ids=ids)
                row = out.logits[0, -1, :].detach().to(torch.float32).cpu().contiguous()
                sha = C.sha256_bytes(row.numpy().tobytes())
                first_sha = sha if first_sha is None else (first_sha if sha == first_sha else "MOVED:" + first_sha[:16] + "/" + sha[:16])
                g_ms, gen = clock.ms(model.generate, input_ids=ids, attention_mask=enc["attention_mask"].to(device),
                                     max_new_tokens=args.max_new, do_sample=False, num_beams=1,
                                     pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id))
                new, returns = C.split_generated(prompt_ids, gen[0].tolist())
                ids_runs.append(new)
                pre.append(p_ms)
                dec.append(max(g_ms - p_ms, 0.0))
                if gen_text is None:
                    gen_text = tok.decode(new, skip_special_tokens=False)
        extra = dict(generate_returns=returns, generate_ms_runs=[p + d for p, d in zip(pre, dec)])
        return C.cell_from_runs(n, len(ids_runs[0]), ids_runs, first_sha, gen_text, pre, dec, extra)

    try:
        one(prompts[0][1], 1)  # the warm-up, never recorded
    except Exception as e:  # noqa: BLE001
        arm["warmup_error"] = str(e)[:300]
    for pid, text in prompts:
        try:
            cell = one(text, args.runs)
        except Exception as e:  # noqa: BLE001
            cell = C.refused_cell(e)
        arm["prompts"][pid] = cell
        print(f"# CELL {arm_name}/{pid} {cell['verdict']} prefill={cell.get('prefill_ms_per_token')} "
              f"decode={cell.get('decode_ms_per_token')}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(arm, fh, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local model directory (config.json inside)")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt"))
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--column", default=None, help="the column label, e.g. nvidia-h100-torch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arms", default="fast,deterministic")
    ap.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    C.require_model_dir(args.model)
    if args.arm:
        return run_arm(args)
    if not args.column:
        raise SystemExit("--column is required")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("fast", "deterministic"):
            raise SystemExit(f"unknown arm {a!r}; fast or deterministic")
    prompts = C.read_prompts(args.prompts)
    protocol = dict(max_new=args.max_new, greedy=True, runs=args.runs, statistic="median",
                    warm_prompt=prompts[0][0], prompts_file=os.path.abspath(args.prompts),
                    prompts_sha256=C.prompts_sha256(args.prompts), device_requested=args.device, arms=arms,
                    timing="prefill = one forward over the prompt; decode = generate minus that run's "
                    "prefill; per token = whole phase over its token count; device synchronized around each")
    rec = C.new_record("torch", args.column, C.model_record(args.model, os.path.basename(args.model.rstrip("/"))),
                       protocol, library={})
    C.write_record(args.out, rec)
    for a in arms:
        fd, tmp = tempfile.mkstemp(prefix="torch_twin_", suffix=".json")
        os.close(fd)
        cmd = [sys.executable, os.path.abspath(__file__), "--model", args.model, "--prompts", args.prompts,
               "--max-new", str(args.max_new), "--runs", str(args.runs), "--device", args.device,
               "--arm", a, "--out", tmp]
        env = dict(os.environ)
        if a != "deterministic":
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0 or not os.path.getsize(tmp):
            name = C.TORCH_DETERMINISTIC_ARM if a == "deterministic" else "torch-fast"
            rec["arms"][name] = dict(prompts={pid: C.refused_cell(f"arm process exit {proc.returncode}: " + proc.stderr[-500:])
                                              for pid, _ in prompts}, arm=name, error=proc.stderr[-1500:])
        else:
            with open(tmp, encoding="utf-8") as fh:
                arm = json.load(fh)
            rec["arms"][arm["arm"]] = arm
            if not rec["library"]:
                rec["library"] = {k: v for k, v in arm["library"].items() if k not in ("arm", "switches")}
        os.unlink(tmp)
        C.write_record(args.out, rec)
    rec["complete"] = True
    C.write_record(args.out, rec)
    cells = [c for a in rec["arms"].values() for c in a["prompts"].values()]
    n_ref = sum(1 for c in cells if c["verdict"] == "REFUSED")
    n_mov = sum(1 for c in cells if c["verdict"] == "MOVED")
    print(f"cells={len(cells)} stable={len(cells) - n_ref - n_mov} moved={n_mov} refused={n_ref} "
          f"torch={rec['library'].get('torch')} transformers={rec['library'].get('transformers')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
