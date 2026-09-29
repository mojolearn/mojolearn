#!/usr/bin/env python3
"""Paired model inference probe, ready for committed fixed15 integration.

Uses already-staged model/prompts, never downloads or enables a profile.
Times stateful prefill and actual step calls; never subtracts timings.
Full logits from fixed token contexts are saved for cross-vendor comparison.
This is a latency/identity probe, not a task-quality evaluation.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
from mojolearn import _numeric_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench/model"))
import _common as common


def run_once(model, ids, decode_ids, target=None):
    # Tokenization/allocation of fresh state excluded; all model calls and
    # their transfers included. Identical contexts across profiles/vendors.
    state = model.allocate_state(1, ids.shape[1] + len(decode_ids))
    arrays = [np.array([[t]], dtype=np.int32) for t in decode_ids]
    start = time.perf_counter_ns()
    logits = model.forward(ids, state=state)
    prefill_ms = (time.perf_counter_ns() - start) / 1e6
    raw = common.bytes_of(logits)
    h = hashlib.sha256(raw)
    if target:
        target.with_suffix(".prefill.f32").write_bytes(raw)
    steps = []
    for i, token in enumerate(arrays):
        start = time.perf_counter_ns()
        logits = model.step(token, state)
        steps.append((time.perf_counter_ns() - start) / 1e6)
        raw = common.bytes_of(logits)
        h.update(raw)
        if target:
            target.with_suffix(f".decode{i:03d}.f32").write_bytes(raw)
    return {"prefill_ms": prefill_ms, "decode_ms": steps,
            "full_logits_sha256": h.hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="existing staged model directory")
    ap.add_argument("--prompts", required=True, help="existing model-leg prompts file")
    ap.add_argument("--out-dir", required=True, help="must not already exist")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=8)
    args = ap.parse_args()
    if args.runs < 3 or args.decode_tokens < 1:
        ap.error("need at least three runs and one decode token")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "mojolearn.lowbit_model_probe.v1", "status": "STARTED",
              "scope": "stateful latency and full-logit identity, fixed contexts; not quality or training",
              "rows": [], "commit": os.environ.get("MOJOLEARN_COMMIT", common.git_commit()),
              "protocol": {"runs": args.runs, "decode_tokens": args.decode_tokens,
                           "arm_order": "alternating, model loaded separately, warmup before timing",
                           "state_allocation_timed": False}}
    try:
        # Fail closed before loading weights. Do not bypass the registry.
        try:
            _numeric_profile.resolve("fixed15_v1", "model_probe", use="inference")
        except NotImplementedError as exc:
            report.update(status="BLOCKED_NOT_INTEGRATED", reason=str(exc))
            return 3
        from mojolearn.models import CausalLM, Tokenizer
        common.require_model_dir(args.model)
        report["model"] = common.model_record(args.model, Path(args.model).name)
        report["box"] = common.box_record()
        report["prompts_sha256"] = common.prompts_sha256(args.prompts)
        tok = Tokenizer.from_pretrained(args.model)
        contexts = []
        for pid, text in common.read_prompts(args.prompts):
            tokens = list(tok.encode(text))
            if len(tokens) < 2:
                raise ValueError(f"prompt {pid} needs at least two tokens")
            n_decode = min(args.decode_tokens, len(tokens) - 1)
            ids = np.array([tokens[:-n_decode]], dtype=np.int32)
            contexts.append((pid, ids, tokens[-n_decode:]))
        profiles = ("fp32_v1", "fixed15_v1")
        hashes = {}
        for repeat in range(args.runs):
            for profile in profiles if repeat % 2 == 0 else profiles[::-1]:
                model = CausalLM.load(args.model, numeric_profile=profile)
                try:
                    if model.numeric_profile != profile:
                        raise AssertionError("model did not retain requested profile")
                    for index, (pid, ids, continuation) in enumerate(contexts):
                        run_once(model, ids, continuation)  # same-shape warmup
                        key = (profile, index)
                        target = out / f"{profile}.prompt{index:03d}" if repeat == 0 else None
                        result = run_once(model, ids, continuation, target)
                        old = hashes.setdefault(key, result["full_logits_sha256"])
                        if old != result["full_logits_sha256"]:
                            raise AssertionError(f"repeated logits changed: {profile}/{pid}")
                        if repeat == 0:
                            tokens = np.concatenate((ids, np.array([continuation], np.int32)), axis=1)
                            (out / f"prompt{index:03d}.tokens.i32").write_bytes(tokens.tobytes())
                        report["rows"].append(dict(result, profile=profile, prompt=pid, repeat=repeat,
                                                   prompt_tokens=int(ids.shape[1])))
                finally:
                    if hasattr(model, "close"):
                        model.close()
                    del model
                    gc.collect()
        report["status"] = "PASS_REPEAT_IDENTITY_CROSS_VENDOR_PENDING"
        report["summary"] = {}
        for profile in profiles:
            report["summary"][profile] = {
                str(pid): {"prefill_median_ms": statistics.median(r["prefill_ms"] for r in report["rows"]
                                                               if r["profile"] == profile and r["prompt"] == pid),
                           "decode_token_median_ms": statistics.median(t for r in report["rows"]
                               if r["profile"] == profile and r["prompt"] == pid for t in r["decode_ms"])}
                for pid, _, _ in contexts}
        return 0
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
