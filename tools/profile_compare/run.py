#!/usr/bin/env python3
"""Matched resident generation: explicit profiles, checked outputs, no fallback.

This measures current implementations, not the intrinsic value of precision.
Run resident_gate.py first; experimental HIP is never a shipping default.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools/lowbit_blocks"))
from model_logits import token_ids, raw, SEED
from mojolearn.models import CausalLM
from mojolearn._array import Array
from mojolearn._buffer import empty
from mojolearn import _backend


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--experimental-hip", action="store_true")
    args = ap.parse_args()
    if args.rounds < 3:
        ap.error("at least three alternating rounds required")
    report = {"status": "STARTED", "vendor": _backend.vendor(),
              "commit": os.environ["MOJOLEARN_COMMIT"], "rows": [],
              "experimental_hip": args.experimental_hip,
              "protocol": "both resident; setup included; load/pack excluded; hashes outside clock",
              "scope": "current implementation comparison, not precision-only attribution"}
    output = Path(args.out)
    models = {}
    try:
        profiles = ("fp32_v1", "fixed15_v1")
        for profile in profiles:
            start = time.perf_counter()
            models[profile] = CausalLM.load(args.model, numeric_profile=profile)
            report.setdefault("load_pack_s", {})[profile] = time.perf_counter() - start
            if models[profile].numeric_profile != profile:
                raise AssertionError("requested profile not retained")
        report["model_files"] = {
            p.name: digest(p.read_bytes()) for p in sorted(Path(args.model).glob("*"))
            if p.is_file() and (p.suffix == ".safetensors" or p.name == "config.json")}
        for prompt_length in (32, 512):
            prompt = Array.from_list(token_ids(1, prompt_length,
                                    models[profiles[0]].vocab_size, SEED + 7), "<i4")
            for new_tokens in (1, 32, 128):
                expected = {}
                samples = {p: [] for p in profiles}
                def call(profile):
                    model = models[profile]
                    last = empty((1, model.vocab_size), "<f4")
                    start = time.perf_counter_ns()
                    result = model._generate_resident(
                        prompt, new_tokens, prompt_length + new_tokens, last_logits=last,
                        _experimental_int15_resident=args.experimental_hip)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if result is None:
                        raise RuntimeError(f"BLOCKED: {profile} has no resident path on this box")
                    hashes = {"tokens": digest(result.tobytes()), "last_logits": digest(raw(last))}
                    if hashes != expected.setdefault(profile, hashes):
                        raise AssertionError(f"repeat output changed: {profile}")
                    return elapsed
                for profile in profiles:
                    call(profile)  # untimed warmup for the exact shape
                for repeat in range(args.rounds):
                    order = profiles if repeat % 2 == 0 else profiles[::-1]
                    for profile in order:
                        samples[profile].append(call(profile))
                med = {p: statistics.median(samples[p]) for p in profiles}
                row = {"prompt_tokens": prompt_length, "new_tokens": new_tokens,
                       "prompt_sha256": digest(prompt.tobytes()), "samples_ms": samples,
                       "median_ms": med, "output_hashes": expected,
                       "fixed15_over_fp32": med[profiles[1]] / med[profiles[0]]}
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "PASS_RESIDENT_COMPARISON"
    except Exception as exc:
        report.update(status="FAIL_OR_BLOCKED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
