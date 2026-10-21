#!/usr/bin/env python3
"""Public API diagnostic, NOT a whole-model speed or quality benchmark.

Includes Python dispatch, allocation, transfers and activation conversion.
The bindings synchronize and copy output to host before returning. Packed
weights are reused as host planes; this is NOT resident GPU weight storage.
Synthetic inputs diagnose integration costs, not corpus-level performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
from mojolearn import linalg, _numeric_profile


def bits(x):
    return np.asarray(x).tobytes(order="C")


def digest(x):
    return hashlib.sha256(bits(x)).hexdigest()


def values(rows, cols, salt):
    # No random-generator/version dependence; moderate exponents, non-dyadic
    # quantization relative to the row scale, all finite and normal.
    i = np.arange(rows * cols, dtype=np.uint64)
    u = ((i + salt) * 1664525 + 1013904223) & 0xFFFFFFFF
    x = ((u & 0xFFFFF).astype(np.int64) - 524288).astype(np.float32)
    return (x * np.float32(2.0 ** -18)).reshape(rows, cols)


def reference_sample(a, b):
    def quantize(x):
        amax = np.abs(x).max(axis=1)
        e = np.floor(np.log2(amax)).astype(np.int32) - 13
        q = np.clip(np.rint(np.ldexp(x.astype(np.float64), -e[:, None])),
                    -16383, 16383).astype(np.int64)
        return q, e
    qa, ea = quantize(a)
    qb, eb = quantize(b)
    exact = qa @ qb.T  # These sampled products are bounded below 2**53.
    rounded = exact.astype(np.float64).astype(np.float32)
    return np.ldexp(rounded, ea[:, None] + eb[None, :]).astype(np.float32)


def measure(fn):
    start = time.perf_counter_ns()
    result = fn()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return elapsed, result


def run_shape(name, m, n, k, repeats):
    a, b = values(m, k, 11), values(n, k, 73)
    pack_ms, packed = measure(lambda: tuple(linalg.quantize_int15(b)))
    candidate = lambda: linalg.matmul_int15(a, packed)
    baseline = lambda: linalg.matmul(a, b, transpose_b=True)
    expected = candidate()
    baseline_expected = baseline()
    assert bits(expected) == bits(linalg.matmul_int15(a, b)), "packed/raw disagreement"
    sample_rows = sorted(set((0, m // 2, m - 1)))
    sample_cols = sorted(set((0, n // 2, n - 1)))
    want = reference_sample(a[sample_rows], b[sample_cols])
    got = np.asarray(expected)[np.ix_(sample_rows, sample_cols)]
    assert bits(got) == bits(want), "sampled integer reference disagreement"
    corrupt = got.copy()
    corrupt.view(np.uint32).flat[0] ^= np.uint32(1)
    assert bits(corrupt) != bits(want), "comparison failed to detect changed bit"
    if m > 1:
        for idx in sample_rows:
            one = linalg.matmul_int15(np.ascontiguousarray(a[idx:idx + 1]), packed)
            assert bits(one) == bits(np.asarray(expected)[idx:idx + 1]), "row/batch disagreement"
    samples = {"fp32_v1": [], "fixed15_v1": []}
    for repeat in range(repeats):
        order = ("fp32_v1", "fixed15_v1") if repeat % 2 == 0 else ("fixed15_v1", "fp32_v1")
        for profile in order:
            ms, out = measure(baseline if profile == "fp32_v1" else candidate)
            truth = baseline_expected if profile == "fp32_v1" else expected
            assert bits(out) == bits(truth), f"{profile}: output changed across repeats"
            samples[profile].append(ms)
    medians = {p: statistics.median(v) for p, v in samples.items()}
    return {"name": name, "shape_mnk": [m, n, k], "samples_ms": samples,
            "median_ms": medians, "fixed15_over_fp32": medians["fixed15_v1"] / medians["fp32_v1"],
            "one_time_weight_pack_ms": pack_ms, "input_sha256": [digest(a), digest(b)],
            "output_sha256": {"fixed15_v1": digest(expected), "fp32_v1": digest(baseline_expected)},
            "checks": "sampled exact reference, bit-flip comparator, packed/raw, row invariance, repeats"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if args.repeats < 3:
        ap.error("at least three repetitions are required")
    linalg.require_identical()
    row = _numeric_profile.PROFILES["fixed15_v1"]
    report = {"schema": "mojolearn.lowbit_public_probe.v1", "host": platform.node(),
              "commit": os.environ.get("MOJOLEARN_COMMIT", "UNRECORDED"),
              "scope": "synthetic public-API GEMM diagnostic; NOT a model or training-step benchmark",
              "weight_residency": "packed host planes, uploaded by each public call",
              "model_inference": "AVAILABLE_REQUIRES_MODEL_GATE" if row["inference"] else "BLOCKED_NOT_INTEGRATED",
              "model_training": "AVAILABLE_REQUIRES_TRAINING_GATE" if row["training"] else "BLOCKED_UNSUPPORTED",
              "rows": []}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shapes = [("attention_projection_decode1", 1, 960, 960),
              ("attention_projection_decode8", 8, 960, 960),
              ("attention_projection_prefill512", 512, 960, 960),
              ("feedforward_down_prefill512", 512, 960, 2560),
              ("output_head_decode1", 1, 49152, 960)]
    try:
        for shape in shapes:
            result = run_shape(*shape, repeats=args.repeats)
            report["rows"].append(result)
            print(json.dumps(result), flush=True)
            out.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "PASS_PUBLIC_API_ONLY"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "model_inference", "model_training")}), flush=True)


if __name__ == "__main__":
    main()
