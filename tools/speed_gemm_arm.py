# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The cuBLAS arm of the FAST speed lane, in FSPEED lines.

    python3 tools/speed_gemm_arm.py                       # both arms
    python3 tools/speed_gemm_arm.py --rounds 10 --max-macs 5e11

WHY THIS EXISTS BESIDE tools/vendor_gemm_price.py RATHER THAN INSIDE IT
=======================================================================
`vendor_gemm_price.py` is the IDENTITY lane's vendor arm. It answers "what
does the pin cost against the library", it prints `VENDORPRICE` lines, and it
reports one median per shape. This run asks a different question -- what does
the FAST path, which is the arm an ordinary mojolearn user gets, cost against
cuBLAS on the vendor's own silicon -- and it needs PER ROUND lines so the
shared table (`tools/fast_speed_table.py`) can show the spread and catch a
box that was throttling.

So the SHAPE TABLE AND THE DEVICE DETECTION ARE IMPORTED, not copied. That
file already parses `bench/gemm_shapes.mojo`, which is the single source of
truth for the twenty shapes, and a second hand-maintained copy is a table
that drifts. What is written fresh here is only the timing loop, which is a
dozen lines and has to emit per round.

THE ONE THING THAT WILL RUIN THIS BENCHMARK IF IT IS MISSED
===========================================================
**TF32.** On Ampere and later cuBLAS may satisfy an FP32 matmul with TF32
tensor cores: 10 explicit mantissa bits instead of 23. Measured in this
repository on an H100 at `llama8b.qkv.t512`: 44.4 TFLOP/s with
`allow_tf32=False` and 207.5 TFLOP/s with it on. That is a factor of five,
and it is a precision cut rather than an optimization.

Both are timed and both are reported as separate arms, `cublas-fp32` and
`cublas-tf32`, and the table is expected to be read with the arm name in
view. Which one is the fair opponent depends on what our arm did:

  * Our IDENTICAL kernel is strict FP32 by contract, so `cublas-fp32` is its
    opponent and `cublas-tf32` would charge our contract for someone else's
    precision cut.
  * Our FAST path calls MAX's `linalg.matmul`, which on an H100 measured 200
    TFLOP/s at that same shape -- matching the TF32 column and not the FP32
    one. **So the FAST arm's honest opponent is `cublas-tf32`.** This run is
    the FAST arm, and that is why both columns are here rather than only the
    strict one.

This file never decides which comparison to quote. It measures both and
labels them, and the label is what makes the table readable.
"""

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The shape table and the device detection come from the identity lane's
# vendor arm so there is exactly one parser for bench/gemm_shapes.mojo.
from vendor_gemm_price import (  # noqa: E402
    OP_NN,
    OP_NT,
    OP_TN,
    load_shapes,
    pick_device,
)


def fnv1a64(data):
    """The same recurrence core/identity_trace.mojo uses, byte at a time."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def build(torch, dev, sh):
    """Operands in the orientation the row asks for.

    Transcribed from `vendor_gemm_price.time_matmul` so the library is handed
    the SAME logical product our kernel computes. It is a transcription and
    it says so; the alternative was to import a function whose contract is a
    median rather than a call.
    """
    m, n, k, op = sh["m"], sh["n"], sh["k"], sh["op"]
    g = torch.Generator(device="cpu").manual_seed(0x5EED0000 + sh["i"])
    if op == OP_NT:
        a = torch.rand(m, k, generator=g, dtype=torch.float32).to(dev)
        b = torch.rand(n, k, generator=g, dtype=torch.float32).to(dev)
        return a, b, (lambda: torch.matmul(a, b.t()))
    if op == OP_TN:
        a = torch.rand(k, m, generator=g, dtype=torch.float32).to(dev)
        b = torch.rand(k, n, generator=g, dtype=torch.float32).to(dev)
        return a, b, (lambda: torch.matmul(a.t(), b))
    a = torch.rand(m, k, generator=g, dtype=torch.float32).to(dev)
    b = torch.rand(k, n, generator=g, dtype=torch.float32).to(dev)
    return a, b, (lambda: torch.matmul(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-macs", type=float, default=0.0,
                    help="skip shapes above this MAC count; 0 means no cap")
    ap.add_argument("--hash", action="store_true",
                    help="hash the result each round. Costs a device-to-host "
                         "copy OUTSIDE the timed region and is off by default "
                         "because at these sizes the copy dominates the run.")
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:
        print("FSPEED-REFUSED lane=gemm arm=cublas reason=torch import failed: %s"
              % str(exc)[:120])
        return 0

    try:
        dev, libname, devname, build_s = pick_device(torch)
    except SystemExit as exc:
        print("FSPEED-REFUSED lane=gemm arm=cublas reason=%s" % str(exc).splitlines()[0])
        return 0

    shapes = load_shapes()

    # BOTH ARMS ARE ONLY MEANINGFUL ON CUDA. On MPS and on ROCm the
    # allow_tf32 switch either does nothing or reaches a different mode
    # (CDNA3's XF32), and an arm named tf32 that did not run tf32 is worse
    # than no arm. So off CUDA only one arm runs and it is named for what it
    # is: the backend's default.
    if dev == "cuda" and getattr(torch.version, "hip", None) is None:
        arms = [("cublas-fp32", False), ("cublas-tf32", True)]
    else:
        arms = [("%s-default" % libname.split("/")[0].lower(), None)]

    for armname, tf32 in arms:
        print("FSPEED-HEADER family=gemm lane=gemm arm=%s mode=FAST device=%s "
              "rounds=%d size=shipped" % (armname, devname, args.rounds))
        print("FSPEED-NOTE lane=gemm arm=%s library=%s build=%s torch=%s "
              "allow_tf32=%s" % (armname, libname, build_s, torch.__version__, tf32))
        if tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            # READ IT BACK. Setting a backend flag and the backend honoring it
            # are two claims, and torch has moved this switch's spelling more
            # than once. A run whose tf32 arm silently stayed strict would
            # print two identical columns and read as "tf32 does not help".
            got = torch.backends.cuda.matmul.allow_tf32
            if got != tf32:
                print("FSPEED-REFUSED lane=gemm arm=%s reason=allow_tf32 asked "
                      "for %s and reads back %s" % (armname, tf32, got))
                continue

        for sh in shapes:
            macs = float(sh["m"]) * float(sh["n"]) * float(sh["k"])
            if args.max_macs and macs > args.max_macs:
                print("FSPEED-NOTE lane=gemm arm=%s shape=%s SKIPPED %.3g MACs "
                      "above --max-macs %.3g" % (armname, sh["name"], macs, args.max_macs))
                continue
            try:
                a, b, call = build(torch, dev, sh)
            except Exception as exc:
                print("FSPEED-REFUSED lane=gemm arm=%s reason=%s did not "
                      "allocate: %s" % (armname, sh["name"], str(exc)[:100]))
                continue

            def sync():
                if dev == "cuda":
                    torch.cuda.synchronize()
                elif dev == "mps":
                    torch.mps.synchronize()

            try:
                for _ in range(args.warmup):
                    out = call()
                sync()
                # The warm-up is timed and printed, never averaged in. On a
                # cold context the first matmul pays for kernel selection and
                # a reader who cannot see that cannot tell it from a cost.
                t0 = time.perf_counter()
                out = call()
                sync()
                print("FSPEED-WARMUP lane=gemm arm=%s shape=%s ms=%.6f"
                      % (armname, sh["name"], (time.perf_counter() - t0) * 1000.0))

                for r in range(1, args.rounds + 1):
                    t0 = time.perf_counter()
                    out = call()
                    sync()
                    ms = (time.perf_counter() - t0) * 1000.0
                    h = "-"
                    if args.hash:
                        h = "%016x" % fnv1a64(
                            out.detach().to("cpu").contiguous().numpy().tobytes())
                    print("FSPEED lane=gemm arm=%s shape=%s round=%d ms=%.6f hash=%s"
                          % (armname, sh["name"], r, ms, h))
            except Exception as exc:
                print("FSPEED-REFUSED lane=gemm arm=%s reason=%s raised: %s"
                      % (armname, sh["name"], str(exc)[:100]))
            finally:
                del a, b
                if dev == "cuda":
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
