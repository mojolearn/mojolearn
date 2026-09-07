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

THE DETERMINISTIC ARM, AND WHAT THE VENDOR ACTUALLY PROMISES (DEVIATION 2109)
=============================================================================
`--arm cublas-deterministic` (2026-09-07, the NVIDIA identity-cost grid)
runs the SAME two precisions under the vendor's DOCUMENTED deterministic
configuration and names them `cublas-fp32-deterministic` and
`cublas-tf32-deterministic`: `CUBLAS_WORKSPACE_CONFIG=:4096:8` is placed in
the environment BEFORE torch is imported (torch is imported inside `main`,
and `vendor_gemm_price` imports it lazily too, so the variable precedes the
runtime; if torch is somehow already imported the process re-executes
itself with the variable set rather than proceeding with a runtime that
was initialized without it), then `torch.use_deterministic_algorithms(True)`,
read back with `torch.are_deterministic_algorithms_enabled()`.

What that column MEANS depends on what cuBLAS already promises by default,
so here is the vendor's statement, fetched 2026-09-07 from
https://docs.nvidia.com/cuda/cublas/index.html, section "Results
Reproducibility" (2.1.4), quoted as the fetch returned it:

    "By design, all cuBLAS API routines from a given toolkit version,
    generate the same bit-wise results at every run when executed on GPUs
    with the same architecture and the same number of SMs. However,
    bit-wise reproducibility is not guaranteed across toolkit versions
    because the implementation might differ due to some implementation
    changes. This guarantee no longer holds when multiple CUDA streams are
    active or fixed-point emulation is used."

and, on what to do when multiple streams are active:

    "set a debug environment variable `CUBLAS_WORKSPACE_CONFIG` to `:16:8`
    (may limit overall performance) or `:4096:8` (will increase library
    footprint in GPU memory by approximately 24MiB)."

with the alternatives "provide a separate workspace for each used stream
using the cublasSetWorkspace() function, or have one cuBLAS handle per
stream". The section also records that `cublas<t>symv()` / `cublas<t>hemv()`
have atomics-based faster implementations that are NOT bit-wise
reproducible; no such routine is called here (these arms are GEMMs).

SO: **cuBLAS's DEFAULT is already documented as run-to-run bitwise
reproducible on one GPU model, on a single stream.** This process is
single-stream, so on the paper's reading the `cublas-*-deterministic`
column is NOT "the price of determinism" the way `n_streams=1` is for
cuML or `use_deterministic_algorithms` is for a torch op with a
nondeterministic default kernel. It is the price of the workspace
configuration the vendor documents for the multi-stream case, applied
here where it should change nothing but the workspace size -- and the
`hash=` column, which this arm turns ON by default (the copy is outside
the timed region), is what says whether the default arm was in fact
repeatable on this box. A `cublas-fp32` row whose hash never moves is the
vendor's promise holding; a `-deterministic` row that is not slower is
the expected result, not a null one. Read the column that way.
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


#: The vendor's documented workspace configuration for deterministic results
#: (docstring: DEVIATION 2109). `:4096:8` rather than `:16:8` because the
#: vendor marks the latter "may limit overall performance" and a column
#: meant to price determinism must not also price a workspace starvation.
CUBLAS_WORKSPACE = ":4096:8"


def _deterministic_env_before_torch(arm):
    """Put `CUBLAS_WORKSPACE_CONFIG` in the environment BEFORE torch exists.

    DEVIATION 2109. cuBLAS reads the variable when its handle is created,
    which torch does lazily on the first CUDA matmul, so setting it before
    `import torch` is sufficient and is also what the torch documentation
    tells a user to do. The re-exec covers the one way that ordering can be
    violated in this file: something imported torch before `main` ran (it
    is not imported at module scope here, and `vendor_gemm_price` imports
    it lazily, so on the leg this branch is never taken -- it exists so the
    guarantee does not depend on that staying true)."""
    if arm != "cublas-deterministic":
        return
    if "torch" in sys.modules and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
        os.execv(sys.executable, [sys.executable] + sys.argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE


def _reason(exc, deterministic):
    """The refusal text. The fast arm's spelling is unchanged (first 100
    characters); the deterministic arm prints the torch error's FIRST LINE
    (that line is where torch names the op without a deterministic
    implementation), which is what a reader of that column needs."""
    if deterministic:
        return " ".join(str(exc).splitlines()[0].split())[:240] if str(exc) else exc.__class__.__name__
    return str(exc)[:100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-macs", type=float, default=0.0,
                    help="skip shapes above this MAC count; 0 means no cap")
    ap.add_argument("--hash", dest="hash", action="store_true", default=None,
                    help="hash the result each round. Costs a device-to-host "
                         "copy OUTSIDE the timed region and is off by default "
                         "for --arm cublas because at these sizes the copy "
                         "dominates the run; ON by default for --arm "
                         "cublas-deterministic, where the hash column is the "
                         "point (DEVIATION 2109).")
    ap.add_argument("--no-hash", dest="hash", action="store_false")
    ap.add_argument("--arm", choices=("cublas", "cublas-deterministic"), default="cublas",
                    help="cublas (default; arms cublas-fp32 and cublas-tf32, "
                         "unchanged) or cublas-deterministic (the same two "
                         "precisions under CUBLAS_WORKSPACE_CONFIG=%s and "
                         "torch.use_deterministic_algorithms(True), named "
                         "cublas-fp32-deterministic / cublas-tf32-deterministic; "
                         "DEVIATION 2109)" % CUBLAS_WORKSPACE)
    args = ap.parse_args()
    deterministic = args.arm == "cublas-deterministic"
    if args.hash is None:
        args.hash = deterministic
    suffix = "-deterministic" if deterministic else ""

    _deterministic_env_before_torch(args.arm)

    try:
        import torch
    except Exception as exc:
        print("FSPEED-REFUSED lane=gemm arm=%s reason=torch import failed: %s"
              % (args.arm, str(exc)[:120]))
        return 0

    try:
        dev, libname, devname, build_s = pick_device(torch)
    except SystemExit as exc:
        print("FSPEED-REFUSED lane=gemm arm=%s reason=%s" % (args.arm, str(exc).splitlines()[0]))
        return 0

    shapes = load_shapes()

    # BOTH ARMS ARE ONLY MEANINGFUL ON CUDA. On MPS and on ROCm the
    # allow_tf32 switch either does nothing or reaches a different mode
    # (CDNA3's XF32), and an arm named tf32 that did not run tf32 is worse
    # than no arm. So off CUDA only one arm runs and it is named for what it
    # is: the backend's default.
    if dev == "cuda" and getattr(torch.version, "hip", None) is None:
        arms = [("cublas-fp32" + suffix, False), ("cublas-tf32" + suffix, True)]
    elif deterministic:
        # The deterministic configuration quoted in the docstring is
        # cuBLAS's. On MPS or ROCm the variable reaches no cuBLAS and the
        # arm name would be a lie; refuse by name rather than time the
        # backend's default under a deterministic label.
        print("FSPEED-REFUSED lane=gemm arm=cublas-deterministic reason=%s is not "
              "cuBLAS; CUBLAS_WORKSPACE_CONFIG has no meaning here and no vendor "
              "deterministic configuration is documented for it in this file"
              % libname)
        return 0
    else:
        arms = [("%s-default" % libname.split("/")[0].lower(), None)]

    if deterministic:
        torch.use_deterministic_algorithms(True)
        got = bool(torch.are_deterministic_algorithms_enabled())
        env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if not got or env != CUBLAS_WORKSPACE:
            # READ BACK, the same rule the tf32 switch obeys below: a
            # deterministic column whose switch did not take is a fast
            # column under the wrong name.
            print("FSPEED-REFUSED lane=gemm arm=cublas-deterministic reason=asked "
                  "for use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG=%s, "
                  "read back enabled=%s env=%s" % (CUBLAS_WORKSPACE, got, env))
            return 0

    for armname, tf32 in arms:
        print("FSPEED-HEADER family=gemm lane=gemm arm=%s mode=FAST device=%s "
              "rounds=%d size=shipped" % (armname, devname, args.rounds))
        print("FSPEED-NOTE lane=gemm arm=%s library=%s build=%s torch=%s "
              "allow_tf32=%s" % (armname, libname, build_s, torch.__version__, tf32))
        if deterministic:
            print("FSPEED-NOTE lane=gemm arm=%s deterministic=on "
                  "use_deterministic_algorithms=%s CUBLAS_WORKSPACE_CONFIG=%s; the "
                  "vendor documents its DEFAULT as bit-wise reproducible run to "
                  "run on one GPU model on a single stream (docstring, DEVIATION "
                  "2109), so this column prices the documented multi-stream "
                  "workspace configuration, not a kernel change"
                  % (armname, torch.are_deterministic_algorithms_enabled(),
                     os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
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
                      "allocate: %s" % (armname, sh["name"], _reason(exc, deterministic)))
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
                # Under the deterministic arm a torch op with no deterministic
                # implementation raises RuntimeError here; the first line of
                # that error names the op and the row continues.
                print("FSPEED-REFUSED lane=gemm arm=%s reason=%s raised: %s"
                      % (armname, sh["name"], _reason(exc, deterministic)))
            finally:
                del a, b
                if dev == "cuda":
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
