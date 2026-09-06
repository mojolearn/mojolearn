# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The VENDOR LIBRARY arm: cuBLAS, hipBLASLt or MPS on the same shapes.

    python3 tools/vendor_gemm_price.py                 # auto-detect backend
    python3 tools/vendor_gemm_price.py --repeats 20
    python3 tools/vendor_gemm_price.py --out card.tsv

WHY THIS FILE EXISTS AND WHY IT IS NOT PART OF gemm_price_main.mojo
===================================================================
`bench/gemm_price_main.mojo`'s `vendor` arm is MAX's `linalg.matmul`. That is
a real and fair baseline -- it is what `core/gemm.mojo` actually calls under
FAST, so it is the thing a mojolearn user gets -- but it is NOT cuBLAS, and a
paper that writes "against CUDA" while timing MAX has mislabelled its own
control. Modular's kernels are Modular's. Whether `linalg.matmul` dispatches
to cuBLAS internally is not something this repository knows, so the question
is settled by measuring the vendor library directly through the one binding
that reliably reaches it on all three platforms, which is torch.

    torch.matmul on device cuda   -> cuBLAS      (NVIDIA)
    torch.matmul on device cuda   -> hipBLASLt   (AMD, ROCm torch)
    torch.matmul on device mps    -> MPS/MPSGraph (Apple)

THE SHAPE TABLE IS NOT COPIED HERE. It is PARSED out of
`bench/gemm_shapes.mojo`, which stays the single source of truth. A second
hand-maintained copy of twenty shapes is a table that drifts, and a drifted
baseline is worse than no baseline because it still prints numbers.

TF32 IS THE WHOLE ARGUMENT ON NVIDIA AND IT IS MEASURED BOTH WAYS
=================================================================
On Ampere and later, cuBLAS may satisfy an FP32 matmul with TF32 tensor cores:
10 explicit mantissa bits instead of 23. It is faster and it is not FP32. Our
kernel is strict FP32 by contract. Comparing strict FP32 against TF32 and
calling the gap "the cost of identity" would charge our contract for someone
else's precision cut.

So both are timed and both are reported, as separate arms:

    strict   torch.backends.cuda.matmul.allow_tf32 = False   (true FP32)
    tf32     torch.backends.cuda.matmul.allow_tf32 = True    (the default a
             user usually gets, and NOT the same arithmetic)

On Apple the flag does not apply. On AMD it DOES -- CDNA3 has an XF32 matrix
mode that the same `allow_tf32` switch reaches -- but this harness does not
touch the flag on the hip backend, so the AMD row is torch's default, which is
off. That makes it strict FP32 BY DEFAULT rather than by assertion, and an
XF32 arm is OWED there. Only `strict` is emitted on both, and the reason is
printed rather than left as a missing row.

IT REFUSES RATHER THAN FALLING BACK
====================================
A CPU torch on a rented GPU box would produce a perfectly good millisecond
for the wrong device, and nothing downstream could tell. This script asserts
an accelerator is present, prints the device name and the torch build
(`torch.version.cuda` / `torch.version.hip`), and exits non-zero otherwise.

WHAT THIS DOES NOT MEASURE. One library call at one shape on one box. No
epilogue, no fusion, no batching, no autotune warm cache beyond the warm-up
below, and no claim that these shapes are the ones a served model spends its
time in. It is a denominator for the identity price and nothing more.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES_MOJO = os.path.join(REPO, "bench", "gemm_shapes.mojo")

OP_NT, OP_TN, OP_NN = 0, 1, 2
OP_NAME = {OP_NT: "NT", OP_TN: "TN", OP_NN: "NN"}


# ---------------------------------------------------------------------------
# Parsing bench/gemm_shapes.mojo, so there is exactly one shape table
# ---------------------------------------------------------------------------
def _ladder(src, fn):
    """Evaluate one `gemm_shape_*` if-ladder for every i, from the source.

    The ladders are deliberately simple and the parser refuses anything it
    does not recognize rather than guessing. A silently mis-parsed shape is
    the failure this whole file exists to avoid.
    """
    m = re.search(r"^def %s\(i: Int\)[^\n]*:\n(.*?)(?=\n\ndef |\n\n#|\Z)" % fn, src, re.S | re.M)
    if not m:
        raise SystemExit("vendor_gemm_price: cannot find %s in %s" % (fn, SHAPES_MOJO))
    body = m.group(1)
    rules = []          # (predicate, value)
    default = None
    for raw in body.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith('"'):
            continue
        eq = re.match(r"if i == (\d+):$", line)
        anyof = re.match(r"if i == (\d+)(?: or i == (?:\d+))+:$", line)
        rng = re.match(r"if i >= (\d+) and i <= (\d+):$", line)
        le = re.match(r"if i <= (\d+):$", line)
        ret = re.match(r"return (.+?)(?:\s*#.*)?$", line)
        if anyof:
            # `if i == 8 or i == 11 or i == 14 or i == 17:` -- the transformer
            # rows, where m is the TOKEN COUNT and the same count recurs at
            # qkv, mlp_up, mlp_down and lm_head. Matched BEFORE the plain `eq`
            # rule, which would otherwise fail the anchor and fall through to
            # the raise.
            rules.append(("pending", ("set", tuple(int(v) for v in re.findall(r"i == (\d+)", line)))))
        elif eq:
            rules.append(("pending", ("eq", int(eq.group(1)))))
        elif rng:
            rules.append(("pending", ("rng", int(rng.group(1)), int(rng.group(2)))))
        elif le:
            rules.append(("pending", ("le", int(le.group(1)))))
        elif ret:
            val = ret.group(1).strip()
            if val.startswith("String("):
                val = val[len("String("):].rstrip(")").strip('"')
            elif val in ("OP_NT", "OP_TN", "OP_NN"):
                val = {"OP_NT": OP_NT, "OP_TN": OP_TN, "OP_NN": OP_NN}[val]
            else:
                # digit separators only; strip AFTER the symbol cases, or
                # `OP_TN` becomes `OPTN` and int() raises on it
                val = int(val.replace("_", ""))
            if rules and rules[-1][0] == "pending":
                rules[-1] = (rules[-1][1], val)
            else:
                default = val
        else:
            raise SystemExit("vendor_gemm_price: unparsed line in %s: %r" % (fn, line))
    if default is None:
        raise SystemExit("vendor_gemm_price: %s has no trailing default return" % fn)

    def evaluate(i):
        for pred, val in rules:
            if pred[0] == "eq" and i == pred[1]:
                return val
            if pred[0] == "set" and i in pred[1]:
                return val
            if pred[0] == "rng" and pred[1] <= i <= pred[2]:
                return val
            if pred[0] == "le" and i <= pred[1]:
                return val
        return default

    return evaluate


def load_shapes():
    src = open(SHAPES_MOJO).read()
    cnt = re.search(r"comptime GEMM_SHAPE_COUNT = (\d+)", src)
    if not cnt:
        raise SystemExit("vendor_gemm_price: no GEMM_SHAPE_COUNT")
    n = int(cnt.group(1))
    getm, getn, getk = _ladder(src, "gemm_shape_m"), _ladder(src, "gemm_shape_n"), _ladder(src, "gemm_shape_k")
    getop, getname = _ladder(src, "gemm_shape_op"), _ladder(src, "gemm_shape_name")
    return [
        {"i": i, "name": getname(i), "op": getop(i), "m": getm(i), "n": getn(i), "k": getk(i)}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# The device, named out loud, or nothing
# ---------------------------------------------------------------------------
def pick_device(torch):
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        hip = getattr(torch.version, "hip", None)
        cuda = getattr(torch.version, "cuda", None)
        if hip:
            return "cuda", "hipBLASLt/rocBLAS", name, "ROCm " + str(hip)
        return "cuda", "cuBLAS", name, "CUDA " + str(cuda)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "MPS/MPSGraph", "Apple GPU", "torch " + torch.__version__
    raise SystemExit(
        "vendor_gemm_price: REFUSED. No accelerator visible to torch, and a CPU\n"
        "matmul timed here would be a perfectly good number for the wrong device.\n"
        "torch %s, cuda_available=%s" % (torch.__version__, torch.cuda.is_available())
    )


def time_matmul(torch, dev, sh, repeats, warmup):
    """One shape, median of `repeats` timed calls after `warmup` untimed ones.

    Operands are built in the orientation the row asks for, so the library is
    handed the SAME logical product our kernel computes:
        NT  A (m,k) . B (n,k)^T
        TN  A (k,m)^T . B (k,n)
        NN  A (m,k) . B (k,n)
    """
    m, n, k = sh["m"], sh["n"], sh["k"]
    op = sh["op"]
    g = torch.Generator(device="cpu").manual_seed(0x5EED0000 + sh["i"])
    if op == OP_NT:
        a = torch.rand(m, k, generator=g, dtype=torch.float32).to(dev)
        b = torch.rand(n, k, generator=g, dtype=torch.float32).to(dev)
        call = lambda: torch.matmul(a, b.t())
    elif op == OP_TN:
        a = torch.rand(k, m, generator=g, dtype=torch.float32).to(dev)
        b = torch.rand(k, n, generator=g, dtype=torch.float32).to(dev)
        call = lambda: torch.matmul(a.t(), b)
    else:
        a = torch.rand(m, k, generator=g, dtype=torch.float32).to(dev)
        b = torch.rand(k, n, generator=g, dtype=torch.float32).to(dev)
        call = lambda: torch.matmul(a, b)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        call()
    sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    del a, b
    return statistics.median(samples), min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-macs", type=float, default=0.0,
                    help="skip (and REPORT) rows above this many MACs; 0 = no cap")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("vendor_gemm_price: REFUSED. torch is not importable in this environment.")

    dev, libname, devname, build = pick_device(torch)
    shapes = load_shapes()

    arms = [("strict", False)]
    tf32_applies = dev == "cuda" and getattr(torch.version, "hip", None) is None
    if tf32_applies:
        arms.append(("tf32", True))

    print("== tools/vendor_gemm_price.py ==")
    print("  device      : %s (%s)" % (devname, dev))
    print("  library     : %s" % libname)
    print("  build       : %s, torch %s" % (build, torch.__version__))
    print("  shapes      : %d, parsed from bench/gemm_shapes.mojo" % len(shapes))
    print("  repeats     : %d timed, %d warm-up, median reported" % (args.repeats, args.warmup))
    if tf32_applies:
        print("  arms        : strict (allow_tf32=False, true FP32) and tf32 (allow_tf32=True,")
        print("                10 explicit mantissa bits -- NOT the same arithmetic as ours)")
    else:
        print("  arms        : strict only, meaning torch's DEFAULT for this backend.")
        print("                On ROCm that default is off, but CDNA3 does have an XF32")
        print("                matrix mode that the same allow_tf32 switch reaches, so")
        print("                this is strict FP32 by default rather than by assertion.")
        print("                An XF32 arm is OWED here the way tf32 is measured on CUDA.")
    print()

    rows = []
    for sh in shapes:
        macs = float(sh["m"]) * sh["n"] * sh["k"]
        if args.max_macs and macs > args.max_macs:
            print("  SKIPPED %-32s %s  m=%d n=%d k=%d  (%.3g MACs > --max-macs %.3g)"
                  % (sh["name"], OP_NAME[sh["op"]], sh["m"], sh["n"], sh["k"], macs, args.max_macs))
            rows.append(dict(sh, skipped=True))
            continue
        rec = dict(sh, skipped=False, macs=macs)
        for armname, tf32 in arms:
            if tf32_applies:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                torch.backends.cudnn.allow_tf32 = tf32
            try:
                med, best = time_matmul(torch, dev, sh, args.repeats, args.warmup)
            except RuntimeError as e:
                rec[armname] = None
                rec[armname + "_error"] = str(e)[:160]
                print("  FAILED  %-32s arm=%s  %s" % (sh["name"], armname, str(e)[:100]))
                continue
            rec[armname] = med
            rec[armname + "_best"] = best
            gf = (2.0 * macs / 1e9) / (med / 1000.0)
            print("VENDORPRICE %-10s %-32s %s m=%-7d n=%-7d k=%-8d  %10.4f ms  %9.1f GF/s"
                  % (armname, sh["name"], OP_NAME[sh["op"]], sh["m"], sh["n"], sh["k"], med, gf))
        rows.append(rec)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"device": devname, "backend": dev, "library": libname,
                       "build": build, "torch": torch.__version__,
                       "repeats": args.repeats, "rows": rows}, f, indent=1)
        print()
        print("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
