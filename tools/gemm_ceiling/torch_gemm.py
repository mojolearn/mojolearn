#!/usr/bin/env python3
"""PyTorch's GEMM at the int15 price harness's shapes, GPU-resident, CUDA events.

    python tools/gemm_ceiling/torch_gemm.py <harness.log> <out.json>

Reads every `== <name>  m=.. n=.. k=..` row the harness printed and times
C = A @ B^T (the harness's OP_NT orientation; backward rows are timed the same
way at their printed m, n, k) with operands already on the device: FP32 with
TF32 off, FP32 with TF32 on, and BF16. Median of 30 after 5 warm-ups.
"""
import json, re, statistics, sys
import torch

rows = []
for line in open(sys.argv[1]):
    m = re.match(r"== (\S+)\s+m=(\d+) n=(\d+) k=(\d+)", line)
    if m:
        rows.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
rows.append(("square.8192", 8192, 8192, 8192))
dev = torch.device("cuda")
out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": []}


def timed(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


for name, m, n, k in rows:
    g = torch.Generator(device=dev).manual_seed(7)
    a = torch.randn(m, k, device=dev, generator=g); b = torch.randn(n, k, device=dev, generator=g)
    rec = {"name": name, "m": m, "n": n, "k": k}
    for arm, dt, tf32 in (("torch.fp32", torch.float32, False), ("torch.tf32", torch.float32, True),
                          ("torch.bf16", torch.bfloat16, False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        x, y = a.to(dt), b.to(dt)
        ms = timed(lambda: x @ y.t())
        rec[arm] = {"ms": ms, "tflops": 2 * m * n * k / ms / 1e9}
    out["rows"].append(rec)
    print(json.dumps(rec), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=2)
