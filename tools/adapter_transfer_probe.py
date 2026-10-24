#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time one torch.compile(backend="mojolearn") call: its transfer against its compute. A probe, never product code.

    python tools/adapter_transfer_probe.py --device cuda:0|mps|cpu --rows 4096 --features 64 \
        [--hidden 256] [--outputs 8] [--out FILE.json]

The backend's per-call path (python/mojolearn/adapters/_torch_compile.py, _Compiled.__call__) copies each
input to a host NumPy array, runs the imported program, and rebuilds each output as a tensor on the input's
device. This probe compiles a Linear-ReLU-Linear model of the given shape, makes one warm call (the import
and weight upload), then times ONE call split into those three steps, each closed by a device sync:

    transfer_in_ms    x.detach().cpu().contiguous().numpy() for every input
    compute_ms        ImportedProgram.run on the host arrays (upload, program, download inside mojolearn)
    transfer_out_ms   torch.tensor(y, device=input device) for every output
    call_ms           one whole compiled(x) call, for the sum check

It prints one line `TRANSFER_PROBE device=... rows=... transfer_in_ms=... compute_ms=... transfer_out_ms=...
call_ms=...` and writes the same fields as JSON to --out. The numbers go in the `# measured` note beside the
transfer path in _torch_compile.py; zero-copy device handoff is a later lane. One call per arm (CLAUDE.md
"Measurements run ONE run per arm"); no repeats.
"""
import argparse
import json
import pathlib
import sys
import time


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps" if sys.platform == "darwin" else "cuda:0")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--features", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--outputs", type=int, default=8)
    ap.add_argument("--numeric-mode", default=None)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from mojolearn.adapters import _torch_compile as tc

    tc.register()
    device = torch.device(a.device)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(a.features, a.hidden), torch.nn.ReLU(),
                              torch.nn.Linear(a.hidden, a.outputs)).eval().to(device)
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((a.rows, a.features), dtype=np.float32)).to(device)
    options = {} if a.numeric_mode is None else {"numeric_mode": a.numeric_mode}
    compiled = torch.compile(net, backend="mojolearn", fullgraph=True, dynamic=False, options=options or None)
    with torch.no_grad():
        compiled(x)  # warm: capture, import, weight upload
        sync(torch, device)
        t = time.perf_counter()
        compiled(x)
        sync(torch, device)
        call_ms = (time.perf_counter() - t) * 1e3

    # the same three steps as _Compiled.__call__, on the program the warm call imported
    entries = [e for per_gm in tc._PROGRAMS.values() for e in per_gm.values()]
    if len(entries) != 1:
        raise SystemExit(f"adapter_transfer_probe: expected one imported program, found {len(entries)}")
    program = entries[0][0]
    sync(torch, device)
    t0 = time.perf_counter()
    arrays = [x.detach().cpu().contiguous().numpy()]
    t1 = time.perf_counter()
    got = program.run(*arrays)
    t2 = time.perf_counter()
    got = got if isinstance(got, tuple) else (got,)
    outs = tuple(torch.tensor(y, device=device) for y in got)
    sync(torch, device)
    t3 = time.perf_counter()
    del outs

    row = dict(device=str(device), rows=a.rows, features=a.features, hidden=a.hidden, outputs=a.outputs,
               numeric_mode=a.numeric_mode or "identical", torch=torch.__version__,
               transfer_in_ms=round((t1 - t0) * 1e3, 4), compute_ms=round((t2 - t1) * 1e3, 4),
               transfer_out_ms=round((t3 - t2) * 1e3, 4), call_ms=round(call_ms, 4))
    print("TRANSFER_PROBE " + " ".join(f"{k}={v}" for k, v in row.items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(row, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
