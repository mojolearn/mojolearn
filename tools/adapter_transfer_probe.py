#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time one torch.compile(backend="mojolearn") call per route: its transfer against its compute. A probe, never product code.

    python tools/adapter_transfer_probe.py --device cuda:0|mps|cpu --rows 4096 --features 64 \
        [--hidden 256] [--outputs 8] [--out FILE.json]

The backend (python/mojolearn/adapters/_torch_compile.py, _Compiled) has two per-call routes:

  device  CUDA and ROCm tensors on GPU 0: the inputs' device addresses go to ops_imported_run_device,
          which writes the outputs into torch.empty tensors on the same device (no host copy).
  host    every other tensor (CPU, MPS): each input is copied to a host NumPy array, ImportedProgram.run
          runs, and each output is rebuilt as a tensor on the input's device.

This probe compiles a Linear-ReLU-Linear model of the given shape, makes one warm call (the import and
weight upload), times ONE whole compiled(x) call, then times each route's steps once, each closed by a
device sync:

  host    transfer_in_ms   x.detach().cpu().contiguous().numpy() for every input
          compute_ms       ImportedProgram.run on the host arrays (upload, program, download inside mojolearn)
          transfer_out_ms  torch.tensor(y, device=input device) for every output
  device  transfer_in_ms   x.detach().contiguous() and the torch.empty outputs, then the stream drain
          compute_ms       ops_imported_run_device (device-to-device copies in and out, the program)
          transfer_out_ms  the closing stream drain

It prints one line per route, `TRANSFER_PROBE route=<device|host> device=... rows=... transfer_in_ms=...
compute_ms=... transfer_out_ms=... call_ms=...` (call_ms: the whole compiled call, which takes the route
_Compiled._route picks; route_taken names it). A route this device cannot take prints `status=refused
reason=...` instead of times (the device route on MPS and CPU). --out writes both rows as JSON. The numbers
go in the `# measured:` block beside _Compiled.__call__. One call per arm (CLAUDE.md "Measurements run ONE
run per arm"); no repeats.
"""
import argparse
import gc
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

    # each route's steps on the program and the callable the warm call made
    entries = [e for per_gm in tc._PROGRAMS.values() for e in per_gm.values()]
    if len(entries) != 1:
        raise SystemExit(f"adapter_transfer_probe: expected one imported program, found {len(entries)}")
    program = entries[0][0]
    compiled_obj = next(o for o in gc.get_objects() if isinstance(o, tc._Compiled) and o._program is program)
    route_taken, reason = compiled_obj._route([x])
    base = dict(device=str(device), rows=a.rows, features=a.features, hidden=a.hidden, outputs=a.outputs,
                numeric_mode=a.numeric_mode or "identical", torch=torch.__version__, route_taken=route_taken,
                call_ms=round(call_ms, 4))
    rows = []

    # host route
    with torch.no_grad():
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
    rows.append(dict(route="host", status="ok", **base, transfer_in_ms=round((t1 - t0) * 1e3, 4),
                     compute_ms=round((t2 - t1) * 1e3, 4), transfer_out_ms=round((t3 - t2) * 1e3, 4)))

    # device route
    if route_taken != "device":
        rows.append(dict(route="device", status="refused", reason=reason.replace(" ", "_"), **base))
    else:
        with torch.no_grad():
            sync(torch, device)
            t0 = time.perf_counter()
            xs = [x.detach().contiguous()]
            outs = tuple(torch.empty(tuple(shape), dtype=torch.int32 if dt == "int32" else torch.float32,
                                     device=device) for _n, shape, dt in program.outputs)
            stream = torch.cuda.current_stream(device)
            stream.synchronize()
            t1 = time.perf_counter()
            program._run_device(program, [(v.data_ptr(), tuple(v.shape), tc._dtype_name(v.dtype)) for v in xs],
                                [y.data_ptr() for y in outs])
            t2 = time.perf_counter()
            stream.synchronize()
            t3 = time.perf_counter()
            del outs, xs
        rows.append(dict(route="device", status="ok", **base, transfer_in_ms=round((t1 - t0) * 1e3, 4),
                         compute_ms=round((t2 - t1) * 1e3, 4), transfer_out_ms=round((t3 - t2) * 1e3, 4)))

    for row in rows:
        print("TRANSFER_PROBE " + " ".join(f"{k}={v}" for k, v in row.items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
