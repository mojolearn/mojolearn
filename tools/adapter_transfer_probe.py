#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time one torch.compile(backend="mojolearn") call per route: its transfer against its compute. A probe, never product code.

    python tools/adapter_transfer_probe.py --device cuda:0|mps|cpu --rows 4096 --features 64 \
        [--hidden 256] [--outputs 8] [--out FILE.json]

The backend (python/mojolearn/adapters/_torch_compile.py, _Compiled) has two per-call routes:

  device  CUDA and ROCm tensors on GPU 0: the inputs' device addresses go to ops_imported_run_device,
          which writes the outputs into torch.empty tensors on the same device (no host copy).
  host    every other tensor (CPU, MPS): native ATen calls stage host tensors;
          their raw buffers enter the program executor and outputs return to the input device.

This probe compiles a Linear-ReLU-Linear model of the given shape, makes one warm call (the import and
weight upload), then times ONE whole compiled(x) call per supported route, closed by a device sync.
The native execution entry records the phases of that same call:

  host    transfer_in_ms   admission, native host copies and output allocation
          compute_ms       raw-buffer program execution (upload, program, download inside mojolearn)
          transfer_out_ms  native ATen copies to the input device
  device  transfer_in_ms   admission, native contiguous copies, allocation and device synchronization
          compute_ms       ops_imported_run_device (device-to-device copies in and out, the program)
          transfer_out_ms  closing device synchronization and current-device restoration

The forced host comparison on CUDA/ROCm runs through the same native entry.
It receives one excluded warmup before its scored call. No Python reconstruction
of the route is used as a substitute for the product execution boundary.

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

    # each route's steps on the program and the callable the warm call made
    entries = [e for per_gm in tc._PROGRAMS.values() for e in per_gm.values()]
    if len(entries) != 1:
        raise SystemExit(f"adapter_transfer_probe: expected one imported program, found {len(entries)}")
    program = entries[0][0]
    compiled_obj = next(o for o in gc.get_objects() if isinstance(o, tc._Compiled) and o._program is program)
    route_taken, reason = compiled_obj._route([x])
    base = dict(device=str(device), rows=a.rows, features=a.features, hidden=a.hidden, outputs=a.outputs,
                numeric_mode=a.numeric_mode or "identical", torch=torch.__version__, route_taken=route_taken,
                clock_scope="native_execution", samples=1, warmups=1)
    rows = []
    context = compiled_obj._execution_context
    context["profile"] = True
    try:
        for route in (["device", "host"] if route_taken == "device" else ["host"]):
            context["force_host"] = route == "host"
            with torch.no_grad():
                if route != route_taken:
                    compiled(x)  # excluded warmup for the forced host route
                sync(torch, device)
                started = time.perf_counter()
                out = compiled(x)
                sync(torch, device)
                call_ms = (time.perf_counter() - started) * 1e3
                measured, incoming_ns, compute_ns, outgoing_ns = context["last_profile"]
                if measured != route:
                    raise RuntimeError(f"native execution took {measured}, requested {route}")
                rows.append(dict(route=route, status="ok", **base, call_ms=round(call_ms, 4),
                                 transfer_in_ms=round(incoming_ns / 1e6, 4),
                                 compute_ms=round(compute_ns / 1e6, 4),
                                 transfer_out_ms=round(outgoing_ns / 1e6, 4)))
                del out
    finally:
        context.pop("profile", None)
        context.pop("force_host", None)
    if route_taken != "device":
        rows.append(dict(route="device", status="refused", reason=reason.replace(" ", "_"), **base))

    for row in rows:
        print("TRANSFER_PROBE " + " ".join(f"{k}={v}" for k, v in row.items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
