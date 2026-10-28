#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure one public ImportedProgram.run_torch call, including buffer transfers.

The explicit ONNX export/import is preparation. One excluded warmup precedes
one scored whole call, closed by device synchronization. call_ms includes native
admission, allocation, transfers, execution and returned output. This bounded
probe is not a full-workload performance qualification. Phase times are not
inferred from the whole clock. --out retains the input artifact and result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps" if sys.platform == "darwin" else "cuda:0")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--outputs", type=int, default=8)
    parser.add_argument("--numeric-mode", default="identical")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from mojolearn import import_model
    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact = args.out.with_suffix(".onnx")
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(args.features, args.hidden), torch.nn.ReLU(),
                                torch.nn.Linear(args.hidden, args.outputs)).eval().to(device)
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((args.rows, args.features), dtype=np.float32)).to(device)
    torch.onnx.export(model, (x,), str(artifact), opset_version=17, dynamo=False)
    program = import_model(str(artifact), numeric_mode=args.numeric_mode)
    route, reason = program._run_torch_native(program, (x,), None, True)
    with torch.no_grad():
        program.run_torch(x)
        sync(torch, device)
        started = time.perf_counter()
        result = program.run_torch(x)
        sync(torch, device)
        elapsed = (time.perf_counter() - started) * 1e3
    arrays = args.out.with_suffix(".npz")
    np.savez(arrays, inputs=x.detach().cpu().numpy(), outputs=result.detach().cpu().numpy())
    record = dict(status="ok", route=route, route_reason=reason, device=str(device),
                  rows=args.rows, features=args.features, hidden=args.hidden, outputs=args.outputs,
                  numeric_mode=args.numeric_mode, torch=torch.__version__, call_ms=elapsed,
                  clock_scope="public run_torch plus closing synchronization; export/import excluded",
                  samples=1, warmups=1, artifact=str(artifact),
                  artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), arrays=str(arrays),
                  pending=["phase breakdown", "forced-host comparison", "full-workload evaluation"])
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print("TRANSFER_PROBE " + json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
