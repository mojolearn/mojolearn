#!/usr/bin/env python3
"""Bounded real framework probe, root-only after build; no all-native claim."""
import argparse
import json
from pathlib import Path
import traceback


def run(device, folder):
    import numpy as np
    import torch
    import mojolearn
    from mojolearn._verify_native_runtime import audit_native_call
    torch.manual_seed(83)
    network = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).eval().to(device)
    x = torch.tensor(np.random.default_rng(83).standard_normal((5, 3)).astype('f4'), device=device)
    reports = []
    artifact = folder / "model.onnx"
    # Explicit external test preparation; retained artifact is the product input.
    torch.onnx.export(network, (x,), str(artifact), opset_version=17, dynamo=False)
    admission = audit_native_call(mojolearn.import_model, str(artifact), numeric_mode="identical")
    imported = admission.assert_native()
    reports.append(admission.report())
    with torch.no_grad():
        audit = audit_native_call(imported.run_torch, x)
        first = audit.assert_native()
        reports.append(audit.report())
        second = imported.run_torch(x)
        assert torch.equal(first, second)
        assert torch.allclose(first, network(x), rtol=1e-4, atol=1e-5)
        reference = np.asarray(imported.run(x.detach().cpu().numpy()))
        assert first.detach().cpu().numpy().tobytes() == reference.tobytes()
        for weight in network.parameters():
            weight.add_(.01)
        assert torch.equal(imported.run_torch(x), first)
        updated = folder / "model-updated.onnx"
        torch.onnx.export(network, (x,), str(updated), opset_version=17, dynamo=False)
        replacement = mojolearn.import_model(str(updated), numeric_mode="identical")
        changed = replacement.run_torch(x)
        assert torch.allclose(changed, network(x), rtol=1e-4, atol=1e-5)
        assert torch.equal(changed, replacement.run_torch(x.clone()))
    np.savez(folder / "outputs.npz", original=first.detach().cpu().numpy(),
             updated=changed.detach().cpu().numpy(), array_route=reference)
    return dict(status='selected_behavior_passed', device=device,
                actual_shape=list(first.shape), dtype=str(first.dtype), runtime_audits=reports,
                pending=['Other device routes not executed by this invocation'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = run(args.device, args.output.parent)
    except BaseException:
        result = dict(status='failed', traceback=traceback.format_exc())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'evidence': str(args.output)}))
    return result['status'] == 'failed'


if __name__ == '__main__':
    raise SystemExit(main())
