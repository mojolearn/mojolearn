#!/usr/bin/env python3
"""Bounded real framework probe, root-only after build; no all-native claim."""
import argparse
import json
from pathlib import Path
import traceback


def run(device):
    import numpy as np
    import torch
    import mojolearn
    from mojolearn.adapters import _torch_compile as tc
    from mojolearn._verify_native_runtime import audit_native_call
    tc.register()
    torch.manual_seed(83)
    network = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).eval().to(device)
    x = torch.tensor(np.random.default_rng(83).standard_normal((5, 3)).astype('f4'), device=device)
    compiled = torch.compile(network, backend='mojolearn', fullgraph=True, dynamic=False)
    reports = []
    with torch.no_grad():
        first = compiled(x)
        second = compiled(x)
        assert torch.equal(first, second)
        assert torch.allclose(first, network(x), rtol=1e-4, atol=1e-5)
        imported = mojolearn.import_model(network, example_inputs=x)
        reference = np.asarray(imported.run(x.detach().cpu().numpy()))
        assert first.detach().cpu().numpy().tobytes() == reference.tobytes()
        audit = audit_native_call(compiled, x)
        reports.append(audit.report())
        # Capture/export and its version/reimport wrapper remain documented
        # Python debt. Do not convert a private native helper result to a
        # fully-native public-adapter qualification.
        for weight in network.parameters():
            weight.add_(.01)
        changed = compiled(x)
        assert torch.allclose(changed, network(x), rtol=1e-4, atol=1e-5)
        again = compiled(x.clone())
        assert torch.equal(changed, again)
    return dict(status='selected_behavior_passed_native_migration_pending', device=device,
                actual_shape=list(first.shape), dtype=str(first.dtype), runtime_audits=reports,
                pending=['Torch graph capture/export and parameter-version reimport',
                         'Other device routes not executed by this invocation'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        result = run(args.device)
    except BaseException:
        result = dict(status='failed', traceback=traceback.format_exc())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'evidence': str(args.output)}))
    return result['status'] == 'failed'


if __name__ == '__main__':
    raise SystemExit(main())
