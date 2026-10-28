#!/usr/bin/env python3
"""Authored bounded Mac public probe; execute only after integration compilation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


# Test evidence only; preserve failed audits as structured output too.
RUNTIME_AUDITS = []


def qualify():
    RUNTIME_AUDITS.clear()
    import numpy as np
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
    from mojolearn.models import CausalLM

    os.environ['MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD'] = '0'
    os.environ['MOJOLEARN_GENERATION_NATIVE_SESSION'] = '0'
    results = []
    for family in ('llama', 'mamba', 'mamba2'):
        with tempfile.TemporaryDirectory(prefix='ml-neural-descriptor-') as folder:
            cfg, tensors = family_fixture(family, False)
            _write_checkpoint(folder, cfg, tensors)
            models = []
            for switch in ('0', '1'):
                os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = switch
                models.append(CausalLM.load(folder, device='gpu', numeric_profile='fp32_v1'))
            baseline, candidate = models
            assert candidate._native_vendor == 'metal', candidate._native_vendor
            assert baseline._native_numeric_mode == candidate._native_numeric_mode == 1
            assert 'native_primitives' not in baseline._native_entries
            assert 'native_primitives' in candidate._native_entries
            ids = np.array([[1, 4, 7]], dtype='i4')
            states = [model.allocate_state(1, 48) for model in models]
            for repeat in range(8):
                baseline_output = baseline.forward(ids, states[0])
                if repeat in (0, 7):
                    # Profile the actual candidate operation, with input/oracle
                    # preparation outside. Baseline poisoning below is separate.
                    audit = audit_native_call(candidate.forward, ids, states[1])
                    RUNTIME_AUDITS.append(dict(family=family, operation='forward', repeat=repeat,
                                               report=audit.report()))
                    candidate_output = audit.assert_native()
                else:
                    candidate_output = candidate.forward(ids, states[1])
                outputs = [np.asarray(value).copy() for value in (baseline_output, candidate_output)]
                assert np.array_equal(outputs[0].view('u4'), outputs[1].view('u4')), (family, repeat)
                assert states[0].positions == states[1].positions
                for a, b in zip(states[0].layers, states[1].layers):
                    # Nonzero return values are essential: first outputs alone
                    # missed the original native-session out-parameter failure.
                    for attr in ('cached_tokens', 'buffered_tokens', 'absolute_position'):
                        if hasattr(a, attr):
                            assert getattr(a, attr) == getattr(b, attr), (family, repeat, attr)
            baseline_generated = baseline.generate(ids, 5, greedy=False, seed=7, top_k=2)
            audit = audit_native_call(candidate.generate, ids, 5, greedy=False, seed=7, top_k=2)
            RUNTIME_AUDITS.append(dict(family=family, operation='generate', report=audit.report()))
            candidate_generated = audit.assert_native()
            generated = [np.asarray(value).copy() for value in (baseline_generated, candidate_generated)]
            assert np.array_equal(*generated), family
            diagnostic = next(module.neural_native_dispatch_info for module in tuple(sys.modules.values())
                              if module is not None and hasattr(module, 'neural_native_dispatch_info'))
            counts = {key: diagnostic(value) for key, value in candidate._native_entries.items()
                      if key in ('native_primitives', 'native_mamba')}
            assert counts['native_primitives'][1][0] > 0 and counts['native_primitives'][1][4] > 0
            if family != 'llama':
                assert sum(counts['native_mamba'][1]) > 0
            calls = []
            def poison(*args):
                calls.append(True)
                raise RuntimeError('poisoned legacy primitive dispatch')
            for model in models:
                model._native_entries['embedding_forward'] = poison
            try:
                baseline.forward(ids, baseline.allocate_state(1, 12))
            except Exception as problem:
                assert 'poisoned legacy primitive dispatch' in str(problem)
            else:
                raise AssertionError('legacy primitive negative control did not fire')
            n = len(calls)
            candidate.forward(ids, candidate.allocate_state(1, 12))
            assert len(calls) == n
            try:
                candidate.forward(np.array([[candidate.vocab_size]], dtype='i4'), candidate.allocate_state(1, 12))
            except Exception:
                pass
            else:
                raise AssertionError('invalid token negative control did not fire')
            candidate.forward(ids, candidate.allocate_state(1, 12))
            results.append(dict(family=family, counts=counts, logits_sha256=hashlib.sha256(outputs[1].tobytes()).hexdigest(),
                                tokens=generated[1].tolist(), status='passed'))
    return dict(status='passed', families=results, runtime_audits=RUNTIME_AUDITS, pending=['Samba and Mamba3 public execution', 'concurrency/busy',
                'CPU fallback', 'owned Mamba3', 'full-workload timing', 'other targets'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = qualify()
    except BaseException:
        report = dict(status='failed', traceback=traceback.format_exc(), runtime_audits=RUNTIME_AUDITS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], evidence=str(args.output))))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
