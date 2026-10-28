#!/usr/bin/env python3
"""Bounded public A/B correctness probe; authored here, root runs rebuilt Mac.

Controller NumPy/checkpoint work is test-only. This makes no timing claim.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import traceback


def qualify():
    import numpy as np
    from mojolearn import Array
    from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn.models import CausalLM

    os.environ['MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD'] = '0'
    os.environ['MOJOLEARN_GENERATION_NATIVE_SESSION'] = '1'
    os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = '1'
    os.environ.pop('MOJOLEARN_TRANSFORMER_LEGACY_SETUP', None)
    reports = []
    def call(arm, method, *args, **kwargs):
        os.environ['MOJOLEARN_GENERATION_NATIVE_ENGINE'] = str(arm)
        return method(*args, **kwargs)
    def state_record(state):
        return dict(positions=state.positions, usable=getattr(state, '_usable', True),
                    cached=[layer.cached_tokens for layer in state.layers])
    def same(a, b):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    with tempfile.TemporaryDirectory(prefix='ml-native-engine-') as folder:
        cfg, tensors = family_fixture('llama', False)
        _write_checkpoint(folder, cfg, tensors)
        models = [CausalLM.load(folder, device='gpu', numeric_profile='fp32_v1') for _ in range(2)]
        assert all(m._native_vendor == 'metal' and m._native_numeric_mode == 1 for m in models)
        assert all('native_primitives' in m._native_entries and 'native_session_provider' in m._native_entries for m in models)
        ids = np.array([[1, 4, 7], [2, 5, 8]], dtype='i4')
        states = [m.allocate_state(2, 48) for m in models]
        for repeat in range(8):
            a = call(0, models[0].forward, ids, states[0])
            os.environ['MOJOLEARN_GENERATION_NATIVE_ENGINE'] = '1'
            audit = audit_native_call(models[1].forward, ids, states[1])
            reports.append(audit.report())
            b = audit.assert_native()
            same(a, b)
            assert state_record(states[0]) == state_record(states[1])
            assert states[1]._native_execution_route == 'transformer-nonresident-native-engine-v1'
            assert states[1]._native_engine_forwards == 1
        digest = hashlib.sha256(np.asarray(b).tobytes()).hexdigest()
        options = dict(greedy=False, seed=17, top_k=5, top_p=0.83, temperature=0.8)
        streams = [m.stream(ids, 6, **options) for m in models]
        for count in (2, 1, 3, 1):
            a = call(0, streams[0].read, count)
            b = call(1, streams[1].read, count)
            same(a, b)
            assert streams[0].info()['generated_tokens'] == streams[1].info()['generated_tokens']
        stream = models[1].stream(ids, 6, **options)
        call(1, stream.read, 2)
        saved = pickle.loads(pickle.dumps(stream.state_dict()))
        tail = np.asarray(call(1, stream.read, 4)).copy()
        same(tail, call(1, models[1].resume_generation(saved).read, 4))

        # Fresh metadata/owner capture on the next operation, no stale global
        # head pointer. Same exact Array replacement on both arms.
        heads = [m._head for m in models]
        for m, head in zip(models, heads):
            m._head = Array.from_buffer(np.zeros_like(np.asarray(head)))
        gc.collect()
        same(call(0, models[0].forward, ids), call(1, models[1].forward, ids))
        for m, head in zip(models, heads):
            m._head = head

        # A refusal in layer2 retains layer1 progress and invalidates global
        # state, exactly as legacy. No forged address or unsafe storage test.
        failed = [m.allocate_state(2, 12) for m in models]
        for state in failed:
            state.layers[1].k_cache = Array.from_buffer(np.zeros((1,), dtype='f4'))
        for arm, (m, state) in enumerate(zip(models, failed)):
            try:
                call(arm, m.forward, ids, state)
            except Exception as error:
                assert 'k_cache' in str(error) or 'shape' in str(error) or 'span' in str(error)
            else:
                raise AssertionError('malformed later cache was accepted')
        assert state_record(failed[0]) == state_record(failed[1])
        assert failed[1].layers[0].cached_tokens == 3 and failed[1].positions == 0

        # Capacity preflight remains recoverable (outside execution failure).
        for arm, m in enumerate(models):
            short = m.allocate_state(2, 1)
            try:
                call(arm, m.forward, ids, short)
            except Exception:
                assert getattr(short, '_usable', True)
            else:
                raise AssertionError('capacity refusal missing')
            call(arm, m.forward, ids, m.allocate_state(2, 12))

        # Existing session close/refusal contract, then supported replacement.
        for arm, m in enumerate(models):
            fields = object.__getattribute__(m.blocks[1], '__dict__')
            m._native_entries['block_binding'].transformer_session_close(fields['_native_session'])
            bad = m.allocate_state(2, 12)
            try:
                call(arm, m.forward, ids, bad)
            except Exception as error:
                assert 'busy or closed' in str(error)
                assert bad.layers[0].cached_tokens == 3 and bad.positions == 0 and not bad._usable
            else:
                raise AssertionError('closed-session negative control missing')
            fields['_native_session'] = m._native_entries['create']()
            call(arm, m.forward, ids, m.allocate_state(2, 12))
    return dict(status='passed', logits_sha256=digest, runtime_audits=reports,
                checks=['8 repeated forward bit comparisons and native route diagnostics',
                        'multi-token stream reads and resume', 'owner replacement',
                        'later-layer failure progress', 'capacity recovery', 'closed session/replacement'],
                pending=['concurrent contention', 'tied session alias fixture', 'all unsupported routes',
                         'full-workload timing', 'other vendors'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        report = qualify()
    except BaseException:
        report = dict(status='failed', traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], evidence=str(args.output))))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
