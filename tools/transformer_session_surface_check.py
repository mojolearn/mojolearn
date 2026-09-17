#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Selected public-wrapper checks against an explicitly loaded native binding.

Only the Transformer implementation and dependencies are imported, avoiding
unrelated family bindings. Actual native calls and numeric-mode checks remain.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import types
from unittest.mock import patch

import numpy as np
from bench_neural_decode import weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binding', type=Path, required=True)
    ap.add_argument('--backend', choices=('cpu', 'metal', 'cuda', 'hip'), required=True)
    ap.add_argument('--group', choices=('state', 'serialization', 'threads'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not __debug__:
        ap.error('assertions required (no -O)')
    # Import the public class without importing the full package/other lanes.
    package = types.ModuleType('mojolearn')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'python/mojolearn')]
    sys.modules['mojolearn'] = package
    impl = importlib.import_module('mojolearn._transformer_impl')
    Block = impl.TransformerBlock
    spec = importlib.util.spec_from_file_location(args.binding.stem, args.binding.resolve())
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    assert native.transformer_vendor() == args.backend
    assert native.transformer_numeric_mode() == 1
    reuse = hasattr(native, 'transformer_session_forward')
    data = weights('transformer', 32)
    x = np.random.default_rng(330).uniform(-.5, .5, (1, 1, 32)).astype(np.float32)
    outputs = {}

    def block(window=0):
        return Block(dict(zip(Block._W_NAMES, data)), n_heads=2,
                     n_kv_heads=1, window=window, numeric_mode='identical')

    def record(name, value):
        array = np.asarray(value).copy()
        assert np.isfinite(array).all()
        outputs[name] = array
        return array

    with patch.object(Block, '_bind', lambda self: native):
        model = block()
        assert model._native_session is None
        if args.group == 'state':
            for window in (0, 3):
                model = block(window)
                state = model.allocate_state(1, 8)
                for pos in range(4):
                    record(f'w{window}.pos{pos}', model.step(x, state))
                    assert state.cached_tokens == pos + 1
                if reuse:
                    info = native.transformer_session_info(model._native_session)
                    assert list(info)[:3] == [1, 1, False]
                    assert 0 < info[3] <= 64 * 1024 * 1024
                # Same state object reset, then same-address weight edits.
                state.cached_tokens = 0
                np.asarray(state.k_cache).fill(0)
                np.asarray(state.v_cache).fill(0)
                reset = record(f'w{window}.reset', model.step(x, state))
                assert reset.tobytes() == outputs[f'w{window}.pos0'].tobytes()
                original = data[2].copy()
                data[2].flat[0] += np.float32(.125)
                record(f'w{window}.edited', model.step(x, state))
                data[2][:] = original
                record(f'w{window}.k', state.k_cache)
                record(f'w{window}.v', state.v_cache)
                xx = np.concatenate([x, -x], axis=1)
                first = record(f'w{window}.fresh', model.forward(xx))
                second = record(f'w{window}.fresh-repeat', model.forward(xx))
                assert first.tobytes() == second.tobytes()
                record(f'w{window}.resume-after-fresh', model.step(x, state))
            if reuse:
                owner = model._native_session
                with patch.dict(os.environ, {'MOJOLEARN_TRANSFORMER_LEGACY_SETUP': '1'}):
                    model.step(x, model.allocate_state(1, 8))
                assert model._native_session is None
                assert native.transformer_session_info(owner)[2] is True
        elif args.group == 'serialization':
            want = record('original', model.step(x, model.allocate_state(1, 8)))
            for name, clone in [('pickle', pickle.loads(pickle.dumps(model))),
                                ('copy', copy.copy(model)), ('deepcopy', copy.deepcopy(model))]:
                assert clone._native_session is None
                assert clone._runtime_lock is not model._runtime_lock
                got = record(name, clone.step(x, clone.allocate_state(1, 8)))
                assert got.tobytes() == want.tobytes()
                if reuse:
                    assert clone._native_session is not model._native_session
        else:
            # Same-model calls serialize the complete state read/native call/
            # cached-token update, including first-use session initialization.
            state = model.allocate_state(1, 8)
            def same_model(_):
                return np.asarray(model.step(x, state)).copy()
            with ThreadPoolExecutor(max_workers=2) as pool:
                threaded = list(pool.map(same_model, range(4)))
            assert state.cached_tokens == 4
            serial = block()
            reference = serial.allocate_state(1, 8)
            expected = [np.asarray(serial.step(x, reference)).copy() for _ in range(4)]
            # Thread completion order is unspecified; state progression is not.
            assert sorted(a.tobytes() for a in threaded) == sorted(a.tobytes() for a in expected)
            record('same-model.k', state.k_cache)
            record('same-model.v', state.v_cache)
            assert np.asarray(state.k_cache).tobytes() == np.asarray(reference.k_cache).tobytes()
            assert np.asarray(state.v_cache).tobytes() == np.asarray(reference.v_cache).tobytes()
            models = [block(), block()]
            def separate(i):
                return np.asarray(models[i].step(x, models[i].allocate_state(1, 8))).copy()
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(separate, range(2)))
            assert results[0].tobytes() == results[1].tobytes() == expected[0].tobytes()
            record('separate-models', results[0])
        if not reuse:
            assert model._native_session is None
    np.savez(args.out, **outputs)
    report = dict(backend=args.backend, group=args.group, arrays=len(outputs), reuse=reuse,
                  binding_sha256=hashlib.sha256(args.binding.read_bytes()).hexdigest())
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
