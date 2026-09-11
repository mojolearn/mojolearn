#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded actual-native default/GQA gradient, update and evaluation gate.

Root execution only. Eight training steps and eight evaluations total. No build,
rental, installation, opponent timing or learning claim. Explicit CPU FP64 oracle
supports native Metal, whose PyTorch backend cannot execute float64 arithmetic.
Existing preset oracle tolerances are not adapted to measured errors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import byte_lm_gradient_oracle as oracle

CASES = [('default', (2, 32, 32, 4, 2, 8, 64)),
         ('alternate_gqa', (3, 7, 24, 3, 1, 8, 40)),
         ('one_layer_vocab257', (1, 5, 16, 2, 1, 8, 24, 1, 257)),
         ('three_layers_vocab513', (2, 7, 24, 3, 1, 8, 40, 3, 513))]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def state_witness(state):
    # Includes metadata/counters as well as every authoritative array bit.
    return {key: {'dtype': str(value.dtype), 'shape': list(value.shape),
                  'sha256': sha(value.tobytes())}
            if hasattr(value, 'dtype') else value for key, value in state.items()}


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resident', action='store_true', help='Exercise owned native sessions')
    parser.add_argument('--native-vendor', choices=('metal', 'cuda', 'hip'), required=True)
    parser.add_argument('--oracle-device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error('--run is required to execute actual model operations')
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('set MOJOLEARN_NUMERIC_MODE=identical')
    args.out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from mojolearn import SmallByteLanguageModelTrainer, ByteLanguageModelConfig
    torch, _ = oracle._torch_device(device=args.oracle_device)
    torch.set_num_threads(1)
    if args.oracle_device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    source = Path(__file__).resolve().parents[1]
    metadata = {'schema': 'mojolearn.byte-lm.runtime-numerical.v1',
                'resident': args.resident, 'python': sys.version, 'platform': platform.platform(),
                'numpy': np.__version__, 'torch': torch.__version__,
                'oracle_device': args.oracle_device, 'native_vendor': args.native_vendor,
                'oracle_device_name': torch.cuda.get_device_name() if args.oracle_device == 'cuda' else 'CPU FP64',
                'tolerances_atol_rtol': oracle.TOLERANCES,
                'qualification': 'numerical tolerance and local eval invariance; not cross-vendor bitwise identity',
                'source_sha256': {name: sha((source / name).read_bytes()) for name in (
                    'tools/byte_lm_runtime_numerical_check.py', 'tools/byte_lm_gradient_oracle.py',
                    'training/byte_lm.mojo', 'training/byte_lm_config.mojo')}}
    write_json(args.out / 'metadata.json', metadata)
    results = []
    for case_index, (name, fields) in enumerate(CASES):
        cfg = ByteLanguageModelConfig(*fields)
        entries = oracle.registry(fields)
        count = entries[-1]['offset'] + entries[-1]['count']
        assert count == cfg.n_total
        public_registry = SmallByteLanguageModelTrainer.parameter_registry(cfg)
        assert [(e['name'], list(e['shape']), e['offset'], e['size']) for e in public_registry] == [
            (e['name'], e['shape'], e['offset'], e['count']) for e in entries]
        rng = np.random.Generator(np.random.PCG64(20260910 + case_index))
        initial = rng.normal(0, .08, count).astype(np.float32)
        for entry in entries:
            if 'norm' in entry['name']:
                start = entry['offset']
                initial[start:start + entry['count']] += np.float32(1)
        trainer = SmallByteLanguageModelTrainer(initial,
            data_schedule={'kind': 'bounded synthetic numerical fixture', 'seed': 20260910 + case_index},
            shape=cfg, resident=args.resident)
        runtime = trainer.run_metadata()
        assert runtime['native_vendor'] == args.native_vendor, runtime
        write_json(args.out / f'{name}-runtime.json', runtime)
        case = {'name': name, 'shape': list(fields), 'profile': cfg.profile, 'steps': [], 'eval': []}
        for step in range(2):
            ids = rng.integers(0, cfg.vocab_size, (cfg.batch, cfg.length + 1), dtype=np.int32)
            # Repeated IDs exercise embedding gradient accumulation; endpoints
            # and distinct rows exercise byte range, shift and batch ownership.
            ids[:, 0] = 0
            ids[:, 1] = 7
            ids[:, -1] = cfg.vocab_size - 1
            before = trainer.state_dict()
            reference_loss, gradients = oracle.reference(before['parameters'], ids,
                model_shape=fields, oracle_device=args.oracle_device)
            reference_flat = np.concatenate([gradients[e['name']].reshape(-1) for e in entries])
            result = trainer.train_step(ids)
            if runtime['step_result'] == 'lean':
                # DEVIATION 2514: the lean step leaves the gradient on the
                # device; export the last completed step's so the checks
                # and the capture below read the same `flat_gradients`.
                result = dict(result, **trainer.export_gradients())
            after = trainer.state_dict()
            grad_checks = {}
            for entry in entries:
                start, end = entry['offset'], entry['offset'] + entry['count']
                grad_checks[entry['name']] = oracle._compare(result['flat_gradients'][start:end],
                    gradients[entry['name']].reshape(-1), oracle.TOLERANCES['gradient'])
            loss_check = oracle._compare(np.asarray([result['loss']]), np.asarray([reference_loss]), oracle.TOLERANCES['loss'])
            expected_update = oracle.adamw_reference(before['parameters'], before['m'], before['v'],
                result['flat_gradients'], before['config'], before['completed_steps'], oracle_device=args.oracle_device)
            update_checks = {key: oracle._compare(after[state], expected_update[key], oracle.TOLERANCES[key])
                             for key, state in [('post_p', 'parameters'), ('post_m', 'm'), ('post_v', 'v')]}
            assert after['completed_steps'] == before['completed_steps'] + 1 == step + 1
            assert np.array_equal(after['flags'], before['flags']), 'AdamW must preserve initialization flags'
            assert after['parameters'].tobytes() != before['parameters'].tobytes(), 'update did not move parameters'
            controls = {'negated_gradient_detected': not oracle._compare(-result['flat_gradients'], reference_flat,
                         oracle.TOLERANCES['gradient'])['passed']}
            # Keep each block's SiLU values while deleting its sigmoid derivative.
            # All layer controls must be detected without moving loss.
            if step == 0:
                for block in range(cfg.n_layers):
                    bad_loss, bad = oracle.reference(before['parameters'], ids, wrong_silu_block=block,
                        model_shape=fields, oracle_device=args.oracle_device)
                    bad_flat = np.concatenate([bad[e['name']].reshape(-1) for e in entries])
                    controls[f'block{block}_wrong_silu_detected'] = not oracle._compare(
                        bad_flat, reference_flat, oracle.TOLERANCES['gradient'])['passed']
                    controls[f'block{block}_forward_unchanged'] = oracle._compare(
                        np.asarray([bad_loss]), np.asarray([reference_loss]), oracle.TOLERANCES['loss'])['passed']
            arrays = {'ids': ids, 'initial_p': before['parameters'], 'initial_m': before['m'],
                      'initial_v': before['v'], 'initial_flags': before['flags'],
                      'grad': result['flat_gradients'], 'post_p': after['parameters'],
                      'post_m': after['m'], 'post_v': after['v'], 'post_flags': after['flags'],
                      'loss': np.asarray([result['loss']], np.float32),
                      'reference_loss': np.asarray([reference_loss]), 'reference_grad': reference_flat,
                      **{'reference_' + k: v for k, v in expected_update.items()}}
            capture_path = args.out / f'{name}-step{step}.npz'
            with capture_path.open('xb') as stream:
                np.savez(stream, **arrays)
            record = {'step': step + 1, 'gradient': grad_checks, 'loss': loss_check,
                      'updates': update_checks, 'controls': controls,
                      'capture_sha256': sha(capture_path.read_bytes())}
            # Evaluate the updated state using the same supplied token array.
            eval_before = trainer.state_dict()
            eval_loss = trainer.evaluate(ids)
            eval_after = trainer.state_dict()
            eval_reference, _ = oracle.reference(eval_before['parameters'], ids,
                model_shape=fields, oracle_device=args.oracle_device)
            eval_record = {'state_before': state_witness(eval_before), 'state_after': state_witness(eval_after),
                           'loss': oracle._compare(np.asarray([eval_loss]), np.asarray([eval_reference]),
                                                   oracle.TOLERANCES['loss'])}
            eval_record['state_identical'] = eval_record['state_before'] == eval_record['state_after']
            record['passed'] = (all(c['passed'] for c in grad_checks.values()) and loss_check['passed']
                                and all(c['passed'] for c in update_checks.values()) and all(controls.values()))
            case['steps'].append(record)
            case['eval'].append(eval_record)
            # Persist failures before rejecting, with no threshold adaptation.
            write_json(args.out / f'{name}-step{step}.json', {'step': record, 'eval': eval_record})
            assert record['passed'], f'{name} step{step}: numerical/control failure; inspect evidence'
            assert eval_record['state_identical'] and eval_record['loss']['passed'], f'{name}: evaluation failure'
        results.append(case)
    write_json(args.out / 'verdict.json', {'passed': True, 'cases': results})
    print('PASS actual native runtime byte LM: 4 shapes, 8 full gradient/update steps, 8 eval state-invariance checks')


if __name__ == '__main__':
    main()
