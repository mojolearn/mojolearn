#!/usr/bin/env python3
"""Bounded, interleaved full-step resident/reconstructed LM pilot; no opponent run.

Parent enforces a process deadline. Production context length alone does not
qualify a small model as the target 125M-scale training workload.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import hashlib
    import resource
    import statistics
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape

    shape = Shape(*args.shape)
    rng = np.random.default_rng(93261)
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if 'norm' in entry['name']:
            weights[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    models = {name: Trainer(weights, shape=shape, resident=(name == 'resident'),
                           data_schedule={'fixture': 'bounded session pilot', 'seed': 93261})
              for name in ('reconstructed', 'resident')}
    events = []

    def emit(value):
        events.append(value)
        with (args.out / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        print(json.dumps(value, allow_nan=False), flush=True)

    # DEVIATION 2514: a trainer running step_result='lean' keeps the gradient
    # on the device; the comparison below fetches it with export_gradients()
    # after the timed call. Under 'full' the result carries it and nothing
    # here changes.
    lean = {name: model.run_metadata()['step_result'] == 'lean' for name, model in models.items()}
    emit({'event': 'setup', 'shape': shape.to_dict(), 'parameters': shape.n_total,
          'runtime': models['resident'].run_metadata(),
          'step_result': {name: 'lean' if lean[name] else 'full' for name in models},
          'attention_path_requested': os.environ.get('MOJOLEARN_TRANSFORMER_ATTN_PATH'),
          'qualification': 'pilot only; target-model default qualification remains separate'})
    for index in range(args.pairs + 1):
        ids = rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
        order = ('reconstructed', 'resident') if index % 2 == 0 else ('resident', 'reconstructed')
        outputs = {}
        for name in order:
            emit({'event': 'call_start', 'pair': index, 'arm': name, 'warmup': index == 0})
            start = time.perf_counter()
            outputs[name] = models[name].train_step(ids)
            elapsed = time.perf_counter() - start
            emit({'event': 'call_end', 'pair': index, 'arm': name,
                  'warmup': index == 0, 'seconds': elapsed})
            if lean[name]:
                outputs[name] = dict(outputs[name], **models[name].export_gradients(named=False))
        state = {name: model.state_dict() for name, model in models.items()}
        arrays = {name: {**{key: state[name][key] for key in ('parameters', 'm', 'v', 'flags')},
                         'gradient': outputs[name]['flat_gradients'],
                         'loss': np.array([outputs[name]['loss']], np.float32)} for name in models}
        left, right = arrays['resident'], arrays['reconstructed']
        bits = {key: left[key].tobytes() == right[key].tobytes() for key in left}
        emit({'event': 'comparison', 'pair': index, 'bits_match': bits,
              'sha256': {key: hashlib.sha256(left[key].tobytes()).hexdigest() for key in left}})
        assert all(bits.values()), bits
        assert state['resident']['completed_steps'] == state['reconstructed']['completed_steps'] == index + 1
        del outputs, state, arrays, left, right
    samples = {name: [e['seconds'] for e in events if e['event'] == 'call_end'
                     and not e['warmup'] and e['arm'] == name] for name in models}
    medians = {name: statistics.median(values) for name, values in samples.items()}
    for model in models.values():
        model.close()
    result = {'passed': True, 'samples_seconds': samples, 'median_seconds': medians,
              'resident_over_reconstructed': medians['resident'] / medians['reconstructed'],
              'process_max_rss_bytes_both_arms': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                  * (1 if sys.platform == 'darwin' else 1024),
              'qualification': 'complete-call pilot; not GPU peak memory, opponent admission or a default gate'}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    emit({'event': 'complete', **result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shape', nargs=9, type=int, default=[1, 2048, 256, 4, 4, 64, 768, 6, 8192])
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300 or not 1 <= args.pairs <= 4:
        parser.error('use 1–300 seconds and 1–4 pairs')
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('requires MOJOLEARN_NUMERIC_MODE=identical')
    if args.worker:
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (args.out / 'worker.log').open('w') as log:
        try:
            proc = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker'],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds)
            status = {'exit_code': proc.returncode, 'timed_out': False}
        except subprocess.TimeoutExpired:
            status = {'exit_code': None, 'timed_out': True}
    status.update(elapsed_seconds=time.monotonic() - start, deadline_seconds=args.seconds)
    (args.out / 'execution.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if status['exit_code'] == 0 else 1)


if __name__ == '__main__':
    main()
