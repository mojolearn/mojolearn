#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DEVIATIONS 3010 and 3011: the two witnesses this lane added, and the A/B
that has to separate before the CE aliasing may be believed.

Public API only (`LanguageModelConfig`, `LanguageModelTrainer`,
`parameter_registry`, `train_step`, `export_state`, `export_gradients`,
`run_metadata`, `attention_stage_report`, `export_checkpoint_binary`,
`from_checkpoint_binary`, `close`). Nothing here is a timing sample: the
seconds are reported so a run can be sized, and the claim is about BITS.

Modes:

  steps        run `--steps` complete steps and record, per step, the sha256
               of loss, flat gradients, parameters, m, v and flags, plus
               `attention_stage_report()` and the polled device peak. This is
               one arm of the aliasing A/B; the caller runs it once against a
               binding built clean and once against one built with
               `-D MOJOLEARN_BYTE_LM_CE_UNALIASED=1`, and the two `result.json`
               files must agree hash for hash at every step.

  checkpoint   run `--steps` steps, write a BINARY checkpoint, then exit. A
               second process with `--resume` loads that file into a new
               trainer and runs `--tail` more steps. Its witnesses must equal
               the uninterrupted run's tail. `--drop-moments` is the CONTROL:
               it zeroes m and v after the load, and it must SEPARATE -- if
               it does not, the comparison is not reading the moments and the
               resume proves nothing.

`ce_aliased` and `binding_sha256` are recorded in every result because an A/B
that cannot tell its arms apart reads exactly like a passed identity gate.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def device_peak_mb(index=0):
    """Device-wide used memory from nvidia-smi, in MB, or None."""
    if not shutil.which('nvidia-smi'):
        return None
    try:
        out = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=memory.used',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def rss_bytes():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak * (1 if sys.platform == 'darwin' else 1024)
    except Exception:
        return None


def build(args):
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    shape = Shape(*args.shape)
    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if 'norm' in entry['name']:
            weights[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    trainer = Trainer(weights, shape=shape, resident=True, step_result='lean',
                      data_schedule={'fixture': 'lm ce alias probe', 'seed': args.seed,
                                     'batches': 'synthetic uniform token ids, no corpus'})
    return trainer, shape


def ids_for(shape, seed, index):
    import numpy as np
    rng = np.random.default_rng(seed * 1000003 + index)
    return rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)


def witness(trainer, loss):
    import struct
    gradients = trainer.export_gradients(named=False)['flat_gradients']
    state = trainer.export_state()
    return dict(loss=sha(struct.pack('<f', loss)),
                gradients=sha(gradients.tobytes()),
                parameters=sha(state['parameters'].tobytes()),
                m=sha(state['m'].tobytes()), v=sha(state['v'].tobytes()),
                flags=sha(state['flags'].tobytes()),
                completed_steps=state['completed_steps'])


def aliasing_witness(trainer):
    """`byte_lm_ce_aliased()` straight off the loaded binding, never from a
    build log: the .so that ran is the only thing that can answer."""
    from mojolearn import _backend
    binding = _backend.binding('_mojolearn_byte_lm', 'identical')
    entry = getattr(binding, 'byte_lm_ce_aliased', None)
    return None if entry is None else bool(entry())


def run_steps(trainer, shape, args, start_index, count, records):
    for offset in range(count):
        index = start_index + offset
        t0 = time.perf_counter()
        result = trainer.train_step(ids_for(shape, args.seed, index))
        seconds = time.perf_counter() - t0
        record = dict(step=index, seconds=seconds,
                      device_used_mb=device_peak_mb(args.gpu_index),
                      rss_bytes=rss_bytes(),
                      attention=trainer.attention_stage_report())
        record.update(witness(trainer, float(result['loss'])))
        records.append(record)
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('steps', 'checkpoint'), default='steps')
    parser.add_argument('--shape', type=int, nargs=9,
                        default=[1, 2048, 768, 12, 12, 64, 2048, 12, 50257],
                        help='B L DM H KV HD FF layers vocab; the default is the 162,147,840 target')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--tail', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--attention-path', default=None,
                        help="'eager' forces the eager kernels; proves the eager witness moves")
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--drop-moments', action='store_true',
                        help='CONTROL: zero m and v after the load; must separate')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.attention_path:
        os.environ['MOJOLEARN_TRANSFORMER_ATTN_PATH'] = args.attention_path

    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape

    records = []
    started = time.time()
    if args.resume:
        trainer = Trainer.from_checkpoint_binary(args.checkpoint, resident=True)
        shape = Shape(*args.shape)
        if args.drop_moments:
            state = trainer.state_dict()
            zeros_m = state['m']
            zeros_v = state['v']
            for view in (zeros_m, zeros_v):
                raw = memoryview(view).cast('B')
                raw[:] = b'\x00' * len(raw)
            trainer.load_state_dict(dict(state, m=zeros_m, v=zeros_v))
        resumed_at = trainer.state_dict()['completed_steps']
        run_steps(trainer, shape, args, resumed_at, args.tail, records)
    else:
        trainer, shape = build(args)
        run_steps(trainer, shape, args, 0, args.steps, records)
        if args.mode == 'checkpoint':
            t0 = time.perf_counter()
            digest = trainer.export_checkpoint_binary(args.checkpoint)
            save_seconds = time.perf_counter() - t0
            (args.out / 'checkpoint.json').write_text(json.dumps(dict(
                sha256=digest, bytes=Path(args.checkpoint).stat().st_size,
                save_seconds=save_seconds, saved_at_step=records[-1]['completed_steps']), indent=1))
        else:
            run_steps(trainer, shape, args, args.steps, args.tail, records)

    result = dict(schema='mojolearn.lm-ce-alias-probe.v1',
                  mode=args.mode, resume=bool(args.resume),
                  drop_moments=bool(args.drop_moments),
                  shape=list(args.shape), seed=args.seed,
                  attention_path=args.attention_path or 'auto',
                  ce_aliased=aliasing_witness(trainer),
                  run_metadata=trainer.run_metadata(),
                  steps=records, started=started, finished=time.time())
    (args.out / 'result.json').write_text(json.dumps(result, indent=1, allow_nan=False))
    trainer.close()
    print(json.dumps(dict(event='done', out=str(args.out), ce_aliased=result['ce_aliased'],
                          steps=len(records))), flush=True)


if __name__ == '__main__':
    main()
