#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DEVIATIONS 3010 and 3011: the two witnesses this lane added, and the A/B
that has to separate before the CE aliasing may be believed.

Public API only (`LanguageModelConfig`, `LanguageModelTrainer`,
`parameter_registry`, `train_step`, `export_state`, `export_gradients`,
`run_metadata`, `attention_stage_report`, `export_checkpoint_binary`,
`from_checkpoint_binary`, `close`). Full-step wall times are recorded before
optional state export; they are session measurements, not isolated-kernel
benchmarks. Exact witnesses qualify the compared arithmetic.

Modes:

  steps        run `--steps` complete steps and record, per step, the loss
               value, the seconds, `attention_stage_report()` and (every
               `--smi-every` steps) the polled device memory. With
               `--witness-every 1` it also records the sha256 of loss, flat
               gradients, parameters, m, v and flags, which is what the
               aliasing A/B compares; the caller runs it once against a
               binding built clean and once against one built with
               `-D MOJOLEARN_BYTE_LM_CE_UNALIASED=1`, and the two
               `result.json` files must agree hash for hash at every step.

               WITNESSES ARE OFF BY DEFAULT BECAUSE THEY ARE NOT FREE: at the
               162,147,840-parameter shape `export_state()` downloads 1.95 GB
               and `export_gradients()` another 0.65 GB, which is several
               times the step itself. A long run that switched them on would
               be measuring the export. `attention_stage_report()` costs
               nothing: it is `len()` of buffers the trainer already owns.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
#: ONE spelling of the pinned-corpus schedule, not a second one. This is the
#: class lane/lm-training-shakedown's long run fed its steps from, so a run
#: here sees the same bytes in the same order at the same step index.
from lm_step_memory_probe import CorpusBatches


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


def build(args, corpus=None):
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
                                     'batches': corpus.describe() if corpus is not None else
                                                'synthetic uniform token ids, no corpus'})
    return trainer, shape


def ids_for(shape, seed, index, corpus=None):
    """The pinned corpus when one was staged, else synthetic uniform ids.

    THE CORPUS IS NOT A DETAIL FOR THE EAGER WITNESS. The fallback this run
    is looking for is triggered by the DATA (a `FUSED_CORNER` hit, or a
    refused regime), so a run on uniform random byte ids is not the same
    experiment as a run on enwik8 and a null result on one says nothing
    about the other. Every result records which it was.
    """
    if corpus is not None:
        return corpus.ids(index)
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


def gemm_stage_witness(entry_name='byte_lm_gemm_stage_ftz'):
    from mojolearn import _backend
    binding = _backend.binding('_mojolearn_byte_lm', 'identical')
    entry = getattr(binding, entry_name, None)
    return None if entry is None else bool(entry())


def sticky_witness():
    """DEVIATION 3110's `byte_lm_attn_sticky_fallback()` straight off the
    loaded binding. False means the build relaunches a refused layer's fused
    kernels and discards them; None means a binding that predates the flag."""
    from mojolearn import _backend
    binding = _backend.binding('_mojolearn_byte_lm', 'identical')
    entry = getattr(binding, 'byte_lm_attn_sticky_fallback', None)
    return None if entry is None else bool(entry())


def kv_guard_witness():
    """DEVIATION 3111's `byte_lm_attn_kv_corner_guard()` off the loaded
    binding. False means the dk/dv corner fires on any -0.0."""
    from mojolearn import _backend
    binding = _backend.binding('_mojolearn_byte_lm', 'identical')
    entry = getattr(binding, 'byte_lm_attn_kv_corner_guard', None)
    return None if entry is None else bool(entry())


def bwd_corner_witness():
    """DEVIATION 3112's `byte_lm_attn_bwd_corner_refuses()` off the loaded
    binding. False means this build's backward never refuses."""
    from mojolearn import _backend
    binding = _backend.binding('_mojolearn_byte_lm', 'identical')
    entry = getattr(binding, 'byte_lm_attn_bwd_corner_refuses', None)
    return None if entry is None else bool(entry())


def run_steps(trainer, shape, args, start_index, count, records, corpus=None):
    """One step, then the cheap witnesses; the expensive ones only if asked."""
    for offset in range(count):
        index = start_index + offset
        t0 = time.perf_counter()
        result = trainer.train_step(ids_for(shape, args.seed, index, corpus))
        seconds = time.perf_counter() - t0
        want_smi = args.smi_every > 0 and (offset % args.smi_every == 0 or offset == count - 1)
        record = dict(step=index, seconds=seconds,
                      loss=float(result['loss']),
                      completed_steps=int(result['completed_steps']),
                      device_used_mb=device_peak_mb(args.gpu_index) if want_smi else None,
                      rss_bytes=rss_bytes() if want_smi else None,
                      attention=trainer.attention_stage_report())
        if args.witness_every > 0 and (offset % args.witness_every == 0 or offset == count - 1):
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
    parser.add_argument('--corpus', type=Path, default=None,
                        help='pinned corpus staged from R2; without it the ids are synthetic')
    parser.add_argument('--witness-every', type=int, default=0,
                        help='sha256 the state every N steps (1 for the A/B; 0 = never, the default, '
                             'because an export at the target shape costs several times the step)')
    parser.add_argument('--smi-every', type=int, default=1,
                        help='poll nvidia-smi every N steps; 0 disables it')
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
    corpus = None
    if args.corpus is not None:
        shape_for_corpus = Shape(*args.shape)
        corpus = CorpusBatches(args.corpus, shape_for_corpus.batch, shape_for_corpus.length)
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
        run_steps(trainer, shape, args, resumed_at, args.tail, records, corpus)
    else:
        trainer, shape = build(args, corpus)
        run_steps(trainer, shape, args, 0, args.steps, records, corpus)
        if args.mode == 'checkpoint':
            t0 = time.perf_counter()
            digest = trainer.export_checkpoint_binary(args.checkpoint)
            save_seconds = time.perf_counter() - t0
            (args.out / 'checkpoint.json').write_text(json.dumps(dict(
                sha256=digest, bytes=Path(args.checkpoint).stat().st_size,
                save_seconds=save_seconds, saved_at_step=records[-1]['completed_steps']), indent=1))
        else:
            run_steps(trainer, shape, args, args.steps, args.tail, records, corpus)

    result = dict(schema='mojolearn.lm-ce-alias-probe.v1',
                  mode=args.mode, resume=bool(args.resume),
                  drop_moments=bool(args.drop_moments),
                  shape=list(args.shape), seed=args.seed,
                  attention_path=args.attention_path or 'auto',
                  witness_every=args.witness_every, smi_every=args.smi_every,
                  corpus=(corpus.describe() if corpus is not None else None),
                  ids='pinned corpus' if corpus is not None else 'synthetic uniform token ids',
                  ce_aliased=aliasing_witness(trainer),
                  gemm_stage_ftz=gemm_stage_witness(),
                  gemm_reuse_group_ws=gemm_stage_witness('byte_lm_gemm_reuse_group_ws'),
                  attn_sticky_fallback=sticky_witness(),
                  attn_kv_corner_guard=kv_guard_witness(),
                  attn_bwd_corner_refuses=bwd_corner_witness(),
                  run_metadata=trainer.run_metadata(),
                  steps=records, started=started, finished=time.time())
    (args.out / 'result.json').write_text(json.dumps(result, indent=1, allow_nan=False))
    trainer.close()
    print(json.dumps(dict(event='done', out=str(args.out), ce_aliased=result['ce_aliased'],
                          attn_sticky_fallback=result['attn_sticky_fallback'],
                          attn_kv_corner_guard=result['attn_kv_corner_guard'],
                          attn_bwd_corner_refuses=result['attn_bwd_corner_refuses'],
                          steps=len(records))), flush=True)


if __name__ == '__main__':
    main()
