#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the IDENTICAL byte LM trainer learn as well as torch? Same init, same batches.

Three subcommands, one JSON each:

  ours   the installed mojolearn wheel (MOJOLEARN_NUMERIC_MODE=identical,
         resident session, step_result='lean'), N training steps. Every
         step's loss is recorded as a float and as its float32 bit pattern.
         Held-out mean loss (trainer.evaluate, no state change) before and
         after training. The final parameters are written to --params-out
         (float32 .npy, registry order) for the torch evaluator.
  torch  torch eager float32, TF32 OFF (the eager_fp32 row of
         tools/torch_lm_step_opponent.py, whose model, init, corpus schedule,
         SDPA backend choice and AdamW this reuses unchanged), N steps, no
         warmup, the same per-step record. Held-out mean loss before and after
         training with torch's own forward, and, with --ours-params, the same
         held-out loss of OUR final parameters under torch's forward (one
         evaluator for both models).
  compare  stdlib only: per-step table, differences, held-out losses, and the
         bitwise verdict between two `ours` files (another vendor's).

Batches: training step k (zero-based) is the probe's schedule, row b reads
bytes [(k*B*L + b*L) % (n - L - 1) : + L + 1] (tools/lm_step_memory_probe.py
CorpusBatches, tools/torch_lm_step_opponent.py Corpus.rows). Held-out rows
come from the corpus manifest's validation_range (enwik8: bytes 90M to 95M),
--heldout-rows rows of L + 1 bytes spaced evenly across it, never read by
training at these step counts (the run refuses if they could be). Loss is
the mean next-byte cross entropy over the row; the held-out loss is the mean
over rows (every row has L targets, so it is the mean over all targets).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import torch_lm_step_opponent as opp  # noqa: E402  (stdlib at module level; torch imported lazily)

SCHEMA = 'mojolearn.lm-quality-vs-torch.v1'


def f32_hex(value):
    return struct.pack('>f', value).hex()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def heldout_starts(corpus, rows):
    lo, hi = corpus.manifest.get('validation_range') or (0, 0)
    L = corpus.length
    if hi - lo < rows * (L + 1):
        opp.refuse('corpus %s has no validation_range that holds %d rows' % (corpus.name, rows))
    stride = (hi - lo - (L + 1)) // max(1, rows - 1)
    return [lo + j * stride for j in range(rows)]


def check_disjoint(corpus, steps, starts):
    last = (steps - 1) * corpus.batch * corpus.length + (corpus.batch - 1) * corpus.length + corpus.length + 1
    if last >= corpus.modulus:
        opp.refuse('training wraps the corpus at %d steps; held-out disjointness not guaranteed' % steps)
    if min(starts) < last:
        opp.refuse('held-out rows start at %d, inside the training bytes [0, %d)' % (min(starts), last))
    return last


def heldout_rows(corpus, starts):
    L = corpus.length
    return [corpus.raw[s:s + L + 1] for s in starts]


def base_record(args, corpus, dims, starts, train_end):
    return dict(schema=SCHEMA, side=args.cmd, shape_name=args.shape,
                shape=dict(zip(opp.SHAPE_FIELDS, dims)), steps=args.steps,
                corpus=corpus.describe(), train_bytes=[0, train_end],
                heldout=dict(rows=len(starts), starts=starts, length=corpus.length + 1,
                             source='manifest validation_range, evenly spaced'),
                init=dict(seed=opp.INIT_SEED, source='numpy default_rng(93261).normal(0, .02) float32, +1 on norms'),
                optimizer=dict(kind='AdamW', lr=opp.LR, betas=list(opp.BETAS), eps=opp.ADAM_EPS,
                               weight_decay=opp.WEIGHT_DECAY, schedule='constant, no warmup, no clipping'),
                python=platform.python_version(), host=platform.node(),
                repo_commit=os.environ.get('MOJOLEARN_REPO_COMMIT'),
                harness_sha256=_sha(Path(__file__).read_bytes()))


def cmd_ours(args, corpus, dims, starts, train_end):
    import numpy as np
    import mojolearn
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        opp.refuse('requires MOJOLEARN_NUMERIC_MODE=identical')
    shape = Shape(*dims)
    rng = np.random.default_rng(opp.INIT_SEED)
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    registry = Trainer.parameter_registry(shape)
    for entry in registry:
        if 'norm' in entry['name']:
            weights[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    init_sha = _sha(weights.tobytes())
    t0 = time.perf_counter()
    trainer = Trainer(weights, shape=shape, resident=True, step_result='lean',
                      data_schedule={'fixture': 'lm quality vs torch', 'seed': opp.INIT_SEED,
                                     'batches': 'pinned corpus ' + corpus.sha256})
    runtime = trainer.run_metadata()

    def ids_of(rows):
        return np.stack([np.frombuffer(r, dtype=np.uint8).astype(np.int32) for r in rows])

    held = [ids_of([r]) for r in heldout_rows(corpus, starts)]

    def heldout():
        losses = [float(trainer.evaluate(ids)) for ids in held]
        return dict(mean=sum(losses) / len(losses), rows=losses, rows_f32=[f32_hex(x) for x in losses])

    before = heldout()
    print(json.dumps(dict(event='heldout_before', mean=before['mean'])), flush=True)
    steps = []
    for k in range(args.steps):
        rows = corpus.rows(k)
        s = time.perf_counter()
        result = trainer.train_step(ids_of(rows))
        seconds = time.perf_counter() - s
        loss = float(result['loss'])
        steps.append(dict(step=k + 1, loss=loss, loss_f32=f32_hex(loss), seconds=seconds,
                          ids_sha256=corpus.ids_sha256(rows)))
        if k < 3 or (k + 1) % 25 == 0:
            print(json.dumps(dict(event='step', **steps[-1])), flush=True)
    after = heldout()
    print(json.dumps(dict(event='heldout_after', mean=after['mean'])), flush=True)
    state = trainer.export_state()
    params = np.frombuffer(state['parameters'].tobytes(), dtype='<f4')
    witness = dict(parameters=_sha(params.tobytes()), m=_sha(state['m'].tobytes()),
                   v=_sha(state['v'].tobytes()), flags=_sha(state['flags'].tobytes()),
                   completed_steps=state['completed_steps'])
    if args.params_out:
        np.save(args.params_out, params)
    trainer.close()
    record = dict(base_record(args, corpus, dims, starts, train_end),
                  mojolearn_version=getattr(mojolearn, '__version__', None),
                  mojolearn_file=mojolearn.__file__, runtime=runtime,
                  numeric_mode=os.environ.get('MOJOLEARN_NUMERIC_MODE'),
                  registry=[dict(name=e['name'], offset=e['offset'], size=e['size']) for e in registry],
                  initial_parameters_sha256=init_sha, wall_seconds=time.perf_counter() - t0,
                  losses=[s['loss'] for s in steps], losses_f32=[s['loss_f32'] for s in steps],
                  step_records=steps, heldout_before=before, heldout_after=after,
                  final_witness=witness, params_out=str(args.params_out) if args.params_out else None)
    return record


def cmd_torch(args, corpus, dims, starts, train_end):
    import torch
    column = opp.COLUMNS['eager_fp32']
    b, l, dm, h, kv, hd, ff, layers, vocab = dims
    shapes = opp.registry(dims)
    device = torch.device(args.device)
    on_gpu = device.type == 'cuda'
    if on_gpu and not torch.cuda.is_available():
        opp.refuse('--device cuda but torch.cuda.is_available() is False')
    precision = opp.set_precision(torch, column['tf32'])
    sync = torch.cuda.synchronize if on_gpu else (lambda: None)
    sdpa = opp.choose_sdpa_backend(torch, args.sdpa_backend, device, dims, sync, None)
    torch.manual_seed(opp.INIT_SEED)
    t0 = time.perf_counter()
    flat, init_source, init_sha = opp.initial_flat(shapes, torch)
    model = opp.build_model(torch, dims, shapes, flat, device)
    del flat
    if sum(p.numel() for p in model.parameters()) != opp.EXPECTED_PARAMETERS[args.shape]:
        opp.refuse('torch module parameter count is not the pinned count')
    optimizer = torch.optim.AdamW(model.parameters(), lr=opp.LR, betas=opp.BETAS, eps=opp.ADAM_EPS,
                                  weight_decay=opp.WEIGHT_DECAY)

    def to_ids(rows):
        return torch.tensor([list(r) for r in rows], dtype=torch.long).to(device)

    held = [to_ids([r]) for r in heldout_rows(corpus, starts)]

    def heldout(m):
        with torch.no_grad():
            losses = [float(m(ids).item()) for ids in held]
        return dict(mean=sum(losses) / len(losses), rows=losses, rows_f32=[f32_hex(x) for x in losses])

    before = heldout(model)
    print(json.dumps(dict(event='heldout_before', mean=before['mean'])), flush=True)
    steps = []
    for k in range(args.steps):
        rows = corpus.rows(k)
        sync()
        s = time.perf_counter()
        ids = to_ids(rows)
        optimizer.zero_grad(set_to_none=True)
        loss_t = model(ids)
        loss_t.backward()
        optimizer.step()
        loss = float(loss_t.item())
        sync()
        seconds = time.perf_counter() - s
        steps.append(dict(step=k + 1, loss=loss, loss_f32=f32_hex(loss), seconds=seconds,
                          ids_sha256=corpus.ids_sha256(rows)))
        if k < 3 or (k + 1) % 25 == 0:
            print(json.dumps(dict(event='step', **steps[-1])), flush=True)
    after = heldout(model)
    print(json.dumps(dict(event='heldout_after', mean=after['mean'])), flush=True)
    ours_eval = None
    if args.ours_params:
        # A failure here is recorded, never allowed to cost the torch record.
        try:
            import numpy as np
            ours = np.load(args.ours_params)
            if ours.dtype != np.float32 or ours.size != opp.EXPECTED_PARAMETERS[args.shape]:
                raise ValueError('--ours-params is not %d float32 values' % opp.EXPECTED_PARAMETERS[args.shape])
            del optimizer
            ours_model = opp.build_model(torch, dims, shapes, torch.from_numpy(ours), device)
            ours_eval = dict(heldout(ours_model), params_sha256=_sha(ours.tobytes()),
                             evaluator='torch eager float32 forward of this file (TF32 off, same SDPA backend)')
            del ours_model
            print(json.dumps(dict(event='heldout_ours_params_torch_eval', mean=ours_eval['mean'])), flush=True)
        except Exception as exc:
            ours_eval = dict(error=repr(exc)[:2000])
            print(json.dumps(dict(event='heldout_ours_params_torch_eval_failed', error=ours_eval['error'])),
                  flush=True)
    return dict(base_record(args, corpus, dims, starts, train_end), column='eager_fp32',
                torch_version=str(torch.__version__), torch_cuda=torch.version.cuda,
                torch_hip=getattr(torch.version, 'hip', None),
                gpu_name=torch.cuda.get_device_name(0) if on_gpu else None,
                precision=precision, sdpa=sdpa, init_source=init_source, initial_parameters_sha256=init_sha,
                wall_seconds=time.perf_counter() - t0,
                losses=[s['loss'] for s in steps], losses_f32=[s['loss_f32'] for s in steps],
                step_records=steps, heldout_before=before, heldout_after=after,
                heldout_ours_params_torch_eval=ours_eval,
                note='torch uses different kernels (fused SDPA, cuBLAS/rocBLAS); its curve is not '
                     'expected to equal ours bitwise')


def load(path):
    return json.loads(Path(path).read_text())


def cmd_compare(args):
    ours, theirs = load(args.ours), load(args.torch)
    lo, lt = ours['losses'], theirs['losses']
    n = min(len(lo), len(lt))
    out = []
    out.append('shape %s, corpus %s, steps ours %d torch %d' % (ours['shape_name'], ours['corpus']['name'],
                                                                 len(lo), len(lt)))
    if ours['initial_parameters_sha256'] != theirs['initial_parameters_sha256']:
        out.append('INIT DIFFERS: ours %s torch %s' % (ours['initial_parameters_sha256'],
                                                        theirs['initial_parameters_sha256']))
    else:
        out.append('init sha256 equal %s' % ours['initial_parameters_sha256'][:16])
    batches_equal = all(a['ids_sha256'] == b['ids_sha256'] for a, b in
                        zip(ours['step_records'][:n], theirs['step_records'][:n]))
    out.append('batch ids sha256 equal on all %d steps: %s' % (n, batches_equal))
    out.append('')
    out.append('| step | ours | torch | ours - torch | relative |')
    out.append('|---|---|---|---|---|')
    marks = [s for s in (1, 10, 50, 100, 150, 200, 250, 300) if s <= n]
    if n not in marks:
        marks.append(n)
    for s in marks:
        a, b = lo[s - 1], lt[s - 1]
        out.append('| %d | %.6f | %.6f | %+.6f | %+.3f%% |' % (s, a, b, a - b, 100 * (a - b) / b))
    diffs = [abs(a - b) for a, b in zip(lo[:n], lt[:n])]
    rels = [abs(a - b) / abs(b) for a, b in zip(lo[:n], lt[:n])]
    imax = max(range(n), key=lambda i: diffs[i])
    out.append('')
    out.append('max |diff| %.6f at step %d (relative %.4f%%); max relative %.4f%% at step %d'
               % (diffs[imax], imax + 1, 100 * rels[imax], 100 * max(rels), 1 + max(range(n), key=lambda i: rels[i])))
    out.append('final |diff| %.6f (relative %.4f%%)' % (diffs[-1], 100 * rels[-1]))
    w = min(50, n)
    mo, mt = sum(lo[n - w:n]) / w, sum(lt[n - w:n]) / w
    out.append('last %d steps mean training loss: ours %.6f torch %.6f (ours - torch %+.6f, %+.4f%%)'
               % (w, mo, mt, mo - mt, 100 * (mo - mt) / mt))
    first_diff = next((i + 1 for i in range(n) if ours['losses_f32'][i] != theirs['losses_f32'][i]), None)
    out.append('first step whose loss bits differ (ours vs torch): %s' % first_diff)
    out.append('')
    ho, ht = ours['heldout_after']['mean'], theirs['heldout_after']['mean']
    out.append('held-out mean loss before training: ours %.6f torch %.6f'
               % (ours['heldout_before']['mean'], theirs['heldout_before']['mean']))
    out.append('held-out mean loss after training (own evaluator): ours %.6f torch %.6f (ours - torch %+.6f, %+.4f%%)'
               % (ho, ht, ho - ht, 100 * (ho - ht) / ht))
    x = theirs.get('heldout_ours_params_torch_eval')
    if x and 'error' in x:
        out.append('torch evaluation of our parameters FAILED: %s' % x['error'])
    elif x:
        out.append('held-out, both final models under the torch evaluator: ours %.6f torch %.6f (%+.4f%%)'
                   % (x['mean'], ht, 100 * (x['mean'] - ht) / ht))
        out.append('ours evaluated by ours vs by torch, same parameters: %.6f vs %.6f (diff %+.2e)'
                   % (ho, x['mean'], ho - x['mean']))
    for other_path in args.ours_other or []:
        other = load(other_path)
        m = min(len(other['losses_f32']), len(ours['losses_f32']))
        same_steps = sum(1 for i in range(m) if other['losses_f32'][i] == ours['losses_f32'][i])
        same_held = (other['heldout_before']['rows_f32'] == ours['heldout_before']['rows_f32']
                     and other['heldout_after']['rows_f32'] == ours['heldout_after']['rows_f32'])
        same_state = other['final_witness'] == ours['final_witness']
        out.append('')
        out.append('ours vs %s: loss bits equal on %d of %d steps; held-out bits equal %s; '
                   'final parameters/m/v/flags sha256 equal %s'
                   % (other_path, same_steps, m, same_held, same_state))
    for other_path in args.torch_other or []:
        other = load(other_path)
        m = min(len(other['losses']), len(lt))
        d = [abs(a - b) for a, b in zip(other['losses'][:m], lt[:m])]
        eq = sum(1 for i in range(m) if other['losses_f32'][i] == theirs['losses_f32'][i])
        fd = next((i + 1 for i in range(m) if other['losses_f32'][i] != theirs['losses_f32'][i]), None)
        out.append('')
        out.append('torch vs %s: loss bits equal on %d of %d steps, first differing step %s, max |diff| %.6f, '
                   'final |diff| %.6f; held-out after %.6f vs %.6f'
                   % (other_path, eq, m, fd, max(d), d[-1], other['heldout_after']['mean'], ht))
    text = '\n'.join(out) + '\n'
    sys.stdout.write(text)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('ours', 'torch'):
        p = sub.add_parser(name)
        p.add_argument('--shape', choices=sorted(opp.SHAPES), default='target')
        p.add_argument('--corpus', choices=opp.CORPORA, default='enwik8')
        p.add_argument('--steps', type=int, default=300)
        p.add_argument('--heldout-rows', type=int, default=32)
        p.add_argument('--out', type=Path, required=True)
        if name == 'ours':
            p.add_argument('--params-out', type=Path, default=None,
                           help='final parameters as a float32 .npy (keep it off the evidence directory)')
        else:
            p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
            p.add_argument('--sdpa-backend', choices=('auto', 'efficient', 'flash', 'math', 'cudnn'),
                           default='auto')
            p.add_argument('--ours-params', type=Path, default=None,
                           help="an `ours` run's --params-out, evaluated on the held-out rows by torch")
    p = sub.add_parser('compare')
    p.add_argument('--ours', required=True)
    p.add_argument('--torch', required=True)
    p.add_argument('--ours-other', action='append', help="another vendor's `ours` JSON: bitwise check")
    p.add_argument('--torch-other', action='append', help="another vendor's `torch` JSON: difference")
    args = parser.parse_args()
    if args.cmd == 'compare':
        return cmd_compare(args)
    if args.out.exists():
        parser.error('%s exists; evidence is never overwritten' % args.out)
    dims = opp.SHAPES[args.shape]
    corpus = opp.Corpus(args.corpus, dims[0], dims[1])
    starts = heldout_starts(corpus, args.heldout_rows)
    train_end = check_disjoint(corpus, args.steps, starts)
    record = (cmd_ours if args.cmd == 'ours' else cmd_torch)(args, corpus, dims, starts, train_end)
    with args.out.open('x') as handle:
        handle.write(json.dumps(record, indent=1, allow_nan=False) + '\n')
    print(json.dumps(dict(event='result', side=args.cmd, out=str(args.out),
                          final_loss=record['losses'][-1], heldout_after=record['heldout_after']['mean'])),
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
