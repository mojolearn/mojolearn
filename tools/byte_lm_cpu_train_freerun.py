#!/usr/bin/env python3
"""A FREE-RUNNING CPU training run against the retained capture (DEVIATION 2680).

WHAT THIS ADDS TO THE GATE, AND WHY THE GATE DOES NOT ALREADY SAY IT.
`tools/byte_lm_cpu_train_gate.py cpu` replays each step from its OWN recorded
starting state: it loads `initial_p`, `initial_m` and `initial_v` for step N,
runs one step, and compares. That is deliberate and it is what makes a failure
at step 87 debuggable without replaying 86 steps first. But every step is
handed the GPU's state, so what it proves is per-step agreement.

The claim a reader actually wants is different: TRAIN ON A CPU AND END AT THE
GPU'S WEIGHTS. That should follow by induction -- if `post_p`, `post_m` and
`post_v` all agree at every step, then step N's output is step N+1's input and
the runs cannot separate -- but an induction argument is not a measurement, and
a re-seed is exactly the place a drift hides. So this runs it.

WHAT COMES FROM THE CAPTURE, AND WHAT DOES NOT. Only step 1's starting state is
read: `initial_p`, `initial_m`, `initial_v`. After that the model keeps its own
parameters and moments and is never touched again. Per step this reads only
`ids`, because token ids are the INPUT DATA a training run consumes, not state.
Reading `initial_*` for step N > 1 would rebuild the replay and prove nothing.

THE OPTIMIZER IS FIXED AT CONSTRUCTION, so a capture whose hyperparameters move
between steps cannot be expressed by one free run. This refuses such a tree by
name rather than silently using step 1's configuration for all 128 steps.

The per-step comparisons against the recorded bytes are diagnostics: they name
the first step at which anything separates. The VERDICT is the final state --
parameters and both moments after the last step -- against that step's recorded
`post_p`, `post_m`, `post_v`.

Exit 0: the free run ended on the recorded bytes. 1: it did not. 2: could not run.
"""
import argparse
import importlib
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load the gate as a module and reuse ITS capture readers. A second decoder here
# would be a second opinion about what the bytes mean, and this file exists to
# strengthen the gate's claim, not to make an independent one beside it.
_spec = importlib.util.spec_from_file_location(
    '_byte_lm_cpu_train_gate', Path(__file__).with_name('byte_lm_cpu_train_gate.py'))
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

#: The arrays a step produces, compared every step as a diagnostic. The verdict
#: uses only the final three.
PRODUCED = ('grad', 'loss', 'post_p', 'post_m', 'post_v')
FINAL = ('post_p', 'post_m', 'post_v')


def optimizer_across(tree, steps):
    """One optimizer configuration for the whole run, or a refusal.

    A free run constructs the trainer once, so the hyperparameters are fixed for
    all of it. If the capture's own steps disagree, say so and name the step and
    the field, rather than running 128 steps under step 1's numbers and
    reporting whatever comes out."""
    first = gate.optimizer_config(tree, steps[0])
    fields = ('lr', 'beta1', 'beta2', 'eps', 'weight_decay')
    for number in steps[1:]:
        other = gate.optimizer_config(tree, number)
        for name in fields:
            if float(other[name]) != float(first[name]):
                raise ValueError(
                    f'the capture changes {name} between step {steps[0]} '
                    f'({first[name]!r}) and step {number} ({other[name]!r}). A free '
                    'run fixes the optimizer at construction, so this tree cannot '
                    'be run as one; the per-step gate is the right tool for it.')
    return first


def freerun(tree, vendor, steps, verify_digests, report_path=None):
    shape = gate.bind_shape({vendor: tree}, argparse.Namespace(shape=None))
    sys.path.insert(0, str(ROOT / 'python'))
    host = importlib.import_module('mojolearn._byte_lm_host')
    trainer = getattr(host, 'LanguageModelHostTrainer', None)
    if trainer is None:
        print('freerun: no CPU training surface (LanguageModelHostTrainer); '
              'build bindings/build_byte_lm_host.sh', file=sys.stderr)
        return 2, None
    le_bytes = importlib.import_module('mojolearn._bufcheck').le_bytes
    frombytes = importlib.import_module('mojolearn._buffer').frombytes
    config = importlib.import_module('mojolearn._byte_lm_config')
    native = config.ByteLanguageModelConfig(**shape.to_json())

    opt = optimizer_across(tree, steps)

    # THE ONLY RECORDED STATE THIS RUN EVER READS. Step 1's starting point.
    first = steps[0]
    desc = gate.descriptors(tree, first) if verify_digests else {}
    model = trainer.from_state(
        frombytes(gate.array(tree, first, 'initial_p', desc.get('initial_p')), '<f4', (gate.N,)),
        frombytes(gate.array(tree, first, 'initial_m', desc.get('initial_m')), '<f4', (gate.N,)),
        frombytes(gate.array(tree, first, 'initial_v', desc.get('initial_v')), '<f4', (gate.N,)),
        completed_steps=first - 1, shape=native,
        lr=opt['lr'], betas=(opt['beta1'], opt['beta2']), eps=opt['eps'],
        weight_decay=opt['weight_decay'])

    compared, mismatches, digests = 0, [], 0
    first_divergence = None
    step_seconds = 0.0
    produced = {}
    for number in steps:
        desc = gate.descriptors(tree, number) if verify_digests else {}
        digests += len(desc)
        # ids only. Never initial_p/m/v -- that would be the replay.
        ids = frombytes(gate.array(tree, number, 'ids', desc.get('ids')), '<i4',
                        (gate.BATCH, gate.LENGTH + 1))
        started = time.perf_counter()
        bits = model.train_step(ids)
        step_seconds += time.perf_counter() - started
        produced = dict(grad=le_bytes(model.gradient_, 'f'),
                        loss=struct.pack('<I', bits),
                        post_p=le_bytes(model.parameters_, 'f'),
                        post_m=le_bytes(model.m_, 'f'),
                        post_v=le_bytes(model.v_, 'f'))
        for key in PRODUCED:
            compared += 1
            want = gate.array(tree, number, key, desc.get(key))
            if produced[key] != want:
                row = gate.locate(produced[key], want, key)
                row.update(step=number, vendor=vendor)
                if first_divergence is None:
                    first_divergence = number
                if len(mismatches) < 20:
                    mismatches.append(row)

    # THE VERDICT: the state this run ARRIVED at, against the recorded state
    # after the same step. Everything above is diagnosis.
    last = steps[-1]
    desc = gate.descriptors(tree, last) if verify_digests else {}
    final_equal = {}
    for key in FINAL:
        final_equal[key] = produced[key] == gate.array(tree, last, key, desc.get(key))
    ended_on_recorded = all(final_equal.values())

    verdict = 'PASS' if ended_on_recorded and not mismatches else 'FAIL'
    report = dict(schema='mojolearn.byte-lm-cpu-train-freerun.v1', mode='freerun',
                  deviation=2680, profile=gate.PROFILE, vendor=vendor,
                  steps=len(steps), first_step=first, last_step=last,
                  reseeded=False, recorded_state_read='step %d initial_p/m/v only' % first,
                  compared=compared, mismatched=len(mismatches),
                  first_divergence=first_divergence,
                  final_state_equal=final_equal,
                  ended_on_recorded_bytes=ended_on_recorded,
                  digests_verified=digests, optimizer=opt,
                  step_seconds=round(step_seconds, 4),
                  seconds_per_step=round(step_seconds / len(steps), 4),
                  first_mismatches=mismatches, verdict=verdict)

    print(f'freerun: {verdict}: {compared - len(mismatches)}/{compared} array comparisons '
          f'equal over {len(steps)} free-running steps against {vendor}, '
          f'NO per-step re-seeding')
    print(f'freerun: recorded state read: step {first} initial_p/m/v only; '
          f'per step, ids only')
    print('freerun: final state after step %d: %s' % (
        last, ', '.join(f'{k}={"equal" if v else "DIFFERENT"}' for k, v in final_equal.items())))
    if first_divergence is not None:
        print(f'freerun: first divergence at step {first_divergence}')
    print(f'freerun: optimizer from the capture: '
          + ' '.join(f'{k}={opt[k]!r}' for k in ('lr', 'beta1', 'beta2', 'eps', 'weight_decay')))
    print(f'freerun: train_step time: {step_seconds:.3f} s over {len(steps)} steps, '
          f'{step_seconds / len(steps) * 1000:.1f} ms per step, reference path, one thread')
    for row in mismatches[:10]:
        print(f'freerun: mismatch {row}')

    if report_path:
        with open(report_path, 'x') as stream:
            stream.write(json.dumps(report, indent=1, sort_keys=True) + '\n')
    return (0 if verdict == 'PASS' else 1), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--vendor', default='apple', choices=sorted(gate.VENDORS))
    parser.add_argument('--steps', default='all',
                        help='all, a comma list, or a-b (default: all). A free run '
                             'needs a contiguous range starting where its state does')
    parser.add_argument('--verify-digests', action='store_true')
    parser.add_argument('--expect-mismatch', action='store_true',
                        help='invert the verdict, so a sabotage build that still '
                             'lands on the recorded bytes is the failure')
    parser.add_argument('--tree', action='append', metavar='VENDOR=PATH',
                        help='read this vendor tree from an explicit path, for a '
                             'capture with no settled home yet. Same spelling as '
                             'the gate, and the shape is read from the capture '
                             'either way')
    parser.add_argument('--report', help='write the JSON report here (must not exist)')
    args = parser.parse_args(argv)
    try:
        trees = gate.present(argparse.Namespace(tree=args.tree))
        tree = trees.get(args.vendor)
        if tree is None:
            print(f'freerun: vendor tree not present: {args.vendor}', file=sys.stderr)
            return 2
        steps = gate.selected(args.steps)
        # A FREE RUN MUST BE CONTIGUOUS. Skipping step 40 would mean feeding step
        # 41's tokens to a model that never consumed step 40's, which is neither
        # the recorded run nor any other run, and its disagreement would mean
        # nothing.
        if steps != list(range(steps[0], steps[-1] + 1)):
            print('freerun: --steps must be a contiguous range; a free run cannot '
                  'skip a step and still be a run', file=sys.stderr)
            return 2
        code, report = freerun(tree, args.vendor, steps, args.verify_digests, args.report)
        if args.expect_mismatch and report is not None:
            inverted = 'PASS' if report['mismatched'] or not report['ended_on_recorded_bytes'] else 'FAIL'
            print(f'freerun: --expect-mismatch inverts the verdict to {inverted}')
            return 0 if inverted == 'PASS' else 1
        return code
    except ValueError as exc:
        print(f'freerun: could not run: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
