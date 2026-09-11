#!/usr/bin/env python3
"""Byte LM GPU logits: device forward against the CPU reference path, byte for byte (DEVIATION 2660).

`LanguageModelTrainer.logits` runs the IDENTICAL device forward the
trainer's loss uses, on Metal, CUDA or HIP. For each selected parameter
state of the retained capture, with seeded random token ids that reach
every byte value, this sweep requires the GPU to equal the CPU reference
path (`LanguageModelInference(threaded=False)`, the oracles as written).

  logits_resident   a resident trainer's logits, byte for byte, at every
                    batch in [1, --max-batch] and every length in
                    [1, configured length]
  logits_stateless  a stateless trainer's logits at the same shapes, on
                    every Nth shape with --stateless-every N, because each
                    call builds and destroys a device context
  next_bytes        the resident trainer's greedy bytes at each shape
  loss_bits         the IEEE-754 bits of the resident trainer's evaluate()
                    on --loss-batches random [2, 33] batches

The ids are drawn in exactly the order tools/byte_lm_host_path_sweep.py
draws them, so the SHA-256 per state over the GPU resident logits equals
that sweep's reference_logits_sha256 whenever the GPU equals the CPU. At the
CPU sweep's defaults the recorded values are checked and a difference fails
the sweep. A negative control requires the logits of two states with
different parameters to differ on the same ids, so a PASS cannot come from a
comparison that compares nothing.

The resident sessions and the stateless calls create device contexts in one
process (the DEVIATION 2494 shape). If the second-context hang is open on
the box, run with MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1 and say so in the record.

Exits 0 when every comparison is equal, 1 on any difference, and 2 when the
sweep could not run.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / 'bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple/full128'
DEVIATION = 2660
#: The defaults of tools/byte_lm_host_path_sweep.py and the reference logits
#: SHA-256 it recorded at them. The loss batches are part of the key because
#: their ids are drawn from the same stream between states.
CPU_SWEEP_DEFAULTS = dict(seed=2640, max_batch=8, states=['initial', 'step000064', 'final'], loss_batches=16)
CPU_SWEEP_LOGITS_SHA256 = {
    'initial': '6db55997f1e4d1e9aa08bfaf082ebe48c1b03eec121ebaf6ddf3a0bdbad7e61f',
    'step000064': '30a89281f2daac140dc9fb440b4476dfe53dd3dcfd5dc4746a84a6fc74ee26f5',
    'final': 'b518e71ecf5ac01e743f5d5b46df6687e9eefa80a9d627580162305a083f70b2',
}


def state_path(name):
    if name == 'initial':
        return CAPTURE / 'initial/initial_p.f32'
    if name == 'final':
        return CAPTURE / 'step000128/post_p.f32'
    return CAPTURE / name / 'initial_p.f32'


def gate_module():
    spec = importlib.util.spec_from_file_location('byte_lm_host_gate', ROOT / 'tools/byte_lm_host_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commit():
    env = os.environ.get('MOJOLEARN_GATE_COMMIT')
    if env:
        return env
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def first_difference(got, want):
    """The flat index of the first differing float32, or None when the byte
    lengths differ."""
    if len(got) != len(want):
        return None
    for at in range(0, len(got), 4):
        if got[at:at + 4] != want[at:at + 4]:
            return at // 4
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--states', default='initial,step000064,final',
                        help="comma list: 'initial', 'final', or a capture step directory such as step000064")
    parser.add_argument('--loss-batches', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2640)
    parser.add_argument('--stateless-every', type=int, default=1,
                        help='compare the stateless path on every Nth shape only (it builds a device context per call)')
    parser.add_argument('--report', type=Path, help='new exclusive JSON report')
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / 'python'))
    try:
        from mojolearn import _backend
        from mojolearn import _byte_lm_impl as impl
        from mojolearn._buffer import frombytes
        from mojolearn._bufcheck import le_bytes
        from mojolearn._byte_lm_config import ByteLanguageModelConfig
        from mojolearn._byte_lm_host import LanguageModelInference, binary_path
        from mojolearn.language_model import LanguageModelTrainer
        gate = gate_module()
    except Exception as exc:  # the import is part of what runs
        print(f'sweep: import failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

    shape = ByteLanguageModelConfig()
    states = [s for s in args.states.split(',') if s]
    if not states or not 1 <= args.max_batch <= impl._LOGITS_MAX_BATCH or args.stateless_every < 1 \
            or args.loss_batches < 0:
        print(f'sweep: --states must name a state, --max-batch must be in [1, {impl._LOGITS_MAX_BATCH}], '
              '--stateless-every must be positive and --loss-batches nonnegative', file=sys.stderr)
        return 2
    for name in states:
        if not state_path(name).exists():
            print(f'sweep: no parameter file for state {name}: {state_path(name)}', file=sys.stderr)
            return 2

    try:
        binding = impl._load(shape)
        missing = [entry for entry in ('byte_lm_logits', 'byte_lm_session_logits', *impl._SESSION_ENTRIES)
                   if not callable(getattr(binding, entry, None))]
        if missing:
            raise ImportError('GPU byte LM binding lacks ' + ', '.join(missing)
                              + '; rebuild bindings/build_byte_lm.sh')
        metadata = impl._binding_metadata(binding)
        gpu = dict(vendor=str(binding.byte_lm_vendor()), arch=_backend.gpu_arch(),
                   numeric_mode=dict(process=_backend.numeric_mode(),
                                     native=int(binding.byte_lm_numeric_mode())),
                   profile=metadata['native_profile'], attention_arm=metadata['native_attention_arm'],
                   path=metadata['binding_file'], sha256=metadata['binding_sha256'])
        cpu = dict(path=binary_path(), sha256=gate.sha256(binary_path()))
    except Exception as exc:
        print(f'sweep: bindings not usable: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    seen = set()
    compared = dict(logits_resident=0, logits_stateless=0, next_bytes=0, loss_bits=0)
    mismatches = []
    gpu_digests = {}
    cpu_digests = {}
    parameter_sha = {}
    schedule = {'tool': 'byte_lm_gpu_logits_sweep', 'deviation': DEVIATION, 'seed': args.seed}
    started = time.perf_counter()

    def ids(batch, width):
        flat = [rng.randrange(shape.vocab_size) for _ in range(batch * width)]
        seen.update(flat)
        return frombytes(struct.pack(f'<{len(flat)}i', *flat), '<i4', (batch, width))

    def parameters(name):
        return frombytes(state_path(name).read_bytes(), '<f4', (shape.n_total,))

    def mismatch(**row):
        if len(mismatches) < 20 and 'got_bytes' in row:
            row['first_cell'] = first_difference(row.pop('got_bytes'), row.pop('want_bytes'))
        row.pop('got_bytes', None)
        row.pop('want_bytes', None)
        mismatches.append(row)

    negative_control = None
    try:
        for name in states:
            params = parameters(name)
            model = LanguageModelInference(params, shape=shape, threaded=False)
            parameter_sha[name] = model.parameters_sha256()
            resident = LanguageModelTrainer(params, data_schedule=schedule, resident=True)
            stateless = LanguageModelTrainer(params, data_schedule=schedule, resident=False)
            try:
                gpu_digest = hashlib.sha256()
                cpu_digest = hashlib.sha256()
                shape_index = 0
                for batch in range(1, args.max_batch + 1):
                    for length in range(1, shape.length + 1):
                        x = ids(batch, length)
                        reference = le_bytes(model.logits(x, threaded=False), 'f')
                        cpu_digest.update(reference)
                        reference_next = model.next_bytes(x, threaded=False)
                        got = le_bytes(resident.logits(x), 'f')
                        gpu_digest.update(got)
                        compared['logits_resident'] += 1
                        if got != reference:
                            mismatch(state=name, what='logits_resident', batch=batch, length=length,
                                     got_bytes=got, want_bytes=reference)
                        if shape_index % args.stateless_every == 0:
                            compared['logits_stateless'] += 1
                            got = le_bytes(stateless.logits(x), 'f')
                            if got != reference:
                                mismatch(state=name, what='logits_stateless', batch=batch, length=length,
                                         got_bytes=got, want_bytes=reference)
                        compared['next_bytes'] += 1
                        got_next = resident.next_bytes(x)
                        if got_next != reference_next:
                            mismatch(state=name, what='next_bytes', batch=batch, length=length,
                                     want=reference_next, got=got_next)
                        shape_index += 1
                for index in range(args.loss_batches):
                    y = ids(shape.batch, shape.length + 1)
                    want = model.loss_bits(y, threaded=False)
                    got = struct.unpack('<I', struct.pack('<f', resident.evaluate(y)))[0]
                    compared['loss_bits'] += 1
                    if got != want:
                        mismatch(state=name, what='loss_bits', batch_index=index,
                                 want=f'{want:08x}', got=f'{got:08x}')
            finally:
                resident.close()
            gpu_digests[name] = gpu_digest.hexdigest()
            cpu_digests[name] = cpu_digest.hexdigest()

        # The negative control draws from its own stream, so the main
        # stream stays the CPU sweep's.
        distinct = []
        for name in states:
            if all(parameter_sha[name] != parameter_sha[other] for other in distinct):
                distinct.append(name)
        if len(distinct) >= 2:
            first, second = distinct[:2]
            control = random.Random(args.seed + 1)
            flat = [control.randrange(shape.vocab_size) for _ in range(2 * shape.length)]
            x = frombytes(struct.pack(f'<{len(flat)}i', *flat), '<i4', (2, shape.length))
            trainer = LanguageModelTrainer(parameters(first), data_schedule=schedule, resident=True)
            try:
                got = le_bytes(trainer.logits(x), 'f')
            finally:
                trainer.close()
            other = LanguageModelInference(parameters(second), shape=shape, threaded=False)
            want = le_bytes(other.logits(x, threaded=False), 'f')
            negative_control = dict(states=[first, second], seed=args.seed + 1, batch=2,
                                    length=shape.length, differ=got != want)
    except Exception as exc:
        print(f'sweep: could not complete: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

    at_defaults = (args.seed == CPU_SWEEP_DEFAULTS['seed'] and args.max_batch == CPU_SWEEP_DEFAULTS['max_batch']
                   and states == CPU_SWEEP_DEFAULTS['states']
                   and args.loss_batches == CPU_SWEEP_DEFAULTS['loss_batches'])
    digests_equal = (all(gpu_digests[name] == CPU_SWEEP_LOGITS_SHA256[name] for name in states)
                     if at_defaults else None)
    verdict = ('PASS' if not mismatches and digests_equal is not False
               and (negative_control is None or negative_control['differ']) else 'FAIL')
    report = dict(
        schema='mojolearn.byte-lm-gpu-logits-sweep.v1', deviation=DEVIATION, commit=commit(),
        host=gate.host_info(), gpu=gpu, cpu_binary=cpu,
        seed=args.seed, max_batch=args.max_batch, length=shape.length, states=states,
        loss_batches=args.loss_batches, stateless_every=args.stateless_every,
        parameter_sha256=parameter_sha, byte_values_seen=len(seen),
        compared=compared, mismatched=len(mismatches), first_mismatches=mismatches[:20],
        gpu_logits_sha256=gpu_digests, cpu_reference_logits_sha256=cpu_digests,
        cpu_sweep_logits_sha256=CPU_SWEEP_LOGITS_SHA256 if at_defaults else None,
        digests_equal_cpu_sweep=digests_equal, negative_control=negative_control,
        seconds=round(time.perf_counter() - started, 3), verdict=verdict)
    if args.report:
        with open(args.report, 'x') as stream:
            stream.write(json.dumps(report, indent=1, sort_keys=True) + '\n')
    total = sum(compared.values())
    print(f"sweep: {verdict}: {total - len(mismatches)}/{total} comparisons equal "
          f"({compared['logits_resident']} logits_resident, {compared['logits_stateless']} logits_stateless, "
          f"{compared['next_bytes']} next_bytes, {compared['loss_bits']} loss_bits) "
          f"over {len(states)} states, batch 1..{args.max_batch}, length 1..{shape.length}, "
          f"{len(seen)} byte values, vendor {gpu['vendor']}")
    for name, value in gpu_digests.items():
        print(f'sweep: gpu logits sha256 {name} {value}')
    print(f'sweep: digests equal the CPU sweep: {digests_equal}')
    print(f'sweep: negative control: {negative_control}')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
