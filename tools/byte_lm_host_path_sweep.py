#!/usr/bin/env python3
"""Byte LM CPU inference: threaded path against reference path, byte for byte (DEVIATION 2640).

The gate (tools/byte_lm_host_gate.py) ties both paths to the retained GPU
loss bytes at the training shape [2, 33] and compares logits probes at three
shapes. This sweep covers the rest of the public surface. For each selected
parameter state of the retained capture, with seeded random token ids that
reach every byte value, it requires the threaded path at every requested
thread count to equal the reference path (the oracles as written):

  logits      byte for byte at every batch in [1, --max-batch] and every
              length in [1, configured length]
  next_bytes  the same greedy bytes at each of those shapes
  loss_bits   the same IEEE-754 bits on --loss-batches random [2, 33] batches

It also records one SHA-256 per state over the reference logits of the whole
sweep, in order, so reports from different CPUs can be compared at every
shape and not only through the gate's probes.

Exit 0: every comparison equal. 1: any difference. 2: the sweep could not run.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--threads', default='1,2,3', help='comma list of thread counts for the threaded path')
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--states', default='initial,step000064,final',
                        help="comma list: 'initial', 'final', or a capture step directory such as step000064")
    parser.add_argument('--loss-batches', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2640)
    parser.add_argument('--report', type=Path, help='new exclusive JSON report')
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / 'python'))
    try:
        from mojolearn._buffer import frombytes
        from mojolearn._bufcheck import le_bytes
        from mojolearn._byte_lm_config import ByteLanguageModelConfig
        from mojolearn._byte_lm_host import LanguageModelInference, binary_path
        gate = gate_module()
    except Exception as exc:  # the import is part of what runs
        print(f'sweep: import failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

    shape = ByteLanguageModelConfig()
    threads = [int(t) for t in args.threads.split(',') if t]
    states = [s for s in args.states.split(',') if s]
    if not threads or min(threads) < 1 or args.max_batch < 1:
        print('sweep: threads and --max-batch must be positive', file=sys.stderr)
        return 2
    for name in states:
        if not state_path(name).exists():
            print(f'sweep: no parameter file for state {name}: {state_path(name)}', file=sys.stderr)
            return 2

    rng = random.Random(args.seed)
    seen = set()
    compared = dict(logits=0, next_bytes=0, loss_bits=0)
    mismatches = []
    digests = {}
    parameter_sha = {}
    started = time.perf_counter()

    def ids(batch, width):
        flat = [rng.randrange(shape.vocab_size) for _ in range(batch * width)]
        seen.update(flat)
        return frombytes(struct.pack(f'<{len(flat)}i', *flat), '<i4', (batch, width))

    for name in states:
        params = frombytes(state_path(name).read_bytes(), '<f4', (shape.n_total,))
        model = LanguageModelInference(params, shape=shape)
        parameter_sha[name] = model.parameters_sha256()
        digest = hashlib.sha256()
        for batch in range(1, args.max_batch + 1):
            for length in range(1, shape.length + 1):
                x = ids(batch, length)
                reference = le_bytes(model.logits(x, threaded=False), 'f')
                digest.update(reference)
                reference_next = model.next_bytes(x, threaded=False)
                for count in threads:
                    compared['logits'] += 1
                    if le_bytes(model.logits(x, threaded=True, threads=count), 'f') != reference:
                        mismatches.append(dict(state=name, what='logits', batch=batch, length=length, threads=count))
                    compared['next_bytes'] += 1
                    if model.next_bytes(x, threaded=True, threads=count) != reference_next:
                        mismatches.append(dict(state=name, what='next_bytes', batch=batch, length=length, threads=count))
        for index in range(args.loss_batches):
            y = ids(shape.batch, shape.length + 1)
            want = model.loss_bits(y, threaded=False)
            for count in threads:
                compared['loss_bits'] += 1
                got = model.loss_bits(y, threaded=True, threads=count)
                if got != want:
                    mismatches.append(dict(state=name, what='loss_bits', batch_index=index, threads=count,
                                           want=f'{want:08x}', got=f'{got:08x}'))
        digests[name] = digest.hexdigest()

    verdict = 'PASS' if not mismatches else 'FAIL'
    report = dict(
        schema='mojolearn.byte-lm-host-path-sweep.v1', deviation=2640, commit=commit(),
        host=gate.host_info(), binary=dict(path=binary_path(), sha256=gate.sha256(binary_path())),
        seed=args.seed, threads=threads, max_batch=args.max_batch, length=shape.length,
        states=states, parameter_sha256=parameter_sha, byte_values_seen=len(seen),
        compared=compared, mismatched=len(mismatches), first_mismatches=mismatches[:20],
        reference_logits_sha256=digests, seconds=round(time.perf_counter() - started, 3), verdict=verdict)
    if args.report:
        with open(args.report, 'x') as stream:
            stream.write(json.dumps(report, indent=1, sort_keys=True) + '\n')
    total = sum(compared.values())
    print(f"sweep: {verdict}: {total - len(mismatches)}/{total} comparisons equal "
          f"({compared['logits']} logits, {compared['next_bytes']} next_bytes, {compared['loss_bits']} loss_bits) "
          f"over {len(states)} states, batch 1..{args.max_batch}, length 1..{shape.length}, threads {threads}, "
          f"{len(seen)} byte values")
    for name, value in digests.items():
        print(f'sweep: reference logits sha256 {name} {value}')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
