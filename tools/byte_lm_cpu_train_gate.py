#!/usr/bin/env python3
"""Byte LM CPU training against the retained three-vendor capture (DEVIATION 2680).

The retained capture holds, for every one of the 128 training steps and on
each of Apple Metal, NVIDIA CUDA and AMD HIP, the full FP32 tensors of one
step: the parameters, both Adam moments and the state flags before the step,
the token ids it consumed, the loss it produced, the gradient it computed,
and the parameters, moments and flags after the update. Nothing is sampled
and nothing is reduced to a digest; the digests in each `capture.json` are an
integrity check layered over the bytes themselves.

That is enough to certify a CPU training step without renting anything. Each
step carries its own starting state, so a step is checked in isolation: load
`initial_p`, `initial_m`, `initial_v` and `ids`, run one step on the CPU, and
require `grad`, `post_p`, `post_m`, `post_v` and the loss to equal the
recorded bytes. A disagreement at step 87 needs no replay of the 86 before it.

Two modes, because one of them is runnable before any CPU training code
exists:

  vendors   re-derive the three-vendor agreement from the raw bytes. For each
            selected step, every array of every pair of vendor trees must be
            equal byte for byte, and every array must match the SHA-256 its
            own capture.json records. The shipped inference gate trusts
            `comparison.json`'s identity_admitted flag instead of re-deriving
            it, so this mode stands on its own.

  cpu       the gate proper. Replays selected steps through the CPU training
            surface and compares against one vendor tree. Refuses with exit 2
            while that surface does not exist, which is the honest answer
            until the host backward pass lands.

A difference is reported by tensor. The parameter registry is a fixed order
of 21 tensors, so a flat element index is localized to a name, an index
inside that tensor and the two IEEE-754 bit patterns, which is the diagnostic
a kernel author actually needs.

WHAT A PASS DOES NOT SAY. Agreement here is agreement at this model profile,
this batch shape and this optimizer configuration. The weight gradient
contracts over the token count, so the same tokens presented as a different
batch or microbatch schedule are a different sum and are not covered by a
pass at this shape.

Exit 0: every comparison equal. 1: any difference. 2: the gate could not run.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESUME = ROOT / 'bench/results/resume'

#: The three retained trees. Apple sits inside the three-vendor bundle; the
#: other two are separate result trees that the bundle's comparator reads by
#: path, so the same indirection is spelled out here rather than hidden.
VENDORS = {
    'apple': RESUME / '2026-09-07-root-byte-lm-three-vendor/apple/full128',
    'cuda': RESUME / '2026-09-07-root-byte-lm-nvidia-common/run2/remote/byte-lm-validation/full128',
    'hip': RESUME / '2026-09-07-root-byte-lm-do-amd/run6/remote/byte-lm-do-output/byte-lm-validation/full128',
}

PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
N = 34944
STEPS = 128
COUNTS = {key: N for key in
          ('initial_p', 'initial_m', 'initial_v', 'post_p', 'post_m', 'post_v', 'grad')}
COUNTS.update(initial_flags=20, post_flags=20, loss=1, ids=66)
#: Every array of a step, in a fixed order, so two runs of this gate compare
#: the same things in the same sequence.
KEYS = ('initial_p', 'initial_m', 'initial_v', 'initial_flags', 'ids',
        'grad', 'loss', 'post_p', 'post_m', 'post_v', 'post_flags')
MAX_FILE = 16 * 1024 * 1024


def registry():
    """The 21 parameter tensors in their flat order, as the capture records
    them. Offsets are what turn a flat element index into a name."""
    shapes = [('embed', [256, 32])]
    for block in range(2):
        shapes += [(f'block{block}.{name}', shape) for name, shape in (
            ('norm1_w', [32]), ('w_q', [32, 32]), ('w_k', [16, 32]),
            ('w_v', [16, 32]), ('w_o', [32, 32]), ('norm2_w', [32]),
            ('w_gate', [64, 32]), ('w_up', [64, 32]), ('w_down', [32, 64]))]
    shapes += [('lm_head', [256, 32])]
    out, offset = [], 0
    for name, shape in shapes:
        count = math.prod(shape)
        out.append(dict(name=name, shape=shape, offset=offset, count=count))
        offset += count
    if offset != N:
        raise ValueError('registry does not sum to the parameter count')
    return out


REGISTRY = registry()


def filename(key):
    return key + ('.i32' if key == 'ids' or key.endswith('flags') else '.f32')


def read(path, size):
    if not path.exists():
        raise ValueError(f'missing capture file: {path}')
    actual = path.stat().st_size
    if actual != size or actual > MAX_FILE:
        raise ValueError(f'{path} is {actual} bytes, expected {size}')
    return path.read_bytes()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def step_dir(tree, number):
    return tree / f'step{number:06d}'


def descriptors(tree, number):
    """The step's own capture.json array descriptors, by key."""
    meta = json.loads(read_text(step_dir(tree, number) / 'capture.json'))
    found = meta.get('arrays', meta)
    return {key: found[key] for key in found if key in COUNTS}


def read_text(path):
    if not path.exists():
        raise ValueError(f'missing capture file: {path}')
    if path.stat().st_size > MAX_FILE:
        raise ValueError(f'{path} is too large to be a manifest')
    return path.read_text()


def array(tree, number, key, desc=None):
    """One array of one step, with its recorded digest checked when the
    capture.json describes it. A mismatch here is a corrupt or edited
    capture, which must fail before any comparison is reported."""
    raw = read(step_dir(tree, number) / filename(key), COUNTS[key] * 4)
    if desc is not None:
        want = desc.get('sha256')
        if want is not None and want != sha(raw):
            raise ValueError(f'capture digest mismatch: step {number} {key}')
        count = desc.get('count')
        if count is not None and count != COUNTS[key]:
            raise ValueError(f'capture count mismatch: step {number} {key}')
    return raw


def locate(got, want, key):
    """The first differing element, named. Floats are reported by their bit
    patterns because that is what equality means here."""
    if len(got) != len(want):
        return dict(key=key, reason='length', got=len(got), want=len(want))
    integral = key == 'ids' or key.endswith('flags')
    code = '<i' if integral else '<f'
    width = 4
    for index in range(len(want) // width):
        lo = index * width
        if got[lo:lo + width] == want[lo:lo + width]:
            continue
        row = dict(key=key, element=index)
        if integral:
            row.update(got=struct.unpack(code, got[lo:lo + width])[0],
                       want=struct.unpack(code, want[lo:lo + width])[0])
        else:
            row.update(got_bits=f'{struct.unpack("<I", got[lo:lo + width])[0]:08x}',
                       want_bits=f'{struct.unpack("<I", want[lo:lo + width])[0]:08x}',
                       got=struct.unpack(code, got[lo:lo + width])[0],
                       want=struct.unpack(code, want[lo:lo + width])[0])
        if COUNTS[key] == N:
            for tensor in REGISTRY:
                if tensor['offset'] <= index < tensor['offset'] + tensor['count']:
                    row.update(tensor=tensor['name'], shape=tensor['shape'],
                               index_in_tensor=index - tensor['offset'])
                    break
        return row
    return None


def selected(argument):
    """`all`, or a comma list of step numbers, or `a-b`."""
    if argument == 'all':
        return list(range(1, STEPS + 1))
    out = []
    for piece in argument.split(','):
        piece = piece.strip()
        if '-' in piece:
            lo, hi = piece.split('-', 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(piece))
    for number in out:
        if not 1 <= number <= STEPS:
            raise ValueError(f'step {number} is outside 1..{STEPS}')
    return sorted(set(out))


def present():
    """The vendor trees that are actually on disk."""
    return {name: path for name, path in VENDORS.items() if path.is_dir()}


def mode_vendors(args):
    """Re-derive the three-vendor agreement from the raw bytes."""
    trees = present()
    if len(trees) < 2:
        print(f'gate: fewer than two vendor trees present ({sorted(trees)})', file=sys.stderr)
        return 2, None
    names = sorted(trees)
    steps = selected(args.steps)
    compared, mismatches, digests = 0, [], 0
    for number in steps:
        arrays = {}
        for name in names:
            desc = descriptors(trees[name], number) if args.verify_digests else {}
            arrays[name] = {key: array(trees[name], number, key, desc.get(key)) for key in KEYS}
            digests += len(desc)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                for key in KEYS:
                    compared += 1
                    if arrays[left][key] != arrays[right][key]:
                        row = locate(arrays[left][key], arrays[right][key], key)
                        row.update(step=number, left=left, right=right)
                        if len(mismatches) < 20:
                            mismatches.append(row)
    verdict = 'PASS' if not mismatches else 'FAIL'
    report = dict(schema='mojolearn.byte-lm-cpu-train-gate.v1', mode='vendors',
                  deviation=2680, profile=PROFILE, vendors=names,
                  steps=len(steps), arrays_per_step=len(KEYS),
                  compared=compared, digests_verified=digests,
                  mismatched=len(mismatches), first_mismatches=mismatches,
                  verdict=verdict)
    print(f'gate: {verdict}: {compared - len(mismatches)}/{compared} array comparisons equal '
          f'over {len(steps)} steps, {len(KEYS)} arrays per step, vendors {" ".join(names)}'
          f'{"" if not args.verify_digests else f", {digests} recorded digests verified"}')
    for row in mismatches[:5]:
        print(f'gate: mismatch {row}')
    return (0 if verdict == 'PASS' else 1), report


def mode_cpu(args):
    """The gate proper. Replay steps on the CPU and compare with one vendor.

    The CPU training surface does not exist yet. This refuses rather than
    reporting a vacuous pass, and names what it looked for, so the lane that
    builds the host backward knows exactly what has to appear."""
    tree = VENDORS.get(args.vendor)
    if tree is None or not tree.is_dir():
        print(f'gate: vendor tree not present: {args.vendor}', file=sys.stderr)
        return 2, None
    # `mojolearn/__init__.py` calls `_backend.select()` before it exposes any
    # CPU surface, and that refuses in a tree with no binary built. Importing
    # the package would therefore fail for a reason that has nothing to do
    # with this gate, so the surface is looked for in the module that will
    # own it, without importing the package. A built tree still works: the
    # module imports the same way either way.
    trainer = None
    reason = None
    try:
        sys.path.insert(0, str(ROOT / 'python'))
        import importlib
        module = importlib.import_module('mojolearn._byte_lm_host')
        trainer = getattr(module, 'LanguageModelHostTrainer', None)
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
    if trainer is None:
        if reason is not None:
            print(f'gate: could not reach the host module ({reason})', file=sys.stderr)
        print('gate: no CPU training surface. This gate needs '
              'mojolearn.LanguageModelHostTrainer with train_step(ids) returning the '
              'loss and exposing the gradient and the Adam moments, built from the '
              'host backward pass (DEVIATION 2680). Until it exists there is nothing '
              'to compare and this is not a pass.', file=sys.stderr)
        return 2, None
    steps = selected(args.steps)
    compared, mismatches = 0, []
    for number in steps:
        desc = descriptors(tree, number) if args.verify_digests else {}
        start = {key: array(tree, number, key, desc.get(key))
                 for key in ('initial_p', 'initial_m', 'initial_v', 'ids')}
        model = trainer.from_state(start['initial_p'], start['initial_m'], start['initial_v'],
                                   completed_steps=number - 1)
        model.train_step(start['ids'])
        produced = dict(grad=model.gradient_bytes(), loss=model.loss_bytes(),
                        post_p=model.parameter_bytes(), post_m=model.moment_bytes('m'),
                        post_v=model.moment_bytes('v'))
        for key, got in produced.items():
            compared += 1
            want = array(tree, number, key, desc.get(key))
            if got != want:
                row = locate(got, want, key)
                row.update(step=number, vendor=args.vendor)
                if len(mismatches) < 20:
                    mismatches.append(row)
    verdict = 'PASS' if not mismatches else 'FAIL'
    report = dict(schema='mojolearn.byte-lm-cpu-train-gate.v1', mode='cpu',
                  deviation=2680, profile=PROFILE, vendor=args.vendor,
                  steps=len(steps), compared=compared, mismatched=len(mismatches),
                  first_mismatches=mismatches, verdict=verdict)
    print(f'gate: {verdict}: {compared - len(mismatches)}/{compared} array comparisons equal '
          f'over {len(steps)} steps against {args.vendor}')
    return (0 if verdict == 'PASS' else 1), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('mode', choices=('vendors', 'cpu'))
    parser.add_argument('--steps', default='all',
                        help='all, a comma list, or a-b (default: all)')
    parser.add_argument('--vendor', default='apple', choices=sorted(VENDORS),
                        help='which tree the cpu mode compares against')
    parser.add_argument('--verify-digests', action='store_true',
                        help='also check every array against the SHA-256 its capture.json records')
    parser.add_argument('--report', help='write the JSON report here (must not exist)')
    args = parser.parse_args(argv)
    try:
        code, report = mode_vendors(args) if args.mode == 'vendors' else mode_cpu(args)
    except ValueError as exc:
        print(f'gate: could not run: {exc}', file=sys.stderr)
        return 2
    if report is not None and args.report:
        with open(args.report, 'x') as stream:
            stream.write(json.dumps(report, indent=1, sort_keys=True) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
