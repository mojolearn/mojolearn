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
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESUME = ROOT / 'bench/results/resume'

# DEVIATION 2682. The shape used to be five literals here. It now comes from the
# shared module, and from the CAPTURE rather than from an assumption, so this
# gate reads a tree and learns which shape it is holding instead of insisting on
# the only one that existed when it was written. The helper asserts at import
# that its default derivation equals those literals, so the certified b2-l32
# path cannot move underneath this.
_shape_spec = importlib.util.spec_from_file_location(
    '_byte_lm_shape', Path(__file__).with_name('byte_lm_shape.py'))
byte_lm_shape = importlib.util.module_from_spec(_shape_spec)
_shape_spec.loader.exec_module(byte_lm_shape)

#: The three retained trees. Apple sits inside the three-vendor bundle; the
#: other two are separate result trees that the bundle's comparator reads by
#: path, so the same indirection is spelled out here rather than hidden.
VENDORS = {
    'apple': RESUME / '2026-09-07-root-byte-lm-three-vendor/apple/full128',
    'cuda': RESUME / '2026-09-07-root-byte-lm-nvidia-common/run2/remote/byte-lm-validation/full128',
    'hip': RESUME / '2026-09-07-root-byte-lm-do-amd/run6/remote/byte-lm-do-output/byte-lm-validation/full128',
}

STEPS = 128
#: Every array of a step, in a fixed order, so two runs of this gate compare
#: the same things in the same sequence.
KEYS = ('initial_p', 'initial_m', 'initial_v', 'initial_flags', 'ids',
        'grad', 'loss', 'post_p', 'post_m', 'post_v', 'post_flags')
MAX_FILE = 16 * 1024 * 1024

#: The shape the gate is currently reading. It starts as the certified default,
#: which is what every capture written before DEVIATION 2682 is, and `use_shape`
#: replaces it once a tree says otherwise. One binding point, because every
#: reader below needs the same answer and a second source would be a second
#: opinion.
SHAPE = byte_lm_shape.Shape()
PROFILE = SHAPE.profile
N = SHAPE.n_total
BATCH, LENGTH = SHAPE.batch, SHAPE.length
COUNTS = SHAPE.counts()
REGISTRY = SHAPE.registry()
_BOUND = False


def use_shape(shape):
    """Bind the gate to one shape, once, and refuse a second one.

    A single run compares one tree against one replay, so two shapes inside one
    run would mean the counts and the registry changed underneath a comparison
    that had already started."""
    global SHAPE, PROFILE, N, BATCH, LENGTH, COUNTS, REGISTRY, _BOUND
    if _BOUND and shape != SHAPE:
        raise ValueError(f'one run compares one shape; this tree is {shape.profile} '
                         f'and the run is already bound to {SHAPE.profile}')
    _BOUND = True
    SHAPE = shape
    PROFILE = shape.profile
    N = shape.n_total
    BATCH, LENGTH = shape.batch, shape.length
    COUNTS = shape.counts()
    REGISTRY = shape.registry()
    return shape


def shape_of(tree, number=1):
    """The shape a retained tree records, read from a step's own capture.json.

    A capture from before DEVIATION 2682 records no `model_shape` and is the
    default by construction, since that is the only shape that existed when it
    was written."""
    meta = json.loads(read_text(step_dir(tree, number) / 'capture.json'))
    return byte_lm_shape.from_capture(meta.get('config', {}))


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


#: `kind` in a recorded optimizer config. The byte LM trainer admits only
#: AdamW, and so does the CPU surface, so any other value is refused here
#: rather than quietly compared against arithmetic it does not describe.
OPT_ADAMW = 2


def optimizer_config(tree, number):
    """The optimizer configuration that produced this recorded step.

    Read rather than assumed. The capture's own `config.optimizer` block is
    what the GPU ran, and `post_p` is the only array that reads `lr` or
    `weight_decay`, so a default guessed here fails the gate on
    hyperparameters while the backward pass may be agreeing perfectly."""
    meta = json.loads(read_text(step_dir(tree, number) / 'capture.json'))
    config = meta.get('config', {})
    opt = config.get('optimizer')
    if not isinstance(opt, dict):
        raise ValueError(f'step {number} records no optimizer configuration')
    if opt.get('kind') != OPT_ADAMW:
        raise ValueError(f'step {number} was not AdamW (kind {opt.get("kind")!r}); '
                         'the CPU surface admits AdamW only')
    for name in ('max_norm', 'momentum', 'dampening'):
        if float(opt.get(name, 0.0)) != 0.0:
            raise ValueError(f'step {number} used {name}={opt[name]}, which the '
                             'CPU surface refuses')
    if opt.get('nesterov'):
        raise ValueError(f'step {number} used nesterov, which the CPU surface refuses')
    for name in ('lr', 'beta1', 'beta2', 'eps', 'weight_decay'):
        if name not in opt:
            raise ValueError(f'step {number} records no {name}')
    return opt


def selected(argument):
    """`all`, `every:N`, a comma list of step numbers, or `a-b`.

    `every:N` is step 1 and every Nth step after it, always including the
    last, which is how a CI run samples the capture without paying for all
    128 steps of a reference-path replay."""
    if argument == 'all':
        return list(range(1, STEPS + 1))
    if argument.startswith('every:'):
        stride = int(argument.split(':', 1)[1])
        if stride < 1:
            raise ValueError('every:N needs N >= 1')
        return sorted({1, STEPS} | set(range(1, STEPS + 1, stride)))
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


def present(args=None):
    """The vendor trees that are actually on disk.

    `--tree vendor=path` names a tree explicitly, which is how a second shape's
    capture is read before it has a settled home under bench/results."""
    trees = dict(VENDORS)
    named = set()
    for entry in getattr(args, 'tree', None) or ():
        name, _, raw = entry.partition('=')
        if not name or not raw:
            raise ValueError('--tree takes vendor=path')
        trees[name] = Path(raw)
        named.add(name)
    # A MISSING BUILT-IN TREE IS ABSENCE; A MISSING NAMED ONE IS AN ERROR. The
    # built-in paths are filtered because a runner legitimately holds only some
    # of them. A caller who spells out a path is saying to read that tree, so
    # dropping it silently would compare whatever else happened to be there and
    # report a pass over it, which is the failure this gate exists to refuse.
    for name in sorted(named):
        if not trees[name].is_dir():
            raise ValueError(f'--tree {name}={trees[name]} is not a directory')
    return {name: path for name, path in trees.items() if path.is_dir()}


def bind_shape(trees, args):
    """Bind the run to the shape its trees record, and refuse a mixed set.

    Comparing two trees of different shapes would compare arrays of different
    lengths, which is a failure worth naming rather than a mismatch to report."""
    shapes = {name: shape_of(path) for name, path in trees.items()}
    distinct = {s.fields for s in shapes.values()}
    if len(distinct) > 1:
        raise ValueError('trees record different shapes: '
                         + ', '.join(f'{n}={s.profile}' for n, s in sorted(shapes.items())))
    shape = use_shape(next(iter(shapes.values())))
    wanted = getattr(args, 'shape', None)
    if wanted is not None and byte_lm_shape.parse(wanted) != shape:
        raise ValueError(f'--shape {wanted} does not describe these trees, which are {shape.profile}')
    return shape


def mode_vendors(args):
    """Re-derive the vendor agreement from the raw bytes, over whatever trees
    are present.

    THE TREES ARE NOT ALL IN ONE PLACE and a runner may hold only some. Apple
    sits inside the three-vendor bundle; CUDA and HIP are separate result trees
    totalling about 308 MB, which CI does not sparse-checkout. So the honest
    behaviour with one tree is to verify that tree's own recorded digests and
    say that no cross-vendor comparison was possible, not to refuse, and not
    to report a pass that reads like three vendors agreed.

    `--require-vendors N` is how a caller that MEANS to compare demands it.
    """
    trees = present(args)
    names = sorted(trees)
    if names:
        bind_shape(trees, args)
    if len(names) < args.require_vendors:
        print(f'gate: {len(names)} vendor tree(s) present {names}, '
              f'--require-vendors {args.require_vendors} demands more', file=sys.stderr)
        return 2, None
    if not names:
        print('gate: no vendor tree present', file=sys.stderr)
        return 2, None
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
    cross_vendor = len(names) >= 2
    report = dict(schema='mojolearn.byte-lm-cpu-train-gate.v1', mode='vendors',
                  deviation=2680, profile=PROFILE, vendors=names,
                  cross_vendor=cross_vendor,
                  steps=len(steps), arrays_per_step=len(KEYS),
                  compared=compared, digests_verified=digests,
                  mismatched=len(mismatches), first_mismatches=mismatches,
                  verdict=verdict)
    if cross_vendor:
        print(f'gate: {verdict}: {compared - len(mismatches)}/{compared} array comparisons equal '
              f'over {len(steps)} steps, {len(KEYS)} arrays per step, vendors {" ".join(names)}'
              f'{"" if not args.verify_digests else f", {digests} recorded digests verified"}')
    else:
        # One tree cannot agree with anything. Say that, rather than letting a
        # zero-comparison PASS read like a cross-vendor result.
        print(f'gate: {verdict}: NO CROSS-VENDOR COMPARISON, only the {names[0]} tree is present '
              f'({len(steps)} steps, {digests} recorded digests verified, 0 array comparisons). '
              f'The CUDA and HIP trees are about 308 MB and CI does not check them out.')
    for row in mismatches[:5]:
        print(f'gate: mismatch {row}')
    return (0 if verdict == 'PASS' else 1), report


def mode_cpu(args):
    """The gate proper. Replay steps on the CPU and compare with one vendor.

    Each step is replayed from its OWN recorded starting state, its own token
    ids and its own optimizer configuration, so a step is judged in isolation
    and a failure at step 87 needs no replay of the 86 before it. The gradient,
    the loss bits, and the post-step parameters and both Adam moments must equal
    the recorded bytes exactly.

    If the surface is absent this refuses with exit 2 and names what it looked
    for, rather than reporting a pass over nothing."""
    tree = present(args).get(args.vendor)
    if tree is None or not tree.is_dir():
        print(f'gate: vendor tree not present: {args.vendor}', file=sys.stderr)
        return 2, None
    shape = bind_shape({args.vendor: tree}, args)
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
        le_bytes = importlib.import_module('mojolearn._bufcheck').le_bytes
        frombytes = importlib.import_module('mojolearn._buffer').frombytes
        # The surface takes its own config object. Building it from the nine
        # integers the capture recorded is what makes the replay the same shape
        # as the tree, rather than the default the surface would otherwise pick.
        config = importlib.import_module('mojolearn._byte_lm_config')
        native = config.ByteLanguageModelConfig(**shape.to_json())
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
    if trainer is None:
        if reason is not None:
            print(f'gate: could not reach the host module ({reason})', file=sys.stderr)
        print('gate: no CPU training surface. This gate needs '
              'mojolearn._byte_lm_host.LanguageModelHostTrainer, built from the host '
              'backward pass (DEVIATION 2680) with the binding entry '
              'byte_lm_host_train_step. Until it exists there is nothing to compare '
              'and this is not a pass.', file=sys.stderr)
        return 2, None
    steps = selected(args.steps)
    compared, mismatches = 0, []
    used_optimizer = None
    # Time the STEP, not the window around it. An earlier estimate of this cost
    # was taken from a window that contained a binding build and came out about
    # twentyfold high, which is how a 119-step sample got justified. The clock
    # here covers train_step alone, not the file reads or the comparisons.
    step_seconds = 0.0
    for number in steps:
        desc = descriptors(tree, number) if args.verify_digests else {}
        start = {key: array(tree, number, key, desc.get(key))
                 for key in ('initial_p', 'initial_m', 'initial_v', 'ids')}
        # The recorded arrays are raw little-endian bytes, which is what the
        # surface accepts as parameters and moments, and the ids are int32.
        # THE OPTIMIZER COMES FROM THE CAPTURE, NOT FROM DEFAULTS. Each step's
        # capture.json records the configuration that produced it (lr 0.003 and
        # weight_decay 0.01 here, not the surface's 0.001 and 0.0), and post_p
        # is the only array that reads either. Assuming defaults made this gate
        # fail on hyperparameters while the backward pass was in fact agreeing,
        # which is a gate testing the wrong thing.
        opt = optimizer_config(tree, number)
        used_optimizer = opt
        model = trainer.from_state(
            frombytes(start['initial_p'], '<f4', (N,)),
            frombytes(start['initial_m'], '<f4', (N,)),
            frombytes(start['initial_v'], '<f4', (N,)),
            completed_steps=number - 1, shape=native,
            lr=opt['lr'], betas=(opt['beta1'], opt['beta2']), eps=opt['eps'],
            weight_decay=opt['weight_decay'])
        ids = frombytes(start['ids'], '<i4', (BATCH, LENGTH + 1))
        started = time.perf_counter()
        bits = model.train_step(ids)
        step_seconds += time.perf_counter() - started
        produced = dict(grad=le_bytes(model.gradient_, 'f'),
                        loss=struct.pack('<I', bits),
                        post_p=le_bytes(model.parameters_, 'f'),
                        post_m=le_bytes(model.m_, 'f'),
                        post_v=le_bytes(model.v_, 'f'))
        for key, got in produced.items():
            compared += 1
            want = array(tree, number, key, desc.get(key))
            if got != want:
                row = locate(got, want, key)
                row.update(step=number, vendor=args.vendor)
                if len(mismatches) < 20:
                    mismatches.append(row)
    # A SABOTAGE BUILD MUST FAIL THIS GATE. With --expect-mismatch the verdict
    # inverts: agreement becomes the failure, because a control that cannot
    # fire proves nothing about the gate it is meant to validate.
    if args.expect_mismatch:
        verdict = 'PASS' if mismatches else 'FAIL'
    else:
        verdict = 'PASS' if not mismatches else 'FAIL'
    report = dict(schema='mojolearn.byte-lm-cpu-train-gate.v1', mode='cpu',
                  deviation=2680, profile=PROFILE, vendor=args.vendor,
                  steps=len(steps), compared=compared, mismatched=len(mismatches),
                  first_mismatches=mismatches, optimizer=used_optimizer,
                  expect_mismatch=args.expect_mismatch,
                  step_seconds=round(step_seconds, 4),
                  seconds_per_step=(round(step_seconds / len(steps), 4) if steps else None),
                  verdict=verdict)
    print(f'gate: {verdict}: {compared - len(mismatches)}/{compared} array comparisons equal '
          f'over {len(steps)} steps against {args.vendor}'
          f'{" (sabotage build, a mismatch was required)" if args.expect_mismatch else ""}')
    if used_optimizer is not None:
        # State the optimizer, so a hyperparameter error is distinguishable
        # from an arithmetic one without re-deriving it.
        print('gate: optimizer from the capture: '
              + ' '.join(f'{k}={used_optimizer[k]!r}'
                         for k in ('lr', 'beta1', 'beta2', 'eps', 'weight_decay')))
    if steps:
        print(f'gate: train_step time: {step_seconds:.3f} s over {len(steps)} steps, '
              f'{step_seconds / len(steps) * 1000:.1f} ms per step, forward and backward '
              f'and the update, reference path, one thread')
    # NAME THE TENSOR. A verdict of 12/15 with nothing else said is close to
    # useless: it cannot tell a wrong gradient from a wrong hyperparameter.
    for row in mismatches[:10]:
        print(f'gate: mismatch {row}')
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
    parser.add_argument('--expect-mismatch', action='store_true',
                        help='cpu mode only: invert the verdict, so a sabotage build '
                             'that still agrees is the failure')
    parser.add_argument('--require-vendors', type=int, default=1, metavar='N',
                        help='refuse unless at least N vendor trees are present (default 1). '
                             'Pass 3 where a cross-vendor comparison is the point; CI holds '
                             'only the Apple tree, the other two are about 308 MB')
    parser.add_argument('--shape', default=None,
                        help='DEVIATION 2682: assert the trees are this shape, as '
                             'batch,length or nine dimensions. The shape is read '
                             'from the capture either way; this is how a caller '
                             'says which one it meant to be reading')
    parser.add_argument('--tree', action='append', metavar='VENDOR=PATH',
                        help='read this vendor tree from an explicit path, for a '
                             'capture that has no settled home yet')
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
