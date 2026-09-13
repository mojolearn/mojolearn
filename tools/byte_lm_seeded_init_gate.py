#!/usr/bin/env python3
"""The recorded initialization IS a function of a seed, on all three vendors.

WHAT THIS CLOSES. Every byte LM training result reads its starting parameters
out of a capture, so the honest statement has been "given these starting bytes,
this is where training ends". A reader can fairly answer that the run was handed
its initialization. This gate shows the starting bytes were never arbitrary: they
are `fmix32((i + 1) ^ 0x42595445) >> 24`, centered at 128 and divided by 1024,
with the RMS norm vectors set to exactly 1.0. So a seed determines the initial
weights, the initial weights determine the trained weights
(`tools/byte_lm_cpu_train_freerun.py`), and the claim becomes end to end:

    a seed, a corpus and a config determine the trained model's bits,
    on a CPU and on every vendor.

No new capture and no rented GPU is needed for this. The three retained trees
already record the initialization, and they record the SAME one.

WHY THIS GATE DOES NOT IMPORT THE GENERATOR. `tools/byte_lm_real_text_capture.py`
has an `initialize()` that produced these bytes. Comparing that function to its
own output would prove nothing at all. The arithmetic below is an INDEPENDENT
reimplementation, written from the recorded identifier

    u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1

and it is checked against the recorded bytes, against the
`initial_parameters_sha256` each tree pins, and against that identifier string.
Three ways to disagree, and a difference is reported by tensor.

WHY IT IS BIT-EXACT ON ANY MACHINE, rather than measured to be. The draw is
UInt32 integer arithmetic with one float operation: an 8-bit integer divided by
2^-10. Both are exact in FP32 and the quotient is exact, so there is nothing to
round differently anywhere. No accumulator, so no fold order. No transcendental,
which is what would actually break cross-vendor agreement.

Exit 0: every tree reproduced. 1: any difference. 2: the gate could not run.
"""
import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    '_byte_lm_shape', Path(__file__).with_name('byte_lm_shape.py'))
byte_lm_shape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(byte_lm_shape)

_gspec = importlib.util.spec_from_file_location(
    '_byte_lm_cpu_train_gate', Path(__file__).with_name('byte_lm_cpu_train_gate.py'))
gate = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(gate)

#: The identifier the captures record and this file implements.
INIT_ID = 'u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1'
SEED_XOR = 0x42595445
CENTER = 128
DIVISOR = 1024.0
MASK = 0xFFFFFFFF


def fmix32(x):
    """Murmur3's 32-bit finalizer, masked at every step so this is exact in
    Python's unbounded integers."""
    x &= MASK
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & MASK
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & MASK
    x ^= x >> 16
    return x


def draw(index, seed_xor=SEED_XOR):
    """One drawn value, before the norm overwrite. Exact: an 8-bit integer over
    a power of two."""
    return ((fmix32((index + 1) ^ seed_xor) >> 24) - CENTER) / DIVISOR


def regenerate(shape, seed_xor=SEED_XOR):
    """The flat FP32 parameter vector at step 0, as little-endian bytes.

    Rounded through struct so the comparison is on FP32 bits, which is what
    equality means here, not on Python floats."""
    values = [draw(i, seed_xor) for i in range(shape.n_total)]
    for entry in shape.registry():
        if entry['name'].endswith(('norm1_w', 'norm2_w')):
            for k in range(entry['offset'], entry['offset'] + entry['count']):
                values[k] = 1.0
    return struct.pack('<%df' % shape.n_total, *values)


def locate(got, want, shape):
    """The first differing element, named by tensor, with both bit patterns."""
    for index in range(len(want) // 4):
        lo = index * 4
        if got[lo:lo + 4] == want[lo:lo + 4]:
            continue
        row = dict(element=index,
                   got_bits='%08x' % struct.unpack('<I', got[lo:lo + 4])[0],
                   want_bits='%08x' % struct.unpack('<I', want[lo:lo + 4])[0],
                   got=struct.unpack('<f', got[lo:lo + 4])[0],
                   want=struct.unpack('<f', want[lo:lo + 4])[0])
        for entry in shape.registry():
            if entry['offset'] <= index < entry['offset'] + entry['count']:
                row.update(tensor=entry['name'], shape=entry['shape'],
                           index_in_tensor=index - entry['offset'])
                break
        return row
    return None


def summary_of(tree):
    """The capture's own schedule block, which records the identifier and the
    digest this gate pins against. A tree without one is refused rather than
    compared against a default."""
    path = tree.parent / 'summary.json' if not (tree / 'summary.json').exists() \
        else tree / 'summary.json'
    if not path.exists():
        raise ValueError(f'no summary.json beside {tree}')
    return json.loads(path.read_text()).get('schedule', {})


def check_tree(name, tree, seed_xor, verify_digests):
    """One vendor tree: identifier, digest and every byte of step 1's initial_p."""
    shape = gate.shape_of(tree)
    gate.use_shape(shape)
    sched = summary_of(tree)
    rows = []

    recorded_id = sched.get('initialization')
    if recorded_id != INIT_ID:
        rows.append(dict(kind='identifier', got=INIT_ID, want=recorded_id))

    produced = regenerate(shape, seed_xor)
    digest = hashlib.sha256(produced).hexdigest()
    recorded_digest = sched.get('initial_parameters_sha256')
    if recorded_digest is not None and digest != recorded_digest:
        rows.append(dict(kind='digest', got=digest, want=recorded_digest))

    desc = gate.descriptors(tree, 1) if verify_digests else {}
    want = gate.array(tree, 1, 'initial_p', desc.get('initial_p'))
    if produced != want:
        row = locate(produced, want, shape) or dict(kind='length')
        row['kind'] = 'bytes'
        rows.append(row)

    return dict(vendor=name, profile=shape.profile, n_total=shape.n_total,
                recorded_identifier=recorded_id, digest=digest,
                recorded_digest=recorded_digest,
                distinct_values=len(set(struct.unpack('<%df' % shape.n_total, produced))),
                mismatches=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seed-xor', type=lambda s: int(s, 0), default=SEED_XOR,
                        help='the XOR constant; the default is what the captures '
                             'recorded. Change it and this gate must FAIL')
    parser.add_argument('--verify-digests', action='store_true',
                        help="also check the capture array against its capture.json digest")
    parser.add_argument('--expect-mismatch', action='store_true',
                        help='invert the verdict, so a wrong seed that still '
                             'reproduces the bytes is the failure')
    parser.add_argument('--require-vendors', type=int, default=1, metavar='N')
    parser.add_argument('--tree', action='append', metavar='VENDOR=PATH')
    parser.add_argument('--report', help='write the JSON report here (must not exist)')
    args = parser.parse_args(argv)

    try:
        trees = gate.present(argparse.Namespace(tree=args.tree))
    except ValueError as exc:
        print(f'gate: could not run: {exc}', file=sys.stderr)
        return 2
    names = sorted(trees)
    if len(names) < args.require_vendors:
        print(f'gate: {len(names)} vendor tree(s) present {names}, '
              f'--require-vendors {args.require_vendors} demands more', file=sys.stderr)
        return 2
    if not names:
        print('gate: no vendor tree present', file=sys.stderr)
        return 2

    results, bad = [], 0
    for name in names:
        try:
            row = check_tree(name, trees[name], args.seed_xor, args.verify_digests)
        except ValueError as exc:
            print(f'gate: could not run on {name}: {exc}', file=sys.stderr)
            return 2
        results.append(row)
        bad += len(row['mismatches'])

    agreed = not bad
    verdict = ('PASS' if agreed else 'FAIL') if not args.expect_mismatch \
        else ('PASS' if bad else 'FAIL')

    report = dict(schema='mojolearn.byte-lm-seeded-init-gate.v1', deviation=2680,
                  init_id=INIT_ID, seed_xor=hex(args.seed_xor),
                  vendors=names, trees=results, mismatched=bad,
                  expect_mismatch=args.expect_mismatch, verdict=verdict)

    print(f'gate: {verdict}: the recorded initialization is reproduced from the seed '
          f'on {len(names)} vendor tree(s) {" ".join(names)}'
          f'{" (wrong seed, a mismatch was required)" if args.expect_mismatch else ""}')
    for row in results:
        state = 'reproduced' if not row['mismatches'] else 'DIFFERS'
        print(f"gate:   {row['vendor']:6s} {state}: {row['n_total']} values, "
              f"{row['distinct_values']} distinct, sha256 {row['digest'][:16]} "
              f"vs recorded {str(row['recorded_digest'])[:16]}")
        for m in row['mismatches'][:4]:
            print(f'gate:     mismatch {m}')
    if agreed and len(names) >= 2:
        print(f'gate: all {len(names)} trees record the same identifier and the same '
              'initial digest, so the seed statement is cross-vendor')

    if args.report:
        with open(args.report, 'x') as stream:
            stream.write(json.dumps(report, indent=1, sort_keys=True) + '\n')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
