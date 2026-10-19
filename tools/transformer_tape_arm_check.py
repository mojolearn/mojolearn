#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gradient digests of one decoder-layer backward (the autograd tape),
compared bit for bit against the stored digests in
tools/transformer_tape_arm_digests.json (the same on Metal, NVIDIA and AMD).

Loads the given _mojolearn_transformer binding, builds one fixed problem
(seeded inputs, weights and incoming gradient) and runs the two backward
entries:

  saved    transformer_forward_tape + transformer_backward_tape
           (_transformer_saved_backward); needs a build with the forward
           tape enabled (MOJOLEARN_IDN_ACT_RETAIN=3), otherwise reported as
           skipped
  session  transformer_session_backward (_transformer_run_session_backward)

and prints, per path, one `GRAD <path>.<name> <digest>` line per gradient
buffer and a `LOSS <path> <hex bits>` line. LOSS is the float32 bits of
<y, dy> (the scalar whose gradient this backward is) when the path returns
y, else of <x, d_x>. Digests are blake2b of the raw float32 bytes, 16 hex.
The binding does every computed number of the run; this script only makes
the fixed inputs and digests what comes back.

Every printed digest is then compared with the stored file (--expected):
one `STORED <key> MATCH|DIFFER` line per digest and a closing
`STORED all MATCH|DIFFER <equal>/<total>`; the exit status is 1 on any
DIFFER. A path the build skips is reported as SKIP, not compared.
(The hand arm, -D MOJOLEARN_AUTOGRAD_TRANSFORMER_OFF, was deleted
2026-10-19; it printed the same digests.)

    python3 tools/transformer_tape_arm_check.py \
        --binding python/mojolearn/identical/_mojolearn_transformer.so \
        --backend metal --out tape-arm.json

--time N (default 0, off) then times N repeated backwards per path on a
larger shape (TIME_SHAPE below), one warm-up call excluded:

  TIME <arm> session.forward median_ms=<x> n=<N>   forward alone
  TIME <arm> session median_ms=<x> n=<N>           backward alone
  TIME <arm> saved median_ms=<x> n=<N>             saved-tape backward
  LAUNCHES <arm> <path> <n>                        with MOJOLEARN_AG_PROFILE=1

<arm> is tape, as the binding reports. The binding uploads the inputs
and weights before the first clock and syncs the device on both sides of
every clock (kernel-only); this script only takes the median of the
nanoseconds it returns. LAUNCHES counts the launches that go through
DeviceExec; kernels enqueued on the context directly are not counted.
"""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

# One small block, the head_dim of the existing transformer fixtures (16).
B, L, DM, NH, NKV, HD, IT, WINDOW = 2, 128, 64, 4, 2, 16, 128, 0
# --time shape, chosen by cost, not a board row: M = B*L = 4096 rows makes
# the projections and MLP GEMMs dominate the layer (the regime a training
# step lives in), head_dim 64 with 4:1 grouped KV takes the tiled attention
# route, and intermediate ~2.7 * d_model is the usual SwiGLU ratio. One
# layer's stages fit well inside every box's memory.
TIME_SHAPE = (8, 512, 512, 8, 2, 64, 1376, 0)
STORED = Path(__file__).resolve().with_name('transformer_tape_arm_digests.json')
NAMES = ('d_x', 'dw_norm1', 'dw_norm2', 'dw_q', 'dw_k', 'dw_v', 'dw_o',
         'dw_gate', 'dw_up', 'dw_down')


def digest(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=8).hexdigest()


def bits(v):
    return '%08x' % int(np.float32(v).view(np.uint32))


def problem(b=B, l=L, dm=DM, nh=NH, nkv=NKV, hd=HD, it=IT):
    rng = np.random.default_rng(20261019)
    qw, kw = nh * hd, nkv * hd
    shapes = [(dm,), (dm,), (qw, dm), (kw, dm), (kw, dm), (dm, qw),
              (it, dm), (it, dm), (dm, it)]
    w = [(rng.standard_normal(s) * .1).astype(np.float32) for s in shapes]
    w[0] = (1 + w[0]).astype(np.float32)  # RMSNorm weights near one
    w[1] = (1 + w[1]).astype(np.float32)
    x = rng.uniform(-.5, .5, (b, l, dm)).astype(np.float32)
    dy = rng.uniform(-.5, .5, (b, l, dm)).astype(np.float32)
    return x, w, dy


def median_ms(ns):
    """Median of host-clock nanoseconds (measurement tool, not the runtime)."""
    return statistics.median(ns) / 1e6


def launches_text(n):
    return 'n/a' if n < 0 else str(n)


def timing(mod, n, result):
    """N timed calls per path on TIME_SHAPE after one warm-up (dropped)."""
    x, w, dy = problem(*TIME_SHAPE[:7])
    params = list(TIME_SHAPE)
    grads = grads_like(x, w)
    timed = result['timing'] = dict(shape=params, n=n)
    session = mod.transformer_session_create()
    try:
        addrs = [x.ctypes.data] + [a.ctypes.data for a in w] + [dy.ctypes.data] + [g.ctypes.data for g in grads]
        out = list(mod.transformer_session_backward_time(session, addrs, params + [n]))
    finally:
        mod.transformer_session_close(session)
    assert out[0] == 1, 'binding reports a non-tape backward'
    arm = 'tape'
    fwd, bwd = out[2::2][1:], out[3::2][1:]
    assert len(fwd) == n and len(bwd) == n, 'binding returned %d/%d calls' % (len(fwd), len(bwd))
    timed['arm'] = arm
    timed['session.forward_ms'] = median_ms(fwd)
    timed['session_ms'] = median_ms(bwd)
    timed['session_digest_d_x'] = digest(grads[0])
    print('TIME %s session.forward median_ms=%.3f n=%d' % (arm, timed['session.forward_ms'], n), flush=True)
    print('TIME %s session median_ms=%.3f n=%d' % (arm, timed['session_ms'], n), flush=True)
    if out[1] >= 0:
        print('LAUNCHES %s session %s' % (arm, launches_text(out[1])), flush=True)
    if mod.transformer_forward_tape_enabled():
        y = np.full_like(x, np.nan)
        sgrads = grads_like(x, w)
        addrs = [x.ctypes.data] + [a.ctypes.data for a in w] + [y.ctypes.data] + \
            [dy.ctypes.data] + [g.ctypes.data for g in sgrads]
        out = list(mod.transformer_backward_tape_time(addrs, params + [n]))
        bwd = out[2:][1:]
        assert len(bwd) == n, 'binding returned %d calls' % len(bwd)
        timed['saved_ms'] = median_ms(bwd)
        print('TIME %s saved median_ms=%.3f n=%d' % (arm, timed['saved_ms'], n), flush=True)
        if out[1] >= 0:
            print('LAUNCHES %s saved %s' % (arm, launches_text(out[1])), flush=True)
    else:
        print('SKIP time saved forward tape disabled in this build', flush=True)


def grads_like(x, w):
    return [np.full_like(x, np.nan)] + [np.full_like(a, np.nan) for a in w]


def record(result, path, grads, loss):
    for name, g in zip(NAMES, grads):
        assert np.isfinite(g).all(), (path, name, 'non-finite or unwritten gradient')
        d = digest(g)
        result['grads'][path + '.' + name] = d
        print('GRAD %s.%s %s' % (path, name, d), flush=True)
    result['loss'][path] = loss
    print('LOSS %s %s' % (path, loss), flush=True)


def compare_stored(result, path):
    """Equal-string comparison of every digest this run printed with the
    stored file; returns True when all are equal."""
    stored = json.loads(path.read_text())
    if stored['shape'] != result['shape']:
        print('STORED shape %s differs from the stored %s DIFFER' % (result['shape'], stored['shape']), flush=True)
        return False
    got = [('GRAD', k, v, stored['grads'].get(k)) for k, v in result['grads'].items()] + \
          [('LOSS', k, v, stored['loss'].get(k)) for k, v in result['loss'].items()]
    equal = 0
    for kind, key, value, want in got:
        same = value == want
        equal += same
        print('STORED %s %s %s%s' % (kind, key, 'MATCH' if same else 'DIFFER',
                                     '' if same else ' got=%s stored=%s' % (value, want)), flush=True)
    ok = equal == len(got) and len(got) > 0
    print('STORED all %s %d/%d' % ('MATCH' if ok else 'DIFFER', equal, len(got)), flush=True)
    result['stored'] = dict(file=str(path), equal=equal, total=len(got), match=ok)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--binding', type=Path, required=True)
    ap.add_argument('--backend', choices=('metal', 'cuda', 'hip'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--expected', type=Path, default=STORED,
                    help='stored digests to compare with (default: %(default)s)')
    ap.add_argument('--time', type=int, default=0, metavar='N',
                    help='after the digests, time N backwards per path on TIME_SHAPE (0: off)')
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location(args.binding.stem, args.binding.resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vendor = mod.transformer_vendor()
    if vendor != args.backend:
        ap.error('binding vendor is %s, not %s' % (vendor, args.backend))
    if mod.transformer_numeric_mode() != 1:
        ap.error('binding is not an IDENTICAL build')
    x, w, dy = problem()
    params = [B, L, DM, NH, NKV, HD, IT, WINDOW]
    result = dict(binding=str(args.binding),
                  sha256=hashlib.sha256(args.binding.read_bytes()).hexdigest(),
                  backend=vendor, shape=params, grads={}, loss={}, skipped=[])
    print('BINDING %s %s' % (args.binding, result['sha256'][:16]), flush=True)

    # saved: the forward keeps its stages on a tape; the backward reads them
    if mod.transformer_forward_tape_enabled():
        y = np.full_like(x, np.nan)
        tape = mod.transformer_forward_tape([a.ctypes.data for a in [x] + w + [y]], params)
        grads = grads_like(x, w)
        mod.transformer_backward_tape(tape, [dy.ctypes.data] + [g.ctypes.data for g in grads])
        assert np.isfinite(y).all(), 'forward output not written'
        result['grads']['saved.y'] = digest(y)
        print('FWD saved.y %s' % digest(y), flush=True)
        record(result, 'saved', grads, bits(np.dot(y.ravel().astype(np.float64), dy.ravel())))
    else:
        result['skipped'].append('saved: forward tape disabled in this build (MOJOLEARN_IDN_ACT_RETAIN=3)')
        print('SKIP saved forward tape disabled in this build', flush=True)

    # session: forward from a zero cache and backward in one call
    session = mod.transformer_session_create()
    try:
        grads = grads_like(x, w)
        addrs = [x.ctypes.data] + [a.ctypes.data for a in w] + [dy.ctypes.data] + [g.ctypes.data for g in grads]
        mod.transformer_session_backward(session, addrs, params)
    finally:
        mod.transformer_session_close(session)
    record(result, 'session', grads, bits(np.dot(x.ravel().astype(np.float64), grads[0].ravel())))

    if 'saved' in result['loss']:
        same = all(result['grads']['saved.' + n] == result['grads']['session.' + n] for n in NAMES)
        result['saved_equals_session'] = same
        print('PATHS saved session %s' % ('MATCH' if same else 'DIFFER'), flush=True)
    ok = compare_stored(result, args.expected)
    if args.time > 0:
        timing(mod, args.time, result)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
