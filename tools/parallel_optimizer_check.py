#!/usr/bin/env python3
"""RunPod-only bit checks for sharded SGD/Adam/AdamW and whole-registry clip."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn._training_impl import _load
    binding = _load('identical')
    assert binding.optimizer_parallel_available() == 1
    rng = np.random.default_rng(719236)
    checks = []

    def initial(sizes):
        offsets = np.asarray([0] + list(np.cumsum(sizes)), dtype='<i4')
        n = int(offsets[-1])
        p = rng.normal(0, .2, n).astype('<f4')
        g = rng.normal(0, .1, n).astype('<f4')
        m = rng.normal(0, .02, n).astype('<f4')
        v = rng.uniform(.0001, .001, n).astype('<f4')
        # Copy-only transports must retain signed zero and subnormal payloads.
        g.view('<u4')[0] = 0x80000000
        if n > 2:
            g.view('<u4')[1] = 1
        flags = (np.arange(len(sizes)) % 2).astype('<i4')
        return [p, g, m, v, offsets, flags, np.zeros(3, dtype='<f4')]

    def invoke(state, count, params):
        os.environ['MOJOLEARN_OPTIMIZER_DEVICE_COUNT'] = str(count)
        return binding.optimizer_step(*(int(a.ctypes.data) for a in state), params)

    for sizes in ([1], [3, 11, 7, 80], [129, 3, 513, 17, 65], [1025, 7, 3]):
        for kind, momentum, dampening, nesterov in (
                (0, 0., 0., 0), (0, .9, .2, 0), (0, .9, 0., 1),
                (1, 0., 0., 0), (2, 0., 0., 0)):
            for clip in (0., .01, 100.):
                left = initial(sizes)
                right = [a.copy() for a in left]
                for step in (1, 2, 7):
                    params = [len(sizes), kind, step, nesterov, .003,
                              .8, .95, 1e-8, .07, momentum, dampening, clip]
                    assert invoke(left, 1, params) == int(left[4][-1])
                    assert invoke(right, 2, params) == int(right[4][-1])
                    for name, a, b in zip(('p','g','m','v','offsets','flags','clip_info'), left, right):
                        assert a.tobytes() == b.tobytes(), (sizes,kind,clip,step,name)
                checks.append(dict(sizes=sizes, kind=kind, momentum=momentum,
                    dampening=dampening, nesterov=nesterov, clip=clip,
                    sha256=hashlib.sha256(b''.join(a.tobytes() for a in right)).hexdigest()))
    # Worker failures must not publish already-computed sibling ranges or a
    # staged clipped gradient. This probes each scanned input in the last shard.
    refusals = []
    params = [4, 2, 1, 0, .003, .8, .95, 1e-8, .07, 0., 0., .01]
    for slot in range(4):
        state = initial([3, 11, 7, 80])
        state[slot].view('<u4')[-1] = 0x7fc12345
        before = [a.tobytes() for a in state]
        try:
            invoke(state, 2, params)
        except Exception:
            pass
        else:
            raise AssertionError('nonfinite shard accepted: ' + str(slot))
        assert [a.tobytes() for a in state] == before
        refusals.append('nonfinite_' + str(slot))
    for count in (0, 65, 3):
        state = initial([3, 11, 7, 80])
        before = [a.tobytes() for a in state]
        try:
            invoke(state, count, params)
        except Exception:
            pass
        else:
            raise AssertionError('invalid device count accepted on two-GPU gate: ' + str(count))
        assert [a.tobytes() for a in state] == before
        refusals.append('devices_' + str(count))
    os.environ.pop('MOJOLEARN_OPTIMIZER_DEVICE_COUNT', None)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, refusals=refusals,
        scope='Two H100s: all SGD/Adam/AdamW outputs, global clip, split momentum flags and atomic refusal. Host staged optimizer ranges; no full-model capacity claim.'), indent=2) + '\n')
    print('PASS', len(checks), 'optimizer configurations; 3 steps each;', len(refusals), 'atomic refusals')


if __name__ == '__main__':
    main()
