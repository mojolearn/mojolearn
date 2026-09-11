"""Timing and bit-equality harness for the byte LM CPU binding.

usage: cpu_bench.py WORKTREE BINARY LABEL [--threaded] [--reps N]

Loads step-128 parameters from the retained capture, times logits at several
shapes and the [2,33] loss, and records a SHA-256 of every logits output so
two binaries can be compared at every shape, not only at the gate's.
Prints one JSON line.
"""
import hashlib
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
os.environ['MOJOLEARN_BYTE_LM_HOST_BINARY'] = sys.argv[2]
label = sys.argv[3]
threaded = '--threaded' in sys.argv
reps = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 20
# Never the library default (one thread per core) on the shared Mac: 3 at most.
threads = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else 3
assert 1 <= threads <= 3, 'this harness runs at most 3 threads'
sys.path.insert(0, str(root / 'python'))

from mojolearn import LanguageModelInference  # noqa: E402
from mojolearn._buffer import frombytes  # noqa: E402
from mojolearn._bufcheck import le_bytes  # noqa: E402

cap = root / 'bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple/full128'
model = LanguageModelInference.from_checkpoint(cap / 'final.checkpoint.json')

# Deterministic token streams from the held-out and training captures.
stream = []
for i in range(8):
    stream += struct.unpack('<66i', (cap / f'heldout-final/batch{i:02d}.ids.i32').read_bytes())
while len(stream) < 64 * 32:
    stream = stream + stream


def ids(batch, length):
    flat = stream[:batch * length]
    return frombytes(struct.pack(f'<{len(flat)}i', *flat), '<i4', (batch, length))


def timed(fn, n):
    fn()
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples) * 1000


out = dict(label=label, threaded=threaded, reps=reps)
for batch, length in ((1, 1), (1, 32), (2, 32), (8, 32), (32, 32)):
    x = ids(batch, length)
    n = max(3, reps // max(1, batch // 2))
    out[f'ms_{batch}x{length}'] = round(timed(lambda: model.logits(x, threaded=threaded, threads=threads), n), 3)
    out[f'sha_{batch}x{length}'] = hashlib.sha256(le_bytes(model.logits(x, threaded=threaded, threads=threads), 'f')).hexdigest()[:16]
loss_ids = frombytes((cap / 'heldout-final/batch00.ids.i32').read_bytes(), '<i4', (2, 33))
out['ms_loss_2x33'] = round(timed(lambda: model.loss_bits(loss_ids, threaded=threaded, threads=threads), reps), 3)
out['loss_bits'] = f'{model.loss_bits(loss_ids, threaded=threaded, threads=threads):08x}'
print(json.dumps(out))
