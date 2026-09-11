"""End-to-end user path for byte LM CPU inference on this machine.

Loads the real checkpoint file through LanguageModelInference.from_checkpoint
(the path a user takes, which the gate never exercised), and checks it against
the retained three-vendor capture: the parameters are the step-128 state, every
held-out loss's bytes match on both paths, their math.fsum mean equals the
recorded evaluation mean, the logits hash matches every certified CPU, and bad
input is refused. Prints a JSON summary.
"""
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / 'python'))

import mojolearn  # noqa: E402
from mojolearn import LanguageModelInference, _backend  # noqa: E402
from mojolearn._buffer import frombytes  # noqa: E402
from mojolearn._bufcheck import le_bytes  # noqa: E402

CAP = ROOT / 'bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple/full128'
PROBE = '2e2408f47392b4a1b69a791cf4672cdf6e49fcc0f576e9bcd6b7018f0ffaa6c4'
out = dict(cpu=subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip(),
           machine=platform.machine(), python=platform.python_version(),
           vendor=mojolearn.vendor(), cpu_only_import=_backend._CPU_ONLY is not None)

ref = LanguageModelInference.from_checkpoint(CAP / 'final.checkpoint.json')
thr = LanguageModelInference.from_checkpoint(CAP / 'final.checkpoint.json', threaded=True, threads=3)
out['checkpoint_params_equal_step128_post_p'] = (
    ref.parameters_sha256() == hashlib.sha256((CAP / 'step000128/post_p.f32').read_bytes()).hexdigest())
out['profile'] = ref.profile

evaluation = json.loads((CAP / 'heldout-final/evaluation.json').read_text())
equal_ref = equal_thr = 0
losses = []
for i, _ in enumerate(evaluation['batches']):
    ids = frombytes((CAP / f'heldout-final/batch{i:02d}.ids.i32').read_bytes(), '<i4', (2, 33))
    want = struct.unpack('<I', (CAP / f'heldout-final/batch{i:02d}.loss.f32').read_bytes())[0]
    equal_ref += ref.loss_bits(ids) == want
    equal_thr += thr.loss_bits(ids) == want
    losses.append(ref.loss(ids))
out['heldout_final_loss_bytes_equal'] = dict(reference=equal_ref, threaded=equal_thr, of=len(evaluation['batches']))
out['heldout_final_mean_equals_record'] = math.fsum(losses) / len(losses) == evaluation['mean_loss']
out['heldout_final_mean'] = math.fsum(losses) / len(losses)

flat = struct.unpack('<66i', (CAP / 'heldout-final/batch00.ids.i32').read_bytes())
inputs = [v for r in range(2) for v in flat[r * 33: r * 33 + 32]]
rows = frombytes(struct.pack('<64i', *inputs), '<i4', (2, 32))
h_ref = hashlib.sha256(le_bytes(ref.logits(rows), 'f')).hexdigest()
h_thr = hashlib.sha256(le_bytes(thr.logits(rows), 'f')).hexdigest()
out['logits_hash_matches_certified_cpus'] = dict(reference=h_ref == PROBE, threaded=h_thr == PROBE)
out['next_bytes'] = ref.next_bytes(rows)

refusals = {}
for label, call in (
        ('token_out_of_range', lambda: ref.logits(frombytes(struct.pack('<i', 300), '<i4', (1, 1)))),
        ('length_over_config', lambda: ref.logits(frombytes(struct.pack('<33i', *([0] * 33)), '<i4', (1, 33)))),
        ('wrong_loss_shape', lambda: ref.loss_bits(frombytes(struct.pack('<32i', *([0] * 32)), '<i4', (1, 32))))):
    try:
        call()
        refusals[label] = 'NOT REFUSED'
    except Exception as exc:  # the refusal itself is the check
        refusals[label] = type(exc).__name__
out['refusals'] = refusals

for name, model in (('reference', ref), ('threaded', thr)):
    started = time.perf_counter()
    for _ in range(50):
        model.logits(rows)
    out[f'ms_per_2x32_logits_{name}'] = round((time.perf_counter() - started) / 50 * 1000, 3)

out['PASS'] = (out['checkpoint_params_equal_step128_post_p'] and equal_ref == equal_thr == 8
               and out['heldout_final_mean_equals_record']
               and all(out['logits_hash_matches_certified_cpus'].values())
               and all(v != 'NOT REFUSED' for v in refusals.values()))
print(json.dumps(out, indent=1))
sys.exit(0 if out['PASS'] else 1)
