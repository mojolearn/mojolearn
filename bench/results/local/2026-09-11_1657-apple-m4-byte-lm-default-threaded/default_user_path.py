"""The default LanguageModelInference (threaded path) on the real checkpoint, and id refusals on the real binding."""
import hashlib, json, math, struct, sys
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / 'python'))
from mojolearn import LanguageModelInference
from mojolearn._buffer import frombytes
from mojolearn._bufcheck import le_bytes
CAP = ROOT / 'bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple/full128'
out = {}
model = LanguageModelInference.from_checkpoint(CAP / 'final.checkpoint.json')
out['default_threaded'] = model._threaded
out['default_threads'] = model._threads
out['params_equal_step128'] = model.parameters_sha256() == hashlib.sha256((CAP / 'step000128/post_p.f32').read_bytes()).hexdigest()
evaluation = json.loads((CAP / 'heldout-final/evaluation.json').read_text())
equal_default = equal_reference = 0
losses = []
for i in range(len(evaluation['batches'])):
    ids = frombytes((CAP / f'heldout-final/batch{i:02d}.ids.i32').read_bytes(), '<i4', (2, 33))
    want = struct.unpack('<I', (CAP / f'heldout-final/batch{i:02d}.loss.f32').read_bytes())[0]
    equal_default += model.loss_bits(ids) == want
    equal_reference += model.loss_bits(ids, threaded=False) == want
    losses.append(model.loss(ids))
out['heldout_loss_bytes_equal'] = dict(default=equal_default, reference=equal_reference, of=len(evaluation['batches']))
out['heldout_mean_equals_record'] = math.fsum(losses) / len(losses) == evaluation['mean_loss']
flat = struct.unpack('<66i', (CAP / 'heldout-final/batch00.ids.i32').read_bytes())
rows = frombytes(struct.pack('<64i', *[v for r in range(2) for v in flat[r * 33: r * 33 + 32]]), '<i4', (2, 32))
probe = '2e2408f47392b4a1b69a791cf4672cdf6e49fcc0f576e9bcd6b7018f0ffaa6c4'
out['logits_hash_default'] = hashlib.sha256(le_bytes(model.logits(rows), 'f')).hexdigest() == probe
out['logits_hash_reference'] = hashlib.sha256(le_bytes(model.logits(rows, threaded=False), 'f')).hexdigest() == probe
refusals = {}
def refused(label, call):
    try:
        call(); refusals[label] = 'NOT REFUSED'
    except Exception as exc:
        refusals[label] = type(exc).__name__
refused('logits token 300', lambda: model.logits(frombytes(struct.pack('<2i', 1, 300), '<i4', (1, 2))))
refused('logits token -1', lambda: model.logits(frombytes(struct.pack('<i', -1), '<i4', (1, 1))))
bad = list(flat); bad[33 + 7] = 256
refused('loss input 256', lambda: model.loss_bits(frombytes(struct.pack('<66i', *bad), '<i4', (2, 33))))
bad = list(flat); bad[32] = -1
refused('loss target -1', lambda: model.loss_bits(frombytes(struct.pack('<66i', *bad), '<i4', (2, 33))))
out['refusals'] = refusals
ign = list(flat); ign[65] = -100
ign_ids = frombytes(struct.pack('<66i', *ign), '<i4', (2, 33))
a = model.loss_bits(ign_ids); b = model.loss_bits(ign_ids, threaded=False)
out['ignore_index_last_column'] = dict(default=f'{a:08x}', reference=f'{b:08x}', equal=a == b,
                                       differs_from_unignored=a != model.loss_bits(frombytes(struct.pack('<66i', *flat), '<i4', (2, 33))))
out['PASS'] = (out['default_threaded'] is True and out['params_equal_step128']
               and equal_default == equal_reference == 8 and out['heldout_mean_equals_record']
               and out['logits_hash_default'] and out['logits_hash_reference']
               and all(v == 'ValueError' for v in refusals.values())
               and out['ignore_index_last_column']['equal'] and out['ignore_index_last_column']['differs_from_unignored'])
print(json.dumps(out, indent=1))
sys.exit(0 if out['PASS'] else 1)
