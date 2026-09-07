from pathlib import Path
import hashlib
import json
from byte_lm_validation_admit import admit
from byte_lm_state_compare import load_capture, compare_continuous, receipt, sha, COUNTS

root = Path('/Users/andrewhendel/CascadeProjects/mojolearn')
base = root / 'bench/results/resume'
out = base / '2026-09-07-root-byte-lm-three-vendor'
paths = {
    'cuda': base / '2026-09-07-root-byte-lm-nvidia-common/run2/remote/byte-lm-validation',
    'hip': base / '2026-09-07-root-byte-lm-do-amd/run6/remote/byte-lm-do-output/byte-lm-validation',
}
admissions = {vendor: admit(path) for vendor, path in paths.items()}
captures = {vendor: load_capture(path / 'full128', 'continuous', vendor) for vendor, path in paths.items()}
captures['metal'] = load_capture(out / 'apple/full128', 'continuous', 'metal')
metal_receipt = receipt(out / 'apple/full128.root-receipt.json', captures['metal']['summary_sha256'], 'metal')
expected = json.loads((base / '2026-09-07-root-metal-preparation/common-source-expanded.json').read_text())['files']
assert all(c['source'] == expected for c in captures.values())
assert compare_continuous(captures['cuda'], captures['hip']) == 128
assert compare_continuous(captures['cuda'], captures['metal']) == 128
assert all(c['ratio'] <= .9 for c in captures.values())
result = dict(
    schema='mojolearn.byte-lm.continuous-admission.v1',
    status='QUALIFIED_BOUNDED_CONTINUOUS', identity_admitted=True, learning_admitted=True,
    resume_admitted=False, metal_admitted=True, continuous_vendors=['cuda', 'hip', 'metal'],
    source_files=len(expected), compared_steps=128, raw_32bit_cells_per_step=sum(COUNTS.values()),
    checkpoint_sha256=sha(captures['cuda']['checkpoint']),
    initial_heldout_loss=captures['metal']['summary']['initial_heldout']['mean_loss'],
    final_heldout_loss=captures['metal']['summary']['final_heldout']['mean_loss'],
    summary_sha256={vendor: capture['summary_sha256'] for vendor, capture in captures.items()},
    runtimes={vendor: capture['runtime'] for vendor, capture in captures.items()},
    linux_admissions=admissions, metal_guard=metal_receipt,
    independent_gradient_oracle_vendors=['cuda', 'hip'],
    scope='All128 continuous FP32 training states, inputs, heldout bytes and final checkpoint match across three vendors for this fixed two-block 34944-parameter model. No Metal resume, independent Metal FP64 oracle, larger-model or benchmark claim. Metal used the explicitly user-authorized no-free-reserve guard policy; other recorded limits remained active.')
raw = json.dumps(result, sort_keys=True, indent=2) + '\n'
with (out / 'comparison.json').open('x') as stream:
    stream.write(raw)
print(json.dumps({k: result[k] for k in ('status', 'continuous_vendors', 'compared_steps', 'initial_heldout_loss', 'final_heldout_loss', 'checkpoint_sha256')}))
print('comparison_sha256', hashlib.sha256(raw.encode()).hexdigest())
