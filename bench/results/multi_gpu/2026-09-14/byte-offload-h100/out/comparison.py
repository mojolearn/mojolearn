import json
from pathlib import Path
root=Path('/root/offload-out')
for logical in (1,3,5):
    a=json.loads((root/f'public-{logical}.json').read_text())['checks']
    b=json.loads((root/'reference-5090'/f'pool-{logical}.json').read_text())['checks']
    for left,right in zip(a,b,strict=True):
        for key in ('layers','logical_shards','parameters','state_sha256','gradient_sha256','loss_sha256'):
            assert left[key] == right[key], (logical,key)
p=json.loads((root/'public-3.json').read_text())['checks']
f=json.loads((root/'fault.json').read_text())['checks']
for left,right in zip(p,f,strict=True):
    for key in ('state_sha256','gradient_sha256','loss_sha256'):
        assert left[key] == right[key], key
assert json.loads((root/'original-byte.json').read_text()) == json.loads((root/'reference-5090'/'original-byte.json').read_text())
a=json.loads((root/'capacity-pooled.json').read_text())
b=json.loads((root/'capacity-offloaded.json').read_text())
initial=json.loads((root/'capacity-pooled-thread-sampler.json').read_text())
assert initial['receipts'] == a['receipts']
for key in ('status','parameters','shape','logical_shards','corpus_sha256','receipts'):
    assert a[key] == b[key], key
report=dict(status='PASS',small_receipt_groups=9,fault_receipt_groups=3,
    capacity_parameters=a['parameters'],capacity_steps=len(a['receipts']),logical_shards=a['logical_shards'],
    pooled_sampled_peak_mib=a['sampled_peak_mib'],offloaded_sampled_peak_mib=b['sampled_peak_mib'],
    original_byte_receipt_unchanged=True,
    scope='H100 offload versus H100 two-GPU pool and earlier RTX5090 pooled small-model receipts. Offload itself has not run on RTX5090 or AMD/Apple.')
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
