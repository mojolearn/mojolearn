import json
from pathlib import Path
root = Path('/root/model-pool-out')
checks = []
for shards in (1, 3, 5):
    name = f'pool-{shards}.json'
    h100 = json.loads((root/name).read_text())
    rtx = json.loads((Path('/root/reference-5090')/name).read_text())
    assert h100 == rtx, name
    assert h100['status'] == 'PASS'
    for row in h100['checks']:
        checks.append(dict(layers=row['layers'], logical_shards=shards,
                           state_sha256=row['state_sha256'],
                           gradient_sha256=row['gradient_sha256'],
                           loss_sha256=row['loss_sha256']))
assert len(checks) == 9
result = dict(status='PASS', checks=checks,
              scope='Exact complete receipts: two H100s and two RTX 5090s, nine layer-owned byte-LM configurations. Both are NVIDIA architectures.')
(root/'cross-architecture.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS all nine complete H100/RTX5090 model-pool receipts, including state, gradient and loss hashes')
