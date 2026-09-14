#!/bin/sh
set -eu
while [ ! -f /root/jobs/model-capacity-same-driver.done ]; do sleep 10; done
test "$(cat /root/jobs/model-capacity-same-driver.rc)" = 0
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=cvisjmryfcmz6g MOJOLEARN_NUMERIC_MODE=identical
cd /root/gradient-pool
export PYTHONPATH="$PWD/python"
mkdir -p /root/model-pool-final
for shards in 1 3 5; do
    pixi run python tools/byte_lm_model_pool_check.py --cloud --logical-shards "$shards" --corpus training/corpus/enwik8/input.txt --report /root/model-pool-final/pool-${shards}.json > /root/model-pool-final/pool-${shards}.log 2>&1
done
cp /root/model-pool-v2-out/fault/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
pixi run python tools/byte_lm_model_pool_check.py --cloud --faults --corpus training/corpus/enwik8/input.txt --report /root/model-pool-final/fault.json > /root/model-pool-final/fault.log 2>&1
cp /root/model-pool-v2-out/production/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
python3 - <<'PY' > /root/model-pool-final/regression-comparison.log
import json
from pathlib import Path
new=json.loads(Path('/root/model-pool-v2-out/byte-lm.json').read_text())
old=json.loads(Path('/root/replay-out/byte-lm.json').read_text())
assert new == old
p=json.loads(Path('/root/model-pool-final/pool-3.json').read_text())['checks']
f=json.loads(Path('/root/model-pool-final/fault.json').read_text())['checks']
for a,b in zip(p,f,strict=True):
    for key in ('state_sha256','gradient_sha256','loss_sha256'):
        assert a[key] == b[key], key
print('PASS complete original byte-LM receipt unchanged; fault recovery retains all model-pool state, gradient and loss hashes')
PY
