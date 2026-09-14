#!/bin/sh
set -eu
while [ ! -f /root/jobs/replay-classical.done ]; do sleep 5; done
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=cvisjmryfcmz6g MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_120 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/gradient-pool
export PYTHONPATH="$PWD/python"
export MOJOLEARN_BYTE_LM_OUTDIR=/root/gradient-pool-out/production
sh bindings/build_byte_lm.sh > /root/gradient-pool-out/build.log 2>&1
cp "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" > /root/gradient-pool-out/production.sha256
for shards in 2 3 5 8; do
    pixi run python tools/byte_lm_optimizer_pool_check.py --cloud --logical-shards "$shards" --corpus training/corpus/enwik8/input.txt --report /root/gradient-pool-out/pool-${shards}.json > /root/gradient-pool-out/pool-${shards}.log 2>&1
done
pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/gradient-pool-out/byte-lm.json > /root/gradient-pool-out/byte-lm.log 2>&1
python3 - <<'PY' > /root/gradient-pool-out/previous-comparison.log
import json
from pathlib import Path
root=Path('/root/gradient-pool-out'); old=Path('/root/replay-out')
a=json.loads((root/'byte-lm.json').read_text()); b=json.loads((old/'byte-lm.json').read_text())
assert a == b
new=json.loads((root/'pool-3.json').read_text())['checks']
previous=json.loads((old/'byte-pool.json').read_text())['checks']
assert [r['state_sha256'] for r in new] == [r['state_sha256'] for r in previous]
print('PASS complete existing byte-LM receipt and pooled final states unchanged')
PY
export MOJOLEARN_BYTE_LM_OUTDIR=/root/gradient-pool-out/fault
export MOJOLEARN_BUILD_EXTRA_DEFINES='-D MOJOLEARN_BYTE_POOL_FAULT_INJECT=1'
sh bindings/build_byte_lm.sh > /root/gradient-pool-out/build-fault.log 2>&1
cp "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" > /root/gradient-pool-out/fault.sha256
pixi run python tools/byte_lm_optimizer_pool_check.py --cloud --faults --corpus training/corpus/enwik8/input.txt --report /root/gradient-pool-out/fault.json > /root/gradient-pool-out/fault.log 2>&1
cp /root/gradient-pool-out/production/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
