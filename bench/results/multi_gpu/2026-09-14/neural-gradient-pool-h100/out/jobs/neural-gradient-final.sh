#!/bin/sh
set -eu
while [ ! -f /root/jobs/neural-gradient-build.done ]; do sleep 5; done
test "$(cat /root/jobs/neural-gradient-build.rc)" = 0
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
tar xzf /root/neural-gradient-final.tgz
cp /root/neural-gradient-final.tgz /root/neural-gradient-out/source-final.tgz
sha256sum /root/neural-gradient-final.tgz > /root/neural-gradient-out/source-final.sha256
mkdir -p /root/neural-gradient-out/final /root/neural-gradient-production
export PYTHONPATH="$PWD/python"
sh bindings/build_training.sh > /root/neural-gradient-out/final/build-training.log 2>&1
cp python/mojolearn/identical/_mojolearn_training.so /root/neural-gradient-production/
sha256sum python/mojolearn/identical/*.so > /root/neural-gradient-out/final/binaries.sha256
pixi run python tools/parallel_accumulate_check.py --cloud --report /root/neural-gradient-out/final/accumulate.json > /root/neural-gradient-out/final/accumulate.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/neural-gradient-out/final/mlp.json > /root/neural-gradient-out/final/mlp.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --corpus training/corpus/enwik8/input.txt --report /root/neural-gradient-out/final/samba.json > /root/neural-gradient-out/final/samba.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/neural-gradient-out/final/samba-clipped.json > /root/neural-gradient-out/final/samba-clipped.log 2>&1
python3 - <<'PY'
from pathlib import Path
p=Path('bindings/build_training.sh')
s=p.read_text().replace('MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1"','MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_ACCUMULATE_POOL_FAULT=1"')
Path('bindings/build_training_fault.sh').write_text(s)
Path('/root/neural-gradient-out/final/build-training-fault.sh').write_text(s)
PY
sh bindings/build_training_fault.sh > /root/neural-gradient-out/final/build-fault.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_training.so > /root/neural-gradient-out/final/fault.sha256
pixi run python tools/parallel_accumulate_check.py --cloud --faults --report /root/neural-gradient-out/final/fault.json > /root/neural-gradient-out/final/fault.log 2>&1
cp /root/neural-gradient-production/_mojolearn_training.so python/mojolearn/identical/_mojolearn_training.so
for devices in 1 2; do
    pixi run python tools/parallel_accumulate_capacity_check.py --cloud --devices "$devices" --report /root/neural-gradient-out/final/capacity-${devices}.json > /root/neural-gradient-out/final/capacity-${devices}.log 2>&1
done
python3 - <<'PY' > /root/neural-gradient-out/final/comparison.log
import json
from pathlib import Path
r=Path('/root/neural-gradient-out')
for lane in ('mlp','samba','samba-clipped'):
    assert json.loads((r/(lane+'.json')).read_text()) == json.loads((r/'final'/(lane+'.json')).read_text())
p=json.loads((r/'final/accumulate.json').read_text())
f=json.loads((r/'final/fault.json').read_text())
assert p['checks'] == f['checks']
a=json.loads((r/'final/capacity-1.json').read_text())
b=json.loads((r/'final/capacity-2.json').read_text())
assert a['status'] == 'REFUSED' and a['output_unchanged']
assert b['status'] == 'PASS' and b['verified_cells'] == b['n']
print('PASS final neural state receipts unchanged, post-compute fault recovery exact, and beyond-one-H100 accumulation capacity')
PY
