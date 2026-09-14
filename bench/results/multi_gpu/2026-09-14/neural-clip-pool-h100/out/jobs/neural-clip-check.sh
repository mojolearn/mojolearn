#!/bin/sh
set -eu
while [ ! -f /root/jobs/neural-clip-build.done ]; do sleep 5; done
test "$(cat /root/jobs/neural-clip-build.rc)" = 0
test "$(cat /root/jobs/neural-gradient-final-v2.rc)" = 0
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
cp /root/neural-clip-production/_mojolearn_training.so python/mojolearn/identical/_mojolearn_training.so
pixi run python tools/parallel_clip_check.py --cloud --report /root/neural-clip-out/clip.json > /root/neural-clip-out/clip.log 2>&1
pixi run python tools/parallel_optimizer_check.py --cloud --report /root/neural-clip-out/optimizer.json > /root/neural-clip-out/optimizer.log 2>&1
pixi run python tools/parallel_accumulate_check.py --cloud --report /root/neural-clip-out/accumulate.json > /root/neural-clip-out/accumulate.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/neural-clip-out/mlp.json > /root/neural-clip-out/mlp.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/neural-clip-out/samba-clip.json > /root/neural-clip-out/samba-clip.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/neural-clip-out/samba-attention-clip.json > /root/neural-clip-out/samba-attention-clip.log 2>&1
python3 - <<'PY'
from pathlib import Path
s=Path('bindings/build_training.sh').read_text().replace('OUTDIR="python/mojolearn/identical"','OUTDIR="/root/neural-clip-fault"').replace('MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1"','MODE_DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_CLIP_POOL_FAULT=1"')
Path('bindings/build_training_clip_fault.sh').write_text(s)
Path('/root/neural-clip-out/build-fault-script.sh').write_text(s)
PY
sh bindings/build_training_clip_fault.sh > /root/neural-clip-out/build-fault.log 2>&1
sha256sum /root/neural-clip-fault/_mojolearn_training.so > /root/neural-clip-out/fault.sha256
cp /root/neural-clip-fault/_mojolearn_training.so python/mojolearn/identical/_mojolearn_training.so
pixi run python tools/parallel_clip_check.py --cloud --faults --report /root/neural-clip-out/fault.json > /root/neural-clip-out/fault.log 2>&1
cp /root/neural-clip-production/_mojolearn_training.so python/mojolearn/identical/_mojolearn_training.so
python3 - <<'PY' > /root/neural-clip-out/comparison.log
import json
from pathlib import Path
r=Path('/root/neural-clip-out')
g=Path('/root/neural-gradient-out')
for lane in ('mlp','optimizer','accumulate'):
    assert json.loads((r/(lane+'.json')).read_text()) == json.loads((g/'final'/(lane+'.json')).read_text()),lane
assert json.loads((r/'samba-attention-clip.json').read_text()) == json.loads((g/'final/samba-clipped.json').read_text())
assert json.loads((r/'clip.json').read_text())['checks'] == json.loads((r/'fault.json').read_text())['checks']
print('PASS unchanged optimizer, accumulation, MLP and clipped attention/dropout Samba receipts; exact post-scale fault recovery')
PY
