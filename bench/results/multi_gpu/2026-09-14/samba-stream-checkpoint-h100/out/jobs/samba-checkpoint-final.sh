#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/samba-checkpoint-final-out
cp /root/samba-checkpoint-final.tgz /root/samba-checkpoint-final-out/
sha256sum /root/samba-checkpoint-final.tgz > /root/samba-checkpoint-final-out/source.sha256
tar xzf /root/samba-checkpoint-final.tgz
cp /root/neural-clip-out/hardware.csv /root/samba-checkpoint-final-out/
cp /root/neural-clip-out/corpus.sha256 /root/samba-checkpoint-final-out/
while [ ! -f /root/jobs/iforest-pool.done ]; do sleep 5; done

pixi run python tools/samba_checkpoint_check.py --cloud --report /root/samba-checkpoint-final-out/checkpoint.json > /root/samba-checkpoint-final-out/checkpoint.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/samba-checkpoint-final-out/samba.json > /root/samba-checkpoint-final-out/samba.log 2>&1
python3 - <<'PY' > /root/samba-checkpoint-final-out/comparison.log
import json
from pathlib import Path
assert json.loads(Path('/root/samba-checkpoint-final-out/samba.json').read_text()) == json.loads(Path('/root/neural-clip-out/samba-attention-clip.json').read_text())
print('PASS unchanged clipped attention/dropout Samba full replay receipts')
PY
sha256sum python/mojolearn/identical/_mojolearn_training.so python/mojolearn/identical/_mojolearn_transformer.so > /root/samba-checkpoint-final-out/binaries.sha256
