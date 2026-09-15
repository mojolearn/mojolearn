#!/bin/sh
set -eu
while [ ! -f /root/jobs/forest-grove-build.done ]; do sleep 5; done
test "$(cat /root/jobs/forest-grove-build.rc)" = 0
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
for layout in separate packed; do
    extra=''
    if [ "$layout" = packed ]; then extra='-D MOJOLEARN_FOREST_PACKED_NODES=1'; fi
    cp /root/forest-grove-$layout/_mojolearn_rf.so /root/forest-grove-$layout/_mojolearn_trees.so python/mojolearn/identical/
    if [ "$layout" = packed ]; then
        pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 $extra -I . training/checks/forest_pool_check.mojo > /root/forest-grove-out/native-$layout.log 2>&1
    fi
    pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 $extra -I . checks/forest_inference_model.mojo > /root/forest-grove-out/resident-$layout.log 2>&1
    pixi run python tools/parallel_forest_pool_check.py --cloud --report /root/forest-grove-out/public-$layout.json > /root/forest-grove-out/public-$layout.log 2>&1
    pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_FOREST_POOL_FAULT=1 $extra -I . training/checks/forest_pool_check.mojo > /root/forest-grove-out/fault-$layout.log 2>&1
done
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/forest-grove-out/mlp.json > /root/forest-grove-out/mlp.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/forest-grove-out/samba.json > /root/forest-grove-out/samba.log 2>&1
python3 - <<'PY' > /root/forest-grove-out/comparison.log
import json
from pathlib import Path
r=Path('/root/forest-grove-out')
assert json.loads((r/'public-separate.json').read_text()) == json.loads((r/'public-packed.json').read_text())
assert json.loads((r/'mlp.json').read_text()) == json.loads(Path('/root/neural-clip-out/mlp.json').read_text())
assert json.loads((r/'samba.json').read_text()) == json.loads(Path('/root/samba-checkpoint-final-out/samba.json').read_text())
print('PASS exact separate/packed forest receipts and unchanged MLP/Samba replay after worker lifetime cleanup')
PY
cp /root/forest-grove-separate/_mojolearn_rf.so /root/forest-grove-separate/_mojolearn_trees.so python/mojolearn/identical/
