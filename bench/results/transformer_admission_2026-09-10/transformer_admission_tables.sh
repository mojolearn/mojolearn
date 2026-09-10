#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
export PYTHONPATH=/root/transformer-admission/repo/python
cd /root/transformer-admission
MOJOLEARN_TRANSFORMER_ROPE_FULL=1 ./rope_probe > rope_full.log
for shape in narrow wide; do
    python3 transformer_admission_diagnose.py --spec original_speed_torch_seq.py --shape "$shape" --arm reference --output "$shape.npy" --rope-log rope.log --full-rope-log rope_full.log --reference64 > "$shape.tables_reference.log" 2>&1
done
