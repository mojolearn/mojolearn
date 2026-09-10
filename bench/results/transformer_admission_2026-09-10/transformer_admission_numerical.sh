#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH=/root/transformer-admission/repo/python
cd /root/transformer-admission
for shape in narrow wide; do
    python3 transformer_admission_diagnose.py --spec original_speed_torch_seq.py --shape "$shape" --arm ours --output "$shape.npy" > "$shape.ours.log" 2>&1
    python3 transformer_admission_diagnose.py --spec original_speed_torch_seq.py --shape "$shape" --arm reference --output "$shape.npy" --rope-log rope.log > "$shape.reference.log" 2>&1
done
