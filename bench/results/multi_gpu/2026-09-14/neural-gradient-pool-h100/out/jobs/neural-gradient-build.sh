#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
mkdir -p /root/mojolearn /root/neural-gradient-out
cd /root/mojolearn
tar xzf /root/neural-gradient-source.tgz
cp /root/neural-gradient-source.tgz /root/neural-gradient-out/source.tgz
sha256sum /root/neural-gradient-source.tgz > /root/neural-gradient-out/source.sha256
export PYTHONPATH="$PWD/python"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /root/neural-gradient-out/hardware.csv
for family in base training linalg mamba transformer; do
    if [ "$family" = base ]; then script=bindings/build.sh; else script=bindings/build_${family}.sh; fi
    sh "$script" > /root/neural-gradient-out/build-${family}.log 2>&1
done
sha256sum python/mojolearn/identical/*.so > /root/neural-gradient-out/binaries.sha256
pixi run python tools/parallel_accumulate_check.py --cloud --report /root/neural-gradient-out/accumulate.json > /root/neural-gradient-out/accumulate.log 2>&1
pixi run python tools/parallel_optimizer_check.py --cloud --report /root/neural-gradient-out/optimizer.json > /root/neural-gradient-out/optimizer.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/neural-gradient-out/mlp.json > /root/neural-gradient-out/mlp.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --corpus training/corpus/enwik8/input.txt --report /root/neural-gradient-out/samba.json > /root/neural-gradient-out/samba.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/neural-gradient-out/samba-clipped.json > /root/neural-gradient-out/samba-clipped.log 2>&1
