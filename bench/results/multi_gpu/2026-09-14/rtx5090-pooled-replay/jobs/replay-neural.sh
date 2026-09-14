#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=cvisjmryfcmz6g MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_120 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
sha256sum training/corpus/enwik8/input.txt > /root/replay-out/corpus.sha256
for name in base training byte_lm linalg mamba transformer; do
    script=bindings/build_${name}.sh
    if [ "$name" = base ]; then script=bindings/build.sh; fi
    sh "$script" > /root/replay-out/build-${name}.log 2>&1
    sha256sum python/mojolearn/identical/*.so > /root/replay-out/neural-binaries.sha256
done
pixi run python tools/parallel_optimizer_check.py --cloud --report /root/replay-out/optimizer.json > /root/replay-out/optimizer.log 2>&1
pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/replay-out/byte-lm.json > /root/replay-out/byte-lm.log 2>&1
pixi run python tools/byte_lm_optimizer_pool_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/replay-out/byte-pool.json > /root/replay-out/byte-pool.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/replay-out/mlp.json > /root/replay-out/mlp.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --corpus training/corpus/enwik8/input.txt --report /root/replay-out/samba.json > /root/replay-out/samba.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --corpus training/corpus/enwik8/input.txt --report /root/replay-out/samba-attention.json > /root/replay-out/samba-attention.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/replay-out/samba-clip.json > /root/replay-out/samba-clip.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/replay-out/samba-attention-clip.json > /root/replay-out/samba-attention-clip.log 2>&1
