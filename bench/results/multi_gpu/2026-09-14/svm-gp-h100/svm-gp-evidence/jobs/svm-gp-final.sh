#!/bin/sh
set -eu
export RUNPOD_POD_ID=ok7m278wjgyo04 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mv /root/svm-out/gate.log /root/svm-out/gate-unsupported-cache.log
mv /root/gp-out/gate.log /root/gp-out/gate-unsupported-alpha.log
pixi run python tools/parallel_svm_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/svm-out/svm.json > /root/svm-out/gate.log 2>&1
pixi run python tools/parallel_gp_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/gp-out/gp.json > /root/gp-out/gate.log 2>&1
cp commit.txt /root/svm-out/source-base-commit.txt
sha256sum python/mojolearn/identical/_mojolearn_svm.so python/mojolearn/identical/_mojolearn.so > /root/svm-out/binaries.sha256
sha256sum python/mojolearn/identical/_mojolearn_gp.so > /root/gp-out/binaries.sha256
sha256sum training/corpus/enwik8/input.txt > /root/svm-out/corpus.sha256
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /root/svm-out/gpu.txt
pixi run mojo --version > /root/svm-out/toolchain.txt 2>&1
