#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/gram_parallel_check.mojo > /root/gram-out/native-bitwise.log 2>&1
pixi run python tools/parallel_gram_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/gram-out/gram.json > /root/gram-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_estimators.so > /root/gram-out/binaries.sha256
