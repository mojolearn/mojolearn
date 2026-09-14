#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_boosting_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/boost-out/boosting.json > /root/boost-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_gbdt.so > /root/boost-out/binaries.sha256
