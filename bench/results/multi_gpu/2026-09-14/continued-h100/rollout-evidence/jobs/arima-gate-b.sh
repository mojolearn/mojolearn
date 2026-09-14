#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_arima_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/arima-out/arima.json > /root/arima-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_arima.so > /root/arima-out/binaries.sha256
