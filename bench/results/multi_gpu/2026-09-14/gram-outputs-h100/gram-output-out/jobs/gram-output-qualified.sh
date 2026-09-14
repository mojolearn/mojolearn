#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
while [ ! -f /root/jobs/gram-output-final.done ]; do sleep 5; done
test "$(cat /root/jobs/gram-output-final.rc)" = 0
mkdir -p /root/gram-output-out/before-extent-sentinel
cp core/gemm.mojo core/gram_multi_gpu.mojo /root/gram-output-out/before-extent-sentinel/
mv /root/gram-output-out/build.log /root/gram-output-out/native.log /root/gram-output-out/public.log /root/gram-output-out/report.json /root/gram-output-out/regression.log /root/gram-output-out/regression-report.json /root/gram-output-out/binaries.sha256 /root/gram-output-out/before-extent-sentinel/
tar -xzf /root/gram-output-qualified-source.tgz
cp /root/gram-output-qualified-source.tgz /root/gram-output-out/
sh bindings/build_estimators.sh > /root/gram-output-out/build.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_estimators.so python/mojolearn/identical/_mojolearn.so > /root/gram-output-out/binaries.sha256
export PYTHONPATH="$PWD/python"
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/gram_outputs_parallel_check.mojo > /root/gram-output-out/native.log 2>&1
pixi run python tools/parallel_gram_wide_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/gram-output-out/report.json > /root/gram-output-out/public.log 2>&1
pixi run python tools/parallel_gram_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/gram-output-out/regression-report.json > /root/gram-output-out/regression.log 2>&1
