#!/usr/bin/env bash
set -euo pipefail
cd /root/performance
out=/root/jobs/mamba-regime-large
mkdir "$out"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MOJOLEARN_NUMERIC_MODE=identical
sha256sum python/mojolearn/identical/_mojolearn_mamba.so tools/mamba3_regime_probe.py tools/mamba3_regime_summary.py > "$out/source-binary-sha256.txt"
cp /root/jobs/build-mamba.log "$out/build-mamba.log"
.pixi/envs/default/bin/mojo --version > "$out/mojo-version.txt" 2>&1
nvidia-smi -q > "$out/gpu-before.txt"
for arm in 0 1; do
    timeout 480 .pixi/envs/default/bin/python tools/mamba3_regime_probe.py \
      --harness bench/results/mamba3/2026-09-09-statepass/reproduction/seq_py_speed_arm.py \
      --spec bench/results/mamba3/2026-09-09-statepass/reproduction/speed_torch_seq.py \
      --order narrow,wide,narrow,tiny,narrow,wide --passes 2 --rounds 8 \
      --reference-json bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json \
      > "$out/shape-order-$arm.log" 2>&1
    .pixi/envs/default/bin/python tools/mamba3_regime_summary.py "$out/shape-order-$arm.log" > "$out/shape-order-$arm-summary.json"
done
nvidia-smi -q > "$out/gpu-after.txt"
printf 'PASS\n' > "$out/completed.txt"
