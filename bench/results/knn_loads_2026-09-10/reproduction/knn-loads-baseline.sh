set -euo pipefail
cd /root/performance
export PATH=/root/.pixi/bin:$PATH
pixi install > /root/jobs/install.log 2>&1
mkdir -p /root/knn-loads/baseline
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_PHASE_TIMERS=1 bench/knn_reference_price_main.mojo -o /root/knn-loads/phase > /root/knn-loads/phase-build.log 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > /root/knn-loads/gpu.csv
for k in 10 15; do
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=5 /root/knn-loads/phase > /root/knn-loads/phase-k$k.log 2>&1
done
pixi run mojo build --emit asm --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_index_layout_main.mojo -o /root/knn-loads/baseline/layout.s > /root/knn-loads/baseline/build.log 2>&1
python -m pip install --no-deps --target /root/cuda126-tools nvidia-cuda-nvcc-cu12==12.6.85 > /root/knn-loads/tool-install.log 2>&1
for p in /root/knn-loads/baseline/*pinned_distan*.ptx; do
python tools/gemm_cuda_resources.py "$p" "${p%.ptx}-resources" --ptxas /root/cuda126-tools/nvidia/cuda_nvcc/bin/ptxas --cuobjdump /usr/local/cuda/bin/cuobjdump > "${p%.ptx}-report.json"
done
