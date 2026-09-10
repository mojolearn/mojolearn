set -euo pipefail
cd /root/performance
export PATH=/root/.pixi/bin:$PATH
mkdir -p /root/knn-loads/aligned
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_vector_load_trial.mojo -o /root/knn-loads/aligned/trial > /root/knn-loads/aligned/build.log 2>&1
MOJOLEARN_KNN_LAYOUT_LARGE=1 MOJOLEARN_KNN_LAYOUT_ROWS=512 MOJOLEARN_KNN_LAYOUT_COLS=65536 MOJOLEARN_KNN_LAYOUT_D=32 MOJOLEARN_KNN_LAYOUT_SAMPLES=15 /root/knn-loads/aligned/trial > /root/knn-loads/aligned/trial.log 2>&1
pixi run mojo build --emit asm --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_vector_load_trial.mojo -o /root/knn-loads/aligned/trial.s > /root/knn-loads/aligned/asm-build.log 2>&1
for p in /root/knn-loads/aligned/*pinned_distan*.ptx; do
python tools/gemm_cuda_resources.py "$p" "${p%.ptx}-resources" --ptxas /root/cuda126-tools/nvidia/cuda_nvcc/bin/ptxas --cuobjdump /usr/local/cuda/bin/cuobjdump > "${p%.ptx}-report.json"
done
