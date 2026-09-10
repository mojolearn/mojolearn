set -euo pipefail
cd /root/performance
export PATH=/root/.pixi/bin:$PATH
mkdir -p /root/knn-request
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > /root/knn-request/gpu.csv
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_vector_request_trial.mojo -o /root/knn-request/trial > /root/knn-request/build.log 2>&1
for k in 10 15; do
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=31 /root/knn-request/trial > /root/knn-request/k$k.log 2>&1
done
