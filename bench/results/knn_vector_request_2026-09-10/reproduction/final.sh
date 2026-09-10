set -euo pipefail
cd /root/performance
export PATH=/root/.pixi/bin:$PATH
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_VECTOR_REQUEST_CHECK=1 bench/knn_vector_request_main.mojo -o /root/knn-request/final-check > /root/knn-request/final-check-build.log 2>&1
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_reference_price_main.mojo -o /root/knn-request/default > /root/knn-request/default-build.log 2>&1
for k in 10 15; do
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=15 MOJOLEARN_KNN_REF_DUMP_FULL=/root/knn-request/scalar-k$k.bin /root/knn-request/final-check > /root/knn-request/final-check-k$k.log 2>&1
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=15 MOJOLEARN_KNN_REF_DUMP_FULL=/root/knn-request/default-k$k.bin /root/knn-request/default > /root/knn-request/default-k$k.log 2>&1
cmp /root/knn-request/scalar-k$k.bin /root/knn-request/default-k$k.bin
done
