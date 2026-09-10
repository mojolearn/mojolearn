set -euo pipefail
cd /root/performance
export PATH=/root/.pixi/bin:$PATH
for k in 15 10; do
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=31 /root/knn-request/trial > /root/knn-request/reverse-k$k.log 2>&1
done
for n in 400003 400004; do
MOJOLEARN_KNN_REF_INDEX=$n MOJOLEARN_KNN_REF_QUERIES=4001 MOJOLEARN_KNN_REF_FEATURES=32 MOJOLEARN_KNN_REF_K=15 MOJOLEARN_KNN_REF_ROUNDS=7 /root/knn-request/trial > /root/knn-request/ragged-$n.log 2>&1
done
cp neighbors/checks/pinned_distance_tile.mojo /root/knn-request/pinned_distance_tile.mojo
cp neighbors/impl/neighbors/detail/knn_brute_force.mojo /root/knn-request/knn_brute_force.mojo
cp bench/knn_vector_request_trial.mojo /root/knn-request/knn_vector_request_trial.mojo
cp /root/knn-request-sabotage.mojo neighbors/checks/pinned_distance_tile.mojo
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_VECTOR_SABOTAGE=1 bench/knn_vector_request_trial.mojo -o /root/knn-request/sabotage > /root/knn-request/sabotage-build.log 2>&1
set +e
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=10 MOJOLEARN_KNN_REF_ROUNDS=1 /root/knn-request/sabotage > /root/knn-request/sabotage.log 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > /root/knn-request/sabotage.rc
[[ $rc != 0 ]]
grep -E 'bytes moved|invalid or poisoned' /root/knn-request/sabotage.log
cp /root/knn-request/pinned_distance_tile.mojo neighbors/checks/pinned_distance_tile.mojo
