set -euo pipefail
cd /root/performance
export PATH=/root/.pixi/bin:$PATH
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_VECTOR_SABOTAGE=1 bench/knn_vector_load_trial.mojo -o /root/knn-loads/sabotage > /root/knn-loads/sabotage-build.log 2>&1
set +e
/root/knn-loads/sabotage > /root/knn-loads/sabotage.log 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > /root/knn-loads/sabotage.rc
[[ $rc != 0 ]]
grep 'distance.*mismatch' /root/knn-loads/sabotage.log
for suffix in 8b3e6f9482a72b6b d2c629f604a66e26; do
python /root/query_knn_occupancy.py /root/knn-loads/interior/trial_neighbors_checks_pinned_distan6A6A6A6A_${suffix}-resources/kernel.cubin neighbors_checks_pinned_distan6A6A6A6A_${suffix} > /root/knn-loads/interior/occupancy-${suffix}.json
done
