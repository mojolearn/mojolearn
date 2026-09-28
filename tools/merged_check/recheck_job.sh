#!/bin/bash
# lane/merged re-check on the fixed tree: build, then the lanes the fixes touch
# (clean, CPU threads 1/3/default), the neighbors e2e sabotage, then the tests.
cd "$(dirname "$0")/../.."
OUT=/root/ev-merged-re; mkdir -p $OUT
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_GPU_ARCHS=sm_89
P="/root/.pixi/bin/pixi run -e default python -u tools/merged_check/merged_check.py"
if [ ! -f $OUT/build.done ]; then
  /root/.pixi/bin/pixi install -e default > $OUT/pixi_install.log 2>&1
  $P build --out $OUT --jobs 6 > $OUT/build.log 2>&1; echo "exit $?" > $OUT/build.done
fi
grep -q "^exit 0" $OUT/build.done || { echo "BUILD NOT OK: $(tail -2 $OUT/build.log)"; exit 1; }
LANES=$(tr '\n' ' ' < tools/merged_check/recheck_lanes.txt)
python3 - "$OUT" $LANES <<'PY'
import json, sys
out, lanes = sys.argv[1], sys.argv[2:]
p = json.load(open(out + "/plan.json")); p["lanes"] = lanes
json.dump(p, open(out + "/plan_re.json", "w"))
PY
mkdir -p $OUT/re && cp $OUT/plan_re.json $OUT/re/plan.json
$P clean --out $OUT/re --shard 0/1 --cpu-threads 1,3,default
$P sabotage --out $OUT --patch x_neighbors/checks/sabotage/e2e_existing_device.patch --lanes knn,knn-sqeuclidean,knn-manhattan,knn-chebyshev,knn-cosine,knn-minkowski-p3,knn-rbc,knn-clf,knn-clf-distance,knn-reg,knn-reg-distance,radius,radius-manhattan,radius-chebyshev,radius-minkowski-p3,kde,kde-tophat-sqeuclidean,kde-epanechnikov-l1,kde-exponential-chebyshev,kde-linear-cosine,kde-cosine-minkowski,kde-weighted,svc,svc-linear,svc-poly,svr,svr-linear,gp,gp-normalize-y,gp-sample-y,gp-sample-y-normalize,gp-optimize,gp-optimize-restarts,gp-matern12,gp-matern32,gp-matern52-ard,gpc,gpc-multiclass,kernel-ridge,kernel-ridge-poly,kernel-ridge-sigmoid,kernel-ridge-laplacian,nystroem,nystroem-poly,nystroem-sigmoid,nystroem-laplacian,rbf-sampler
bash tools/merged_check/tests_job.sh $OUT
