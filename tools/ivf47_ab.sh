#!/bin/bash
# PR #47 (IVF scan grouped by list): released 0.8.32 vs branch (scan staged default, and MOJOLEARN_IVF_SCAN_GROUPED=0),
# 400,000 x 220, 1024 lists, 32 probes, k 10, 4,000 queries; walls, ANN stages, sha of (distances, indices).
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/ivf47-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; else BK=hip; AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=4; unset PYTHONPATH
pixi install > $O/pixi.log 2>&1
MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR bash bindings/build_ivf.sh > $O/build-ivf.log 2>&1; echo "build rc=$?" >> $O/rc.txt
MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR timeout 3600 pixi run check-ivf > $O/check-ivf.log 2>&1; echo "check-ivf rc=$?" >> $O/rc.txt
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
for arm in before after; do $base -m venv $O/venv-$arm; P=$O/venv-$arm/bin/python; $P -m pip -q install mojolearn==0.8.32 numpy==2.5.2 scipy==1.18.0 > $O/pip-$arm.log 2>&1
  if [ $arm = after ]; then S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn; cp $S/_version.py /tmp/v.py; cp python/mojolearn/*.py $S/; cp /tmp/v.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
    so=$(ls -t python/mojolearn/identical/_mojolearn_ivf.so python/mojolearn/_mojolearn_ivf.so 2>/dev/null | head -1); cp $so $S/$BK/$AR/identical/; fi; done
cat > $O/run.py <<'PY'
import numpy as np, time, hashlib, mojolearn as ml
rng=np.random.default_rng(7); X=rng.standard_normal((400000,220)).astype(np.float32); q=rng.standard_normal((4000,220)).astype(np.float32)
idx=ml.IVFIndex(n_lists=1024,n_probes=32,n_neighbors=10,kmeans_n_iters=20,metric="sqeuclidean",random_state=7,numeric_mode="identical")
t=time.perf_counter(); idx.fit(X); t1=time.perf_counter(); out=idx.search(q); t2=time.perf_counter()
d,i=out if isinstance(out,tuple) else (out,None)
h=hashlib.sha256(np.ascontiguousarray(np.asarray(d)).tobytes()+(np.ascontiguousarray(np.asarray(i)).tobytes() if i is not None else b"")).hexdigest()[:16]
print("RESULT fit %.1f ms search %.1f ms sha %s" % ((t1-t)*1000,(t2-t1)*1000,h))
PY
cd /tmp
for run in before after after-ungrouped; do arm=${run%%-*}; extra=X=1; [ $run = after-ungrouped ] && extra=MOJOLEARN_IVF_SCAN_GROUPED=0
  for rep in 1 2; do env $extra MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_ANN_STAGES=1 $O/venv-$arm/bin/python $O/run.py > $O/$run-$rep.log 2>&1; echo "$run rep$rep $(grep RESULT $O/$run-$rep.log)" >> $O/results.txt; done; done
echo done > $O/done
