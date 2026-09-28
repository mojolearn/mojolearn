#!/bin/bash
# lane/py-bugs: the before/after bit record on one shared NVIDIA pod (x86 CPU
# column and NVIDIA column), LIGHT and in ONE tree (the pod disks are small):
# `git apply -R` of the lane's code patch gives the base (lane/apple2-merged at
# the merge), each lane's GPU and CPU arm once, the probe once per column; then
# `git apply` restores the lane and the same runs again (only bindings whose
# sources moved rebuild); then base vs new part by part. The patch is
# re-applied on every exit path; build outputs this job made are deleted at the end.
set -u
NEW=/root/mojolearn-py-bugs
EV=${EV:-/root/ev-py-bugs/$(date -u +%m%d-%H%M)}
PATCH=$NEW/tools/py_bugs/lane_vs_base.patch
mkdir -p "$EV"
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
LANES=${LANES:-x-cluster-bgmm,x-cluster-bgmm-inits,x-cluster-bgmm-covtypes,x-cluster-gmm-options,x-logistic-cv,x-logistic-cv-w,x-huber,x-ridge-clf,x-metrics-classification,x-metrics-cluster,x-metrics-search,x-metrics-splitters,x-cnn-trainer,x-cnn-trainer-options,x-prep-iterative-options,x-prep-user-objects,x-neighbors-svgp,x-neighbors-svc-probability,x-decomp-grp,x-decomp-srp,sequence-autoarima,sequence-rnn,sequence-lstm,sequence-gru}
cd "$NEW"
touch "$EV/.start"
echo "$(date -u +%FT%TZ) $(hostname) out $EV"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; lscpu | grep -m1 'Model name'; df -h /root | tail -1
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
$PIXI install -e default > "$EV/pixi.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$EV/pixi.log"; exit 1; }
cp tools/py_bugs/check.py tools/py_bugs/probe.py "$EV/"   # the driver itself is not in the patch
P="$PIXI run -e default python -u $EV/check.py"
git apply --check -R "$PATCH" || { echo "PATCH DOES NOT REVERSE"; exit 1; }
restore() { git apply --check "$PATCH" 2>/dev/null && git apply "$PATCH" && echo "restored the lane"; }
trap restore EXIT
git apply -R "$PATCH" && echo "base: the lane's code patch reversed"
$P arms --tree "$NEW" --out "$EV/base" --lanes "$LANES" 2>&1 | tee "$EV/base.out" | grep -v '^\s*$' | tail -60
restore; trap - EXIT
$P arms --tree "$NEW" --out "$EV/new" --lanes "$LANES" 2>&1 | tee "$EV/new.out" | grep -v '^\s*$' | tail -60
$P cross --base "$EV/base" --new "$EV/new" --lanes "$LANES" 2>&1 | tee "$EV/cross.txt"
$PIXI install -e test > "$EV/pixi_test.log" 2>&1 || echo "PIXI TEST ENV INSTALL FAIL"
for col in gpu cpu; do
    echo "== pytest test_py_bugs ($col)"
    ( cd python && if [ $col = cpu ]; then export MOJOLEARN_VENDOR=cpu; fi
      MOJOLEARN_NUMERIC_MODE=identical $PIXI run -e test python -m pytest -q -p no:cacheprovider \
          mojolearn/tests/test_py_bugs.py 2>&1 | tail -15 )
done
# small footprint: drop the build outputs this job made (bindings, the math helper)
find python/mojolearn -newer "$EV/.start" \( -name '*.so' -o -name '*.dylib' -o -name '*.lanecheck-stamp' \) -delete 2>/dev/null
du -sh "$EV" | tail -1; df -h /root | tail -1
echo "JOB END $(date -u +%FT%TZ)"
