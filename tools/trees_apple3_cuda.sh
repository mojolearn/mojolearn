#!/bin/bash
# tools/trees_apple3_cuda.sh (lane trees-apple3, 2026-09-28): the lane's ONE
# small NVIDIA queue job. The lane changed code that is not Apple-only
# (ensemble/randomforest.mojo `fit_forest` -> `fit_forest_prepared`, the forest
# data session in bindings/_mojolearn_rf.mojo, the non-symmetric GBDT driver's
# result loop, the node split publish kernel's parameter); its switches are
# FAST and Apple only. This job shows that the IDENTICAL digests on CUDA are
# the Apple IDENTICAL digests of lane/apple2-merged (the same forests and
# boosted models before this lane), with the data session off and on.
#
#   tools/nvidia_central.sh sync trees-apple3 ~/mojolearn-wt/trees-apple3
#   tools/nvidia_central.sh submit trees-apple3 --gpus 1 --cap 90 \
#       /root/mojolearn-trees-apple3/tools/trees_apple3_cuda.sh
set -u
T=/root/mojolearn-trees-apple3
EV=${EV:-/root/ev-trees-apple3/$(date -u +%m%d-%H%M)}
mkdir -p "$EV"
cd "$T" || exit 1
echo "$(date -u +%FT%TZ) $(hostname) out $EV tree $(git log -1 --format=%h 2>/dev/null)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
ls "$HOME/datasets/gbm-bench/taxi" > /dev/null 2>&1 || {
    echo "trees_apple3_cuda: no taxi data under ~/datasets/gbm-bench (stage it from R2: tools/dataset_store.sh stage)"; exit 2; }
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
$PIXI install -e default > "$EV/pixi.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$EV/pixi.log"; exit 1; }
export MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-6}
for b in build.sh build_estimators.sh build_x_trees.sh build_rf.sh build_gbdt.sh; do
    t0=$(date +%s)
    if $PIXI run -e default sh "bindings/$b" > "$EV/build_$b.log" 2>&1; then
        echo "BUILT $b $(( $(date +%s) - t0 ))s"
    else
        echo "BUILD FAIL $b"; grep -v '^\s*$' "$EV/build_$b.log" | tail -60; exit 1
    fi
done
CELLS="gbdt:gbdt-symmetric:taxi gbdt:gbdt-depthwise:taxi gbdt:gbdt-lossguide:taxi rf:taxi rf:taxireg xt:dt:taxi xt:dt:taxireg xt:bagging:taxi xt:dart:taxi xt:dart:taxireg xt:adaboost:taxi xt:adaboost:taxireg"
for s in 0 1; do
    MOJOLEARN_FOREST_SESSION=$s TAP_ROUNDS=1 TAP_CELLS="$CELLS" TAP_OUT="$EV/tap$s" \
        sh tools/trees_apple_speed.sh 2>&1 | sed "s/^/[session$s] /" | tee "$EV/session$s.txt"
done
$PIXI run -e default python - "$EV" <<'PY'
import json, re, sys
ev = sys.argv[1]
# Apple IDENTICAL, 1,000,000 rows, lane/apple2-merged (docs/lanes/progress/trees-apple2.md,
# m4pro-a steward 1790619284873; dart:taxi from trees-apple.md)
apple = {
    "gbdt-symmetric:taxi": "8e760782efae56c8", "gbdt-depthwise:taxi": "5694af7699036c65",
    "gbdt-lossguide:taxi": "b1761eecc6dfbc73", "rf:taxi": "452a173087f86a9d",
    "rf:taxireg": "58ec783b7afbd7a7", "dt:taxi": "86b6487420c736c8", "dt:taxireg": "e735a53b7d74025a",
    "bagging:taxi": "16aaba82631a5774", "dart:taxi": "8375ab8d60172694",
    "dart:taxireg": "86f40254833745ec", "adaboost:taxi": "e8529a04f218dbab",
    "adaboost:taxireg": "160452cbaf288200",
}
def read(path):
    out, cur = {}, None
    for line in open(path):
        line = re.sub(r"^\[\w+\] ", "", line.rstrip())
        m = re.match(r"=== gbdt:([\w-]+):(\w+)$", line)
        if m:
            cur = m[1] + ":" + m[2]
            continue
        m = re.match(r"GTP FIT .*trees=100 .*digest=(\w+)", line)
        if m and cur:
            out[cur] = m[1]
        m = re.match(r"FTRAIN lane=(\w+) dataset=(\w+) .*hash=(\w+)", line)
        if m:
            out[m[1] + ":" + m[2]] = m[3]
        if line.startswith("TAP {"):
            j = json.loads(line[4:])
            out[j["est"] + ":" + j["dataset"]] = j["digests"][-1]
    return out
off, on = read(ev + "/session0.txt"), read(ev + "/session1.txt")
bad = 0
for cell, want in apple.items():
    a, b = off.get(cell), on.get(cell)
    v = ["cuda " + str(a), "session " + ("EQUAL" if a == b and a else "DIFFERS"),
         "apple " + ("EQUAL" if a == want else "DIFFERS (" + want + ")")]
    if not a or a != b or a != want:
        bad += 1
    print("TREES-CUDA %-22s %s" % (cell, "  ".join(v)))
print("TREES-CUDA VERDICT", "CLEAN" if bad == 0 else "%d cells differ" % bad)
sys.exit(1 if bad else 0)
PY
