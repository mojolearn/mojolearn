#!/bin/sh
# Focused AMD GPU trial: shipped IDENTICAL kNN against the already-gated
# shared-memory distance tile plus block top-k candidate.
set -u
R=/root/mojolearn
O=/root/gemm_leg_out/knn-amd-smem
CTD=/root/gemm_leg_out/knn-amd-smem-setup
DATA=/root/ctd-data
MODELS=/root/knn-amd-smem-models
P=$R/.pixi/envs/default/bin/python3
mkdir -p "$O"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942}
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-13}
export MOJOLEARN_BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-13}
note() { printf '%s %s\n' "$*" "$(date -u +%H:%M:%S)" | tee -a "$O/progress.txt"; }
step() {
    n=$1 cap=$2; shift 2; t=$(date +%s)
    timeout -k 30 "$cap" "$@" > "$O/$n.log" 2>&1
    rc=$?; printf '%s\t%s\t%s\n' "$n" "$rc" "$(( $(date +%s) - t ))" >> "$O/status.tsv"
    note "$n=$rc"; return "$rc"
}

note start
{ date -u; uname -a; nproc; rocminfo 2>&1 | grep -E 'Marketing Name|Name: +gfx|Compute Unit' | head -20; } > "$O/box.txt" 2>&1

# The common setup stages exactly the Taxi and Istella-S blocks from R2 and
# builds the shipped binding while decode work runs in parallel.
MOJOLEARN_CTD_OUT="$CTD" MOJOLEARN_CTD_LANES=knn \
MOJOLEARN_CTD_PHASES=setup,prep MOJOLEARN_CTD_DATASETS=taxi,istella \
MOJOLEARN_CTD_BODY_SECONDS=3000 MOJOLEARN_CTD_BODY_START="$(date +%s)" \
sh "$R/tools/classical_two_datasets_leg.sh" > "$O/setup.console" 2>&1
rc=$?; printf 'setup\t%s\t0\n' "$rc" >> "$O/status.tsv"
test "$rc" -eq 0 || exit 29

# Source-tree GPU bindings do not build the package's CPU math helper. The
# package imports it while defining neural defaults even for a kNN process.
step portable_math 300 env PYTHONPATH="$R/packaging/portable_math" "$P" -c \
  "import pathlib, stage; stage.build(pathlib.Path('$R/python/mojolearn/.libs/libMojolearnMath.so'))"
test $? -eq 0 || exit 30

mkdir -p /root/knn-amd-bins/base /root/knn-amd-bins/topk
cp "$R/python/mojolearn/identical/_mojolearn.so" /root/knn-amd-bins/base/
sha256sum /root/knn-amd-bins/base/_mojolearn.so > "$O/base.sha256"
rm -f "$R/python/mojolearn/identical/_mojolearn.so"
step build_topk 1800 env \
  MOJOLEARN_BUILD_EXTRA_DEFINES='-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1 -D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_WIDE_ONLY=1' \
  sh "$R/bindings/build.sh"
test $? -eq 0 || exit 31
cp "$R/python/mojolearn/identical/_mojolearn.so" /root/knn-amd-bins/topk/
sha256sum /root/knn-amd-bins/topk/_mojolearn.so > "$O/topk.sha256"

make_tree() {
  a=$1; rm -rf "/root/knn-amd-$a"; mkdir -p "/root/knn-amd-$a"
  (cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' .) | (cd "/root/knn-amd-$a" && tar xf -)
  ln -s "$R/.pixi" "/root/knn-amd-$a/.pixi"
  mkdir -p "/root/knn-amd-$a/python/mojolearn/identical"
  ln -s "$R/python/mojolearn/.libs" "/root/knn-amd-$a/python/mojolearn/.libs"
  cp "/root/knn-amd-bins/$a/_mojolearn.so" "/root/knn-amd-$a/python/mojolearn/identical/"
}
make_tree base
make_tree topk

step fit_models 900 env PYTHONPATH=/root/knn-amd-base/python "$P" \
  "$R/bench/speed/classical_ladder_infer.py" fit --data "$DATA" --models "$MODELS" --lanes knn
test $? -eq 0 || exit 32
cat > "$O/arms.json" <<EOF
{
 "base": {"python": "$P", "cwd": "/root/knn-amd-base", "cpu": false,
          "env": {"PYTHONPATH": "/root/knn-amd-base/python:/root/knn-amd-base/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}},
 "topk": {"python": "$P", "cwd": "/root/knn-amd-topk", "cpu": false,
          "env": {"PYTHONPATH": "/root/knn-amd-topk/python:/root/knn-amd-topk/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}}
}
EOF

# Six alternating outer passes and three samples per visit. Every sample
# hashes the complete distance/index outputs.
step race 1200 env PYTHONPATH=/root/knn-amd-base/python "$P" \
  "$R/bench/speed/classical_ladder_infer.py" race --data "$DATA" --models "$MODELS" \
  --arms "$O/arms.json" --out "$O/race" --lanes knn --datasets taxi,istella \
  --outer 6 --rounds 3 --warmup 1
test $? -eq 0 || exit 33
cp "$O/race"/*.json "$O/" 2>/dev/null || true

# Make identity a hard gate. Timing instability remains evidence and is
# judged after retrieval; an output mismatch makes the remote body fail.
step judge_identity 60 "$P" - "$O/race" "$O/verdict.json" <<'PY'
import json, pathlib, sys
race, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cells = []
ok = True
for dataset in ("taxi", "istella"):
    cell = json.loads((race / ("knn-" + dataset + ".json")).read_text())
    equal = cell["digest_equal_across_arms_per_rows"].get("4000") is True
    complete = all(cell["arms"][a].get("ok_rounds") == 6 for a in ("base", "topk"))
    ratio = cell["ratios"]["topk"]["paired_ratio"]
    cells.append({"dataset": dataset, "bitwise_equal": equal,
                  "complete": complete, "paired_ratio": ratio})
    ok = ok and equal and complete
record = {"identity_and_completeness_pass": ok, "cells": cells}
out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
test $? -eq 0 || exit 34
note finished
