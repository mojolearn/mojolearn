#!/bin/sh
# Focused Hot Aisle MI300X trial: shipped AMD IDENTICAL k-NN against the
# already-gated shared-memory distance tile plus block top-k candidate.
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

# Reuse the repository's pinned R2-aware setup and block preparation. It
# builds the shipped core binding while the decoded Taxi and Istella caches
# stage in parallel.
MOJOLEARN_CTD_OUT="$CTD" MOJOLEARN_CTD_LANES=knn \
MOJOLEARN_CTD_PHASES=setup,prep MOJOLEARN_CTD_DATASETS=taxi,istella \
MOJOLEARN_CTD_BODY_SECONDS=3000 MOJOLEARN_CTD_BODY_START="$(date +%s)" \
sh "$R/tools/classical_two_datasets_leg.sh" > "$O/setup.console" 2>&1
echo "setup\t$?\t0" >> "$O/status.tsv"

# Source-tree GPU bindings do not build the package's CPU math helper.  The
# public package imports it while defining neural defaults even for a k-NN
# only process, so use the repository's owned Linux recipe before cloning the
# per-arm trees (the same step as gpu_class_gaps_amd_leg.sh).
step portable_math 300 env PYTHONPATH="$R/packaging/portable_math" "$P" -c \
  "import pathlib, stage; stage.build(pathlib.Path('$R/python/mojolearn/.libs/libMojolearnMath.so'))"
test $? -eq 0 || exit 30

mkdir -p /root/knn-amd-bins/base /root/knn-amd-bins/topk
cp "$R/python/mojolearn/identical/_mojolearn.so" /root/knn-amd-bins/base/
sha256sum /root/knn-amd-bins/base/_mojolearn.so > "$O/base.sha256"
rm -f "$R/python/mojolearn/identical/_mojolearn.so"
step build_topk 1800 env \
  MOJOLEARN_BUILD_EXTRA_DEFINES='-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1' \
  sh "$R/bindings/build.sh"
cp "$R/python/mojolearn/identical/_mojolearn.so" /root/knn-amd-bins/topk/
sha256sum /root/knn-amd-bins/topk/_mojolearn.so > "$O/topk.sha256"

make_tree() {
  a=$1; rm -rf "/root/knn-amd-$a"; mkdir -p "/root/knn-amd-$a"
  (cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' .) | (cd "/root/knn-amd-$a" && tar xf -)
  ln -s "$R/.pixi" "/root/knn-amd-$a/.pixi"
  mkdir -p "/root/knn-amd-$a/python/mojolearn/identical"
  # The package's portable math shim is a CPU support library, not an arm
  # binding.  The source archive excludes *.so, so share the setup tree's
  # immutable copy explicitly.
  ln -s "$R/python/mojolearn/.libs" "/root/knn-amd-$a/python/mojolearn/.libs"
  cp "/root/knn-amd-bins/$a/_mojolearn.so" "/root/knn-amd-$a/python/mojolearn/identical/"
}
make_tree base
make_tree topk

step fit_models 900 env PYTHONPATH=/root/knn-amd-base/python "$P" \
  "$R/bench/speed/classical_ladder_infer.py" fit --data "$DATA" --models "$MODELS" --lanes knn
test $? -eq 0 || exit 31
cat > "$O/arms.json" <<EOF
{
 "base": {"python": "$P", "cwd": "/root/knn-amd-base", "cpu": false,
          "env": {"PYTHONPATH": "/root/knn-amd-base/python:/root/knn-amd-base/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}},
 "topk": {"python": "$P", "cwd": "/root/knn-amd-topk", "cpu": false,
          "env": {"PYTHONPATH": "/root/knn-amd-topk/python:/root/knn-amd-topk/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}}
}
EOF

# The conductor alternates arm order. Six outer passes exceed the requested
# five interleaved rounds; three samples per visit expose drift. Each record
# carries complete output digests.
step race 1200 env PYTHONPATH=/root/knn-amd-base/python "$P" \
  "$R/bench/speed/classical_ladder_infer.py" race --data "$DATA" --models "$MODELS" \
  --arms "$O/arms.json" --out "$O/race" --lanes knn --datasets taxi,istella \
  --outer 6 --rounds 3 --warmup 1
test $? -eq 0 || exit 32
cp "$O/race"/*.json "$O/" 2>/dev/null || true
note finished
