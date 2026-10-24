#!/bin/bash
# tools/box_merged_verify.sh nvidia|amd <sha> -- the post-build GPU check of one
# commit on one rented box (2026-10-21): the merged bindings built from the tree
# (identical and host), `python -m mojolearn verify` in identical mode, then
# the incremental reference-column record of the lanes whose source closure
# changed (ONLY_CHANGED=table tools/record_identity_column.sh), so a new verify
# lane gets its NVIDIA/AMD digests without re-fitting every row. Everything
# logs in a NEW /root/results-merged-* directory (BOX_OUT_DIR overrides it).
# BOX_REPO_DIR defaults to /root/mojolearn and must already be clean at <sha>.
# Sets build serially: merged-set peak memory has no qualified parallel budget.
# Compiler workers follow packaging/linux/build_sets.sh: AMD GPU 8, NVIDIA 2,
# host 2. BOX_GPU_COMPILE_JOBS / BOX_HOST_COMPILE_JOBS override individual sets.
# Old installed sets are preserved under previous/ before new sets activate.
#
# Then the owed GPU evidence (2026-10-21, docs/plans/BOX_JOBS_20261021.md), each
# a function guarded by what the box has, each step `st`-logged, none aborting
# the chain. BOX_JOBS (empty by default) explicitly picks them:
#   dist      physical multi-GPU checks, only with 2+ GPUs on the box
#   graphopt  captured-graph optimizer A/B on the saved full workloads (both vendors)
#   witness   native NVIDIA witness: four PTX discrepancies + four tree batch probes
#   stage64   NVIDIA MOJOLEARN_IDN_GEMM_STAGE64 A/B (MLP + recurrent, HIGGS)
# The Mac stages the data first (tools/stage_from_r2.sh): corpus/enwik8/input.txt
# for dist, gbm-bench/higgs/higgs_speed.npz for stage64.
set -uo pipefail
CONTROLLER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
[ $# = 2 ] || { echo "usage: box_merged_verify.sh nvidia|amd <sha>" >&2; exit 2; }
V=$1; SHA=$2
case $V in nvidia|amd) ;; *) echo "unknown vendor: $V" >&2; exit 2;; esac
[[ $SHA =~ ^[0-9a-fA-F]{7,40}$ ]] || { echo "expected a commit SHA" >&2; exit 2; }
O=${BOX_OUT_DIR:-/root/results-merged-$V-${SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$}
case $O in /*) ;; *) echo "BOX_OUT_DIR must be absolute" >&2; exit 2;; esac
# Optional legacy jobs pass paths as shell words. Refuse ambiguous paths up front.
[[ $O != *[[:space:]]* ]] || { echo "BOX_OUT_DIR must have no whitespace" >&2; exit 2; }
mkdir "$O" || { echo "BOX_OUT_DIR must be a new directory: $O" >&2; exit 2; }
O=$(cd "$O" && pwd -P)
echo "box_merged_verify: outputs $O"
exec >> "$O/job.log" 2>&1
st() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/status.txt"; }
finish() { local rc=$?; printf '%s\n' "$rc" > "$O/exit-code"; st "DONE rc=$rc"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
die() { st "REFUSED $*"; exit 3; }
st "START $V $SHA"
cp "$CONTROLLER" "$O/controller.sh" || die "cannot retain controller source"
sha256sum "$O/controller.sh" > "$O/controller-sha256.txt" || die "cannot hash controller source"
# The provisioned pixi and ROCm tools win over anything else on PATH. A box
# that keeps them elsewhere (and the controller's tests, whose shims must not be
# shadowed by a real /root/.pixi/bin on a GPU pod) names them in BOX_TOOL_PATH.
export PATH="${BOX_TOOL_PATH:-/root/.pixi/bin:/opt/rocm/bin}:$PATH"
command -v git >/dev/null || die "git missing; provision the box first"
command -v pixi >/dev/null || die "pixi missing; provision the box first"
cd "${BOX_REPO_DIR:-${BOX_REPO:-/root/mojolearn}}" || die "checkout missing; prepare it at $SHA first"
ROOT=$(pwd -P)
case $O in "$ROOT"|"$ROOT"/*) die "BOX_OUT_DIR must be outside the checkout";; esac
EXPECTED=$(git rev-parse --verify "$SHA^{commit}") || die "commit $SHA is unavailable locally"
command -v flock >/dev/null || die "flock missing; provision the box first"
exec 9> "$(git rev-parse --git-path box-merged-verify.lock)" || die "cannot open checkout build lock"
flock -n 9 || die "another qualification runner holds this checkout"
check_tree() {
  [ "$(git rev-parse HEAD)" = "$EXPECTED" ] || die "checkout is not at $EXPECTED; no files changed"
  git status --porcelain --untracked-files=all > "$O/checkout-status.txt" || die "cannot inspect checkout"
  git diff --quiet HEAD -- || die "tracked/index changes; see checkout-status.txt (no reset performed)"
  local path
  while IFS= read -r path; do case $path in
    python/mojolearn/.libs/libMojolearnMath.so) ;; # generated, retained on activation
    *) die "untracked source $path; see checkout-status.txt (no reset performed)";;
  esac; done < <(git ls-files --others --exclude-standard)
}
check_tree
SHA=$EXPECTED
BOX_BUILDS=${BOX_BUILDS:-gpu-identical host}
[ "$BOX_BUILDS" != all ] || BOX_BUILDS="gpu-identical host"
have_gpu=0; have_host=0
for b in $BOX_BUILDS; do case $b in
  gpu-identical) [ "$have_gpu" = 0 ] || die "duplicate build $b"; have_gpu=1;;
  host) [ "$have_host" = 0 ] || die "duplicate build $b"; have_host=1;;
  *) die "unsupported BOX_BUILDS entry $b; Linux qualification uses gpu-identical host";;
esac; done
[ "$have_gpu:$have_host" = 1:1 ] || die "qualification requires both gpu-identical and host; stale sets cannot be reused"
case ${BOX_RECORD:-1} in 0|1) ;; *) die "BOX_RECORD must be 0 or 1";; esac
for j in ${BOX_JOBS:-}; do case $j in dist|graphopt|witness|stage64) ;; *) die "unknown BOX_JOBS entry $j";; esac; done
# A wheel/plugin set or explicit binding override would shadow the fresh flat set.
for d in cuda hip cpu; do
  [ ! -d "python/mojolearn/$d" ] || [ -z "$(find "python/mojolearn/$d" -name '*.so' -print -quit)" ] || die "vendor binaries in $d shadow checkout builds"
done
for name in $(compgen -v MOJOLEARN_); do case $name in
  MOJOLEARN_HOST_DIR|MOJOLEARN_*_BINARY|MOJOLEARN_GPU_ARCH|MOJOLEARN_VENDOR_FORCE)
    [ -z "${!name}" ] || die "$name overrides qualification binding selection";;
esac; done
export MOJOLEARN_NUMERIC_MODE=identical
if [ "$V" = amd ]; then
  export MOJOLEARN_VENDOR=hip
  ARCH=gfx942
  GPU=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -o 'MI3[0-9A-Z]*' | tr 'A-Z' 'a-z')
  LABEL=amd-${GPU:-mi325x}-$ARCH
  NGPU=$(rocm-smi --showid 2>/dev/null | grep -o 'GPU\[[0-9]*\]' | sort -u | wc -l)
else
  export MOJOLEARN_VENDOR=cuda
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
  [[ $CC =~ ^[0-9]{2,3}$ ]] || die "cannot determine NVIDIA architecture"
  ARCH=sm_$CC
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed -E 's/^NVIDIA //; s/^(GeForce|RTX|Tesla) //g' | awk '{print tolower($1)}')
  LABEL=nvidia-${GPU:-gpu}-$ARCH
  NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
fi
# Match the release worker policy, independently of set concurrency. Its AMD
# evidence records a merged -j1 compiler crash and a successful -j8 build.
# A caller's generic MOJOLEARN_COMPILE_JOBS must not silently put AMD back on
# that known failing default or change the byte-stable host worker policy.
GPU_COMPILE_JOBS=2
if [ "$V" = amd ]; then GPU_COMPILE_JOBS=${MOJOLEARN_AMD_COMPILE_JOBS:-8}; fi
GPU_COMPILE_JOBS=${BOX_GPU_COMPILE_JOBS:-$GPU_COMPILE_JOBS}
HOST_COMPILE_JOBS=${BOX_HOST_COMPILE_JOBS:-2}
for workers in "$GPU_COMPILE_JOBS" "$HOST_COMPILE_JOBS"; do
  [[ $workers =~ ^[1-9][0-9]?$ && $workers -le 32 ]] || die "per-set compiler workers must be 1..32"
done
st "label $LABEL gpus=$NGPU; concurrent sets=1; compiler workers gpu=$GPU_COMPILE_JOBS host=$HOST_COMPILE_JOBS"
printf 'commit\t%s\nvendor\t%s\narch\t%s\nbuilds\t%s\njobs\t1\ngpu_compile_jobs\t%s\nhost_compile_jobs\t%s\n' \
  "$SHA" "$V" "$ARCH" "$BOX_BUILDS" "$GPU_COMPILE_JOBS" "$HOST_COMPILE_JOBS" > "$O/run.tsv"
pixi install --locked > "$O/pixi.log" 2>&1; rc=$?; st "pixi rc=$rc"
[ "$rc" = 0 ] || die "pixi install failed; no build or verification"
PY=$(pixi run -q python -c 'import sys; print(sys.executable)') || die "cannot locate Python"
[ -x "$PY" ] || die "Python is not executable: $PY"
pixi run -q mojo --version > "$O/compiler-version.txt" 2>&1 || die "cannot identify the selected compiler"
COMPILER=$(pixi run -q python -c 'import shutil; print(shutil.which("mojo") or "")') || die "cannot locate the selected compiler"
[ -f "$COMPILER" ] || die "selected compiler path is absent: $COMPILER"
sha256sum "$COMPILER" pixi.lock pixi.toml bindings/merged_build.sh tools/merged_bindings.py \
  > "$O/toolchain-input-sha256.txt" || die "cannot retain toolchain inputs"
printf 'variable\tvalue\n' > "$O/build-environment.tsv"
for name in MOJOLEARN_LINUX_CPU MOJOLEARN_COMPILE_JOBS MOJOLEARN_BUILD_DEFINE_FLAGS \
  MOJOLEARN_AMD_COMPILE_JOBS BOX_GPU_COMPILE_JOBS BOX_HOST_COMPILE_JOBS \
  MOJOLEARN_BUILD_EXTRA_DEFINES MOJOLEARN_MOJO_BUILD_FLAGS MOJOLEARN_MERGED_PYTHON \
  MOJOLEARN_EXTRA_DEFINES MOJOLEARN_MAMBA_DEFINES MOJOLEARN_ATTENTION_MEMORY_PROFILE \
  MOJOLEARN_ALLOW_NO_GPU CC CFLAGS LDFLAGS LD_LIBRARY_PATH TMPDIR; do
  printf '%s\t%s\n' "$name" "${!name-<unset>}" >> "$O/build-environment.tsv"
done
TIMEV=()
/usr/bin/time --version >/dev/null 2>&1 && TIMEV=(/usr/bin/time -v)
printf 'build\texit_code\n' > "$O/build-status.tsv"
mb() {
  local tag=$1 dir=$2; shift 2
  printf '%q ' env "$@" MOJOLEARN_PACKAGE_BYTE_LM=1 pixi run -q sh bindings/merged_build.sh "${tag%%-*}" \
    --outdir "$O/artifacts/$dir" > "$O/build-$tag.command"
  printf '\n' >> "$O/build-$tag.command"
  "${TIMEV[@]}" env "$@" MOJOLEARN_PACKAGE_BYTE_LM=1 pixi run -q sh bindings/merged_build.sh "${tag%%-*}" \
    --outdir "$O/artifacts/$dir" > "$O/build-$tag.log" 2> "$O/build-$tag.time"
  local rc=$?
  printf '%s\t%s\n' "$tag" "$rc" >> "$O/build-status.tsv"
  st "BUILD $tag rc=$rc peak_rss_gb=$(awk '/Maximum resident/ {printf "%.1f", $6/1048576}' "$O/build-$tag.time")"
  return "$rc"
}
failed=0
for b in $BOX_BUILDS; do case $b in
  gpu-identical) mb "$b" identical MOJOLEARN_GPU_ARCHS="$ARCH" MOJOLEARN_TARGET_COLUMN="$V" \
    MOJOLEARN_COMPILE_JOBS="$GPU_COMPILE_JOBS" || failed=1;;
  host) mb "$b" host MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_GPU_ARCHS= \
    MOJOLEARN_COMPILE_JOBS="$HOST_COMPILE_JOBS" || failed=1;;
esac; done
[ "$failed" = 0 ] || die "required build failed; previous artifacts untouched; no verification"
for d in identical host; do
  "$PY" tools/merged_bindings.py check-index --outdir "$O/artifacts/$d" --dir "$d" --platform linux --byte-lm 1 \
    > "$O/index-$d.log" 2>&1 || die "$d build index is incomplete; no activation or verification"
done
# The owned host math helper: a source checkout has none until the tree's own recipe
# builds it (packaging/portable_math/stage.py; the wheel ships it). Without it every
# CPU-arm lane refuses with "libMojolearnMath.so: cannot open" (2026-10-21 AMD run).
MATHLIB=python/mojolearn/.libs/libMojolearnMath.so
PYTHONPATH=packaging/portable_math pixi run -q python -c 'import pathlib, stage, sys; stage.build(pathlib.Path(sys.argv[1]))' \
  "$O/artifacts/.libs/libMojolearnMath.so" > "$O/mathlib.log" 2>&1
rc=$?; st "MATHLIB rc=$rc"
[ "$rc" = 0 ] || die "host math build failed; no activation or verification"
check_tree
mkdir -p "$O/previous" python/mojolearn/.libs || die "cannot prepare activation"
for d in identical host; do
  # These generated directories contain no tracked source. Preserve them whole,
  # including per-binding overrides; only the complete staged set is installed.
  if [ -e "python/mojolearn/$d" ] || [ -L "python/mojolearn/$d" ]; then
    mv "python/mojolearn/$d" "$O/previous/$d" || die "cannot preserve previous $d set"
  fi
  cp -a "$O/artifacts/$d" "python/mojolearn/$d" || die "cannot activate $d set; previous bytes in $O/previous"
done
if [ -e "$MATHLIB" ] || [ -L "$MATHLIB" ]; then
  mv "$MATHLIB" "$O/previous/libMojolearnMath.so" || die "cannot preserve host math helper"
fi
cp -p "$O/artifacts/.libs/libMojolearnMath.so" "$MATHLIB" || die "cannot activate host math helper"
find "$O/artifacts" -type f -exec sha256sum {} \; > "$O/artifact-sha256.txt" || die "cannot record artifact hashes"
check_tree
export PYTHONPATH=$ROOT/python
(cd / && "$PY" -m mojolearn verify) > "$O/verify-identical.log" 2>&1
rc=$?; printf 'identical\t%s\n' "$rc" > "$O/verify-status.tsv"
st "VERIFY identical rc=$rc $(grep -m1 '^RESULT' "$O/verify-identical.log" | cut -c1-200)"
[ "$rc" = 0 ] || die "identical verification failed; no reference recording or optional jobs"
if [ "${BOX_RECORD:-1}" = 1 ]; then
  ONLY_CHANGED=table REPEATS=1 bash tools/record_identity_column.sh "$LABEL" "$O/column" > "$O/record.log" 2>&1
  rc=$?; printf '%s\n' "$rc" > "$O/record-exit-code"
  st "RECORD rc=$rc $(tail -n 1 "$O/record.log" | cut -c1-160)"
  [ "$rc" = 0 ] || die "reference recording failed"
else
  st "RECORD SKIP BOX_RECORD=0"
fi

# RunPod's marker gates the cloud-only checks; an ssh shell may not carry it, PID 1 does.
[ -n "${RUNPOD_POD_ID:-}" ] || export RUNPOD_POD_ID=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p')
[ -n "$RUNPOD_POD_ID" ] || unset RUNPOD_POD_ID
CORPUS=$PWD/training/corpus/enwik8/input.txt
HIGGS=/root/datasets/gbm-bench/higgs/higgs_speed.npz
# run <dir> <name> <seconds> cmd...: one logged step; rc 124 is a timeout. Never aborts the chain.
run() { local d=$1 n=$2 t=$3; shift 3; mkdir -p $d; timeout -k 30 $t "$@" > $d/$n.log 2>&1; local rc=$?
  printf '%s\t%s\n' "$n" "$rc" >> $d/status.tsv; st "$(basename $d) $n rc=$rc $(grep -v '^\s*$' $d/$n.log | tail -n 1 | cut -c1-160)"; return $rc; }

# 1. Physical multi-GPU checks (docs/multi_gpu/README.md "Cloud gates", the distributed workers of
# docs/plans/CORE_COMPLETION_HANDOFF_20261020.md, the byte-LM layer-pool two-GPU case owed since Oct 18).
# One device proves nothing here, so a one-GPU box records SKIP and runs none.
job_dist() {
  local D=$O/dist; mkdir -p $D
  [ "$NGPU" -ge 2 ] || { st "DIST SKIP gpus=$NGPU (needs 2+)"; return 0; }
  local be=cuda; [ $V = amd ] && be=hip
  ( cd / && run $D distributed-classical 1800 $PY $OLDPWD/tools/distributed_classical_check.py --devices 0,1 --out $D/classical.json )
  ( cd / && run $D parallel-cv 2400 $PY $OLDPWD/tools/parallel_cross_val_check.py --require-backend $be --devices 0,1 --out $D/cv )
  # Distributed Program workers (autograd): the tests skip unless two real device ids are named.
  pixi install -e test > $D/pixi-test.log 2>&1; st "DIST pixi test env rc=$?"
  MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical run $D distributed-program 1800 \
    pixi run -q -e test python -m pytest -q -p no:cacheprovider -rs --junitxml=$D/distributed-program.xml \
    python/mojolearn/tests/test_distributed_program.py python/mojolearn/tests/test_distributed_program_boundaries.py
  [ -n "${RUNPOD_POD_ID:-}" ] || { st "DIST cloud gates SKIP: no RUNPOD_POD_ID (they refuse off RunPod)"; return 0; }
  # The native two-GPU checks (each compiles here; RUNPOD-gated in main()).
  for c in byte_lm_layer_pool byte_lm_model_pool ordered_gradient gram_parallel dot_parallel gmm_parallel \
           iforest_parallel svm_parallel gp_parallel; do
    # The flags of tools/parallel_cloud_leg.sh's ordered-gradient phase.
    run $D native-$c 2400 pixi run -q mojo run --target-accelerator $ARCH -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
      -I . training/checks/${c}_check.mojo
  done
  [ -f $CORPUS ] || st "DIST corpus gates SKIP: $CORPUS not staged (tools/stage_from_r2.sh ... corpus/enwik8/input.txt)"
  local f n
  for f in tools/parallel_*_check.py tools/byte_lm_parallel_check.py; do
    n=$(basename $f _check.py)
    case $n in
      parallel_cross_val) continue ;;                       # ran above, both vendors
      parallel_accumulate_capacity|parallel_reference_capacity)
        [ "${BOX_DIST_CAPACITY:-0}" = 1 ] || { st "DIST $n SKIP (memory-capacity gate; BOX_DIST_CAPACITY=1 runs it)"; continue; } ;;
    esac
    local a=(--cloud --report $D/$n.json)
    grep -q "'--corpus'" $f && { [ -f $CORPUS ] || continue; a+=(--corpus $CORPUS); }
    grep -q "'--traces'" $f && a+=(--traces $D/$n-traces)
    case $n in
      parallel_training) for lane in mlp forest samba; do run $D $n-$lane 2400 $PY $f --cloud --lane $lane --corpus $CORPUS --report $D/$n-$lane.json; done
        run $D $n-samba-attention 2400 $PY $f --cloud --lane samba --attention-dropout --corpus $CORPUS --report $D/$n-samba-attention.json ;;
      parallel_accumulate_capacity) run $D $n 2400 $PY $f "${a[@]}" --devices 2 ;;
      *) run $D $n 2400 $PY $f "${a[@]}" ;;
    esac
  done
  st "DIST done: $(awk -F'\t' '$2==0{p++} $2!=0{f++} END{printf "pass=%d fail=%d", p, f}' $D/status.tsv)"
}

# 2. Captured-graph optimizer A/B (docs/plans/GRAPH_OPTIMIZER_RUNNER_20261020.md): the saved full workloads,
# baseline (optimize=False) vs candidate (optimize=True), ONE scored run per arm, from an installed copy of
# the package (the runner refuses a checkout path). A workload the Mac staged under /root/stage/graph-opt/
# <name>/ (the other vendor's retained bytes) is used as is, so `compare` sees the same workload on both.
job_graphopt() {
  local G=$O/graph-opt; mkdir -p $G
  local be=cuda; [ $V = amd ] && be=hip
  local VENV=$G/venv
  $PY -m venv --system-site-packages $VENV > $G/venv.log 2>&1; st "GRAPHOPT venv rc=$?"
  local VPY=$VENV/bin/python SP
  SP=$($VPY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  cp -a python/mojolearn $SP/mojolearn; st "GRAPHOPT installed copy $SP/mojolearn"
  # Per-member receipt (own file or the merged file that serves it), checked by `run` against each loaded __file__.
  env -u PYTHONPATH $VPY tools/bench_graph_optimizer.py provenance --package $SP/mojolearn --source-commit $SHA \
    --out $G/provenance.json > $G/provenance.log 2>&1
  st "GRAPHOPT provenance rc=$? $(tail -n 1 $G/provenance.log)"
  local name recipe prep req
  for name in program-run transformer-forward; do
    if [ -f /root/stage/graph-opt/$name/workload.json ]; then
      W=/root/stage/graph-opt/$name; st "GRAPHOPT $name staged workload $W"
    else
      W=$G/workload/$name
      [ $name = program-run ] && prep=prepare-control || prep=prepare-transformer
      run $G inputs-$name 900 env -u PYTHONPATH $VPY tools/prepare_graph_optimizer_inputs.py $name --out $G/inputs/$name
      run $G capture-$name 1800 env -u PYTHONPATH MOJOLEARN_NUMERIC_MODE=identical $VPY tools/bench_graph_optimizer.py $prep \
        --recipe $G/inputs/$name/recipe.json --out $W
    fi
    req=(); [ $name = transformer-forward ] && req=(--require-pass permutation_composition --require-pass dead_intermediates)
    run $G ab-$name 2400 env -u PYTHONPATH MOJOLEARN_NUMERIC_MODE=identical $VPY tools/bench_graph_optimizer.py run \
      --vendor $be --workload $W/workload.json --binary-provenance $G/provenance.json --out $G/$V-$name --rounds 1 --retain-local "${req[@]}"
    $VPY - $G/$V-$name/result.json $name $V >> $G/ab-table.tsv 2>> $G/ab-table.err <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); a = r.get("arms", {})
ms = lambda arm: a.get(arm, {}).get("rounds", [{}])[0].get("whole_operation_ms", float("nan"))
ratio = (r.get("candidate_over_baseline") or [float("nan")])[0]
print("\t".join([sys.argv[3], sys.argv[2], "%.3f" % ms("baseline"), "%.3f" % ms("candidate"), "%.4f" % ratio,
                 ",".join(r.get("active_passes", [])) or "-", r.get("status", "?")]))
EOF
    st "GRAPHOPT $(tail -n 1 $G/ab-table.tsv)"
  done
  st "GRAPHOPT table: vendor workload baseline_ms candidate_ms candidate/baseline passes status -> $G/ab-table.tsv"
}

# 3. The native NVIDIA witness the PTX column cannot give itself (CORE_COMPLETION_HANDOFF_20261020.md item 4;
# ~/mojolearn-evidence/HANDOFF_2026-10-20_ptx-packaging.json): the four retained PTX discrepancies
# (sequence-adamax train, sequence-nadam train, sequence-lr-schedulers train+batch) and the four weighted-tree
# batch probes (et-clf-weighted, et-reg-weighted, gbdt-class-weights, gbdt-multiclass-offgrid), on the
# native sm_NN set: an admissible column record of exactly these lanes plus the verifier's verdict.
job_witness() {
  [ $V = nvidia ] || { st "WITNESS SKIP: NVIDIA only"; return 0; }
  local X=$O/witness; mkdir -p $X
  local L=sequence-adamax,sequence-nadam,sequence-lr-schedulers,et-clf-weighted,et-reg-weighted,gbdt-class-weights,gbdt-multiclass-offgrid
  LANES=$L REPEATS=1 run $X record 5400 bash tools/record_identity_column.sh $LABEL $X/column
  ( cd / && MOJOLEARN_NUMERIC_MODE=identical run $X verify-neural 3600 $PY -m mojolearn verify --neural-training --all \
      --fixtures base --lanes sequence-adamax,sequence-nadam,sequence-lr-schedulers --json )
  ( cd / && MOJOLEARN_NUMERIC_MODE=identical run $X verify-trees 3600 $PY -m mojolearn verify --all --fixtures base \
      --lanes et-clf-weighted,et-reg-weighted,gbdt-class-weights,gbdt-multiclass-offgrid --json )
  # One row per lane/part: the native value next to the 0.8.55 PTX value and the old reference.
  $PY - $X/verify-neural.log $X/verify-trees.log > $X/witness.tsv 2> $X/witness.err <<'EOF'
import json, sys
ptx = {("sequence-adamax", "train"): ("26cc2c2648fa8bb2", "f5d88e3cf2ab67ee"),
       ("sequence-nadam", "train"): ("c3e6f1be4f0f537f", "c84124a296495f2f"),
       ("sequence-lr-schedulers", "batch"): ("920e438adb20c152", "5058b755ea82d792"),
       ("sequence-lr-schedulers", "train"): ("fc55dad78a22e94f", "ab6a6ae5537bfe43")}
want = {"sequence-adamax", "sequence-nadam", "sequence-lr-schedulers",
        "et-clf-weighted", "et-reg-weighted", "gbdt-class-weights", "gbdt-multiclass-offgrid"}
rows = {}
def walk(x):
    if isinstance(x, dict):
        if x.get("lane") in want and x.get("part"):
            key = (x["lane"], x["part"], x.get("fixture", "base"))
            rows.setdefault(key, (str(x.get("state", "?")), str(x.get("value", "-")), str(x.get("reference", "-"))))
        for v in x.values(): walk(v)
    elif isinstance(x, list):
        for v in x: walk(v)
for path in sys.argv[1:]:
    text = open(path).read(); start = text.find("{")
    try: walk(json.loads(text[start:]))
    except Exception as e: print("unparsed", path, repr(e)[:120], file=sys.stderr)
print("lane\tpart\tfixture\tnative_state\tnative_value\ttable_reference\tptx_0855\tnative_eq_ptx")
for (lane, part, fx), (state, value, ref) in sorted(rows.items()):
    p = ptx.get((lane, part), ("-", "-"))[0]
    print("\t".join([lane, part, fx, state, value, ref, p, "-" if p == "-" else ("yes" if value.startswith(p) or p.startswith(value) else "NO")]))
EOF
  st "WITNESS rows=$(($(wc -l < $X/witness.tsv)-1)) native==ptx: $(awk -F'\t' 'NR>1 && $8=="yes"{y++} NR>1 && $8=="NO"{n++} END{printf "yes=%d NO=%d", y, n}' $X/witness.tsv)"
}

# 4. NVIDIA STAGE64 A/B (sequence/ops.mojo GEMM_STAGE_V2: default off since 2026-10-19 on the AMD numbers;
# NVIDIA never measured). Per-binding x_sequence builds into python/mojolearn/identical, where a member's own
# file shadows the merged one; both arms build the same way, one run per arm, then the own file is removed so
# the merged binding serves again. Same bits is the contract (one fold chain): a digest that moves is a bug.
job_stage64() {
  [ $V = nvidia ] || { st "STAGE64 SKIP: NVIDIA only (AMD measured 2026-10-19)"; return 0; }
  local S=$O/stage64; mkdir -p $S
  [ -f $HIGGS ] || { st "STAGE64 SKIP: $HIGGS not staged (tools/stage_from_r2.sh ... gbm-bench/higgs/higgs_speed.npz)"; return 0; }
  local own=python/mojolearn/identical/_mojolearn_x_sequence.so pair tag defs algos
  for pair in "default:" "stage64:MOJOLEARN_IDN_GEMM_STAGE64=1" "stage64-rec:MOJOLEARN_IDN_GEMM_STAGE64=1,MOJOLEARN_SEQ_REC_STAGE64=1"; do
    tag=${pair%%:*}; defs=${pair#*:}; rm -f $own
    MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_DEFINES=$defs \
      run $S build-$tag 2400 pixi run -q sh bindings/build_x_sequence.sh || continue
    for algos in lstm,gru,rnn mlp; do
      MOJOLEARN_NUMERIC_MODE=identical run $S race-$tag-${algos//,/-} 2400 $PY tools/sequence_speed.py --algos $algos --data $HIGGS \
        --rows 1000000 --out $S/race-$tag-${algos//,/-}.json
      $PY -c "import json,sys; d=json.load(open(sys.argv[1])); [print('$tag\t%s\t%.4f\t%s' % (r['algo'], r['fit_s'], r['digest'])) for r in d['records']]" \
        $S/race-$tag-${algos//,/-}.json >> $S/ab.tsv 2>> $S/ab.err
    done
  done
  rm -f $own; st "STAGE64 own x_sequence removed; merged serves again"
  $PY - $S/ab.tsv > $S/verdict.tsv 2>> $S/ab.err <<'EOF'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
base = {a: (float(s), d) for t, a, s, d in rows if t == "default"}
print("arm\talgo\tdefault_s\tarm_s\tarm/default\tsame_bits")
for t, a, s, d in rows:
    if t != "default" and a in base:
        print("%s\t%s\t%.4f\t%.4f\t%.4f\t%s" % (t, a, base[a][0], float(s), float(s) / base[a][0], "yes" if d == base[a][1] else "NO"))
EOF
  while read -r line; do st "STAGE64 $line"; done < <(tail -n +2 $S/verdict.tsv)
}

for j in ${BOX_JOBS:-}; do
  case $j in dist|graphopt|witness|stage64) st "JOB $j start"; job_$j; st "JOB $j end";; *) st "JOB $j unknown; skipped";; esac
done
st "qualification complete; optional job results remain separate in their status.tsv files"
