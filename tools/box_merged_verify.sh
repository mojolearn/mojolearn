#!/bin/bash
# tools/box_merged_verify.sh nvidia|amd <sha> -- the post-build GPU check of one
# commit on one rented box (2026-10-21): the merged bindings built from the tree
# (fast, identical, host), `python -m mojolearn verify` in identical mode, then
# the incremental reference-column record of the lanes whose source closure
# changed (ONLY_CHANGED=table tools/record_identity_column.sh), so a new verify
# lane gets its NVIDIA/AMD digests without re-fitting every row. Everything
# logs under /root/out; the Mac pulls that folder, so a teardown loses nothing.
#
# Then the owed GPU evidence (2026-10-21, docs/plans/BOX_JOBS_20261021.md), each
# a function guarded by what the box has, each step `st`-logged, none aborting
# the chain. BOX_JOBS (default "dist graphopt witness stage64") picks them:
#   dist      physical multi-GPU checks, only with 2+ GPUs on the box
#   graphopt  captured-graph optimizer A/B on the saved full workloads (both vendors)
#   witness   native NVIDIA witness: four PTX discrepancies + four tree batch probes
#   stage64   NVIDIA MOJOLEARN_IDN_GEMM_STAGE64 A/B (MLP + recurrent, HIGGS)
# The Mac stages the data first (tools/stage_from_r2.sh): corpus/enwik8/input.txt
# for dist, gbm-bench/higgs/higgs_speed.npz for stage64.
set -u; V=$1; SHA=$2; O=/root/out; mkdir -p $O; exec >> $O/job.log 2>&1
st() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a $O/status.txt; }
st "START $V $SHA"
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
command -v pixi >/dev/null || { curl -fsSL https://pixi.sh/install.sh | bash; export PATH=/root/.pixi/bin:$PATH; }
cd /root/mojolearn 2>/dev/null || { cd /root && git clone -q https://github.com/mojolearn/mojolearn.git mojolearn && cd mojolearn; }
git fetch -q origin && git reset -q --hard && git checkout -q -f --detach $SHA
case "$(git rev-parse HEAD)" in $SHA*) st "tree on $SHA";; *) st "TREE NOT ON $SHA; stop"; st DONE; exit 3;; esac
pixi install > $O/pixi.log 2>&1; st "pixi rc=$?"
PY=$(pixi run -q python -c "import sys; print(sys.executable)")
if [ $V = amd ]; then ARCHENV="MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_TARGET_COLUMN=amd"; else
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
  ARCHENV="MOJOLEARN_GPU_ARCHS=sm_$CC MOJOLEARN_TARGET_COLUMN=nvidia"
fi
st "arch env: $ARCHENV ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)$(rocm-smi --showproductname 2>/dev/null | grep -m1 -o 'MI3[0-9A-Z]*'))"
find python/mojolearn -name "*.so" -delete
command -v /usr/bin/time >/dev/null || apt-get install -y -qq time > /dev/null 2>&1
mb() { tag=$1; shift; /usr/bin/time -v env "$@" pixi run -q sh bindings/merged_build.sh ${tag%%-*} > $O/build-$tag.log 2> $O/build-$tag.time; rc=$?
  st "BUILD $tag rc=$rc seconds=$(grep -oP 'Elapsed \(wall clock\) time.*: \K.*' $O/build-$tag.time) peak_rss_gb=$(awk '/Maximum resident/ {printf "%.1f", $6/1048576}' $O/build-$tag.time) errors=$(grep -c ': error:' $O/build-$tag.log)"; }
mb gpu-identical $ARCHENV MOJOLEARN_NUMERIC_MODE=identical &
mb gpu-fast $ARCHENV MOJOLEARN_NUMERIC_MODE=fast &
mb host MOJOLEARN_TARGET_COLUMN=cpu &
wait
find python/mojolearn -name "_mojolearn_native*.so" -exec ls -l {} \; | awk '{print $5, $9}' > $O/merged-files.txt; st "merged files: $(wc -l < $O/merged-files.txt)"
export PYTHONPATH=$PWD/python
( cd / && MOJOLEARN_NUMERIC_MODE=identical $PY -m mojolearn verify > $O/verify-identical.log 2>&1; st "VERIFY identical rc=$? $(grep -m1 '^RESULT' $O/verify-identical.log | cut -c1-200)" )
( cd / && MOJOLEARN_NUMERIC_MODE=fast $PY -m mojolearn verify > $O/verify-fast.log 2>&1; st "VERIFY fast rc=$? $(grep -m1 '^RESULT' $O/verify-fast.log | cut -c1-200)" )
# The record's label is nvidia-<gpu>-<arch> or amd-<gpu>-<arch> (record_identity_column.sh refuses a bare
# vendor), the table's existing spelling: nvidia-l40s-sm_89, amd-mi325x-gfx942.
if [ $V = amd ]; then
  ARCH=gfx942; GPU=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -o 'MI3[0-9A-Z]*' | tr 'A-Z' 'a-z'); LABEL=amd-${GPU:-mi325x}-$ARCH
  NGPU=$(rocm-smi --showid 2>/dev/null | grep -o 'GPU\[[0-9]*\]' | sort -u | wc -l)
else
  ARCH=sm_$CC; GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed -E 's/^NVIDIA //; s/^(GeForce|RTX|Tesla) //g' | awk '{print tolower($1)}')
  LABEL=nvidia-${GPU:-gpu}-$ARCH; NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
fi
st "label $LABEL gpus=$NGPU"
ONLY_CHANGED=table REPEATS=1 bash tools/record_identity_column.sh $LABEL $O/column > $O/record.log 2>&1; st "RECORD rc=$? $(tail -n 1 $O/record.log | cut -c1-160) $(ls $O/column 2>/dev/null | tr '\n' ' ' | cut -c1-200)"

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
  local VENV=/root/graph-opt-venv
  rm -rf $VENV; $PY -m venv --system-site-packages $VENV > $G/venv.log 2>&1; st "GRAPHOPT venv rc=$?"
  local VPY=$VENV/bin/python SP
  SP=$($VPY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  cp -a python/mojolearn $SP/mojolearn; st "GRAPHOPT installed copy $SP/mojolearn"
  $VPY - $G/provenance.json $SHA $SP/mojolearn > $G/provenance.log 2>&1 <<'EOF'
import hashlib, json, pathlib, sys
out, sha, pkg = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
rows = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(pkg.rglob("*.so"))]
pathlib.Path(out).write_text(json.dumps({"source_commit": sha, "built_by": "tools/box_merged_verify.sh merged builds", "binaries": rows}, indent=1) + "\n")
print(len(rows), "binaries")
EOF
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

for j in ${BOX_JOBS:-dist graphopt witness stage64}; do
  case $j in dist|graphopt|witness|stage64) st "JOB $j start"; job_$j; st "JOB $j end";; *) st "JOB $j unknown; skipped";; esac
done
st DONE
