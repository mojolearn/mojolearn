#!/bin/bash
# tools/py_consolidated/job.sh: THE ONE CHECK of the twelve merged py-* lanes
# (docs/lanes/progress/py-consolidated.md), one NVIDIA queue job on one shared
# pod, the pod's x86 CPU as the CPU column. ONE tree (the pod disks are shared):
#   base  `git apply -R tools/py_consolidated/base.patch` (lane/apple2-merged's
#         code), every selected lane's GPU and CPU arm once (tools/py_consolidated/check.py
#         arms: build once, GPU == CPU per lane, the py-bugs probe per column),
#         the py-lm witness on both columns, then a snapshot of the base package
#         (python/mojolearn with its built .so) for the timing pass;
#   head  `git apply` back; only bindings whose sources moved rebuild; the same.
#   cross base vs head per lane and column, part by part; witness base vs head.
#   tests the relevant pytest files on the head tree.
#   ab    the lanes' in-build reference arms (same build, before and after, with
#         the answers compared): fsum/labels micro, arena ranges 0/1/1/0, metrics
#         epilogues, IterativeImputer/Calibrated routes, model selection routes,
#         decomp native vs Python arms, SVC Mojo twins, CNN epoch entry.
#   timing one small interleaved pass, base snapshot vs head tree, GPU in the
#         order base head head base, CPU base then head.
#   sabotage one arm per NEW device path that has a patch (py-dn-ann resident
#         index, py-dn-kern fused chains), inside this job.
# The patch is re-applied on every exit path; build outputs and the base
# snapshot are deleted at the end. PHASES=a,b,... runs a subset.
set -u
T=/root/mojolearn-py-consolidated
EV=${EV:-/root/ev-py-consolidated/$(date -u +%m%d-%H%M)}
PATCH=$T/tools/py_consolidated/base.patch
PHASES=${PHASES:-base,head,cross,tests,sabotage,ab,timing}
has() { case ",$PHASES," in *",$1,"*) return 0 ;; esac; return 1; }
mkdir -p "$EV"; touch "$EV/.start"
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-6}
# the py-decomp-nbrs Kit sends a call to the GPU executor only above this many
# elements; 1 makes the lanes' small fixtures really take the device path
export MOJOLEARN_XD_RES_DEV_MIN=1
export MOJOLEARN_LANE_CHECK_ARM_TIMEOUT=${MOJOLEARN_LANE_CHECK_ARM_TIMEOUT:-1800}
cd "$T" || exit 1
LANES=$(grep -v '^#' tools/py_consolidated/lanes.txt | grep . | paste -sd, -)
echo "$(date -u +%FT%TZ) $(hostname) out $EV; $(echo "$LANES" | tr ',' '\n' | wc -l) lanes"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; lscpu | grep -m1 'Model name'; df -h /root | tail -1
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
$PIXI install -e default > "$EV/pixi.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$EV/pixi.log"; exit 1; }
$PIXI install -e test > "$EV/pixi_test.log" 2>&1 || echo "PIXI TEST ENV INSTALL FAIL"
# the drivers run from copies outside the tree (the base patch does not carry them)
cp tools/py_bugs/probe.py tools/py_lm/witness.py tools/py_dn_ann/bench_ann.py \
   tools/py_misc/metrics_time.py tools/py_consolidated/*.py "$EV/"
P="$PIXI run -e default python -u"

host_env() {  # host_env <python dir>: the CPU column's variables for that package
  echo "MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$1/mojolearn/host MOJOLEARN_FOREST_HOST_BINARY=$1/mojolearn/host/_mojolearn_forest_host.so MOJOLEARN_BYTE_LM_HOST_BINARY=$1/mojolearn/host/_mojolearn_byte_lm_host.so"
}
col_env() {  # col_env <python dir> <gpu|cpu>
  local e="PYTHONPATH=$1 MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1"
  [ "$2" = cpu ] && e="$e $(host_env "$1")"
  echo "$e"
}
lm_build() {  # the bindings the py-lm witness loads beyond the lanes' own
  $P - "$EV/lm_build_$1.log" <<'PY'
import sys; sys.path.insert(0, "tools")
import algos_lane_check as a
need = {"_mojolearn_transformer", "_mojolearn_training", "_mojolearn_mamba", "_mojolearn_neural_host",
        "_mojolearn_transformer_host", "_mojolearn_mamba_host", "_mojolearn_training_host"}
need = {b for b in need if (a.ROOT / "bindings" / a.script_for(b)).is_file()}
a.ensure_built(sorted(need), sys.argv[1])
print("LM BUILT", sorted(need))
PY
}
witness() {  # witness <tag>
  for col in gpu cpu; do
    env $(col_env "$T/python" $col) $PIXI run -e default python -u "$EV/witness.py" --device $col \
      --out "$EV/witness.$1.$col.json" --timing > "$EV/witness.$1.$col.log" 2>&1
    echo "witness $1 $col: exit $? $(tail -1 "$EV/witness.$1.$col.log")"
  done
}

restore() { if git apply --check "$PATCH" 2>/dev/null; then git apply "$PATCH" && echo "restored the head tree"; fi; }
if has base; then
  git apply --check -R "$PATCH" || { echo "PATCH DOES NOT REVERSE"; exit 1; }
  trap restore EXIT
  git apply -R "$PATCH" && echo "== BASE: the merged lanes' code patch reversed ($(wc -l < tools/py_consolidated/base.paths) paths)"
  $P "$EV/check.py" arms --tree "$T" --out "$EV/base" --lanes "$LANES" 2>&1 | tee "$EV/base.out" | grep -v '^\s*$' | tail -200
  lm_build base 2>&1 | tail -2
  witness base
  rm -rf "$EV/base_py"; mkdir -p "$EV/base_py"
  (cd python && tar --exclude='__pycache__' -cf - mojolearn) | tar -xf - -C "$EV/base_py"
  echo "base package snapshot: $(du -sh "$EV/base_py" | cut -f1)"
  restore; trap - EXIT
fi
git apply --check -R "$PATCH" >/dev/null 2>&1 || { echo "THE TREE IS NOT THE HEAD TREE"; exit 1; }

if has head; then
  echo "== HEAD"
  $P "$EV/check.py" arms --tree "$T" --out "$EV/new" --lanes "$LANES" 2>&1 | tee "$EV/new.out" | grep -v '^\s*$' | tail -200
  lm_build head 2>&1 | tail -2
  witness head
fi

if has cross; then
  echo "== CROSS base vs head, per lane and column"
  $P "$EV/check.py" cross --base "$EV/base" --new "$EV/new" --lanes "$LANES" 2>&1 | tee "$EV/cross.txt"
  for col in gpu cpu; do
    echo "== witness $col base vs head (GPT-3 guard: causal token streams, samba and byte LM losses/params)"
    $P "$EV/witness.py" --compare "$EV/witness.base.$col.json" "$EV/witness.head.$col.json" 2>&1 | tee "$EV/witness_cmp.$col.txt" | tail -25
  done
  echo "== witness head gpu vs cpu (common cells)"
  $P "$EV/witness.py" --common --compare "$EV/witness.head.gpu.json" "$EV/witness.head.cpu.json" 2>&1 | tail -4
fi

if has tests; then
  echo "== PYTEST (head)"
  D=python/mojolearn/tests
  TESTS=$(ls $D/test_arena_ranges.py $D/test_portable_math_fast.py $D/test_labels_native.py $D/test_hotpath_native.py \
    $D/test_x_metrics_repeat.py $D/test_x_metrics_sanity.py $D/test_x_prep_*.py $D/test_model_selection_numpy_free.py \
    $D/test_py_bugs.py $D/test_x_sequence_*.py $D/test_holtwinters_*.py $D/test_parallel_forecasting.py \
    $D/test_arima_surface.py $D/test_byte_lm_host.py $D/test_byte_lm_host_trainer.py $D/test_byte_lm_session.py \
    $D/test_byte_lm_surface.py $D/test_byte_lm_trainer_logits.py $D/test_samba_surface.py $D/test_samba_attention_wiring.py \
    $D/test_transformer_surface.py $D/test_verify_causal_lm.py $D/test_causal_lm_cli.py $D/test_parallel_causal_lm.py \
    $D/test_native_dense_coo.py $D/test_spectral_embedding.py $D/test_spectral_predict.py $D/test_spectral_no_dependencies.py \
    $D/test_svc_probability.py $D/test_svc_multiclass.py $D/test_svc_poly.py $D/test_host_model_svm.py \
    $D/test_x_decomp_repeat.py $D/test_ivf_surface.py $D/test_distributed_ivf.py $D/test_x_ann_repeat.py \
    $D/test_cpu_training_par_classical.py $D/test_cpu_training_embedding_ivf.py $D/test_x_cnn_trainer.py \
    $D/test_x_cnn_repeat.py $D/test_classification_metrics.py $D/test_regression_metrics.py \
    $D/test_parallel_model_selection.py $D/test_host_surface.py $D/test_numpy_free_core.py \
    $D/test_numpy_free_features.py 2>/dev/null)
  MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$T/python" timeout 3600 $PIXI run -e test python -m pytest -q \
    -p no:cacheprovider $TESTS > "$EV/pytest.gpu.log" 2>&1; echo "pytest (GPU default) exit $?"; tail -25 "$EV/pytest.gpu.log"
  echo "== pytest test_py_bugs + test_portable_math_fast on the CPU column"
  env $(col_env "$T/python" cpu) timeout 1200 $PIXI run -e test python -m pytest -q -p no:cacheprovider \
    $D/test_py_bugs.py $D/test_portable_math_fast.py $D/test_native_dense_coo.py > "$EV/pytest.cpu.log" 2>&1
  echo "pytest (CPU) exit $?"; tail -8 "$EV/pytest.cpu.log"
fi

if has sabotage; then
  echo "== SABOTAGE, one arm per new device path (clean AGREE, sabotaged DISAGREE, restored AGREE)"
  PIXI=$PIXI sh tools/algos_lane_check.sh ivf,x-ann-ivf-pq --sabotage x_ann/checks/sabotage/resident_index_py_dn_ann.patch \
    --out "$EV/sab_ann" 2>&1 | grep -E 'RESULT|CLEAN:|SABOTAGED:|RESTORED:' | sed 's/^/SAB ann /'
  PIXI=$PIXI sh tools/algos_lane_check.sh x-neighbors-kpca --sabotage x_neighbors/checks/sabotage/fused_chain_device.patch \
    --out "$EV/sab_kern" 2>&1 | grep -E 'RESULT|CLEAN:|SABOTAGED:|RESTORED:' | sed 's/^/SAB kern /'
fi

if has ab; then
  echo "== IN-BUILD REFERENCE ARMS (head build; each prints before, after and whether the answers agree)"
  G=$(col_env "$T/python" gpu); C=$(col_env "$T/python" cpu)
  env $G $P bench/py_shared_micro.py --n 1000000 --reps 3 2>&1 | grep -E 'PYSHARED|Error'
  for arm in 0 1 1 0; do
    env $G MOJOLEARN_ARENA_RANGES=$arm $P bench/x_prep_speed.py --rows 1000000 --reps 1 2>&1 | grep -E '^XPSPEED|Error' | sed "s/^/RANGES=$arm /"
    env $G MOJOLEARN_ARENA_RANGES=$arm $P bench/x_metrics_speed.py --rows 1000000 --reps 1 2>&1 | grep -E '^XMSPEED|Error' | sed "s/^/RANGES=$arm /"
  done
  env $G $P "$EV/metrics_time.py" gpu 2>&1 | tail -20 | sed 's/^/METRICS gpu /'
  env $C $P "$EV/metrics_time.py" cpu 2>&1 | tail -20 | sed 's/^/METRICS cpu /'
  (cd "$T" && env $G $P tools/py_misc_prep/ab.py bits 2>&1 | tail -6 | sed 's/^/PREP bits gpu /'
            env $C $P tools/py_misc_prep/ab.py bits 2>&1 | tail -6 | sed 's/^/PREP bits cpu /'
            env $G $P tools/py_misc_prep/ab.py time 2>&1 | tail -10 | sed 's/^/PREP time gpu /'
            env $C $P tools/py_misc_prep/ab.py time 2>&1 | tail -10 | sed 's/^/PREP time cpu /')
  env $G $P tools/py_misc_msel/check.py equal 2>&1 | tail -6 | sed 's/^/MSEL equal gpu /'
  env $C $P tools/py_misc_msel/check.py equal 2>&1 | tail -6 | sed 's/^/MSEL equal cpu /'
  env $G $P "$EV/ab_arms.py" 2>&1 | grep -E '^ARM'
  env $C $P "$EV/ab_arms.py" 2>&1 | grep -E '^ARM'
  env $G $P "$EV/svc_equal.py" 2>&1 | grep -E 'FAIL|EQUAL RESULT' | tail -6 | sed 's/^/SVC_EQUAL gpu /'
  env $G $P tools/py_misc/cnn_epoch.py "$T" 20000 1 2>&1 | grep -E 'PYMISC-(SAME|TIME)' | sed 's/^/CNN gpu /'
  env $C $P tools/py_misc/cnn_epoch.py "$T" 2000 1 2>&1 | grep -E 'PYMISC-(SAME|TIME)' | sed 's/^/CNN cpu /'
fi

if has timing; then
  echo "== TIMING, base snapshot vs head tree (GPU: base head head base; CPU: base head)"
  [ -d "$EV/base_py/mojolearn" ] || echo "NO BASE SNAPSHOT"
  bench() {  # bench <tree tag> <gpu|cpu> <script> [ENV=V ...]
    local dir=$T/python; [ "$1" = base ] && dir=$EV/base_py
    local tag=$1 col=$2 s=$3; shift 3
    env $(col_env "$dir" $col) "$@" $PIXI run -e default python -u "$EV/$s" 2>&1 \
      | grep -E '^(TIME|BENCH|probA)|FAILED|ERROR' | sed "s/^/T $tag $(basename $s .py) /"
  }
  for tag in base head head base; do
    bench $tag gpu timing.py
    bench $tag gpu kern_bench.py SCALE=cpu
    bench $tag gpu svc_bench.py NB=20000 NM=5000 K=10 NQ=50000
    bench $tag gpu bench_decomp.py ONLY=mcd-20k,lda-online-20kx500,lda-batch-100kx1000,mds-nm-1500
    bench $tag gpu bench_ann.py SIZE=200000 DIM=128 BATCHES=10 BQ=500
  done
  for tag in base head; do
    bench $tag cpu timing.py
    bench $tag cpu kern_bench.py SCALE=cpu
    bench $tag cpu svc_bench.py NB=5000 NM=2000 K=10 NQ=10000
    bench $tag cpu bench_decomp.py ONLY=mcd-20k,lda-online-20kx500
    bench $tag cpu bench_ann.py SIZE=200000 DIM=128 BATCHES=10 BQ=500 WHAT=flat,dist DEVICES=0
  done
fi


# small footprint: the build outputs this job made and the base snapshot
find python/mojolearn -newer "$EV/.start" \( -name '*.so' -o -name '*.dylib' -o -name '*.lanecheck-stamp' -o -name '*.stamp.json' \) -delete 2>/dev/null
rm -rf "$EV/base_py"
du -sh "$EV" | tail -1; df -h /root | tail -1
echo "JOB END $(date -u +%FT%TZ)"
