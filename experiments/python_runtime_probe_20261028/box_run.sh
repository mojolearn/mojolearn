# Box command for tools/runpod_cpu_leg.sh (python-audit-runtime lane, 2026-10-28).
# Builds the CPU host development set (O0, identical, all allocated cores),
# activates what built (partially if needed), then runs the probe.
# Everything lands in $LEG_OUT, which the leg fetches to the Mac.
set -u
T0=$(date +%s)
O="$LEG_OUT"
mkdir -p "$O"
PY="$PWD/.pixi/envs/default/bin/python3"
[ -x "$PY" ] || PY=python3
"$PY" - > "$O/resources.json" <<'PYEOF'
import json, os, sys
sys.path.insert(0, 'tools')
from build_sizing import box_resources
cores, avail = box_resources()
print(json.dumps(dict(allocated_cores=cores, affinity=len(os.sched_getaffinity(0)),
                      available_bytes=avail, nproc_host=os.cpu_count())))
PYEOF
CORES=$("$PY" -c "import json;print(json.load(open('$O/resources.json'))['allocated_cores'])")
AVAIL_GIB=$("$PY" -c "import json;print(int(json.load(open('$O/resources.json'))['available_bytes']/2**30))")
PER_JOB_GIB=12
# Attempt 1 (2026-10-28, pod 5tuclj99stxbe8): at 7 GiB the RSS guard stopped
# tokenizer/runtime_classical/x_sequence (exit 124), and three groups do not
# lower at O0 (llvm.masked.gather alignment must be a constant), so those
# three build at the default optimization level in a second cache.
DEFAULT_OPT="_mojolearn_dev_x_decomp_host _mojolearn_dev_estimators_host _mojolearn_dev_x_linear_host"
NB=3; WB=3
CORES_A=$(( CORES - NB * WB )); [ "$CORES_A" -ge 4 ] || CORES_A=4
J_MEM=$(( (AVAIL_GIB * 8 / 10 - NB * PER_JOB_GIB) / PER_JOB_GIB )); [ "$J_MEM" -ge 1 ] || J_MEM=1
W=$(( CORES_A / J_MEM )); [ "$W" -ge 2 ] || W=2
J=$(( CORES_A / W )); [ "$J" -ge 1 ] || J=1
echo "cores=$CORES avail_gib=$AVAIL_GIB O0: jobs=$J workers=$W; default-opt: jobs=$NB workers=$WB; per_job_gib=$PER_JOB_GIB" | tee "$O/build_settings.txt"
git rev-parse HEAD > "$O/source_commit.txt" 2>/dev/null || true
"$PWD/.pixi/envs/default/bin/mojo" --version > "$O/compiler_version.txt" 2>&1
CACHE_A=/root/.cache/mojolearn-development
CACHE_B=/root/.cache/mojolearn-development-defaultopt
"$PY" tools/development_build.py plan --sets host --json "$O/plan.json" > "$O/plan.log" 2>&1
GROUPS_A=$("$PY" -c "
import json;skip=set('$DEFAULT_OPT'.split())
print(' '.join('--group '+g['stem'] for g in json.load(open('$O/plan.json'))['groups'] if g['stem'] not in skip))")
GROUPS_B=$(for g in $DEFAULT_OPT; do printf -- '--group %s ' "$g"; done)
B0=$(date +%s)
( MOJOLEARN_COMPILE_JOBS=$W MOJOLEARN_MOJO_BUILD_FLAGS="--optimization-level 0" PYTHONUNBUFFERED=1 \
  timeout -k 30 2280 "$PY" tools/development_build.py build --sets host $GROUPS_A --cache "$CACHE_A" --jobs "$J" \
    --memory-per-job-gib "$PER_JOB_GIB" --evidence "$O/devbuild" > "$O/devbuild.log" 2>&1; echo $? > "$O/devbuild.rc" ) &
PA=$!
sleep 90  # let attempt A's generator finish before B's runs
( MOJOLEARN_COMPILE_JOBS=$WB PYTHONUNBUFFERED=1 \
  timeout -k 30 $(( 2280 - 90 )) "$PY" tools/development_build.py build --sets host $GROUPS_B --cache "$CACHE_B" --jobs "$NB" \
    --memory-per-job-gib "$PER_JOB_GIB" --evidence "$O/devbuild-defaultopt" > "$O/devbuild-defaultopt.log" 2>&1; echo $? > "$O/devbuild-defaultopt.rc" ) &
PB=$!
wait $PA; wait $PB
echo "build rcA=$(cat $O/devbuild.rc) rcB=$(cat $O/devbuild-defaultopt.rc) seconds=$(( $(date +%s) - B0 ))" | tee -a "$O/build_settings.txt"
PROBE_PLANS="[{\"cache\": \"$CACHE_A\", \"flags\": \"--optimization-level 0\", \"jobs\": $W}, {\"cache\": \"$CACHE_B\", \"flags\": \"\", \"jobs\": $WB}]" \
  MOJOLEARN_COMPILE_JOBS=$W "$PY" experiments/python_runtime_probe_20261028/partial_activate.py "$O/partial_activation.json" \
    > "$O/partial_activate.log" 2>&1
echo "activation rc=$? $(tail -n 1 $O/partial_activate.log)" | tee -a "$O/build_settings.txt"
# Binary manifest (the binaries stay in the box cache; their hashes come home).
( cd python/mojolearn && find host -type f \( -name '*.so' -o -name '*.json' -o -name '*.dylib' \) -print0 \
    | xargs -0 sha256sum ) > "$O/binary_manifest.sha256" 2>/dev/null
find python/mojolearn -maxdepth 3 -name 'libMojolearnMath*' -print0 | xargs -0 -r sha256sum >> "$O/binary_manifest.sha256"
cp python/mojolearn/host/_mojolearn_merged.json "$O/host_index.json" 2>/dev/null

# Import sanity, then the probe.
MOJOLEARN_VENDOR=cpu PYTHONPATH=python "$PY" -c "import mojolearn,mojolearn._backend as b;print('vendor',mojolearn.vendor());print('missing',len(b._MISSING),sorted(b._MISSING)[:60])" \
    > "$O/import_check.log" 2>&1
echo "import rc=$?" | tee -a "$O/build_settings.txt"
ELAPSED=$(( $(date +%s) - T0 ))
LEFT=$(( 50 * 60 - ELAPSED ))
DEADLINE=$(( LEFT - 330 )); [ "$DEADLINE" -ge 60 ] || DEADLINE=60
echo "probe deadline=${DEADLINE}s workers=$CORES" | tee -a "$O/build_settings.txt"
MOJOLEARN_VENDOR=cpu "$PY" tools/python_runtime_probe.py --out "$O/probe" --workers "$CORES" \
    --group-timeout 300 --call-timeout 60 --deadline-seconds "$DEADLINE" > "$O/probe.log" 2>&1
PRC=$?
echo "$PRC" > "$O/probe.rc"
echo "probe rc=$PRC total_seconds=$(( $(date +%s) - T0 ))" | tee -a "$O/build_settings.txt"
exit 0
