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
PER_JOB_GIB=7
J_MEM=$(( AVAIL_GIB * 8 / 10 / PER_JOB_GIB )); [ "$J_MEM" -ge 1 ] || J_MEM=1
W=$(( CORES / J_MEM )); [ "$W" -ge 2 ] || W=2
J=$(( CORES / W )); [ "$J" -ge 1 ] || J=1
export MOJOLEARN_COMPILE_JOBS="$W"
export MOJOLEARN_MOJO_BUILD_FLAGS="--optimization-level 0"
echo "cores=$CORES avail_gib=$AVAIL_GIB jobs=$J compiler_workers=$W per_job_gib=$PER_JOB_GIB opt=O0" | tee "$O/build_settings.txt"
git rev-parse HEAD > "$O/source_commit.txt" 2>/dev/null || cp .leg_commit "$O/source_commit.txt" 2>/dev/null || true
"$PY" -c "import subprocess;print(subprocess.run(['$PWD/.pixi/envs/default/bin/mojo','--version'],capture_output=True,text=True).stdout)" > "$O/compiler_version.txt" 2>&1

# Build: hard cap 30 minutes so the probe and the fetch fit the 60-minute lease.
B0=$(date +%s)
timeout -k 30 1800 "$PY" tools/development_build.py build --sets host --jobs "$J" \
    --memory-per-job-gib "$PER_JOB_GIB" --evidence "$O/devbuild" > "$O/devbuild.log" 2>&1
BRC=$?
echo "$BRC" > "$O/devbuild.rc"
echo "build rc=$BRC seconds=$(( $(date +%s) - B0 ))" | tee -a "$O/build_settings.txt"
if [ "$BRC" != 0 ]; then
    "$PY" experiments/python_runtime_probe_20261028/partial_activate.py "$O/partial_activation.json" \
        > "$O/partial_activate.log" 2>&1
    echo "partial activation rc=$?" | tee -a "$O/build_settings.txt"
fi
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
LEFT=$(( 46 * 60 - ELAPSED ))
DEADLINE=$(( LEFT - 330 )); [ "$DEADLINE" -ge 60 ] || DEADLINE=60
echo "probe deadline=${DEADLINE}s workers=$CORES" | tee -a "$O/build_settings.txt"
MOJOLEARN_VENDOR=cpu "$PY" tools/python_runtime_probe.py --out "$O/probe" --workers "$CORES" \
    --group-timeout 300 --call-timeout 60 --deadline-seconds "$DEADLINE" > "$O/probe.log" 2>&1
PRC=$?
echo "$PRC" > "$O/probe.rc"
echo "probe rc=$PRC total_seconds=$(( $(date +%s) - T0 ))" | tee -a "$O/build_settings.txt"
exit 0
