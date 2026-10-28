#!/bin/bash
# tools/pod_smoke.sh: the import gate for a branch, run on a rented RunPod CPU box (never the Mac).
# Run it through the CPU leg runner with the binding cache staged:
#   tools/runpod_cpu_leg.sh --lane <tag> --cmd 'bash tools/pod_smoke.sh' --worktree <wt> \
#        --vcpu 32 --lease 60 --release-bincache          (dry run; add --rent to create the pod)
# 1. builds every host family (bindings/build_*_host.sh) through the R2 binding cache, using every
#    allocated core (parallel builds x per-build jobs = sched_getaffinity count);
# 2. import smoke: import mojolearn and the main public modules on the CPU-only host set;
# 3. the board preflight probe: mojolearn._backend.tier_dir('identical');
# 4. a CPU-column runtime check: linalg.matmul, Ridge and KMeans fit+predict on 50 rows, identical mode.
# Exits non-zero on any failure. Results land in $LEG_OUT (summary.txt, results.tsv, logs/, *.log).
set -u
OUT=${LEG_OUT:?LEG_OUT is set by the CPU leg runner}
mkdir -p "$OUT/logs"
CORES=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
PAR=${POD_SMOKE_PARALLEL:-4}
[ "$PAR" -gt "$CORES" ] && PAR=$CORES
JOBS=$(( CORES / PAR )); [ "$JOBS" -lt 1 ] && JOBS=1
{ echo "cores_affinity=$CORES"; echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo none)"
  echo "parallel_builds=$PAR"; echo "jobs_per_build=$JOBS"
  echo "mojo_version=$(pixi run -e default mojo --version 2>&1 | tail -n 1)"; } > "$OUT/resources.txt"
if ! sh tokenizer/tools/gen_unicode_table.sh > "$OUT/logs/gen_unicode.log" 2>&1; then
    echo "gen_unicode rc=1" >> "$OUT/summary.txt"; echo "POD_SMOKE status=FAIL" >> "$OUT/summary.txt"; exit 1
fi
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS MOJOLEARN_COMPILE_JOBS=$JOBS \
       MOJOLEARN_LINUX_CPU=x86-64-v3 MOJOLEARN_BINCACHE_OUT="$OUT/bincache" OUT
fam() {
    local f=$1 t0 rc
    t0=$(date +%s)
    # Declare the one output: builds run in parallel in one tree, and an undeclared build's
    # snapshot diff would archive a neighbour's .so under this family's cache key.
    MOJOLEARN_BINCACHE_OUTPUTS="python/mojolearn/host/_mojolearn_${f}_host.so" \
        pixi run -e default python tools/bincache.py build "bindings/build_${f}_host.sh" > "$OUT/logs/host_$f.log" 2>&1
    rc=$?
    printf 'host\t%s\trc=%s\t%ss\n' "$f" "$rc" "$(( $(date +%s) - t0 ))" >> "$OUT/results.tsv"
}
export -f fam
ls bindings/build_*_host.sh | sed 's/.*build_//; s/_host.sh//' > "$OUT/families.txt"
xargs -P "$PAR" -I{} bash -c 'fam "$@"' _ {} < "$OUT/families.txt"
sha256sum python/mojolearn/host/*.so > "$OUT/so_sha256.txt" 2>/dev/null
cat > "$OUT/smoke.py" <<'EOF'
import importlib, sys
import mojolearn
mods = ["mojolearn.ensemble", "mojolearn.neighbors", "mojolearn.decomposition", "mojolearn.svm", "mojolearn.cluster",
        "mojolearn.linear_model", "mojolearn.preprocessing", "mojolearn.metrics", "mojolearn.linalg"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, repr(e)))
missing = [n for n, v in vars(mojolearn).items() if type(v).__name__ == "_MissingUpperTier"]
print("missing_bindings", len(missing), missing[:40])
for f in failed:
    print("FAIL", f)
sys.exit(1 if failed or missing else 0)
EOF
cat > "$OUT/runtime_check.py" <<'EOF'
import numpy as np
from mojolearn import linalg
from mojolearn.linear_model import Ridge
from mojolearn.cluster import KMeans
a = np.arange(12, dtype=np.float32).reshape(4, 3)
b = np.ones((3, 2), dtype=np.float32)
c = np.asarray(linalg.matmul(a, b))
assert np.allclose(c, a @ b)
print("matmul ok", c.shape)
rng = np.random.default_rng(0)
X = rng.standard_normal((50, 4)).astype(np.float32)
y = (X @ np.arange(1, 5)).astype(np.float32)
p = np.asarray(Ridge(alpha=1.0).fit(X, y).predict(X))
print("ridge ok", p.shape, float(np.abs(p - y).max()))
labels = np.asarray(KMeans(n_clusters=3, n_init=1, random_state=0).fit(X).predict(X))
print("kmeans ok", labels.shape)
print("RUNTIME-OK")
EOF
step() {
    local name=$1 rc; shift
    "$@" > "$OUT/$name.log" 2>&1; rc=$?
    echo "$name rc=$rc" >> "$OUT/summary.txt"
    return $rc
}
total=$(wc -l < "$OUT/families.txt"); ok=$(grep -c 'rc=0' "$OUT/results.tsv")
echo "host_builds total=$total ok=$ok" >> "$OUT/summary.txt"
s=0
step import_smoke pixi run -e default python "$OUT/smoke.py" || s=1
step probe_tier_dir pixi run -e default python -c "import mojolearn._backend as b; print(b.tier_dir('identical'))" || s=1
step runtime_check pixi run -e default python "$OUT/runtime_check.py" || s=1
[ "$ok" = "$total" ] || s=1
echo "POD_SMOKE status=$([ $s = 0 ] && echo PASS || echo FAIL)" >> "$OUT/summary.txt"
exit $s
