#!/bin/sh
# lane/knn-tiled-distance, 2026-09-17: the on-pod sequence (RunPod NVIDIA,
# CUDA 13 driver). Run ON THE POD from /root/mojolearn (the branch's source,
# shipped by tools/trees_leg.sh) with /root/mojolearn-base holding main's
# source (shipped separately). POSIX sh. Stages:
#
#   sh tools/knn_tiled_body.sh setup      pixi, blocks from the staged R2 npz,
#                                         every binding arm, per-arm trees
#   sh tools/knn_tiled_body.sh phase      the phase attribution (timer builds)
#   sh tools/knn_tiled_body.sh race       kNN and KDE arm races, clf/reg probe
#   sh tools/knn_tiled_body.sh identity   cuda columns per arm, cpu column, diffs
#   sh tools/knn_tiled_body.sh opponents  cuML / cuVS on the same blocks
#
# Every stage writes under /root/ktd_out/<stage>; `pull` the whole directory
# home after each stage. Arms (core binding, IDENTICAL, sm_89):
#   base   main's source                     (DEVIATIONs 3000..3003 absent)
#   after0 the branch, rows OFF              (resident doors only)
#   smem   + MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1        (DEVIATION 3000)
#   topk   + ..._SMEM_TILE=1 + ..._KNN_BLOCK_TOPK=1        (3000 + 3001)
#   sabo   topk + MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1       (reach control)
#   phase0 / phasesmem / phasetopk: the same three with MOJOLEARN_KNN_PHASE_TIMERS=1
set -u
STAGE=${1:?stage}
R=/root/mojolearn
B=/root/mojolearn-base
case "$STAGE" in _*) OUT=/root/ktd_out/${KTD_PARENT:-setup} ;; *) OUT=/root/ktd_out/$STAGE; export KTD_PARENT=$STAGE ;; esac
mkdir -p "$OUT/logs"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_COMPILE_JOBS=16 MOJOLEARN_BUILD_JOBS=16
PIXI="$HOME/.pixi/bin/pixi"
P=$R/.pixi/envs/default/bin/python3
DATA=/root/ctd-data
MODELS=/root/ktd_models
note() { echo "$* $(date -u +%H:%M:%S)" | tee -a "$OUT/progress.txt"; }
step() {
    _n=$1; _cap=$2; shift 2
    _t=$(date +%s)
    timeout -k 30 "$_cap" "$@" > "$OUT/logs/$_n.log" 2>&1
    _rc=$?
    printf '%s\t%s\t%s\n' "$_n" "$_rc" "$(( $(date +%s) - _t ))" >> "$OUT/status.tsv"
    note "$_n=$_rc"
    return $_rc
}
KNN_LANES="knn,knn-chebyshev,knn-clf,knn-clf-distance,knn-cosine,knn-manhattan,knn-minkowski-p3,knn-rbc,knn-reg,knn-reg-distance,knn-sqeuclidean,radius,radius-chebyshev,radius-manhattan,radius-minkowski-p3"
KDE_LANES="kde,kde-cosine-minkowski,kde-epanechnikov-l1,kde-exponential-chebyshev,kde-linear-cosine,kde-tophat-sqeuclidean,kde-weighted"
ALL_LANES="$KNN_LANES,$KDE_LANES"
FIX5="base,ties,odd,dupes,wide"

# one core binding build in tree $1 with extra defines $2, copied to /root/gpubins/$3
build_core() {
    ( cd "$1" && MOJOLEARN_GPU_ARCHS=sm_89 MOJOLEARN_BUILD_EXTRA_DEFINES="$2" bash bindings/build.sh ) || return 1
    mkdir -p "/root/gpubins/$3" && cp "$1/python/mojolearn/identical/_mojolearn.so" "/root/gpubins/$3/_mojolearn.so"
}
build_estimators() {
    ( cd "$1" && MOJOLEARN_GPU_ARCHS=sm_89 MOJOLEARN_BUILD_EXTRA_DEFINES="$2" bash bindings/build_estimators.sh ) || return 1
    mkdir -p "/root/gpubins/$3" && cp "$1/python/mojolearn/identical/_mojolearn_estimators.so" "/root/gpubins/$3/_mojolearn_estimators.so"
}
# a source tree with no binaries at /root/t-$1 from tree $2, then the arm's .so files
make_tree() {
    rm -rf "/root/t-$1" && mkdir -p "/root/t-$1"
    ( cd "$2" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd "/root/t-$1" && tar xf - )
    mkdir -p "/root/t-$1/python/mojolearn/identical"
    cp "/root/gpubins/$1/_mojolearn.so" "/root/t-$1/python/mojolearn/identical/"
    cp "/root/gpubins/$3/_mojolearn_estimators.so" "/root/t-$1/python/mojolearn/identical/"
}
tree_env() { echo "PYTHONPATH=/root/t-$1/python:/root/t-$1/tools MOJOLEARN_NUMERIC_MODE=identical"; }

case "$STAGE" in
_build_core) build_core "$2" "$3" "$4"; exit $? ;;
_build_est) build_estimators "$2" "$3" "$4"; exit $? ;;
setup)
    note start branch="$(cat "$R/SHIPPED_COMMIT.txt")" base="$(cat "$B/SHIPPED_COMMIT.txt" 2>/dev/null)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$OUT/gpu.txt" 2>&1
    { nproc; grep -m1 'model name' /proc/cpuinfo; uname -a; python3 --version; free -g | head -2; } > "$OUT/box.txt" 2>&1
    cd "$R" || exit 9
    [ -x "$HOME/.pixi/bin/pixi" ] || curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/logs/pixi_get.log" 2>&1
    step pixi_install 1500 pixi install
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
    step prep_blocks 1500 pixi run python3 tools/classical_two_datasets.py prep --data "$DATA" --lanes knn,kde --datasets taxi,istella
    # the base tree shares the pixi environment through a symlink
    [ -e "$B/.pixi" ] || ln -s "$R/.pixi" "$B/.pixi"
    step build_base_core 1800 sh "$0" _build_core "$B" "" base
    step build_base_est 1800 sh "$0" _build_est "$B" "" base
    step build_after0_core 1800 sh "$0" _build_core "$R" "" after0
    step build_after0_est 1800 sh "$0" _build_est "$R" "" after0
    step build_smem_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1" smem
    step build_topk_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1" topk
    step build_sabo_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1 -D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1" sabo
    step build_phase0_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_KNN_PHASE_TIMERS=1" phase0
    step build_phasesmem_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_KNN_PHASE_TIMERS=1 -D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1" phasesmem
    step build_phasetopk_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_KNN_PHASE_TIMERS=1 -D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1" phasetopk
    # host (CPU column) sets, branch and base
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/after step host_core_after 1500 sh bindings/build_core_host.sh
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/after step host_est_after 1500 sh bindings/build_estimators_host.sh
    ( cd "$B" && MOJOLEARN_HOST_OUTDIR=/root/hostbins/base step host_core_base 1500 sh bindings/build_core_host.sh )
    ( cd "$B" && MOJOLEARN_HOST_OUTDIR=/root/hostbins/base step host_est_base 1500 sh bindings/build_estimators_host.sh )
    sha256sum /root/gpubins/*/*.so /root/hostbins/*/*.so > "$OUT/so_sha256.txt" 2>&1
    for a in after0 smem topk sabo phase0 phasesmem phasetopk; do make_tree "$a" "$R" after0; done
    make_tree base "$B" base
    # CPU-column tree: no GPU .so at all
    rm -rf /root/mojolearn-cpu && mkdir -p /root/mojolearn-cpu && ( cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd /root/mojolearn-cpu && tar xf - )
    for a in base after0 topk; do
        ( cd "/root/t-$a" && env $(tree_env "$a") "$P" -c "import mojolearn as ml; m = ml.NearestNeighbors(); print('$a import OK', ml.vendor(), ml.numeric_mode())" ) >> "$OUT/imports.txt" 2>&1
    done
    step fit_models 1500 env PYTHONPATH=/root/t-after0/python "$P" bench/speed/classical_ladder_infer.py fit --data "$DATA" --models "$MODELS" --lanes knn,kde
    note setup_done
    : > "$OUT/setup.done"
    ;;

rebuild)
    # The branch arms again after a source push (the base arms and the host
    # sets stand); then the trees, the import check and the fits.
    cd "$R" || exit 9
    step build_after0_core 1800 sh "$0" _build_core "$R" "" after0
    step build_after0_est 1800 sh "$0" _build_est "$R" "" after0
    step build_smem_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1" smem
    step build_topk_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1" topk
    step build_sabo_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1 -D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1" sabo
    step build_phase0_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_KNN_PHASE_TIMERS=1" phase0
    step build_phasesmem_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_KNN_PHASE_TIMERS=1 -D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1" phasesmem
    step build_phasetopk_core 1800 sh "$0" _build_core "$R" "-D MOJOLEARN_KNN_PHASE_TIMERS=1 -D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1 -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1" phasetopk
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/after step host_core_after 1500 sh bindings/build_core_host.sh
    MOJOLEARN_HOST_OUTDIR=/root/hostbins/after step host_est_after 1500 sh bindings/build_estimators_host.sh
    sha256sum /root/gpubins/*/*.so /root/hostbins/*/*.so > "$OUT/so_sha256.txt" 2>&1
    for a in after0 smem topk sabo phase0 phasesmem phasetopk; do make_tree "$a" "$R" after0; done
    make_tree base "$B" base
    rm -rf /root/mojolearn-cpu && mkdir -p /root/mojolearn-cpu && ( cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' . ) | ( cd /root/mojolearn-cpu && tar xf - )
    : > "$OUT/imports.txt"
    for a in base after0 smem topk; do
        ( cd "/root/t-$a" && env $(tree_env "$a") "$P" -c "import mojolearn as ml; m = ml.NearestNeighbors(); print('$a import OK', ml.vendor(), ml.numeric_mode())" ) >> "$OUT/imports.txt" 2>&1
    done
    step fit_models 1500 env PYTHONPATH=/root/t-after0/python "$P" bench/speed/classical_ladder_infer.py fit --data "$DATA" --models "$MODELS" --lanes knn,kde
    note rebuild_done
    : > "$OUT/rebuild.done"
    ;;

phase)
    cat > "$OUT/phase_probe.py" <<'PY'
import json, os, re, statistics, subprocess, sys, time
import numpy as np
data = "/root/ctd-data"
arm = sys.argv[1]
out = {"arm": arm, "cells": []}
child = r'''
import sys, time, json
import numpy as np, mojolearn as ml
lane, ds, k, rows = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with np.load("/root/ctd-data/knn-%s.npz" % ds) as z:
    index = np.ascontiguousarray(z["index"]); q = np.ascontiguousarray(z["queries"][:rows])
m = ml.NearestNeighbors(n_neighbors=k).fit(index)
m.kneighbors(q)  # warmup and the resident upload
ms = []
for _ in range(3):
    print("PROBE_CALL_BEGIN", flush=True)
    t0 = time.perf_counter(); d, i = m.kneighbors(q); ms.append((time.perf_counter() - t0) * 1000.0)
    print("PROBE_CALL_END", flush=True)
print("PROBE_MS", json.dumps(ms, separators=(",", ":")), "DIGEST", __import__("hashlib").sha256(d.tobytes() + i.tobytes()).hexdigest()[:16], flush=True)
'''
for ds in ("istella", "taxi"):
    for k in (1, 10, 64):
        for rows in (4000, 1):
            r = subprocess.run([sys.executable, "-c", child, "knn", ds, str(k), str(rows)], capture_output=True, text=True)
            lines = r.stdout.splitlines()
            calls, cur = [], None
            for ln in lines:
                if ln.startswith("PROBE_CALL_BEGIN"): cur = []
                elif ln.startswith("PROBE_CALL_END"): calls.append(cur); cur = None
                elif cur is not None and ln.startswith("KNN_PHASE_TIMERS"): cur.append(ln)
            ms = None; digest = None
            for ln in lines:
                if ln.startswith("PROBE_MS"):
                    parts = ln.split(); ms = json.loads(parts[1]); digest = parts[3]
            cell = {"dataset": ds, "k": k, "rows": rows, "ms": ms, "digest": digest, "phase_lines": calls, "stderr_tail": r.stderr[-2000:], "rc": r.returncode}
            out["cells"].append(cell)
            print("PHASE %s %s k=%d rows=%d ms=%s" % (arm, ds, k, rows, ms), flush=True)
            for c in calls[-1:]:
                for ln in c: print("   ", ln, flush=True)
json.dump(out, open("/root/ktd_out/phase/phase_%s.json" % arm, "w"), indent=1)
PY
    for a in phase0 phasesmem phasetopk; do
        ( cd "/root/t-$a" && env $(tree_env "$a") "$P" "$OUT/phase_probe.py" "$a" > "$OUT/phase_$a.console" 2>&1 )
        note "phase_$a=$?"
    done
    note phase_done
    : > "$OUT/phase.done"
    ;;

race)
    cat > "$OUT/arms_knn.json" <<JSON
{
 "base":   {"python": "$P", "cwd": "/root/t-base",   "cpu": false, "env": {"PYTHONPATH": "/root/t-base/python:/root/t-base/tools",     "MOJOLEARN_NUMERIC_MODE": "identical"}},
 "after0": {"python": "$P", "cwd": "/root/t-after0", "cpu": false, "env": {"PYTHONPATH": "/root/t-after0/python:/root/t-after0/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}},
 "smem":   {"python": "$P", "cwd": "/root/t-smem",   "cpu": false, "env": {"PYTHONPATH": "/root/t-smem/python:/root/t-smem/tools",     "MOJOLEARN_NUMERIC_MODE": "identical"}},
 "topk":   {"python": "$P", "cwd": "/root/t-topk",   "cpu": false, "env": {"PYTHONPATH": "/root/t-topk/python:/root/t-topk/tools",     "MOJOLEARN_NUMERIC_MODE": "identical"}}
}
JSON
    cat > "$OUT/arms_kde.json" <<JSON
{
 "base":   {"python": "$P", "cwd": "/root/t-base",   "cpu": false, "env": {"PYTHONPATH": "/root/t-base/python:/root/t-base/tools",     "MOJOLEARN_NUMERIC_MODE": "identical"}},
 "after0": {"python": "$P", "cwd": "/root/t-after0", "cpu": false, "env": {"PYTHONPATH": "/root/t-after0/python:/root/t-after0/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}}
}
JSON
    note race_knn_start
    env PYTHONPATH=/root/t-after0/python "$P" "$R/bench/speed/classical_ladder_infer.py" race --data "$DATA" --models "$MODELS" --arms "$OUT/arms_knn.json" --out "$OUT/race_knn" --lanes knn --datasets taxi,istella --outer 5 --rounds 3 --warmup 1 > "$OUT/race_knn.console" 2>&1
    note "race_knn=$?"
    env PYTHONPATH=/root/t-after0/python "$P" "$R/bench/speed/classical_ladder_infer.py" race --data "$DATA" --models "$MODELS" --arms "$OUT/arms_kde.json" --out "$OUT/race_kde" --lanes kde --datasets taxi,istella --outer 5 --rounds 3 --warmup 1 > "$OUT/race_kde.console" 2>&1
    note "race_kde=$?"
    # k sweep and the classifier / regressor resident doors: per-call floor at 1 and 4000 queries
    cat > "$OUT/floor_probe.py" <<'PY'
import hashlib, json, statistics, sys, time
import numpy as np, mojolearn as ml
arm = sys.argv[1]
out = {"arm": arm, "vendor": ml.vendor(), "cells": []}
def cell(name, ds, rows, k, call, extra=None):
    call()
    ms = []
    for _ in range(5):
        t0 = time.perf_counter(); o = call(); ms.append((time.perf_counter() - t0) * 1000.0)
    h = hashlib.sha256()
    for part in (o if isinstance(o, tuple) else (o,)):
        h.update(np.ascontiguousarray(part).tobytes())
    c = {"lane": name, "dataset": ds, "rows": rows, "k": k, "median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms), "digest": h.hexdigest()[:16]}
    if extra: c.update(extra)
    out["cells"].append(c)
    print("FLOOR %s %s %s rows=%d k=%d median=%.3f min=%.3f max=%.3f digest=%s" % (arm, name, ds, rows, k, c["median_ms"], c["min_ms"], c["max_ms"], c["digest"]), flush=True)
for ds in ("istella", "taxi"):
    with np.load("/root/ctd-data/knn-%s.npz" % ds) as z:
        index = np.ascontiguousarray(z["index"]); queries = np.ascontiguousarray(z["queries"])
    for k in (1, 10, 64):
        nn = ml.NearestNeighbors(n_neighbors=k).fit(index)
        for rows in (1, 4000):
            q = np.ascontiguousarray(queries[:rows])
            cell("knn", ds, rows, k, lambda: nn.kneighbors(q))
    labels = (np.arange(index.shape[0]) % 7).astype(np.int64)
    clf = ml.KNeighborsClassifier(n_neighbors=10).fit(index, labels)
    reg = ml.KNeighborsRegressor(n_neighbors=10).fit(index, np.ascontiguousarray(index[:, 0]))
    for rows in (1, 4000):
        q = np.ascontiguousarray(queries[:rows])
        cell("knn-clf", ds, rows, 10, lambda: clf.predict(q), {"resident": getattr(clf, "_resident", None) is not None})
        cell("knn-reg", ds, rows, 10, lambda: reg.predict(q), {"resident": getattr(reg, "_resident", None) is not None})
    with np.load("/root/ctd-data/kde-%s.npz" % ds) as z:
        X = np.ascontiguousarray(z["X"]); Xq = np.ascontiguousarray(z["Xq"])
    rec = json.load(open("/root/ctd-data/kde-%s.json" % ds))
    kd = ml.KernelDensity(bandwidth=rec["kde"]["bandwidth"], kernel="gaussian").fit(X)
    for rows in (1, 2000):
        q = np.ascontiguousarray(Xq[:rows])
        cell("kde", ds, rows, 0, lambda: kd.score_samples(q), {"resident": getattr(kd, "_resident", None) is not None})
json.dump(out, open("/root/ktd_out/race/floor_%s.json" % arm, "w"), indent=1)
PY
    for a in base after0 smem topk; do
        ( cd "/root/t-$a" && env $(tree_env "$a") "$P" "$OUT/floor_probe.py" "$a" > "$OUT/floor_$a.console" 2>&1 )
        note "floor_$a=$?"
    done
    note race_done
    : > "$OUT/race.done"
    ;;

identity)
    BRANCH_SHA=$(cat "$R/SHIPPED_COMMIT.txt")
    BASE_SHA=$(cat "$B/SHIPPED_COMMIT.txt")
    gpu_run() { # arm lanes fixtures commit
        ( cd "/root/t-$1" && env $(tree_env "$1") MOJOLEARN_COMMIT="$4" \
          "$PIXI" run --manifest-path "$R/pixi.toml" python3 tools/identity_break.py --require-backend cuda --lanes "$2" --fixtures "$3" --repeats 2 \
          --json "$OUT/cuda-$1.json" > "$OUT/logs/cuda-$1.log" 2>&1; echo "cuda-$1 rc=$?" ) | tee -a "$OUT/progress.txt"
    }
    cpu_run() { # name hostdir commit lanes fixtures
        ( cd /root/mojolearn-cpu && env PYTHONPATH=/root/mojolearn-cpu/python:/root/mojolearn-cpu/tools MOJOLEARN_HOST_DIR="$2" MOJOLEARN_COMMIT="$3" MOJOLEARN_NUMERIC_MODE=identical \
          "$PIXI" run --manifest-path "$R/pixi.toml" python3 tools/identity_break.py --require-backend cpu --lanes "$4" --fixtures "$5" --repeats 2 \
          --json "$OUT/$1.json" > "$OUT/logs/$1.log" 2>&1; echo "$1 rc=$?" ) | tee -a "$OUT/progress.txt"
    }
    note identity_start
    cpu_run cpu-after /root/hostbins/after "$BRANCH_SHA" "$ALL_LANES" "$FIX5" &
    cpu_run cpu-base /root/hostbins/base "$BASE_SHA" "$ALL_LANES" "$FIX5" &
    gpu_run base "$ALL_LANES" "$FIX5" "$BASE_SHA"
    gpu_run after0 "$ALL_LANES" "$FIX5" "$BRANCH_SHA"
    gpu_run smem "$KNN_LANES" "$FIX5" "$BRANCH_SHA"
    gpu_run topk "$ALL_LANES" "$FIX5" "$BRANCH_SHA"
    gpu_run sabo "$KNN_LANES" "$FIX5" "$BRANCH_SHA"
    wait
    cd "$R"
    D() { PYTHONPATH="$R/python:$R/tools" "$PIXI" run python3 tools/identity_break.py --diff "$@"; }
    D "$OUT/cuda-base.json" "$OUT/cuda-after0.json" "$OUT/cuda-smem.json" "$OUT/cuda-topk.json" > "$OUT/diff.cuda.base-after0-smem-topk.txt" 2>&1; echo "diff cuda arms rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-base.json" "$OUT/cpu-base.json" > "$OUT/diff.before.cuda-vs-cpu.txt" 2>&1; echo "diff before cuda-cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-topk.json" "$OUT/cpu-after.json" > "$OUT/diff.after.cuda-topk-vs-cpu.txt" 2>&1; echo "diff after cuda-cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-after0.json" "$OUT/cpu-after.json" > "$OUT/diff.after.cuda-after0-vs-cpu.txt" 2>&1; echo "diff after0 cuda-cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cpu-base.json" "$OUT/cpu-after.json" > "$OUT/diff.cpu.base-vs-after.txt" 2>&1; echo "diff cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-topk.json" "$OUT/cuda-sabo.json" > "$OUT/diff.sabotage.topk-vs-sabo.txt" 2>&1; echo "diff sabotage rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cpu-after.json" "$OUT/cuda-sabo.json" > "$OUT/diff.sabotage.cpu-vs-sabo.txt" 2>&1; echo "diff sabotage cpu rc=$?" | tee -a "$OUT/progress.txt"
    note identity_done
    : > "$OUT/identity.done"
    ;;

pip)
    # The opponents' Python, installable while the GPU stages run.
    SYSPY=$(command -v python3)
    note "pip start syspy=$SYSPY"
    step pip_base 900 "$SYSPY" -m pip install --no-input --disable-pip-version-check scikit-learn pyarrow threadpoolctl
    step pip_cuml 1800 "$SYSPY" -m pip install --no-input --disable-pip-version-check --extra-index-url=https://pypi.nvidia.com cuml-cu12==26.8.0
    step pip_cuvs 1200 "$SYSPY" -m pip install --no-input --disable-pip-version-check --extra-index-url=https://pypi.nvidia.com cuvs-cu12==26.8.0
    "$SYSPY" -m pip freeze > "$OUT/pip_freeze.txt" 2>&1
    note pip_done
    : > "$OUT/pip.done"
    ;;

opponents)
    SYSPY=$(command -v python3)
    note "opponents start syspy=$SYSPY"
    while [ ! -f /root/ktd_out/pip/pip.done ]; do sleep 15; done
    OURS="$PIXI run --manifest-path $R/pixi.toml python3"
    for ds in istella taxi; do
        ( cd "$R" && MOJOLEARN_REPO_COMMIT="$(cat "$R/SHIPPED_COMMIT.txt")" "$SYSPY" tools/classical_two_datasets.py race --lane knn --dataset $ds --data "$DATA" --out "$OUT/ctd_knn" --work /root/ctd-work --root /root/t-topk --arms ours,cuml-gpu --rounds 5 --ours-python "$OURS" --theirs-python "$SYSPY" > "$OUT/ctd_knn_$ds.console" 2>&1 ); note "ctd_knn_$ds=$?"
        ( cd "$R" && MOJOLEARN_REPO_COMMIT="$(cat "$R/SHIPPED_COMMIT.txt")" "$SYSPY" tools/classical_two_datasets.py race --lane kde --dataset $ds --data "$DATA" --out "$OUT/ctd_kde" --work /root/ctd-work --root /root/t-after0 --arms ours,cuml-gpu --rounds 5 --ours-python "$OURS" --theirs-python "$SYSPY" > "$OUT/ctd_kde_$ds.console" 2>&1 ); note "ctd_kde_$ds=$?"
    done
    cat > "$OUT/cuvs_probe.py" <<'PY'
import hashlib, json, statistics, time
import numpy as np, cupy as cp
from cuvs.neighbors import brute_force
import cuml
out = {"cuvs": __import__("cuvs").__version__, "cuml": cuml.__version__, "cells": []}
for ds in ("istella", "taxi"):
    with np.load("/root/ctd-data/knn-%s.npz" % ds) as z:
        index = np.ascontiguousarray(z["index"]); queries = np.ascontiguousarray(z["queries"])
    di = cp.asarray(index); dq = cp.asarray(queries)
    cp.cuda.runtime.deviceSynchronize()
    for k in (1, 10, 64):
        for rows in (4000, 1):
            q = dq[:rows]
            idx = brute_force.build(di, metric="sqeuclidean")
            cp.cuda.runtime.deviceSynchronize()
            d, i = brute_force.search(idx, q, k); cp.cuda.runtime.deviceSynchronize()
            ms = []
            for _ in range(5):
                t0 = time.perf_counter(); d, i = brute_force.search(idx, q, k); cp.cuda.runtime.deviceSynchronize(); ms.append((time.perf_counter() - t0) * 1000.0)
            dh = cp.asnumpy(cp.asarray(d)); ih = cp.asnumpy(cp.asarray(i))
            c = {"library": "cuvs.brute_force", "dataset": ds, "rows": rows, "k": k, "metric": "sqeuclidean", "median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms), "ms": ms, "index_digest": hashlib.sha256(np.ascontiguousarray(ih).tobytes()).hexdigest()[:16]}
            out["cells"].append(c); print("CUVS %s rows=%d k=%d median=%.3f min=%.3f max=%.3f" % (ds, rows, k, c["median_ms"], c["min_ms"], c["max_ms"]), flush=True)
    from cuml.neighbors import NearestNeighbors
    for k in (1, 10, 64):
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean", output_type="cupy").fit(di)
        for rows in (4000, 1):
            q = dq[:rows]
            nn.kneighbors(q); cp.cuda.runtime.deviceSynchronize()
            ms = []
            for _ in range(5):
                t0 = time.perf_counter(); d, i = nn.kneighbors(q); cp.cuda.runtime.deviceSynchronize(); ms.append((time.perf_counter() - t0) * 1000.0)
            c = {"library": "cuml.NearestNeighbors(brute)", "dataset": ds, "rows": rows, "k": k, "median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms), "ms": ms, "index_digest": hashlib.sha256(np.ascontiguousarray(cp.asnumpy(i)).tobytes()).hexdigest()[:16]}
            out["cells"].append(c); print("CUML %s rows=%d k=%d median=%.3f min=%.3f max=%.3f" % (ds, rows, k, c["median_ms"], c["min_ms"], c["max_ms"]), flush=True)
    from cuml.neighbors import KernelDensity
    with np.load("/root/ctd-data/kde-%s.npz" % ds) as z:
        X = cp.asarray(np.ascontiguousarray(z["X"])); Xq = cp.asarray(np.ascontiguousarray(z["Xq"]))
    rec = json.load(open("/root/ctd-data/kde-%s.json" % ds))
    kd = KernelDensity(bandwidth=rec["kde"]["bandwidth"], kernel="gaussian", metric="euclidean", output_type="cupy").fit(X)
    for rows in (2000, 1):
        q = Xq[:rows]
        kd.score_samples(q); cp.cuda.runtime.deviceSynchronize()
        ms = []
        for _ in range(5):
            t0 = time.perf_counter(); s = kd.score_samples(q); cp.cuda.runtime.deviceSynchronize(); ms.append((time.perf_counter() - t0) * 1000.0)
        c = {"library": "cuml.KernelDensity", "dataset": ds, "rows": rows, "k": 0, "median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms), "ms": ms}
        out["cells"].append(c); print("CUML-KDE %s rows=%d median=%.3f min=%.3f max=%.3f" % (ds, rows, c["median_ms"], c["min_ms"], c["max_ms"]), flush=True)
json.dump(out, open("/root/ktd_out/opponents/opponents_probe.json", "w"), indent=1)
PY
    "$SYSPY" "$OUT/cuvs_probe.py" > "$OUT/opponents_probe.console" 2>&1; note "opponents_probe=$?"
    note opponents_done
    : > "$OUT/opponents.done"
    ;;
rebuild2)
    # Second round (the rank loop, the exact-chain admission, the split
    # selector launch): every arm that compiles smem_distance_tile.mojo's
    # reached code, plus the exact arms and their reach control.
    cd "$R" || exit 9
    SM="-D MOJOLEARN_EXPERIMENTAL_KNN_SMEM_TILE=1"
    TK="$SM -D MOJOLEARN_EXPERIMENTAL_KNN_BLOCK_TOPK=1"
    EX="-D MOJOLEARN_EXPERIMENTAL_KNN_EXACT_CHAIN=1"
    PT="-D MOJOLEARN_KNN_PHASE_TIMERS=1"
    step build_smem_core 1800 sh "$0" _build_core "$R" "$SM" smem
    step build_topk_core 1800 sh "$0" _build_core "$R" "$TK" topk
    step build_sabo_core 1800 sh "$0" _build_core "$R" "$TK -D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1" sabo
    step build_smemx_core 1800 sh "$0" _build_core "$R" "$SM $EX" smemx
    step build_topkx_core 1800 sh "$0" _build_core "$R" "$TK $EX" topkx
    step build_sabox_core 1800 sh "$0" _build_core "$R" "$TK $EX -D MOJOLEARN_KNN_EXACT_CHAIN_SABOTAGE=1" sabox
    step build_phasesmem_core 1800 sh "$0" _build_core "$R" "$PT $SM" phasesmem
    step build_phasetopk_core 1800 sh "$0" _build_core "$R" "$PT $TK" phasetopk
    step build_phasesmemx_core 1800 sh "$0" _build_core "$R" "$PT $SM $EX" phasesmemx
    step build_phasetopkx_core 1800 sh "$0" _build_core "$R" "$PT $TK $EX" phasetopkx
    sha256sum /root/gpubins/*/*.so > "$OUT/so_sha256.txt" 2>&1
    for a in smem topk sabo smemx topkx sabox phasesmem phasetopk phasesmemx phasetopkx; do make_tree "$a" "$R" after0; done
    : > "$OUT/imports.txt"
    for a in smem topk smemx topkx; do
        ( cd "/root/t-$a" && env $(tree_env "$a") "$P" -c "import mojolearn as ml; m = ml.NearestNeighbors(); print('$a import OK', ml.vendor(), ml.numeric_mode())" ) >> "$OUT/imports.txt" 2>&1
    done
    note rebuild2_done
    : > "$OUT/rebuild2.done"
    ;;

phase2)
    cp /root/ktd_out/phase/phase_probe.py "$OUT/phase_probe.py"
    sed -i 's|/root/ktd_out/phase/phase_%s.json|/root/ktd_out/phase2/phase_%s.json|' "$OUT/phase_probe.py"
    for a in phasesmem phasetopk phasesmemx phasetopkx; do
        ( cd "/root/t-$a" && env $(tree_env "$a") "$P" "$OUT/phase_probe.py" "$a" > "$OUT/phase_$a.console" 2>&1 )
        note "phase_$a=$?"
    done
    note phase2_done
    : > "$OUT/phase2.done"
    ;;

race2)
    {
        echo "{"
        first=1
        for a in base after0 smem topk smemx topkx; do
            [ $first = 1 ] || echo ","
            first=0
            printf ' "%s": {"python": "%s", "cwd": "/root/t-%s", "cpu": false, "env": {"PYTHONPATH": "/root/t-%s/python:/root/t-%s/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}}' "$a" "$P" "$a" "$a" "$a"
        done
        echo; echo "}"
    } > "$OUT/arms_knn.json"
    note race2_knn_start
    env PYTHONPATH=/root/t-after0/python "$P" "$R/bench/speed/classical_ladder_infer.py" race --data "$DATA" --models "$MODELS" --arms "$OUT/arms_knn.json" --out "$OUT/race_knn" --lanes knn --datasets taxi,istella --outer 5 --rounds 3 --warmup 1 > "$OUT/race_knn.console" 2>&1
    note "race_knn=$?"
    cp /root/ktd_out/race/floor_probe.py "$OUT/floor_probe.py"
    sed -i 's|/root/ktd_out/race/floor_%s.json|/root/ktd_out/race2/floor_%s.json|' "$OUT/floor_probe.py"
    for a in smem topk smemx topkx; do
        ( cd "/root/t-$a" && env $(tree_env "$a") "$P" "$OUT/floor_probe.py" "$a" > "$OUT/floor_$a.console" 2>&1 )
        note "floor_$a=$?"
    done
    note race2_done
    : > "$OUT/race2.done"
    ;;

identity2)
    BRANCH_SHA=$(cat "$R/SHIPPED_COMMIT.txt")
    I1=/root/ktd_out/identity
    gpu_run() { # arm lanes fixtures commit
        ( cd "/root/t-$1" && env $(tree_env "$1") MOJOLEARN_COMMIT="$4" \
          "$PIXI" run --manifest-path "$R/pixi.toml" python3 tools/identity_break.py --require-backend cuda --lanes "$2" --fixtures "$3" --repeats 2 \
          --json "$OUT/cuda-$1.json" > "$OUT/logs/cuda-$1.log" 2>&1; echo "cuda-$1 rc=$?" ) | tee -a "$OUT/progress.txt"
    }
    note identity2_start
    for a in smem topk smemx topkx sabo sabox; do gpu_run "$a" "$KNN_LANES" "$FIX5" "$BRANCH_SHA"; done
    cd "$R"
    D() { PYTHONPATH="$R/python:$R/tools" "$PIXI" run python3 tools/identity_break.py --diff "$@"; }
    D "$I1/cuda-base.json" "$OUT/cuda-smem.json" "$OUT/cuda-topk.json" "$OUT/cuda-smemx.json" "$OUT/cuda-topkx.json" > "$OUT/diff.cuda.base-smem-topk-smemx-topkx.txt" 2>&1; echo "diff cuda arms rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-topkx.json" "$I1/cpu-after.json" > "$OUT/diff.after.cuda-topkx-vs-cpu.txt" 2>&1; echo "diff topkx cuda-cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-smemx.json" "$I1/cpu-after.json" > "$OUT/diff.after.cuda-smemx-vs-cpu.txt" 2>&1; echo "diff smemx cuda-cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-topk.json" "$I1/cpu-after.json" > "$OUT/diff.after.cuda-topk-vs-cpu.txt" 2>&1; echo "diff topk cuda-cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-topk.json" "$OUT/cuda-sabo.json" > "$OUT/diff.sabotage.topk-vs-sabo.txt" 2>&1; echo "diff sabotage rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-topkx.json" "$OUT/cuda-sabox.json" > "$OUT/diff.sabotage.topkx-vs-sabox.txt" 2>&1; echo "diff exact sabotage rc=$?" | tee -a "$OUT/progress.txt"
    D "$I1/cpu-after.json" "$OUT/cuda-sabox.json" > "$OUT/diff.sabotage.cpu-vs-sabox.txt" 2>&1; echo "diff exact sabotage cpu rc=$?" | tee -a "$OUT/progress.txt"
    note identity2_done
    : > "$OUT/identity2.done"
    ;;

cuvs)
    SYSPY=$(command -v python3)
    "$SYSPY" -m pip freeze > "$OUT/pip_freeze.txt" 2>&1
    "$SYSPY" /root/ktd_out/opponents/cuvs_probe.py > "$OUT/opponents_probe.console" 2>&1; note "opponents_probe=$?"
    cp /root/ktd_out/opponents/opponents_probe.json "$OUT/" 2>/dev/null
    note cuvs_done
    : > "$OUT/cuvs.done"
    ;;

final-build)
    # The shipped defaults (no -D): the arm the merge ships.
    cd "$R" || exit 9
    step build_final_core 1800 sh "$0" _build_core "$R" "" final
    make_tree final "$R" after0
    ( cd /root/t-final && env $(tree_env final) "$P" -c "import mojolearn as ml; print('final import OK', ml.vendor(), ml.numeric_mode())" ) > "$OUT/imports.txt" 2>&1
    note final_build_done
    : > "$OUT/final-build.done"
    ;;

final-run)
    BRANCH_SHA=$(cat "$R/SHIPPED_COMMIT.txt")
    I1=/root/ktd_out/identity
    ( cd /root/t-final && env $(tree_env final) MOJOLEARN_COMMIT="$BRANCH_SHA" \
      "$PIXI" run --manifest-path "$R/pixi.toml" python3 tools/identity_break.py --require-backend cuda --lanes "$ALL_LANES" --fixtures "$FIX5" --repeats 2 \
      --json "$OUT/cuda-final.json" > "$OUT/logs/cuda-final.log" 2>&1; echo "cuda-final rc=$?" ) | tee -a "$OUT/progress.txt"
    cd "$R"
    D() { PYTHONPATH="$R/python:$R/tools" "$PIXI" run python3 tools/identity_break.py --diff "$@"; }
    D "$OUT/cuda-final.json" "$I1/cpu-after.json" "$I1/cpu-base.json" > "$OUT/diff.final.cuda-vs-cpu-after-cpu-base.txt" 2>&1; echo "diff final cuda-cpu rc=$?" | tee -a "$OUT/progress.txt"
    D "$I1/cuda-base.json" "$OUT/cuda-final.json" > "$OUT/diff.final.cuda-base-vs-final.txt" 2>&1; echo "diff final cuda base-final rc=$?" | tee -a "$OUT/progress.txt"
    D "$OUT/cuda-final.json" "/root/ktd_out/identity2/cuda-sabox.json" > "$OUT/diff.final.cuda-final-vs-sabox.txt" 2>&1; echo "diff final vs sabox rc=$?" | tee -a "$OUT/progress.txt"
    cat > "$OUT/arms_knn.json" <<JSON
{
 "base":  {"python": "$P", "cwd": "/root/t-base",  "cpu": false, "env": {"PYTHONPATH": "/root/t-base/python:/root/t-base/tools",   "MOJOLEARN_NUMERIC_MODE": "identical"}},
 "final": {"python": "$P", "cwd": "/root/t-final", "cpu": false, "env": {"PYTHONPATH": "/root/t-final/python:/root/t-final/tools", "MOJOLEARN_NUMERIC_MODE": "identical"}}
}
JSON
    env PYTHONPATH=/root/t-after0/python "$P" "$R/bench/speed/classical_ladder_infer.py" race --data "$DATA" --models "$MODELS" --arms "$OUT/arms_knn.json" --out "$OUT/race_knn" --lanes knn,kde --datasets taxi,istella --outer 5 --rounds 3 --warmup 1 > "$OUT/race_knn.console" 2>&1
    note "race_final=$?"
    cp /root/ktd_out/race/floor_probe.py "$OUT/floor_probe.py"
    sed -i 's|/root/ktd_out/race/floor_%s.json|/root/ktd_out/final-run/floor_%s.json|' "$OUT/floor_probe.py"
    ( cd /root/t-final && env $(tree_env final) "$P" "$OUT/floor_probe.py" final > "$OUT/floor_final.console" 2>&1 ); note "floor_final=$?"
    SYSPY=$(command -v python3)
    ( cd "$R" && MOJOLEARN_REPO_COMMIT="$BRANCH_SHA" "$SYSPY" tools/classical_two_datasets.py race --lane knn --dataset istella --data "$DATA" --out "$OUT/ctd_knn" --work /root/ctd-work --root /root/t-final --arms ours,cuml-gpu --rounds 5 --ours-python "$PIXI run --manifest-path $R/pixi.toml python3" --theirs-python "$SYSPY" > "$OUT/ctd_knn_istella.console" 2>&1 ); note "ctd_knn_istella=$?"
    ( cd "$R" && MOJOLEARN_REPO_COMMIT="$BRANCH_SHA" "$SYSPY" tools/classical_two_datasets.py race --lane knn --dataset taxi --data "$DATA" --out "$OUT/ctd_knn" --work /root/ctd-work --root /root/t-final --arms ours,cuml-gpu --rounds 5 --ours-python "$PIXI run --manifest-path $R/pixi.toml python3" --theirs-python "$SYSPY" > "$OUT/ctd_knn_taxi.console" 2>&1 ); note "ctd_knn_taxi=$?"
    note final_run_done
    : > "$OUT/final-run.done"
    ;;

*)
    echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
