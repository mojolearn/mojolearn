#!/bin/sh
# THE AMD CONFIRMATION BODY for the trees/classical speed flips merged
# 2026-09-11/12, plus DEVIATION 2680 (the Jacobi launch/fold width split).
#
# WHAT GAP THIS CLOSES. Every one of those flips was measured and bit-verified
# on an NVIDIA H100 ONLY. mojolearn's headline claim is bitwise identity across
# Apple Metal, NVIDIA CUDA and AMD HIP, so until a gfx942 build has COMPILED
# and RUN the flipped branches, the claim is unclosed for two days of merges.
#
# PRIORITY ORDER, and the phases follow it, so a deadline strands the least
# important work first:
#   1. IDENTITY on AMD   -- do the flipped defaults give the same bits?
#      Two independent ways, because they answer different questions:
#        (a) CROSS-VENDOR: identity_break at the shipped default, diffed on the
#            Mac against the Apple M4 and H100 baselines of the SAME default.
#            This is the headline claim and needs no switch.
#        (b) BEFORE/AFTER on this box: for the flips that still HAVE a switch,
#            rebuild the old arm and diff. Most of these flips have no switch
#            at all (the old code was deleted, not gated), so (a) is the only
#            question that can be asked of them.
#   2. THE SHIPPED GATE  -- every build here is a SHIPPED build, no trial
#      define and no EVERY_COLUMN knob. The neural lanes' recorded trap is that
#      legs measured TRIAL arms against OLD shipped defaults, so no shipped AMD
#      build had ever compiled the flipped branches; that gap hid a real NVIDIA
#      correctness hole. A measurement cannot find that, only a shipped build.
#   3. SPEED -- deliberately NOT here. This leg is correctness only.
#
# A MOJOLEARN_GEMM_LEG_EXTRA body for tools/hotaisle_leg.sh. POSIX sh only.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/amd-conf
mkdir -p "$OUT/logs" "$OUT/ib"
JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
G="$OUT/gate.txt"
ok=1
TIER=python/mojolearn/identical
# The ROCm image's bare python3 has no numpy and identity_break imports it.
PY="pixi run python"
export MOJOLEARN_SPEED_PY="pixi run python"

say() { echo "$@" >> "$G"; }
# Every step records name, exit and seconds. A red step does NOT stop the next
# one: a red step is a finding and the rest of the board still matters.
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    _t1=$(date +%s)
    echo "$_n	$_e	$((_t1 - _t0))" >> "$OUT/status.tsv"
    [ "$_e" = 0 ] || ok=0
    return "$_e"
}

say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "vendor=${MOJOLEARN_TARGET_COLUMN:-unset} archs=${MOJOLEARN_GPU_ARCHS:-unset} jobs=$JOBS"
say "commit=$(cat /root/mojolearn/COMMIT 2>/dev/null || echo unknown)"
# Read the arch back from the device rather than trusting the runner's word.
(rocminfo 2>/dev/null | grep -m3 -oE "gfx[0-9a-z]+"; rocm-smi --showproductname 2>/dev/null) \
    > "$OUT/logs/device.txt" 2>&1
say "device_gfx=$(grep -o -m1 'gfx[0-9a-z]*' "$OUT/logs/device.txt" 2>/dev/null || echo unknown)"

# ===================================================== PHASE 1: SHIPPED BUILDS
# The bindings the flipped lanes live in, built FIRST so a deadline cannot
# strand the gate. MOJOLEARN_NUMERIC_MODE=identical is the shipped tier for
# everything that is not a tree lane (DEVIATION 2490).
#   base       kmeans, knn          (DEVIATIONS 2631, 2672)
#   estimators pca, ols, tsvd       (2620-2622, 2671, 2680)
#   rf/trees   rf, extratrees       (2637, 2663)
#   gbdt       gradient boosting    (2634, 2635)
#   svm        svc AND iforest      (2623, 2665, 2666, 2638)
BUILD_ENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$JOBS"
run build-base $BUILD_ENV sh bindings/build.sh
for b in estimators rf trees gbdt svm; do
    run "build-$b" $BUILD_ENV sh "bindings/build_$b.sh"
done
sha256sum "$TIER"/*.so > "$OUT/bindings.sha256" 2>&1
say "bindings_built=$(grep -c . "$OUT/bindings.sha256" 2>/dev/null || echo 0)"
cat "$OUT/bindings.sha256" >> "$G" 2>/dev/null
# Snapshot the shipped tier as the A/B helper's baseline AND default set.
mkdir -p /root/bins/baseline /root/bins/dflt
cp "$TIER"/*.so /root/bins/baseline/ 2>/dev/null
cp "$TIER"/*.so /root/bins/dflt/ 2>/dev/null

IB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python"

# ================================================ PHASE 2: CROSS-VENDOR BITS
#   forest -- EXACTLY the five lanes and nine fixtures of the Apple M4 baseline
#             (bench/results/forest_finish_2026-09-11/logs/ib_2663_apple_m4.json,
#             45 cells) and of the H100 `dflt` set, so the Mac can diff three
#             vendors cell for cell. The bits question for the forest flips.
#   wide   -- the other flipped lanes. No same-default baseline exists on
#             another vendor yet, so these come home as WITNESSES, not a diff.
#             Each cell is still fitted twice in one process, so a run-to-run
#             mover on this box shows as MOVED with no second vendor needed.
run ib-forest $IB $PY tools/identity_break.py \
    --lanes rf-clf,rf-reg,et-clf,et-reg,iforest \
    --vendor amd-mi300x-gfx942 --json "$OUT/ib/amd_forest.json"
run ib-wide $IB $PY tools/identity_break.py \
    --lanes kmeans,pca,ols,knn,knn-clf,knn-reg,kde,svc,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse \
    --vendor amd-mi300x-gfx942 --json "$OUT/ib/amd_wide.json"
for _f in forest wide; do
    grep -E "^summary:|MOVED|DIVERGENT|REFUSED" "$OUT/logs/ib-$_f.log" 2>/dev/null \
        | head -20 | sed "s/^/ib_$_f: /" >> "$G"
done

# ================================================== PHASE 3: THE NAMED CHECKS
# The gates HANDOFF section 3 owes on gfx942. Each carries its own sabotage
# arms; none reads a dataset, so nothing here needs taxi or Istella-S staged.
CHK="tools/with_identical_mode.sh pixi run mojo run -I ."
run check-if             $CHK isolation_forest/checks/if_check.mojo
run check-et-batched     $CHK extratrees/checks/device_batched_check.mojo
run check-knn-identity   $CHK neighbors/checks/knn_identity_check.mojo
run check-knn            $CHK neighbors/knn_main.mojo
run check-knn-qbatch     $CHK neighbors/checks/query_batch_check.mojo
run check-kde            $CHK kde/checks/kde_check.mojo
run kde-stage-profile    $CHK kde/checks/kde_stage_profile.mojo
run check-svm            $CHK svm/svc_main.mojo
run check-sub-byte       $CHK gbdt/methods/greedy_subsets_searcher/kernel/sub_byte_layout_gate.mojo
run check-forest-layouts bash tools/check_forest_resident_layouts.sh "$OUT/logs/forest_layouts"

# =============================================== PHASE 4: DEVIATION 2680, AMD
# The Jacobi eigensolver's LAUNCH width is not its numeric FOLD width. It was
# verified on NVIDIA only, it is an IDENTICAL-path change, and OLS, PCA, SVD
# and lstsq_min_norm all route through it -- exactly the shape of change that
# can hold its bits on one vendor and move them on another. NVIDIA refused a
# 1024-thread block outright (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES), so AMD's
# viable rungs may differ; the gate prints the widths it actually ran.
run check-jacobi $CHK decomposition/checks/jacobi_check.mojo
run check-ols    $CHK glm/checks/ols_check.mojo
run check-pca    $CHK decomposition/checks/pca_check.mojo
{
    grep -hE "check_jacobi_is_launch_invariant|cells differing|block" "$OUT/logs/check-jacobi.log" 2>/dev/null | tail -40 | sed 's/^/jacobi: /'
    grep -hE "^check_ols_is_launch_invariant|^check_ols_rank_guard|OK|FAIL" "$OUT/logs/check-ols.log" 2>/dev/null | tail -6 | sed 's/^/ols: /'
    grep -hE "OK|FAIL" "$OUT/logs/check-pca.log" 2>/dev/null | tail -4 | sed 's/^/pca: /'
} >> "$G" 2>/dev/null

# ============================================ PHASE 5: SVC CROSS-VENDOR HASHES
# These three fit hashes are already recorded for the H100, H200, M4 and an
# MI300X, so matching them on a SHIPPED gfx942 build carrying 2623/2665/2666 is
# a direct bit-level cross-vendor witness rather than a local PASS. The probe is
# INLINED because the leg bundle excludes bench/results/, where the original
# svm_probe.py lives; the arithmetic is copied from it verbatim.
run svc-hash $IB $PY - <<'PY'
import hashlib, os, sys
import numpy as np
sys.path.insert(0, "/root/mojolearn/python")
os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
import mojolearn                                   # noqa: E402
from mojolearn import _svm_impl as m               # noqa: E402
print("ext", m._extension(None).__file__, flush=True)
EXPECT = {400: "457e29b82bca9df9", 600: "733a383c5699f427", 2000: "2b66bc991a9c9ed0"}
rng = np.random.default_rng(0)
bad = 0
for n in (400, 600, 2000):
    X = rng.standard_normal((n, 220)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.standard_normal(n) > 0).astype(np.float32)
    est = m.SVC(); est.fit(X, y)
    h = hashlib.sha256()
    for name in ("dual_coef_", "support_", "intercept_", "support_vectors_", "n_support_"):
        if hasattr(est, name):
            arr = np.ascontiguousarray(np.asarray(getattr(est, name)))
            h.update(name.encode()); h.update(arr.tobytes())
    dec = np.ascontiguousarray(np.asarray(est.decision_function(X[:256]), dtype=np.float32))
    h.update(dec.tobytes())
    got = h.hexdigest()[:16]
    if got != EXPECT[n]:
        bad += 1
    print("SVCHASH n=%d %s want=%s %s" % (n, got, EXPECT[n], "MATCH" if got == EXPECT[n] else "DIFFERENT"), flush=True)
print("SVCHASH_OK=%d" % (0 if bad else 1), flush=True)
PY
grep -hE "SVCHASH" "$OUT/logs/svc-hash.log" 2>/dev/null | sed 's/^/svc_hash: /' >> "$G"
grep -q "SVCHASH_OK=1" "$OUT/logs/svc-hash.log" 2>/dev/null || { say "SVC FIT HASHES DID NOT MATCH"; ok=0; }

# ======================================= PHASE 6: BEFORE/AFTER ON THIS BOX
# Only for the flips that still have a switch. tools/trees_identical_ab.sh is
# the lane's proven helper and it sets BOTH define variable names, which
# matters: build_trees/rf/gbdt read MOJOLEARN_EXTRA_DEFINES while
# build.sh/build_svm read MOJOLEARN_BUILD_EXTRA_DEFINES, and passing the wrong
# one builds the DEFAULT arm and hands back a false "bits unchanged".
#
# REACH IS PROVEN BY THE ARTIFACT, NOT BY THE NUMBER: each arm's .so sha256
# must DIFFER from the default's. An equal hash means the define selected
# nothing and the comparison is inert, which is reported as a FAILURE here.
AB="sh tools/trees_identical_ab.sh"
ab_arm() {   # <name> <binding> <so> <lanes> <defines...>
    _name=$1; _bind=$2; _so=$3; _lanes=$4; shift 4
    # the default side, restricted to the same lanes
    $AB use dflt > "$OUT/logs/ab-$_name-use-dflt.log" 2>&1
    run "ab-$_name-ib-dflt" $IB $PY tools/identity_break.py --lanes "$_lanes" \
        --vendor amd-gfx942 --json "$OUT/ib/ab_${_name}_dflt.json"
    # the old arm
    run "ab-$_name-build" $AB build "$_name" "$_bind" "$@"
    _h_d=$(sha256sum "/root/bins/dflt/$_so" 2>/dev/null | cut -c1-16)
    _h_a=$(sha256sum "/root/bins/$_name/$_so" 2>/dev/null | cut -c1-16)
    say "ab_$_name: so_dflt=$_h_d so_arm=$_h_a defines='$*'"
    if [ -z "$_h_a" ] || [ "$_h_d" = "$_h_a" ]; then
        say "ab_$_name: REACH FAILED -- the arm's .so equals the default's, so the define selected NOTHING; this comparison is inert"
        ok=0
        $AB use dflt > /dev/null 2>&1
        return 1
    fi
    $AB use "$_name" > "$OUT/logs/ab-$_name-use-arm.log" 2>&1
    run "ab-$_name-ib-arm" $IB $PY tools/identity_break.py --lanes "$_lanes" \
        --vendor amd-gfx942 --json "$OUT/ib/ab_${_name}_arm.json"
    run "ab-$_name-diff" $IB $PY tools/identity_break.py --diff \
        "$OUT/ib/ab_${_name}_dflt.json" "$OUT/ib/ab_${_name}_arm.json"
    grep -hE "^summary:|DIVERGENT|MOVED|REFUSED" "$OUT/logs/ab-$_name-diff.log" 2>/dev/null \
        | head -12 | sed "s/^/ab_$_name: /" >> "$G"
    $AB use dflt > /dev/null 2>&1
}

# DEVIATION 2663: ExtraTrees frontier batch width 16384 -> the old 4096.
ab_arm et2663 trees _mojolearn_trees.so et-clf,et-reg -D MOJOLEARN_ET_DEVICE_BATCH_4096=1
# DEVIATION 2631: the kNN query tile and radix scratch, back to the old rule.
ab_arm knn2631 base _mojolearn.so knn,knn-clf,knn-reg \
    -D MOJOLEARN_KNN_LEGACY_QUERY_TILE=1 -D MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH=1
# DEVIATIONS 2623/2666: the SVM block-solve schedule, back to the halving trees.
# The isolation forest shares this extension, so 2638's lane rides along.
ab_arm svm2666 svm _mojolearn_svm.so svc,iforest -D MOJOLEARN_SVM_TREE_FOLDS=1
# DEVIATIONS 2634/2635: GBDT CTR prep and the binarization bound walks.
ab_arm gbdt2634 gbdt _mojolearn_gbdt.so gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse \
    -D MOJOLEARN_2634_CTR_PREP_OFF=1 -D MOJOLEARN_2635_LINEAR_BOUNDS=1

$AB use dflt > /dev/null 2>&1

# =========================================== PHASE 7: THE REMAINING BINDINGS
# "Build EVERY bindings/build_*.sh" is the recorded rule, because identity_break
# REFUSES a lane whose extension is missing and a partial build looks like a
# result. The lanes above do not need these, so they are built LAST, where a
# deadline costs only completeness.
for f in bindings/build_*.sh; do
    _b=$(basename "$f" .sh); _b=${_b#build_}
    case "$_b" in estimators|rf|trees|gbdt|svm) continue ;; esac
    run "build-$_b" $BUILD_ENV sh "$f"
done

# The helper writes under /root/trees_out, which the runner does NOT fetch.
cp -r /root/trees_out "$OUT/trees_out" 2>/dev/null
cp -r /root/bins "$OUT/bins_sha" 2>/dev/null && find "$OUT/bins_sha" -name '*.so' -delete 2>/dev/null
sha256sum /root/bins/*/*.so > "$OUT/sets_so_sha256.txt" 2>&1

say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "AMD_CONF_OK=$ok"
echo "=== status.tsv ==="; cat "$OUT/status.tsv" 2>/dev/null
echo "=== gate.txt ===";   cat "$G" 2>/dev/null
[ "$ok" = 1 ]
