#!/bin/sh
# AMD gfx942 post-promotion identity gate for the shared-memory k-NN tile.
#
# Run this as the extra body on one guarded Hot Aisle MI300X or DigitalOcean
# MI325X lease.  It builds the promoted defaults, the tile's reach sabotage,
# and the CPU host oracle from the same source.  No dataset download is needed:
# identity_break and the width probe generate fixed inputs locally.
#
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/knn_amd_identity_leg.sh \
#   sh tools/hotaisle_leg.sh ...
#
# POSIX sh.  A missing backend, build, cell, comparison, or reach result is a
# hard failure; the output directory is retained for diagnosis.
set -eu

R=${MOJOLEARN_REMOTE_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_KNN_AMD_IDENTITY_OUT:-/root/gemm_leg_out/knn-amd-identity}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
P=$R/.pixi/envs/default/bin/python3
JOBS=${MOJOLEARN_COMPILE_JOBS:-13}
ARCH=${MOJOLEARN_GPU_ARCHS:-gfx942}
FIXTURES=${MOJOLEARN_KNN_IDENTITY_FIXTURES:-base,ties,hashed,wide,denormal,denormal_ftz,dupes,odd,negative}
KNN_LANES=knn,knn-chebyshev,knn-clf,knn-clf-distance,knn-cosine,knn-manhattan,knn-minkowski-p3,knn-rbc,knn-reg,knn-reg-distance,knn-sqeuclidean,radius,radius-chebyshev,radius-manhattan,radius-minkowski-p3
KDE_LANES=kde,kde-cosine-minkowski,kde-epanechnikov-l1,kde-exponential-chebyshev,kde-linear-cosine,kde-tophat-sqeuclidean,kde-weighted
LANES=$KNN_LANES,$KDE_LANES
SABOTAGE='-D MOJOLEARN_KNN_SMEM_TILE_SABOTAGE=1'

mkdir -p "$OUT/logs" /root/knn-amd-bins/default /root/knn-amd-bins/sabotage /root/knn-amd-host
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_GPU_ARCHS="$ARCH" MOJOLEARN_COMPILE_JOBS="$JOBS" MOJOLEARN_BUILD_JOBS="$JOBS"
export PYTHONUNBUFFERED=1

note() { printf '[%s knn-amd-identity] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$OUT/progress.txt"; }
die() { note "FAIL: $*"; exit 1; }
step() {
    _name=$1 _cap=$2
    shift 2
    _start=$(date +%s)
    note "$_name start"
    if timeout -k 30 "$_cap" "$@" > "$OUT/logs/$_name.log" 2>&1; then
        _rc=0
    else
        _rc=$?
    fi
    printf '%s\t%s\t%s\n' "$_name" "$_rc" "$(( $(date +%s) - _start ))" >> "$OUT/status.tsv"
    note "$_name rc=$_rc"
    [ "$_rc" -eq 0 ] || die "$_name failed; see $OUT/logs/$_name.log"
}

cd "$R"
[ "$ARCH" = gfx942 ] || die "this gate is for gfx942, got MOJOLEARN_GPU_ARCHS=$ARCH"
command -v rocminfo >/dev/null 2>&1 || die "rocminfo is absent; this is not a usable HIP lease"
[ -e /dev/kfd ] || die "/dev/kfd is absent; HIP cannot run"
rocminfo > "$OUT/logs/rocminfo.txt" 2>&1 || die "rocminfo failed"
grep -q 'gfx942' "$OUT/logs/rocminfo.txt" || die "rocminfo did not report gfx942"
{ date -u; uname -a; nproc; grep -m1 'model name' /proc/cpuinfo || true; rocm-smi --showproductname || true; } > "$OUT/box.txt" 2>&1

if [ ! -x "$PIXI" ]; then
    step pixi_get 300 sh -c 'curl -fsSL https://pixi.sh/install.sh | sh'
fi
step pixi_install 1500 "$PIXI" install
"$PIXI" run mojo --version > "$OUT/mojo_version.txt" 2>&1

# The remote archive normally carries SHIPPED_COMMIT.txt.  The generic legs
# instead record it in leg.txt; accept either, but never fabricate a witness.
COMMIT=${MOJOLEARN_COMMIT:-}
if [ -z "$COMMIT" ] && [ -s "$R/SHIPPED_COMMIT.txt" ]; then COMMIT=$(head -1 "$R/SHIPPED_COMMIT.txt"); fi
if [ -z "$COMMIT" ] && [ -s "$R/commit.txt" ]; then COMMIT=$(head -1 "$R/commit.txt"); fi
if [ -z "$COMMIT" ]; then COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1); fi
case "$COMMIT" in
    [0-9a-f][0-9a-f]*) ;;
    *) die "no valid commit witness (got '$COMMIT')" ;;
esac
printf '%s\n' "$COMMIT" > "$OUT/commit.txt"

# The package imports portable math while defining defaults, even though this
# gate only exercises neighbors.
step portable_math 300 env PYTHONPATH="$R/packaging/portable_math" "$P" -c \
    "import pathlib, stage; stage.build(pathlib.Path('$R/python/mojolearn/.libs/libMojolearnMath.so'))"

build_core() {
    _label=$1 _defs=${2:-}
    rm -f "$R/python/mojolearn/identical/_mojolearn.so"
    step "build_${_label}_core" 1800 env MOJOLEARN_BUILD_EXTRA_DEFINES="$_defs" sh "$R/bindings/build.sh"
    cp "$R/python/mojolearn/identical/_mojolearn.so" "/root/knn-amd-bins/$_label/"
}

build_core default ''
step build_default_estimators 1800 sh "$R/bindings/build_estimators.sh"
cp "$R/python/mojolearn/identical/_mojolearn_estimators.so" /root/knn-amd-bins/default/
build_core sabotage "$SABOTAGE"
cp /root/knn-amd-bins/default/_mojolearn_estimators.so /root/knn-amd-bins/sabotage/
step build_host_core 1500 env MOJOLEARN_HOST_OUTDIR=/root/knn-amd-host sh "$R/bindings/build_core_host.sh"
step build_host_estimators 1500 env MOJOLEARN_HOST_OUTDIR=/root/knn-amd-host sh "$R/bindings/build_estimators_host.sh"
sha256sum /root/knn-amd-bins/*/*.so /root/knn-amd-host/*.so > "$OUT/bindings.sha256"

make_tree() {
    _label=$1
    _tree=/root/knn-amd-identity-$_label
    rm -rf "$_tree"
    mkdir -p "$_tree"
    (cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' .) | (cd "$_tree" && tar xf -)
    ln -s "$R/.pixi" "$_tree/.pixi"
    mkdir -p "$_tree/python/mojolearn/identical"
    rm -rf "$_tree/python/mojolearn/.libs"
    ln -s "$R/python/mojolearn/.libs" "$_tree/python/mojolearn/.libs"
    cp "/root/knn-amd-bins/$_label/"*.so "$_tree/python/mojolearn/identical/"
}
make_tree default
make_tree sabotage
CPU=/root/knn-amd-identity-cpu
rm -rf "$CPU" && mkdir -p "$CPU"
(cd "$R" && tar cf - --exclude=.pixi --exclude='*.so' .) | (cd "$CPU" && tar xf -)
ln -s "$R/.pixi" "$CPU/.pixi"
rm -rf "$CPU/python/mojolearn/.libs"
ln -s "$R/python/mojolearn/.libs" "$CPU/python/mojolearn/.libs"

run_identity() {
    _label=$1 _tree=$2 _backend=$3
    shift 3
    step "identity_$_label" 1800 env \
        PYTHONPATH="$_tree/python:$_tree/tools" MOJOLEARN_COMMIT="$COMMIT" "$@" \
        "$PIXI" run --manifest-path "$R/pixi.toml" python3 "$_tree/tools/identity_break.py" \
        --require-backend "$_backend" --fail-on-refused --lanes "$LANES" \
        --fixtures "$FIXTURES" --repeats 2 --vendor "amd-gfx942-$_label" \
        --json "$OUT/$_label.json"
}

# All current identity fixtures have d=16 (odd has d=17), below the promoted
# AMD threshold.  The sabotage must therefore remain invisible in this broad
# matrix; it detects accidental routing changes outside the promoted domain.
run_identity default /root/knn-amd-identity-default hip
run_identity sabotage /root/knn-amd-identity-sabotage hip
run_identity cpu "$CPU" cpu MOJOLEARN_HOST_DIR=/root/knn-amd-host

diff_equal() {
    _name=$1
    shift
    step "diff_$_name" 300 env PYTHONPATH="$R/python:$R/tools" "$PIXI" run \
        --manifest-path "$R/pixi.toml" python3 "$R/tools/identity_break.py" \
        --diff "$@" --lanes "$LANES" --require-columns 2
}
diff_equal default_cpu "$OUT/default.json" "$OUT/cpu.json"
diff_equal narrow_default_sabotage "$OUT/default.json" "$OUT/sabotage.json"

# A separate width probe reaches the promoted d>=32 route.  It covers the
# shipped block-top-k range at k=1,8,16, both L2 spellings, classifier and
# regressor (uniform and distance weights), radius output, KDE, and the other
# public metrics as collateral identity checks.  Every arm runs in a fresh
# process; the JSON stores complete caller-visible array hashes.
cat > "$OUT/width_probe.py" <<'PY'
import hashlib, json, os, sys
import numpy as np
import mojolearn as ml

expected = sys.argv[1]
if ml.vendor() != expected:
    raise SystemExit(f"REFUSING backend {ml.vendor()!r}; expected {expected!r}")
if ml.numeric_mode() != "identical":
    raise SystemExit(f"REFUSING numeric mode {ml.numeric_mode()!r}")

def digest(*values):
    h = hashlib.sha256()
    for value in values:
        if isinstance(value, (tuple, list)):
            h.update(digest(*value).encode())
            continue
        a = np.ascontiguousarray(np.asarray(value))
        if a.dtype.kind in "OUSV":
            raise TypeError(f"non-numeric output {a.dtype}")
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def ragged(result):
    distances, indices = result
    rows_i = [np.asarray(indices[i]).reshape(-1).astype(np.int64) for i in range(len(indices))]
    rows_d = [np.asarray(distances[i], dtype=np.float32).reshape(-1) for i in range(len(distances))]
    lengths = np.asarray([row.size for row in rows_i], dtype=np.int64)
    dd = np.concatenate(rows_d) if rows_d else np.zeros(0, dtype=np.float32)
    ii = np.concatenate(rows_i) if rows_i else np.zeros(0, dtype=np.int64)
    return lengths, dd, ii

def data(d):
    # Integer-exact generation, independent of a random-library version.
    n, q = 4096, 64
    z = np.arange((n + q) * d, dtype=np.uint64)
    z = (z * np.uint64(6364136223846793005) + np.uint64(1442695040888963407))
    x = (((z >> np.uint64(40)) & np.uint64(0xffff)).astype(np.int32) - 32768).astype(np.float32)
    x = np.ascontiguousarray(x.reshape(n + q, d) * np.float32(1.0 / 4096.0))
    return x[:n], x[n:], (np.arange(n) % 7).astype(np.int32), ((np.arange(n) % 101) - 50).astype(np.float32)

out = {"vendor": ml.vendor(), "mode": ml.numeric_mode(), "cases": {}}
for d in (32, 220):
    X, Q, yc, yr = data(d)
    ks = (1, 8, 16) if d == 32 else (1, 16)
    for metric in ("euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine", "minkowski"):
        kw = {"metric": metric}
        if metric == "minkowski": kw["p"] = 3
        for k in ks:
            m = ml.NearestNeighbors(n_neighbors=k, **kw).fit(X)
            out["cases"][f"nn-d{d}-{metric}-k{k}"] = digest(*m.kneighbors(Q))
    if d == 32:
        for weights in ("uniform", "distance"):
            c = ml.KNeighborsClassifier(n_neighbors=16, weights=weights).fit(X, yc)
            out["cases"][f"clf-d32-{weights}"] = digest(c.predict(Q), c.predict_proba(Q), c.kneighbors(Q))
            r = ml.KNeighborsRegressor(n_neighbors=16, weights=weights).fit(X, yr)
            out["cases"][f"reg-d32-{weights}"] = digest(r.predict(Q), r.kneighbors(Q))
        # Radius is derived from this exact input so the result is nonempty and
        # nontrivial on all three implementations.
        rr = ml.RadiusNeighbors(radius=37.0).fit(X).radius_neighbors(Q, sort_results=True)
        out["cases"]["radius-d32"] = digest(*ragged(rr))
        kd = ml.KernelDensity(bandwidth=0.7).fit(X)
        out["cases"]["kde-d32"] = digest(kd.score_samples(Q))
json.dump(out, open(sys.argv[2], "w"), indent=1, sort_keys=True)
print(json.dumps({"vendor": out["vendor"], "cases": len(out["cases"])}, sort_keys=True))
PY

step width_default 1200 env PYTHONPATH=/root/knn-amd-identity-default/python "$P" "$OUT/width_probe.py" hip "$OUT/width-default.json"
step width_sabotage 1200 env PYTHONPATH=/root/knn-amd-identity-sabotage/python "$P" "$OUT/width_probe.py" hip "$OUT/width-sabotage.json"
step width_cpu 1800 env PYTHONPATH="$CPU/python" MOJOLEARN_HOST_DIR=/root/knn-amd-host "$P" "$OUT/width_probe.py" cpu "$OUT/width-cpu.json"

step judge_width 60 "$P" - "$OUT/width-default.json" "$OUT/width-sabotage.json" "$OUT/width-cpu.json" "$OUT/verdict.json" <<'PY'
import json, sys
default, sabotage, cpu = (json.load(open(p)) for p in sys.argv[1:4])
d, s, c = default["cases"], sabotage["cases"], cpu["cases"]
if set(d) != set(s) or set(d) != set(c):
    raise SystemExit("case coverage differs between arms")
mismatch = sorted(k for k in d if d[k] != c[k])
moved = sorted(k for k in d if d[k] != s[k])
# These cases all traverse the Euclidean/squared-Euclidean tile at d>=32.
required = [k for k in d if (k.startswith("nn-") and ("-euclidean-" in k or "-sqeuclidean-" in k))]
required += ["clf-d32-uniform", "clf-d32-distance", "reg-d32-uniform",
             "reg-d32-distance", "radius-d32"]
unreached = sorted(k for k in required if k not in moved)
record = {"default_cpu_bitwise_equal": not mismatch, "default_cpu_mismatches": mismatch,
          "sabotage_moved_count": len(moved), "sabotage_moved": moved,
          "required_reach_count": len(required), "unreached": unreached}
json.dump(record, open(sys.argv[4], "w"), indent=1, sort_keys=True)
print(json.dumps(record, sort_keys=True))
if mismatch or unreached or not moved:
    raise SystemExit(1)
PY

note "PASS: broad identity, CPU oracle, threshold isolation, and d>=32 sabotage reach"
touch "$OUT/PASS"
