#!/bin/sh
# The prep expansion lane's GPU binding: preprocessing additions, naive Bayes,
# discriminant analysis (x_prep/, naive_bayes/; algorithm expansion lane 5).
set -eu
MACOS_FLOOR="11.0"
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh sh bindings/build_x_prep.sh "$@"
fi
# Match existing bindings: environment deployment target disables Metal AOT.
unset MACOSX_DEPLOYMENT_TARGET
# TWO TIERS (2026-09-25, reversing DEVIATION 2490 for classical ML): this
# binding builds IDENTICAL (the default, bitwise across vendors) or FAST
# (per-vendor speed, same quality, no bit promise). The DETERMINISTIC tier
# stays tree-only (build_gbdt.sh, build_rf.sh, build_trees.sh); refusing it
# here, by name, keeps it an unshipped tier rather than an unchecked one.
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" != deterministic ] || {
    echo 'build_x_prep.sh: the deterministic tier ships for the tree lanes (gbdt, rf, trees) only; classical bindings build MOJOLEARN_NUMERIC_MODE=identical (default) or fast.' >&2
    exit 2; }
mode=${MOJOLEARN_NUMERIC_MODE:-identical}
mode_flags=""
outdir=python/mojolearn
case "$mode" in
    fast) expected=0 ;;
    deterministic) expected=2; mode_flags="-D MOJOLEARN_NUMERIC_DETERMINISTIC=1"; outdir=$outdir/deterministic ;;
    identical) expected=1; mode_flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1"; outdir=$outdir/identical ;;
    *) echo "invalid numeric mode: $mode" >&2; exit 2 ;;
esac
link_flags=""
target_flags=""
if [ "$(uname)" = Darwin ]; then
    sdk=$(xcrun --sdk macosx --show-sdk-version)
    link_flags="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $sdk"
    target_flags="--target-cpu apple-m1 --target-accelerator metal:1"
else
    case "$(uname -m)" in
        x86_64) target_flags="--target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3}" ;;
    esac
    if [ -n "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        case "$MOJOLEARN_GPU_ARCHS" in *,*) echo "one GPU architecture required" >&2; exit 2 ;; esac
        target_flags="$target_flags --target-accelerator $MOJOLEARN_GPU_ARCHS"
    fi
fi
column_flags=""
if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
    column_flags="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
fi
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-preprocessing.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out=$tmpdir/_mojolearn_x_prep.so
# Intentionally split compiler option lists, consistent with existing builders.
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib ${MOJOLEARN_MOJO_BUILD_FLAGS:-} \
    $target_flags $link_flags $mode_flags $column_flags -I . -I bindings \
    bindings/_mojolearn_x_prep.mojo -o "$out"
# The gate below needs NumPy in the gating interpreter, which a fresh Linux
# build box does not have; the caller that sets MOJOLEARN_SKIP_BUILD_GATE
# (build_sets.sh, build_release_wheel.sh) owns end-to-end verification, the
# same contract as build_metrics.sh and build_svm.sh. First seen on the AMD
# 0.8.0 release leg: this script was the one binding of 22 that did not build.
if [ -n "${MOJOLEARN_SKIP_BUILD_GATE:-}" ] || [ "$(uname)" != "Darwin" ]; then
    mkdir -p "$outdir"
    mv "$out" "$outdir/_mojolearn_x_prep.so"
    echo "built $outdir/_mojolearn_x_prep.so (gate skipped: non-Darwin or MOJOLEARN_SKIP_BUILD_GATE)"
    exit 0
fi
"${MOJOLEARN_PYTHON:-python3}" - "$out" "$expected" <<'PY'
import importlib.util
import sys
import numpy as np
spec = importlib.util.spec_from_file_location('_mojolearn_x_prep', sys.argv[1])
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
assert b.x_prep_numeric_mode() == int(sys.argv[2])
arena = np.zeros(4 + 12, dtype=np.float32)
arena[:4] = [1, -2, 3, 2]
prog = np.zeros((1, 16), dtype=np.int32)
prog[0, :6] = [1, 2, 0, 2, 2, 4]            # col_stats over the 2 x 2 block
b.x_prep_run(arena.ctypes.data, arena.size, prog.ctypes.data, 1)
np.testing.assert_array_equal(arena[4:8], [2, 2, 2, 0])   # count, mean
print('PASS x_prep native ABI', b.x_prep_numeric_mode(), b.x_prep_vendor())
PY
mkdir -p "$outdir"
mv "$out" "$outdir/_mojolearn_x_prep.so"
echo "built $outdir/_mojolearn_x_prep.so"
