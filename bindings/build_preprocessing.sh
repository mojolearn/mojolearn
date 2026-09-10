#!/bin/sh
# Compile a separate preprocessing extension; launch fit/forward/inverse before install.
set -eu
MACOS_FLOOR="11.0"
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh sh bindings/build_preprocessing.sh "$@"
fi
# Match existing bindings: environment deployment target disables Metal AOT.
unset MACOSX_DEPLOYMENT_TARGET
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
out=$tmpdir/_mojolearn_preprocessing.so
# Intentionally split compiler option lists, consistent with existing builders.
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib \
    $target_flags $link_flags $mode_flags $column_flags -I . -I bindings \
    bindings/_mojolearn_preprocessing.mojo -o "$out"
"${MOJOLEARN_PYTHON:-python3}" - "$out" "$expected" <<'PY'
import importlib.util
import sys
import numpy as np
spec = importlib.util.spec_from_file_location('_mojolearn_preprocessing', sys.argv[1])
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
assert b.preprocessing_numeric_mode() == int(sys.argv[2])
x = np.array([[1, 2], [3, 2]], dtype=np.float32)
stats = np.empty((5, 2), dtype=np.float32)
b.minmax_fit(x.ctypes.data, stats.ctypes.data, [2, 2, 0., 1.])
np.testing.assert_array_equal(stats, [[1, 2], [3, 2], [2, 0], [.5, 1], [-.5, -2]])
y = np.empty_like(x)
b.minmax_transform(x.ctypes.data, stats[3].ctypes.data, stats[4].ctypes.data, y.ctypes.data, [2, 2, 0, 0, 0., 1.])
np.testing.assert_array_equal(y, [[0, 0], [1, 0]])
z = np.empty_like(x)
b.minmax_transform(y.ctypes.data, stats[3].ctypes.data, stats[4].ctypes.data, z.ctypes.data, [2, 2, 1, 0, 0., 1.])
np.testing.assert_array_equal(z, x)
standard = np.empty((3, 2), dtype=np.float32)
b.standard_fit(x.ctypes.data, standard.ctypes.data, [2, 2, 1, 1])
np.testing.assert_array_equal(standard, [[2, 2], [1, 0], [1, 1]])
b.standard_transform(x.ctypes.data, standard[0].ctypes.data, standard[2].ctypes.data, y.ctypes.data, [2, 2, 0, 1, 1])
np.testing.assert_array_equal(y, [[-1, 0], [1, 0]])
b.standard_transform(y.ctypes.data, standard[0].ctypes.data, standard[2].ctypes.data, z.ctypes.data, [2, 2, 1, 1, 1])
np.testing.assert_array_equal(z, x)
print('PASS StandardScaler native ABI fit/transform/inverse')
print('PASS preprocessing native ABI fit/transform/inverse', b.preprocessing_numeric_mode(), b.preprocessing_vendor())
PY
mkdir -p "$outdir"
mv "$out" "$outdir/_mojolearn_preprocessing.so"
echo "built $outdir/_mojolearn_preprocessing.so"
