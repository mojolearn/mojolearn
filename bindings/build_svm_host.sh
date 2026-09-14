#!/bin/sh
# CPU-only build of the svm binding, SVC today (the CPU training lane, phase 1,
# 2026-09-13), bindings/build_forest_host.sh flag for flag.
# No accelerator target. On the Mac, invoke through tools/macos_serial_guard.py.
# IDENTICAL only. MOJOLEARN_BUILD_EXTRA_DEFINES carries trial defines, e.g.
# -D MOJOLEARN_HOST_SABOTAGE=1 for the gate's negative control.
# MOJOLEARN_BUILD_JOBS sets the compile jobs (default 2). The output directory is
# MOJOLEARN_SVM_HOST_OUTDIR, else MOJOLEARN_HOST_OUTDIR (the gate builds the
# whole phase 1 sabotage set into one directory), else python/mojolearn/host.
# The kernel-matrix column is COLUMN_CPU (-D MOJOLEARN_COLUMN_CPU, the CPU
# training lane 2026-09-13); MOJOLEARN_TARGET_COLUMN may only say `cpu`, and
# the binding asserts at build time that it compiled as that column.
set -eu
MACOS_FLOOR="11.0"
host_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$host_root"
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" = identical ] || { echo 'svm host supports only MOJOLEARN_NUMERIC_MODE=identical' >&2; exit 2; }
[ "${MOJOLEARN_TARGET_COLUMN:-cpu}" = cpu ] || { echo "svm host compiles the CPU column only; MOJOLEARN_TARGET_COLUMN=$MOJOLEARN_TARGET_COLUMN is refused" >&2; exit 2; }
host_system=$(uname -s)
if [ "$host_system" = Darwin ]; then
    [ "$(uname -m)" = arm64 ] || { echo 'svm host: macOS requires Apple silicon' >&2; exit 2; }
    unset MACOSX_DEPLOYMENT_TARGET
    host_sdk=$(xcrun --sdk macosx --show-sdk-version)
    set -- --target-cpu apple-m1 \
        -Xlinker -platform_version -Xlinker macos -Xlinker "$MACOS_FLOOR" -Xlinker "$host_sdk"
elif [ "$host_system" = Linux ]; then
    [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] || { echo 'svm host: a CPU build takes no MOJOLEARN_GPU_ARCHS' >&2; exit 2; }
    case "$(uname -m)" in
        x86_64) set -- --target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3} ;;
        aarch64) set -- ;;
        *) echo 'svm host: unsupported Linux host architecture' >&2; exit 2 ;;
    esac
else
    echo 'svm host: requires Linux or macOS' >&2
    exit 2
fi
host_outdir=${MOJOLEARN_SVM_HOST_OUTDIR:-${MOJOLEARN_HOST_OUTDIR:-python/mojolearn/host}}
mkdir -p "$host_outdir"
host_destination="$host_outdir/_mojolearn_svm_host.so"
if [ -e "$host_destination" ] || [ -L "$host_destination" ]; then
    echo 'svm host: output already exists; choose a fresh output directory' >&2
    exit 2
fi
host_tmpdir=$(mktemp -d "$host_outdir/.svm-host-build.XXXXXX")
trap 'rm -rf "$host_tmpdir"' EXIT HUP INT TERM
pixi run mojo build -j "${MOJOLEARN_BUILD_JOBS:-2}" --emit shared-lib "$@" ${MOJOLEARN_BUILD_EXTRA_DEFINES:-} \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings \
    bindings/_mojolearn_svm_host.mojo -o "$host_tmpdir/_mojolearn_svm_host.so"
ln "$host_tmpdir/_mojolearn_svm_host.so" "$host_destination"
echo "built $host_destination"
