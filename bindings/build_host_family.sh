#!/bin/sh
# THE ONE CPU-ONLY BUILDER for the phase 1 host bindings (the CPU training
# lane, 2026-09-13; folded 2026-09-14 when the seventh family, trees, would
# have been the seventh copy of one script). `bindings/build_<family>_host.sh`
# is a two-line wrapper that names its family and execs this file, so the
# workflow, the release legs and a developer keep calling the per-family
# script they always did; the flags live here once.
#
#     sh bindings/build_host_family.sh <family>
#
# builds bindings/_mojolearn_<family>_host.mojo into
# <outdir>/_mojolearn_<family>_host.so, bindings/build_forest_host.sh flag for
# flag. No accelerator target. On the Mac, invoke through
# tools/macos_serial_guard.py. IDENTICAL only. MOJOLEARN_BUILD_EXTRA_DEFINES
# carries trial defines, e.g. -D MOJOLEARN_HOST_SABOTAGE=1 for the gate's
# negative control. MOJOLEARN_BUILD_JOBS sets the compile jobs (default 2).
# The output directory is MOJOLEARN_<FAMILY>_HOST_OUTDIR, else
# MOJOLEARN_HOST_OUTDIR (the gate builds the whole phase 1 sabotage set into
# one directory), else python/mojolearn/host. The kernel-matrix column is
# COLUMN_CPU (-D MOJOLEARN_COLUMN_CPU, the CPU training lane 2026-09-13);
# MOJOLEARN_TARGET_COLUMN may only say `cpu`, and every binding asserts at
# build time that it compiled as that column.
set -eu
family=${1:-}
case "$family" in
    ''|*[!a-z_]*) echo 'build_host_family: usage: build_host_family.sh <family>, lowercase letters and underscores' >&2; exit 2 ;;
esac
FAMILY=$(printf '%s' "$family" | tr 'a-z' 'A-Z')
label="$family host"
MACOS_FLOOR="11.0"
host_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$host_root"
source_file="bindings/_mojolearn_${family}_host.mojo"
[ -f "$source_file" ] || { echo "$label: no such binding source $source_file" >&2; exit 2; }
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" = identical ] || { echo "$label supports only MOJOLEARN_NUMERIC_MODE=identical" >&2; exit 2; }
[ "${MOJOLEARN_TARGET_COLUMN:-cpu}" = cpu ] || { echo "$label compiles the CPU column only; MOJOLEARN_TARGET_COLUMN=$MOJOLEARN_TARGET_COLUMN is refused" >&2; exit 2; }
host_system=$(uname -s)
if [ "$host_system" = Darwin ]; then
    [ "$(uname -m)" = arm64 ] || { echo "$label: macOS requires Apple silicon" >&2; exit 2; }
    unset MACOSX_DEPLOYMENT_TARGET
    host_sdk=$(xcrun --sdk macosx --show-sdk-version)
    set -- --target-cpu apple-m1 \
        -Xlinker -platform_version -Xlinker macos -Xlinker "$MACOS_FLOOR" -Xlinker "$host_sdk"
elif [ "$host_system" = Linux ]; then
    [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] || { echo "$label: a CPU build takes no MOJOLEARN_GPU_ARCHS" >&2; exit 2; }
    case "$(uname -m)" in
        x86_64) set -- --target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3} ;;
        aarch64) set -- ;;
        *) echo "$label: unsupported Linux host architecture" >&2; exit 2 ;;
    esac
else
    echo "$label: requires Linux or macOS" >&2
    exit 2
fi
# MOJOLEARN_<FAMILY>_HOST_OUTDIR, read through eval because the name is
# built from the family; an unset variable reads as empty under `set -u`.
family_outdir=$(eval "printf '%s' \"\${MOJOLEARN_${FAMILY}_HOST_OUTDIR:-}\"")
host_outdir=${family_outdir:-${MOJOLEARN_HOST_OUTDIR:-python/mojolearn/host}}
mkdir -p "$host_outdir"
host_destination="$host_outdir/_mojolearn_${family}_host.so"
if [ -e "$host_destination" ] || [ -L "$host_destination" ]; then
    echo "$label: output already exists; choose a fresh output directory" >&2
    exit 2
fi
host_tmpdir=$(mktemp -d "$host_outdir/.${family}-host-build.XXXXXX")
trap 'rm -rf "$host_tmpdir"' EXIT HUP INT TERM
pixi run mojo build -j "${MOJOLEARN_BUILD_JOBS:-2}" --emit shared-lib "$@" ${MOJOLEARN_BUILD_EXTRA_DEFINES:-} \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings \
    "$source_file" -o "$host_tmpdir/_mojolearn_${family}_host.so"
ln "$host_tmpdir/_mojolearn_${family}_host.so" "$host_destination"
echo "built $host_destination"
