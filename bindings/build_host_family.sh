#!/bin/sh
# THE ONE CPU-ONLY BUILDER for every host binding (the CPU training lane,
# 2026-09-13; folded 2026-09-14 when the seventh family, trees, would have
# been the seventh copy of one script; the byte LM, forest and estimators
# families joined the fold the same day, the host surface manifest lane).
# `bindings/build_<family>_host.sh` is a two-line shim that names its family
# and execs this file, so the workflows, the release legs and a developer
# keep calling the per-family script they always did; the flags live here
# once. The families are declared in python/mojolearn/host_surface.py, and
# python/mojolearn/tests/test_host_surface.py holds the shims to this file.
#
#     sh bindings/build_host_family.sh <family>
#
# builds bindings/_mojolearn_<family>_host.mojo into
# <outdir>/_mojolearn_<family>_host.so. No accelerator target. On the Mac,
# invoke through tools/macos_serial_guard.py. IDENTICAL only. The
# environment contract, unchanged from the copies it replaced:
#   MOJOLEARN_NUMERIC_MODE          identical only (the default); anything else is refused
#   MOJOLEARN_TARGET_COLUMN         cpu only (the default); anything else is refused
#   MOJOLEARN_GPU_ARCHS             refused on Linux: a CPU build takes no GPU arch
#   MOJOLEARN_COMPILE_JOBS          compile jobs, default 2
#   MOJOLEARN_BUILD_EXTRA_DEFINES   trial defines, e.g. "-D MOJOLEARN_HOST_SABOTAGE=1"
#                                   (the gate's negative control for the routed
#                                   set; the byte LM's is
#                                   MOJOLEARN_BYTE_LM_HOST_SABOTAGE, the forest's
#                                   MOJOLEARN_FOREST_HOST_SABOTAGE, the tokenizer's
#                                   MOJOLEARN_TOKENIZER_HOST_SABOTAGE; the manifest
#                                   names each family's)
#   MOJOLEARN_LINUX_CPU             Linux x86-64 --target-cpu, default x86-64-v3
#   MOJOLEARN_<FAMILY>_HOST_OUTDIR  output directory for this family, else
#   MOJOLEARN_HOST_OUTDIR           the shared one (the gate builds the whole
#                                   sabotage set into one directory), else
#                                   python/mojolearn/host. The byte LM and forest
#                                   builds read their OWN variable only, as they
#                                   always did: packaging/linux/build_sets.sh and
#                                   packaging/macos/build_release_wheel.sh unset
#                                   only MOJOLEARN_BYTE_LM_HOST_OUTDIR before the
#                                   wheel build, and the .so must land in the
#                                   package tree whatever else a leg exports.
# The kernel-matrix column is COLUMN_CPU (-D MOJOLEARN_COLUMN_CPU, the CPU
# training lane 2026-09-13); every binding asserts at build time that it
# compiled as that column. An existing output is never overwritten; choose a
# fresh directory.
set -eu
family=${1:-}
case "$family" in
    ''|*[!a-z_]*) echo 'build_host_family: usage: build_host_family.sh <family>, lowercase letters and underscores (byte_lm, forest, tokenizer, core, linalg, estimators, metrics, preprocessing, tsa, solver, svm, trees, rf, gp, gbdt, training, arima)' >&2; exit 2 ;;
esac
FAMILY=$(printf '%s' "$family" | tr 'a-z' 'A-Z')
case "$family" in
    byte_lm) label='byte LM host' ;;
    *) label="$family host" ;;
esac
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
case "$family" in
    byte_lm|forest) host_outdir=${family_outdir:-python/mojolearn/host} ;;
    *) host_outdir=${family_outdir:-${MOJOLEARN_HOST_OUTDIR:-python/mojolearn/host}} ;;
esac
mkdir -p "$host_outdir"
host_destination="$host_outdir/_mojolearn_${family}_host.so"
if [ -e "$host_destination" ] || [ -L "$host_destination" ]; then
    echo "$label: output already exists; choose a fresh output directory" >&2
    exit 2
fi
# The tokenizer's Unicode class table is generated, not tracked (2026-09-15):
# tokenizer/impl/unicode_table_generated.mojo is written from Python's
# unicodedata at the pinned version before the compile that imports it.
if [ "$family" = tokenizer ]; then
    sh tokenizer/tools/gen_unicode_table.sh
fi
host_tmpdir=$(mktemp -d "$host_outdir/.${family}-host-build.XXXXXX")
trap 'rm -rf "$host_tmpdir"' EXIT HUP INT TERM
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib "$@" ${MOJOLEARN_BUILD_EXTRA_DEFINES:-} \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings \
    "$source_file" -o "$host_tmpdir/_mojolearn_${family}_host.so"
ln "$host_tmpdir/_mojolearn_${family}_host.so" "$host_destination"
echo "built $host_destination"
