#!/bin/sh
# Root-only guarded build. No smoke/model/test launch.
# Invoke through the corresponding NVIDIA/AMD/macOS serial guard.
# Requires explicit IDENTICAL mode; Linux additionally needs one GPU target.
set -eu
MACOS_FLOOR="11.0"
byte_lm_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$byte_lm_root"
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" = identical ] || { echo 'byte LM supports only MOJOLEARN_NUMERIC_MODE=identical' >&2; exit 2; }
byte_lm_system=$(uname -s)
if [ "$byte_lm_system" = Darwin ]; then
    [ "$(uname -m)" = arm64 ] || { echo 'byte LM: Metal requires Apple silicon' >&2; exit 2; }
    [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] || { echo 'byte LM: Metal build requires MOJOLEARN_GPU_ARCHS unset' >&2; exit 2; }
    case "${MOJOLEARN_TARGET_COLUMN:-apple}" in
        apple) ;;
        *) echo 'byte LM: Darwin requires the apple target column' >&2; exit 2 ;;
    esac
    # Existing training/Transformer build convention: either an exported
    # deployment target or --target-accelerator suppresses Metal AOT kernels.
    unset MACOSX_DEPLOYMENT_TARGET
    byte_lm_sdk=$(xcrun --sdk macosx --show-sdk-version)
    set -- --target-cpu apple-m1 -D MOJOLEARN_COLUMN_APPLE \
        -Xlinker -platform_version -Xlinker macos -Xlinker "$MACOS_FLOOR" -Xlinker "$byte_lm_sdk"
elif [ "$byte_lm_system" = Linux ]; then
case "${MOJOLEARN_GPU_ARCHS:-}" in
    sm_[0-9]*|gfx[0-9]*) ;;
    *) echo 'byte LM: one explicit sm_NN or gfxNNN target required' >&2; exit 2 ;;
esac
case "$MOJOLEARN_GPU_ARCHS" in
    *[!A-Za-z0-9_]*) echo 'byte LM: invalid/multiple GPU architectures' >&2; exit 2 ;;
esac
# Fixed compiler concurrency; this script never starts a second worker/build.
set -- --target-accelerator "$MOJOLEARN_GPU_ARCHS"
case "$(uname -m)" in
    x86_64) set -- "$@" --target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3} ;;
    aarch64) ;;
    *) echo 'byte LM: unsupported Linux host architecture' >&2; exit 2 ;;
esac
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    '') ;;
    nvidia) set -- "$@" -D MOJOLEARN_COLUMN_NVIDIA ;;
    amd) set -- "$@" -D MOJOLEARN_COLUMN_AMD ;;
    amd_rdna) set -- "$@" -D MOJOLEARN_COLUMN_AMD_RDNA ;;
    *) echo 'byte LM: unsupported target column' >&2; exit 2 ;;
esac
else
    echo 'byte LM: requires Linux CUDA/HIP or Darwin Metal' >&2
    exit 2
fi
byte_lm_outdir=${MOJOLEARN_BYTE_LM_OUTDIR:-python/mojolearn/identical}
mkdir -p "$byte_lm_outdir"
byte_lm_destination="$byte_lm_outdir/_mojolearn_byte_lm.so"
# Refuse prior files, including dangling symlinks. Publish via hard link so a
# path created during compilation cannot be overwritten by a rename race.
if [ -e "$byte_lm_destination" ] || [ -L "$byte_lm_destination" ]; then
    echo 'byte LM: output already exists; choose a fresh output directory' >&2
    exit 2
fi
byte_lm_tmpdir=$(mktemp -d "$byte_lm_outdir/.byte-lm-build.XXXXXX")
trap 'rm -rf "$byte_lm_tmpdir"' EXIT HUP INT TERM
pixi run mojo build -j 2 --emit shared-lib "$@" \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings \
    bindings/_mojolearn_byte_lm.mojo -o "$byte_lm_tmpdir/_mojolearn_byte_lm.so"
ln "$byte_lm_tmpdir/_mojolearn_byte_lm.so" "$byte_lm_destination"
echo "built $byte_lm_destination; root must read byte_lm_numeric_mode/vendor/profile and retain guard exit evidence"
