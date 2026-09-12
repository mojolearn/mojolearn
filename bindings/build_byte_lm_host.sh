#!/bin/sh
# CPU-only build of the byte LM inference binding (DEVIATION 2610).
# No accelerator target. On the Mac, invoke through tools/macos_serial_guard.py.
# IDENTICAL only. MOJOLEARN_BUILD_EXTRA_DEFINES carries trial defines, e.g.
# -D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1 for the gate's negative control.
set -eu
MACOS_FLOOR="11.0"
host_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$host_root"
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" = identical ] || { echo 'byte LM host supports only MOJOLEARN_NUMERIC_MODE=identical' >&2; exit 2; }
host_system=$(uname -s)
if [ "$host_system" = Darwin ]; then
    [ "$(uname -m)" = arm64 ] || { echo 'byte LM host: macOS requires Apple silicon' >&2; exit 2; }
    unset MACOSX_DEPLOYMENT_TARGET
    host_sdk=$(xcrun --sdk macosx --show-sdk-version)
    set -- --target-cpu apple-m1 \
        -Xlinker -platform_version -Xlinker macos -Xlinker "$MACOS_FLOOR" -Xlinker "$host_sdk"
elif [ "$host_system" = Linux ]; then
    [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] || { echo 'byte LM host: a CPU build takes no MOJOLEARN_GPU_ARCHS' >&2; exit 2; }
    case "$(uname -m)" in
        x86_64) set -- --target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3} ;;
        aarch64) set -- ;;
        *) echo 'byte LM host: unsupported Linux host architecture' >&2; exit 2 ;;
    esac
else
    echo 'byte LM host: requires Linux or macOS' >&2
    exit 2
fi
host_outdir=${MOJOLEARN_BYTE_LM_HOST_OUTDIR:-python/mojolearn/host}
mkdir -p "$host_outdir"
host_destination="$host_outdir/_mojolearn_byte_lm_host.so"
if [ -e "$host_destination" ] || [ -L "$host_destination" ]; then
    echo 'byte LM host: output already exists; choose a fresh output directory' >&2
    exit 2
fi
host_tmpdir=$(mktemp -d "$host_outdir/.byte-lm-host-build.XXXXXX")
trap 'rm -rf "$host_tmpdir"' EXIT HUP INT TERM
pixi run mojo build -j 2 --emit shared-lib "$@" ${MOJOLEARN_BUILD_EXTRA_DEFINES:-} \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings \
    bindings/_mojolearn_byte_lm_host.mojo -o "$host_tmpdir/_mojolearn_byte_lm_host.so"
ln "$host_tmpdir/_mojolearn_byte_lm_host.so" "$host_destination"
echo "built $host_destination"
