#!/bin/sh
. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/build_defines.sh"  # MOJOLEARN_BUILD_DEFINES -> $MOJOLEARN_BUILD_DEFINE_FLAGS (compile line)
# Build the MERGED bindings of one installed directory set: one native file per
# group of bindings/merged_groups.json, exporting every member's PyInit, so the
# code the members share (sequence ops, linalg, common kernels, the Mojo stdlib)
# is compiled and embedded ONCE instead of once per binding.
#
#   MOJOLEARN_NUMERIC_MODE=fast      sh bindings/merged_build.sh gpu    -> python/mojolearn/
#   MOJOLEARN_NUMERIC_MODE=identical sh bindings/merged_build.sh gpu    -> python/mojolearn/identical/
#                                    sh bindings/merged_build.sh host   -> python/mojolearn/host/
#
# Options: --group <stem> builds one file of the plan (to run the files of one
# directory in parallel); --outdir <dir> writes elsewhere. Each file lands as
# <outdir>/<stem>.so and is recorded in <outdir>/_mojolearn_merged.json, which
# python/mojolearn/_merged.py reads to load each member under its own module
# name. `python3 tools/merged_bindings.py plan --dir <dir>` prints the plan.
#
# SAME FLAGS AS THE PER-BINDING SCRIPTS, SAME BITS. Every member is compiled with
# exactly the flags its own bindings/build_<x>.sh passes on this platform:
#   macOS   --target-cpu apple-m1, the -platform_version 11.0 link flags, NO
#           MACOSX_DEPLOYMENT_TARGET (it empties every metallib; see the top of
#           bindings/build_gbdt.sh), no --target-accelerator -- except the
#           members of `darwin_metal1` in the manifest, whose scripts pass
#           --target-accelerator metal:1; they compile into <stem>_m1.so.
#           The byte LM's -D MOJOLEARN_COLUMN_APPLE is what TARGET_COLUMN
#           detects on a Metal target (checks/kernel_matrix.mojo), so it joins
#           the default file unchanged.
#   Linux   --target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3} on x86_64, and ONE
#           explicit --target-accelerator from MOJOLEARN_GPU_ARCHS (required:
#           the byte LM script already requires it, and the x_* scripts would
#           otherwise read it from nvidia-smi while the rest default).
#   host    the flags of bindings/build_host_family.sh: IDENTICAL, the CPU
#           column, no accelerator.
# MOJOLEARN_NUMERIC_MODE=identical adds -D MOJOLEARN_NUMERIC_IDENTICAL=1,
# MOJOLEARN_TARGET_COLUMN adds -D MOJOLEARN_COLUMN_<X>, MOJOLEARN_BUILD_DEFINES
# and MOJOLEARN_MOJO_BUILD_FLAGS apply to every member as they apply to every
# per-binding script. Defines that only SOME per-binding scripts read
# (MOJOLEARN_EXTRA_DEFINES, MOJOLEARN_BUILD_EXTRA_DEFINES on a GPU set,
# MOJOLEARN_MAMBA_DEFINES, MOJOLEARN_ATTENTION_MEMORY_PROFILE, and build.sh's
# MOJOLEARN_ALLOW_NO_GPU) are refused: one compile cannot scope them to one
# member. Run that A/B with the per-binding scripts.
#
# The per-binding scripts stay: they compile bindings/entry/entry_<name>.mojo,
# the same module body under its own PyInit. A member's own .so in the output
# directory shadows the merged file for that member (the loader prefers it), so
# this script removes the members' own files there before it installs the
# merged one.
#
# No build gate runs here: the packers read back vendor, mode and kernels from
# every member after the build, as they do for the per-binding files.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

usage() {
    echo "usage: [MOJOLEARN_NUMERIC_MODE=fast|identical] sh bindings/merged_build.sh gpu|host [--group <stem>] [--outdir <dir>]" >&2
}
kind=gpu
only=""
outdir=""
while [ $# -gt 0 ]; do
    case "$1" in
        gpu|host) kind=$1 ;;
        --group) [ $# -ge 2 ] || { usage; exit 2; }; only=$2; shift ;;
        --outdir) [ $# -ge 2 ] || { usage; exit 2; }; outdir=$2; shift ;;
        *) usage; exit 2 ;;
    esac
    shift
done

PY=${MOJOLEARN_MERGED_PYTHON:-python3}   # build glue only: tools/merged_bindings.py
MACOS_FLOOR="11.0"
mode=${MOJOLEARN_NUMERIC_MODE:-identical}
case "$mode" in
    fast|identical) ;;
    *) echo "merged_build: MOJOLEARN_NUMERIC_MODE must be fast or identical, got '$mode'" >&2; exit 2 ;;
esac

refuse_env() {
    echo "merged_build: $1 is set ('$2'). Only some per-binding scripts read it, and one merged" >&2
    echo "  compile cannot scope it to one member; run that build with the per-binding scripts." >&2
    exit 2
}
[ -z "${MOJOLEARN_EXTRA_DEFINES:-}" ] || refuse_env MOJOLEARN_EXTRA_DEFINES "$MOJOLEARN_EXTRA_DEFINES"
[ -z "${MOJOLEARN_MAMBA_DEFINES:-}" ] || refuse_env MOJOLEARN_MAMBA_DEFINES "$MOJOLEARN_MAMBA_DEFINES"
case "${MOJOLEARN_ATTENTION_MEMORY_PROFILE:-estash}" in
    estash) ;;
    *) refuse_env MOJOLEARN_ATTENTION_MEMORY_PROFILE "$MOJOLEARN_ATTENTION_MEMORY_PROFILE" ;;
esac
[ "${MOJOLEARN_ALLOW_NO_GPU:-0}" != 1 ] || refuse_env MOJOLEARN_ALLOW_NO_GPU 1
if [ "$kind" = gpu ] && [ -n "${MOJOLEARN_BUILD_EXTRA_DEFINES:-}" ]; then
    refuse_env MOJOLEARN_BUILD_EXTRA_DEFINES "$MOJOLEARN_BUILD_EXTRA_DEFINES"
fi

system=$(uname -s)
case "$system" in
    Darwin) plat=darwin; byte_lm_default=1 ;;
    Linux) plat=linux; byte_lm_default=0 ;;
    *) echo "merged_build: requires Linux or macOS" >&2; exit 2 ;;
esac
byte_lm=${MOJOLEARN_PACKAGE_BYTE_LM:-$byte_lm_default}
case "$byte_lm" in 0|1) ;; *) echo 'merged_build: MOJOLEARN_PACKAGE_BYTE_LM must be 0 or 1' >&2; exit 2 ;; esac

# ---------------------------------------------------------------- the flags
base_flags=""
link_flags=""
if [ "$plat" = darwin ]; then
    [ "$(uname -m)" = arm64 ] || { echo "merged_build: macOS requires Apple silicon" >&2; exit 2; }
    # NOT EXPORTED, DELIBERATELY -- see bindings/build_gbdt.sh.
    unset MACOSX_DEPLOYMENT_TARGET
    [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] || { echo "merged_build: a macOS build takes no MOJOLEARN_GPU_ARCHS" >&2; exit 2; }
    sdk=$(xcrun --sdk macosx --show-sdk-version)
    link_flags="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $sdk"
    base_flags="--target-cpu apple-m1"
else
    case "$(uname -m)" in
        x86_64) base_flags="--target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3}" ;;
        aarch64) base_flags="" ;;
        *) echo "merged_build: unsupported Linux host architecture" >&2; exit 2 ;;
    esac
fi

column_flags=""
if [ "$kind" = host ]; then
    [ "$mode" = identical ] || { echo "merged_build: the host set supports only MOJOLEARN_NUMERIC_MODE=identical" >&2; exit 2; }
    [ "${MOJOLEARN_TARGET_COLUMN:-cpu}" = cpu ] || { echo "merged_build: the host set compiles the CPU column only" >&2; exit 2; }
    [ -z "${MOJOLEARN_GPU_ARCHS:-}" ] || { echo "merged_build: a CPU build takes no MOJOLEARN_GPU_ARCHS" >&2; exit 2; }
    dir=host
    mode_flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU"
    extra_flags=${MOJOLEARN_BUILD_EXTRA_DEFINES:-}
    default_outdir=${MOJOLEARN_HOST_OUTDIR:-python/mojolearn/host}
else
    dir=$mode
    mode_flags=""
    [ "$mode" = identical ] && mode_flags="-D MOJOLEARN_NUMERIC_IDENTICAL=1"
    extra_flags=""
    if [ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]; then
        column_flags="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
    fi
    if [ "$plat" = linux ]; then
        case "${MOJOLEARN_GPU_ARCHS:-}" in
            '') echo "merged_build: a Linux GPU set needs ONE explicit MOJOLEARN_GPU_ARCHS (sm_NN or gfxNNN)" >&2; exit 2 ;;
            *[!A-Za-z0-9_]*) echo "merged_build: MOJOLEARN_GPU_ARCHS='$MOJOLEARN_GPU_ARCHS': the compiler takes EXACTLY ONE architecture" >&2; exit 2 ;;
        esac
        base_flags="$base_flags --target-accelerator $MOJOLEARN_GPU_ARCHS"
    fi
    if [ "$mode" = identical ]; then default_outdir=python/mojolearn/identical; else default_outdir=python/mojolearn; fi
fi
outdir=${outdir:-$default_outdir}
mkdir -p "$outdir"

plan=$("$PY" tools/merged_bindings.py plan --dir "$dir" --platform "$plat" --byte-lm "$byte_lm")
[ -n "$plan" ] || { echo "merged_build: bindings/merged_groups.json plans nothing for $dir" >&2; exit 2; }
if [ -n "$only" ]; then
    printf '%s\n' "$plan" | awk -v s="$only" '$1==s{f=1} END{exit !f}' \
        || { echo "merged_build: no group '$only' in the $dir plan:" >&2; printf '%s\n' "$plan" | awk '{print "  " $1}' >&2; exit 2; }
fi
if [ "$kind" = host ] && printf '%s\n' "$plan" | grep -q '_mojolearn_tokenizer_host'; then
    sh tokenizer/tools/gen_unicode_table.sh < /dev/null
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-merged-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM

printf '%s\n' "$plan" > "$tmpdir/plan.txt"
while read -r stem cls members; do
    [ -z "$only" ] || [ "$stem" = "$only" ] || continue
    class_flags=""
    case "$cls" in
        default|host) ;;
        metal1) class_flags="--target-accelerator metal:1" ;;
        *) echo "merged_build: unknown flags class '$cls' for $stem" >&2; exit 2 ;;
    esac
    root="$tmpdir/$stem.mojo"
    "$PY" tools/merged_bindings.py root --stem "$stem" --members "$members" --out "$root" < /dev/null
    count=$(printf '%s' "$members" | tr ',' '\n' | grep -c .)
    echo "== $dir/$stem.so: $count members ($cls flags)"
    # shellcheck disable=SC2086  # the flag strings are deliberately word-split
    pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib ${MOJOLEARN_MOJO_BUILD_FLAGS:-} ${MOJOLEARN_BUILD_DEFINE_FLAGS:-} \
        $base_flags $class_flags $link_flags $mode_flags $column_flags $extra_flags \
        -I . -I bindings \
        "$root" -o "$tmpdir/$stem.so" < /dev/null
    if [ "$plat" = darwin ]; then
        got=$(otool -l "$tmpdir/$stem.so" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
        [ "$got" = "$MACOS_FLOOR" ] || { echo "merged_build: $stem.so minos is '$got', want $MACOS_FLOOR" >&2; exit 1; }
    fi
    for n in $(printf '%s' "$members" | tr ',' ' '); do
        if [ -e "$outdir/$n.so" ] || [ -L "$outdir/$n.so" ]; then
            echo "   removing $outdir/$n.so: $stem.so serves $n now, and its own file would shadow it"
            rm -f "$outdir/$n.so"
        fi
    done
    mv "$tmpdir/$stem.so" "$outdir/$stem.so"
    "$PY" tools/merged_bindings.py index --outdir "$outdir" --stem "$stem" --members "$members" < /dev/null
    echo "built $outdir/$stem.so"
done < "$tmpdir/plan.txt"
