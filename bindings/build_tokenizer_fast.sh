#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The FAST-tier tokenizer binding (lane/apple-fast-bpe, 2026-10-03): BPE training and encode_batch
# on the Apple GPU, tokenizer/fast/bpe_device.mojo. FAST ONLY: the IDENTICAL door is
# bindings/build_tokenizer_host.sh (unchanged). Output python/mojolearn/_mojolearn_tokenizer_fast.so
# (the fast tier's directory), which tools/afc_ab_def.sh swaps per arm as binding `tokenizer_fast`.
#
#   MOJOLEARN_NUMERIC_MODE=fast bash bindings/build_tokenizer_fast.sh
#   MOJOLEARN_MOJO_BUILD_FLAGS="-D MOJOLEARN_BPE_TRAIN_DEVICE"   the experiment defines (docs/apple-fast/ab/bpe.md)
#   MOJOLEARN_COMPILE_JOBS                                      compile jobs, default 2
#   MOJOLEARN_SKIP_BUILD_GATE                                   accepted (there is no gate)
set -eu
MACOS_FLOOR="11.0"
cd "$(dirname "$0")/.."
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" = fast ] || {
    echo 'build_tokenizer_fast.sh: FAST only (MOJOLEARN_NUMERIC_MODE=fast); the tokenizer IDENTICAL door is bindings/build_tokenizer_host.sh' >&2
    exit 2; }
unset MACOSX_DEPLOYMENT_TARGET
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
# The Unicode class table the pre-tokenizer imports is generated, not tracked.
sh tokenizer/tools/gen_unicode_table.sh
outdir=python/mojolearn
tmpdir=$(mktemp -d "$outdir/.tokenizer-fast-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out=$tmpdir/_mojolearn_tokenizer_fast.so
# Intentionally split compiler option lists, consistent with existing builders.
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" --emit shared-lib ${MOJOLEARN_MOJO_BUILD_FLAGS:-} \
    $target_flags $link_flags -I . -I bindings \
    bindings/_mojolearn_tokenizer_fast.mojo -o "$out"
mv "$out" "$outdir/_mojolearn_tokenizer_fast.so"
echo "built $outdir/_mojolearn_tokenizer_fast.so"
