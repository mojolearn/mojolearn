#!/bin/sh
. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/build_defines.sh"  # MOJOLEARN_BUILD_DEFINES -> $MOJOLEARN_BUILD_DEFINE_FLAGS (compile line)
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The FAST-tier tokenizer binding: BPE training and encode_batch on the Apple GPU
# (tokenizer/fast/, ported from lane/apple-fast-bpe). FAST ONLY: the IDENTICAL door is
# bindings/build_tokenizer_host.sh (unchanged). Output python/mojolearn/_mojolearn_tokenizer_fast.so
# (the fast tier's directory), which tools/afc_ab_def.sh swaps per arm as binding `tokenizer_fast`.
#
#   MOJOLEARN_NUMERIC_MODE=fast bash bindings/build_tokenizer_fast.sh
#   MOJOLEARN_MOJO_BUILD_FLAGS="-D MOJOLEARN_BPE_TRAIN_DEVICE"   the experiment defines (docs/apple-fast/ab/bpe.md)
#   MOJOLEARN_COMPILE_JOBS                                      compile jobs, default 1
#   MOJOLEARN_SKIP_BUILD_GATE                                   accepted (there is no gate)
# Every define is default off; a build with none compiles no kernel and the Python door keeps
# every call on the host binding.
set -eu
MACOS_FLOOR="11.0"
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh sh bindings/build_tokenizer_fast.sh "$@"
fi
[ "${MOJOLEARN_NUMERIC_MODE:-identical}" = fast ] || {
    echo 'build_tokenizer_fast.sh: FAST only (MOJOLEARN_NUMERIC_MODE=fast); the tokenizer IDENTICAL door is bindings/build_tokenizer_host.sh' >&2
    exit 2; }
[ "$(uname)" = Darwin ] || {
    echo 'build_tokenizer_fast.sh: the FAST tokenizer device path is Apple only; build it on macOS' >&2
    exit 2; }
# Match existing bindings: environment deployment target disables Metal AOT.
unset MACOSX_DEPLOYMENT_TARGET
sdk=$(xcrun --sdk macosx --show-sdk-version)
link_flags="-Xlinker -platform_version -Xlinker macos -Xlinker $MACOS_FLOOR -Xlinker $sdk"
target_flags="--target-cpu apple-m1 --target-accelerator metal:1"
entry_file=bindings/entry/entry__mojolearn_tokenizer_fast.mojo
[ -f "$entry_file" ] || { echo "build_tokenizer_fast.sh: no entry $entry_file; run python3 tools/merged_bindings.py entries" >&2; exit 2; }
# The Unicode class table the pre-tokenizer imports is generated, not tracked.
sh tokenizer/tools/gen_unicode_table.sh
outdir=python/mojolearn
tmpdir=$(mktemp -d "$outdir/.tokenizer-fast-build.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out=$tmpdir/_mojolearn_tokenizer_fast.so
# Intentionally split compiler option lists, consistent with existing builders.
# compiles the entry bindings/entry/entry__mojolearn_tokenizer_fast.mojo, which imports the module body bindings/_mojolearn_tokenizer_fast.mojo
pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-1}" --emit shared-lib ${MOJOLEARN_MOJO_BUILD_FLAGS:-} ${MOJOLEARN_BUILD_DEFINE_FLAGS:-} \
    $target_flags $link_flags -I . -I bindings \
    "$entry_file" -o "$out"
mv "$out" "$outdir/_mojolearn_tokenizer_fast.so"
echo "built $outdir/_mojolearn_tokenizer_fast.so"
