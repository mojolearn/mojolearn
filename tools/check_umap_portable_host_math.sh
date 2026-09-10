#!/usr/bin/env bash
# Run from an activated Mojo toolchain; all invocations are IDENTICAL.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:?usage: check_umap_portable_host_math.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
out=$(cd "$out" && pwd)
mojo_bin=${MOJO:-mojo}
"$mojo_bin" run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I "$repo" "$repo/checks/portable_pow64_check.mojo" > "$out/primitive.log" 2>&1
"$mojo_bin" run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I "$repo" "$repo/umap/checks/portable_host_math_check.mojo" > "$out/host-stages.log" 2>&1
"$mojo_bin" run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I "$repo" "$repo/umap/checks/identity_check.mojo" > "$out/identity.log" 2>&1
"$mojo_bin" run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I "$repo" "$repo/umap/checks/identity_broader_check.mojo" > "$out/identity-broader.log" 2>&1
"$mojo_bin" run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I "$repo" "$repo/umap/checks/transform_check.mojo" > "$out/transform.log" 2>&1
printf '%s\n' 'PASS: portable pow64 and IDENTICAL UMAP numerical/stage gates' > "$out/verdict.txt"
