#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
native_root="$(cd "$(dirname "$0")/../.." && pwd)"
if [ "$#" -lt 2 ]; then
  echo 'Usage: validate.sh PREFIX LOG_DIRECTORY [--gpu] [--apple-fast] [--cross-compile]' >&2
  exit 2
fi
native_prefix="$(cd "$1" && pwd)"
mkdir -p "$2"
native_evidence="$(cd "$2" && pwd)"
shift 2
native_gpu=false
native_fast=false
native_cross=false
for native_arg in "$@"; do
  case "$native_arg" in
    --gpu) native_gpu=true ;;
    --apple-fast) native_fast=true ;;
    --cross-compile) native_cross=true ;;
    *) echo "Unknown argument: $native_arg" >&2; exit 2 ;;
  esac
done
test -f "$native_prefix/lib/mojo/mojolearn.mojoc"
# Compile from outside the checkout: no repository imports or generated source
# may rescue an incomplete package. Preserve binaries and all compiler output.
cd "$native_evidence"
test ! -e results.tsv || { echo 'Use a fresh log directory' >&2; exit 2; }
cp "$native_root/packaging/native/tests/"*.mojo "$native_evidence/"
printf 'check\texit\texpected\n' > results.tsv
run_check() {
  local name="$1"
  shift
  local status=0
  "$@" > "$name.log" 2>&1 || status=$?
  printf '%s\t%s\t0\n' "$name" "$status" >> results.tsv
  printf '%s: exit %s\n' "$name" "$status"
  if [ "$status" -ne 0 ]; then
    tail -n 12 "$name.log"
    exit "$status"
  fi
}
refuse_check() {
  local name="$1" message="$2"
  shift 2
  local status=0
  "$@" > "$name.log" 2>&1 || status=$?
  printf '%s\t%s\tnonzero with diagnostic\n' "$name" "$status" >> results.tsv
  if [ "$status" -eq 0 ] || ! grep -q "$message" "$name.log"; then
    echo "$name: expected refusal missing" >&2
    tail -n 12 "$name.log"
    exit 1
  fi
  echo "$name: expected refusal confirmed"
}
native_flags=(-I "$native_prefix/lib/mojo")
run_check mode-build mojo build "${native_flags[@]}" mode.mojo -o mode
run_check mode ./mode
run_check cpu-build mojo build "${native_flags[@]}" cpu.mojo -o cpu
run_check cpu ./cpu
refuse_check conflicting-modes 'select only one numeric mode' mojo build "${native_flags[@]}" \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_NUMERIC_FAST=1 mode.mojo -o refused
if "$native_gpu"; then
  run_check gpu-build mojo build "${native_flags[@]}" gpu.mojo -o gpu
  run_check gpu ./gpu
fi
if "$native_fast"; then
  run_check fast-mode-build mojo build "${native_flags[@]}" -D MOJOLEARN_NUMERIC_FAST=1 mode.mojo -o fast-mode
  run_check fast-mode ./fast-mode
  if "$native_gpu"; then
    run_check fast-gpu-build mojo build "${native_flags[@]}" -D MOJOLEARN_NUMERIC_FAST=1 gpu.mojo -o fast-gpu
    run_check fast-gpu ./fast-gpu
  fi
fi
if "$native_cross"; then
  for native_target in sm_89 gfx942; do
    run_check "compile-$native_target" mojo build "${native_flags[@]}" \
      --target-accelerator "$native_target" gpu.mojo -o "gpu-$native_target"
    refuse_check "fast-$native_target" 'fast mode requires an Apple GPU target' \
      mojo build "${native_flags[@]}" --target-accelerator "$native_target" \
      -D MOJOLEARN_NUMERIC_FAST=1 mode.mojo -o refused
  done
fi
echo "PASS selected native package checks; full logs and binaries: $native_evidence"
