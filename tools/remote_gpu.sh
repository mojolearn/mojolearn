#!/usr/bin/env bash
# Run this port's checks on a rented NVIDIA or AMD GPU.
#
#   tools/remote_gpu.sh <user@host> [nvidia|amd]
#
# WHY THIS EXISTS
#
# The kernel matrix has four columns and only ONE of them can be tested here.
# `apple` is the machine this was written on; `nvidia` and `amd` resolve to
# different shared-memory budgets, different block sizes and, for AMD, a
# 64-wide wavefront that the bit-identical column pins to 32. Every claim
# about those columns is currently ARITHMETIC, not a measurement, and the
# matrix says so.
#
# That matters more here than it would elsewhere, because two of the port's
# deviations are justified by cross-vendor reasoning that has never run on
# another vendor:
#
#   - `replication_lanes` is pinned at 32 so AMD's 64-wide wavefront cannot
#     change the reduction geometry. Never observed on AMD.
#   - the float-atomic flush branch in the histogram kernels is unreachable
#     on Apple and is the path NVIDIA and AMD would take under
#     `determinism=off`. Never compiled for either.
#
# WHAT IT DOES NOT DO
#
# It does not benchmark. A rented box is shared, throttled and unknown, and
# this repository's own rule is that only interleaved arms inside one process
# compare. This runs the CORRECTNESS checks, which are verdicts rather than
# numbers and do not care about the machine's mood.
set -euo pipefail

HOST="${1:?usage: tools/remote_gpu.sh <user@host> [nvidia|amd]}"
VENDOR="${2:-nvidia}"

case "$VENDOR" in
  nvidia) COLUMN=COLUMN_NVIDIA ;;
  amd)    COLUMN=COLUMN_AMD ;;
  *) echo "vendor must be nvidia or amd, got $VENDOR" >&2; exit 2 ;;
esac

echo "==> target $HOST, column $COLUMN"

REMOTE_DIR="~/cbsym"
echo "==> syncing the port (source only; no build products, no datasets)"
rsync -az --delete \
  --exclude '.git' --exclude '*.trace' --exclude '__pycache__' \
  ./ "$HOST:$REMOTE_DIR/"

echo "==> building against the $VENDOR column and running the checks"
# TARGET_COLUMN is a comptime constant, so the column is a BUILD, not a flag.
# That is the honest shape of the constraint: a threadgroup size cannot follow
# a runtime device query.
ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
cd $REMOTE_DIR
sed -i.bak "s/^comptime TARGET_COLUMN = .*/comptime TARGET_COLUMN = $COLUMN/" \
  original/kernel_matrix.mojo
grep -n "^comptime TARGET_COLUMN" original/kernel_matrix.mojo
mojo run -I . probe_main.mojo
REMOTE

echo
echo "==> done. Correctness only. Any timing from a rented box is not"
echo "    comparable to anything, per the interleaved-arms rule."
