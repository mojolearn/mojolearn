#!/bin/sh
# tools/algos_lane_check.sh <lane[,lane...]> [--sabotage <patch>] [--pass 1|2] [--fixtures f,g] [--out DIR]
#
# THE ONE LANE CHECK OF THE ALGORITHM EXPANSION:
# builds every binding the lanes run (GPU and CPU host) that is missing or
# stale, fits each lane on the GPU and on the CPU, diffs train/infer/model/
# batch and exits 0 only on AGREE (and, with --sabotage, DISAGREE under the
# patch and AGREE again after `git apply -R`). NOTHING COMPARED is a failure.
# Before the diff it runs every per-seam driver in tools/identity_lanes/<f>.checks
# (`<driver>` or `<driver><TAB><sabotage patch>`: PASS, FAIL under the patch, PASS
# after reversal); --pass 2 makes a missing listing or patch a failure.
# Runs on a Linux NVIDIA/AMD pod or a Mac with Metal; the logic is
# tools/algos_lane_check.py, run in the pixi default environment.
set -eu
cd "$(dirname "$0")/.."
PIXI=${PIXI:-pixi}
command -v "$PIXI" >/dev/null 2>&1 || PIXI="$HOME/.pixi/bin/pixi"
exec "$PIXI" run -e default python -u tools/algos_lane_check.py "$@"
