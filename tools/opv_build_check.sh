#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# opv_build_check.sh: compile-only check of the lane/apple-fast-opv OPTICS
# defines (one OPV-BUILD rc line per spec). Builds into a scratch copy of the
# .so, restoring the tree's. Times nothing.
set -u
cd "$(dirname "$0")/.."
RUN() { if command -v pixi > /dev/null; then pixi run "$@"; else "$@"; fi; }
so=python/mojolearn/_mojolearn_x_cluster.so; [ -f "$so" ] && cp "$so" "$HOME/opv_orig.so"
b() {  # $1 mode, $2 script, $3 defines
  MOJOLEARN_NUMERIC_MODE=$1 MOJOLEARN_MOJO_BUILD_FLAGS="$3" MOJOLEARN_SKIP_BUILD_GATE=1 \
    RUN bash "bindings/$2.sh" > "$HOME/opv_build.log" 2>&1
  rc=$?; echo "OPV-BUILD $1 $2 '$3' rc=$rc"
  [ $rc = 0 ] || grep -m 3 -B 2 -A 6 -i error "$HOME/opv_build.log" | cut -c1-250
}
b fast build_x_cluster ""
b identical build_x_cluster ""
b fast build_x_cluster "-D MOJOLEARN_OPTICS_FRONTIER_DEVICE=1"
b fast build_x_cluster "-D MOJOLEARN_OPTICS_LIVEBUF=1"
b fast build_x_cluster "-D MOJOLEARN_OPTICS2_ALL=1"
[ -f "$HOME/opv_orig.so" ] && cp "$HOME/opv_orig.so" "$so.tmp" && mv -f "$so.tmp" "$so"
echo "OPV-BUILD done"
