#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One M3 board (Andrew, 2026-10-04): BOARD.md, BOARD_M3_FAST.md and BOARD_M3_IDENTICAL.md must match a fresh
# render of board.json on origin/main. Laptop, text only. Prints ALERT on a mismatch, nothing else when clean.
# apple_watch.sh calls it:  bash ~/CascadeProjects/mojolearn/tools/af_board_render_watch.sh
repo=${1:-$HOME/CascadeProjects/mojolearn}
d=$HOME/mojolearn-evidence/.render-check
rm -rf "$d"; mkdir -p "$d"
# git archive skips bench/results (export-ignore), so the board files come through git show
B=bench/results/bench_board/m3ultra-0834
if ! (cd "$repo" && git archive origin/main tools | tar -x -C "$d" && mkdir -p "$d/$B" "$d/docs/apple-fast" \
      && for f in $B/board.json $B/BOARD.md docs/apple-fast/BOARD_M3_FAST.md docs/apple-fast/BOARD_M3_IDENTICAL.md; do
           git show "origin/main:$f" > "$d/$f" 2>/dev/null || exit 1; done); then
  echo "ALERT board render check: could not export origin/main"; rm -rf "$d"; exit 1
fi
if [ ! -f "$d/tools/af_board_render.py" ]; then
  echo "NOTE board render check: tools/af_board_render.py not on origin/main yet"; rm -rf "$d"; exit 0
fi
out=$(cd "$d" && python3 tools/af_board_render.py --check 2>&1 | tail -n 4)
rm -rf "$d"
echo "$out" | grep -q 'RENDER CHECK OK' || { echo "ALERT board render check failed on origin/main:"; echo "$out" | cut -c1-200; exit 1; }
