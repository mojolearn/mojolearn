#!/bin/bash
# tools/lowbit_int15/sync_h100.sh -- ON THE LAPTOP: sync this worktree to the
# lane's tree on the shared NVIDIA pod and write the lane's commit beside it
# (a patch-synced tree's HEAD is the merge base, so a record made there would
# otherwise name the wrong commit). Refuses a worktree with uncommitted
# changes to tracked files: a job of record runs a commit, not an edit.
set -eu
cd "$(dirname "$0")/../.."
[ "$(git rev-parse --abbrev-ref HEAD)" = lane/lowbit-int15 ] || { echo "not on lane/lowbit-int15" >&2; exit 2; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "REFUSED: tracked files have uncommitted changes; commit first" >&2
    git status --short --untracked-files=no >&2
    exit 2
fi
C=$(git rev-parse HEAD)
sh tools/nvidia_central.sh sync lowbit-int15 "$PWD" 2>&1 | tail -1
sh tools/nvidia_central.sh sh lowbit-int15 "cd /root/mojolearn-lowbit-int15 && printf '%s\n' $C > .lowbit_int15_commit && cat .lowbit_int15_commit" 2>&1 | tail -1
