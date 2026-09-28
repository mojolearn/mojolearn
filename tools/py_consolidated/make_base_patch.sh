#!/bin/sh
# tools/py_consolidated/make_base_patch.sh [base-ref]: write tools/py_consolidated/base.patch,
# the merged py-* lanes' CODE diff against lane/apple2-merged (default: the merge base
# with origin/lane/apple2-merged), so job.sh can rebuild the base in the SAME tree by
# `git apply -R`. Left out (they do not change what a lane computes): tools/, bench/,
# docs/, prose (*.md), python tests and the sabotage patches. Untracked: it travels
# with `nvidia_central.sh sync` and is never committed.
cd "$(dirname "$0")/../.."
BASE=${1:-$(git merge-base HEAD origin/lane/apple2-merged)}
set -- . ':!tools' ':!bench' ':!docs' ':!*.md' ':!python/mojolearn/tests' ':!*/checks/sabotage/*'
git diff --binary "$BASE" -- "$@" > tools/py_consolidated/base.patch
echo "base $(git rev-parse --short "$BASE"): $(git diff --stat "$BASE" -- "$@" | tail -1)"
git diff --name-only "$BASE" -- "$@" > tools/py_consolidated/base.paths
