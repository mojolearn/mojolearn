#!/usr/bin/env bash
# Create a worktree without the run evidence under bench/results/, except the
# canonical board directory, for lanes that don't read old evidence.
#
#   tools/lean_worktree.sh <path> <branch>
#
# <branch> is checked out if it exists locally or on origin; otherwise it is
# created from origin/main. Widen later with `git sparse-checkout add <dir>`,
# or drop the filter with `git sparse-checkout disable`.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <path> <branch>" >&2
  exit 2
fi
path=$1
branch=$2
canonical=bench/results/bench_board/m3ultra-0834

if git show-ref --verify --quiet "refs/heads/${branch}"; then
  git worktree add --no-checkout "${path}" "${branch}"
elif git show-ref --verify --quiet "refs/remotes/origin/${branch}"; then
  git worktree add --no-checkout -b "${branch}" "${path}" "origin/${branch}"
else
  git worktree add --no-checkout -b "${branch}" "${path}" origin/main
fi

# Non-cone patterns: everything, minus bench/results/, plus the board dir.
git -C "${path}" sparse-checkout set --no-cone \
  '/*' \
  '!/bench/results/' \
  "/${canonical}/"
git -C "${path}" checkout -q "${branch}"
echo "lean worktree: ${path} on ${branch} (bench/results/ left out except ${canonical}/)"
