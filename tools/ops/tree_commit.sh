# SPDX-License-Identifier: Apache-2.0
# tools/ops/tree_commit.sh -- sourced by the box jobs: which commit a tree on a box holds, and whether a staged wheel
# was built from it (2026-11-26).
#
# tools/dev_pod.sh sync leaves the box tree's git HEAD at the worktree's merge base with main and applies the
# worktree's changes on top, so `git rev-parse HEAD` names main, not the code that is there (nvc1-0004, 2026-11-22).
# Every sync records the synced commit in .git/devpod_synced (commit=, worktree_dirty=, base=, ...); a tree without
# that record (a plain checkout), or whose record's base is no longer its HEAD, holds its HEAD.
#
#   tree_commit [DIR]          the commit DIR holds (default .), empty when DIR is not a git tree
#   wheel_commit WHEEL         the +<sha> local version in a wheel file name, empty when it has none
#   wheels_match_tree DIR W..  one line per wheel; 0 when every wheel names DIR's commit, 1 otherwise. A wheel with no
#                              +<sha>, or a tree whose commit is unknown, is a mismatch: the job refuses rather than
#                              run tests of one commit against another commit's wheel.

tree_commit() {
    local dir=${1:-.} git_dir
    git_dir=$(git -C "$dir" rev-parse --absolute-git-dir 2> /dev/null) || return 0
    # a record whose base is no longer HEAD is stale (the tree was checked out since): HEAD holds the code, as
    # tools/bench_board.py repo_sync rules
    local base commit head
    head=$(git -C "$dir" rev-parse HEAD 2> /dev/null)
    if [ -f "$git_dir/devpod_synced" ]; then
        base=$(sed -n 's/^base=//p' "$git_dir/devpod_synced" | head -n 1)
        commit=$(sed -n 's/^commit=//p' "$git_dir/devpod_synced" | head -n 1)
    fi
    if [ -n "${commit:-}" ] && { [ -z "${base:-}" ] || [ "$base" = "$head" ]; }; then
        echo "$commit"
    else
        git -C "$dir" rev-parse HEAD 2> /dev/null
    fi
}

wheel_commit() {
    basename "$1" | sed -n 's/^[^-]*-[^-]*+\([0-9a-f]\{7,40\}\)-.*/\1/p'
}

wheels_match_tree() {
    local dir=$1 tree w sha bad=0
    shift
    tree=$(tree_commit "$dir")
    for w in "$@"; do
        sha=$(wheel_commit "$w")
        if [ -z "$tree" ]; then echo "MISMATCH $(basename "$w"): the tree $dir has no known commit"; bad=1
        elif [ -z "$sha" ]; then echo "MISMATCH $(basename "$w"): no +<sha> local version to compare with ${tree:0:12}"; bad=1
        elif [ "${tree:0:${#sha}}" != "$sha" ]; then echo "MISMATCH $(basename "$w") is commit $sha, the tree holds ${tree:0:12}"; bad=1
        else echo "MATCH $(basename "$w") = ${tree:0:12}"
        fi
    done
    [ "$#" -gt 0 ] || { echo "MISMATCH no wheel given"; bad=1; }
    return $bad
}
