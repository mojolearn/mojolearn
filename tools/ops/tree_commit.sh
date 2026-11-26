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
#   tree_sync_state [DIR]      clean | untracked (worktree_dirty=0 with worktree_untracked=N>0: every tracked file is
#                              exactly the commit, plus N untracked files the sync shipped, e.g. box-local/ scripts
#                              a job runs by name) | dirty (the sync record says worktree_dirty=1: modified tracked
#                              files over the commit) | full-sync-unrecorded (.devpod_manifest from a full sync, with no record of it:
#                              an older tools/dev_pod.sh wrote none, so the tree is a worktree over HEAD of no known
#                              commit, and any record left is an earlier patch sync's)
#   sync_state_ok [DIR]        0 when the state is clean or untracked (named on a MATCH line), or BOX_EXPECT_DIRTY_SYNC=1
#                              states the job expects a dirty one; a
#                              state that is not clean is named on one line either way (`MATCH sync <state> ...
#                              accepted` or `SYNC <state> ... refused`)
#   wheel_commit WHEEL         the commit a wheel was built from: the <sha> of its local version (`+<sha>`, a
#                              measurement build's `+<sha12>.<label>`, e.g. the define build's `.d<hex>`,
#                              tools/release_candidate.py candidate_version), else its recorded provenance (the core
#                              wheel's mojolearn/identity_columns/COMMIT), else BOX_WHEEL_COMMIT (the job's stated
#                              expectation, for a published wheel that records none), else empty
#   wheels_match_tree DIR W..  one line per wheel; 0 when every wheel names DIR's commit and the tree's sync state is
#                              acceptable, 1 otherwise. A wheel of no known commit, or a tree whose commit is unknown, is
#                              a mismatch: the job refuses rather than run tests of one commit against another's wheel.

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
    local sha
    # <dist>-<version>+<sha>[.<label>]-<tags>.whl (a label is lower case letters, digits, dots; PEP 440 local)
    sha=$(basename "$1" | sed -n 's/^[^-]*-[^-+]*+\([0-9a-f]\{7,40\}\)\(\.[a-z0-9_.]*\)\{0,1\}-.*/\1/p')
    [ -n "$sha" ] || sha=$(wheel_recorded_commit "$1")
    [ -n "$sha" ] || sha=${BOX_WHEEL_COMMIT:-}
    echo "$sha"
}

wheel_recorded_commit() {  # the commit witness a wheel carries (the core wheel's identity_columns/COMMIT), or empty
    [ -f "$1" ] || return 0
    python3 -I - "$1" 2> /dev/null << 'PY' || true
import re, sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        names = [n for n in z.namelist() if n.endswith("mojolearn/identity_columns/COMMIT")]
        text = z.read(names[0]).decode().strip() if names else ""
except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError):
    text = ""
print(text if re.fullmatch(r"[0-9a-f]{40}", text) else "")
PY
}

tree_sync_state() {
    local dir=${1:-.} git_dir rec head base psha dirty untracked
    git_dir=$(git -C "$dir" rev-parse --absolute-git-dir 2> /dev/null) || { echo clean; return 0; }
    rec="$git_dir/devpod_synced"
    head=$(git -C "$dir" rev-parse HEAD 2> /dev/null)
    if [ -f "$rec" ]; then
        base=$(sed -n 's/^base=//p' "$rec" | head -n 1)
        psha=$(sed -n 's/^patch_sha256=//p' "$rec" | head -n 1)
        dirty=$(sed -n 's/^worktree_dirty=//p' "$rec" | head -n 1)
        untracked=$(sed -n 's/^worktree_untracked=//p' "$rec" | head -n 1)
    fi
    # a full sync writes .devpod_manifest and a patch sync removes it: a manifest with no full-sync record is an older
    # dev_pod.sh's full sync (it recorded nothing; a record there is an earlier patch sync's, or none)
    if [ -f "$dir/.devpod_manifest" ] && [ "${psha:-}" != none-full-sync ]; then echo full-sync-unrecorded; return 0; fi
    if [ -f "$rec" ] && { [ -z "${base:-}" ] || [ "$base" = "$head" ]; } && [ "${dirty:-0}" = 1 ]; then
        echo dirty; return 0
    fi
    if [ -f "$rec" ] && { [ -z "${base:-}" ] || [ "$base" = "$head" ]; } && [ "${untracked:-0}" -gt 0 ] 2> /dev/null; then
        echo untracked; return 0
    fi
    echo clean
}

sync_state_ok() {
    local dir=${1:-.} state
    state=$(tree_sync_state "$dir")
    [ "$state" = clean ] && return 0
    if [ "$state" = untracked ]; then
        echo "MATCH sync untracked of $dir: every tracked file is commit $(tree_commit "$dir" | cut -c1-12); untracked files beside it (run by name) are not in the commit"
        return 0
    fi
    if [ "${BOX_EXPECT_DIRTY_SYNC:-0}" = 1 ]; then
        # MATCH first: the callers read a line's first word as its verdict (MATCH, else refused)
        echo "MATCH sync $state of $dir accepted (BOX_EXPECT_DIRTY_SYNC=1): the code is not exactly commit $(tree_commit "$dir" | cut -c1-12)"
        return 0
    fi
    echo "SYNC $state $dir: the tree is not exactly commit $(tree_commit "$dir" | cut -c1-12) (refused; a job that expects it passes BOX_EXPECT_DIRTY_SYNC=1)"
    return 1
}

wheels_match_tree() {
    local dir=$1 tree w sha bad=0
    shift
    tree=$(tree_commit "$dir")
    sync_state_ok "$dir" || bad=1
    for w in "$@"; do
        sha=$(wheel_commit "$w")
        if [ -z "$tree" ]; then echo "MISMATCH $(basename "$w"): the tree $dir has no known commit"; bad=1
        elif [ -z "$sha" ]; then echo "MISMATCH $(basename "$w"): no +<sha> local version and no recorded commit to compare with ${tree:0:12} (a published wheel: state it with BOX_WHEEL_COMMIT=<commit>)"; bad=1
        elif [ "${tree:0:${#sha}}" != "$sha" ]; then echo "MISMATCH $(basename "$w") is commit $sha, the tree holds ${tree:0:12}"; bad=1
        else echo "MATCH $(basename "$w") = ${tree:0:12}"
        fi
    done
    [ "$#" -gt 0 ] || { echo "MISMATCH no wheel given"; bad=1; }
    return $bad
}
