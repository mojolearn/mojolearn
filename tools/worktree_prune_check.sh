#!/bin/sh
# Read-only, fail-closed eligibility check for explicitly named worktrees.
# It never removes anything. A caller must separately confirm live-agent,
# cloud-lease and evidence ownership before running `git worktree remove`.
set -eu

[ "$#" -gt 0 ] || {
    echo "usage: $0 /exact/worktree/path [...]" >&2
    exit 2
}

root=$(git rev-parse --show-toplevel)
git -C "$root" rev-parse --verify origin/main >/dev/null
porcelain=$(git -C "$root" worktree list --porcelain)
failed=0

for wt in "$@"; do
    branch=DETACHED
    head=MISSING
    reason=
    if [ ! -d "$wt" ]; then
        reason="missing path"
    elif ! printf '%s\n' "$porcelain" | awk -v p="$wt" '
        $1 == "worktree" { found = substr($0, 10) == p }
        found { seen = 1 }
        END { exit !seen }
    '; then
        reason="not a registered worktree"
    else
        branch=$(git -C "$wt" symbolic-ref --short -q HEAD || printf DETACHED)
        head=$(git -C "$wt" rev-parse HEAD)
        if [ -n "$(git -C "$wt" status --porcelain=v1 -uall)" ]; then
            reason="dirty tracked or untracked content"
        elif printf '%s\n' "$porcelain" | awk -v p="$wt" '
            $1 == "worktree" { here = substr($0, 10) == p }
            here && /^locked/ { found = 1 }
            END { exit !found }
        '; then
            reason="locked"
        elif [ -n "$(git -C "$root" cherry origin/main "$head" | awk '$1 == "+"')" ]; then
            reason="unique stable patch"
        fi
    fi
    if [ -n "$reason" ]; then
        printf 'PRESERVE\t%s\t%s\t%s\t%s\n' "$wt" "$branch" "$head" "$reason"
        failed=1
    else
        printf 'ELIGIBLE_PENDING_OWNERSHIP_CHECK\t%s\t%s\t%s\n' "$wt" "$branch" "$head"
    fi
done
exit "$failed"
