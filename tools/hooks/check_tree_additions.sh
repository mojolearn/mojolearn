#!/usr/bin/env bash
# Refuse a push that ADDS run output to the tree (2026-11-21 tree clean-up; Andrew: "keep measurements; the
# intermediate logs can go; keep the key logs; stop new run output landing in the repo").
#
# Usage: check_tree_additions.sh <local sha> <remote sha | 0000...> <remote name>
# Called by tools/hooks/pre-push once per pushed branch, from the pushed tree's own copy.
#
# It judges only paths ADDED by commits that are not yet on any ref of <remote> (renames count as adds) and
# that still exist in <local sha>'s tree. Deletions and edits of tracked files are always allowed. Refused:
#   - any new file under docs/handoffs/ except docs/handoffs/README.md (handoffs live in ~/mojolearn-evidence/);
#   - any new *.log under bench/results/;
#   - any new file under bench/results/ outside the allowlist below: the live boards and the records that
#     tools write there on purpose. Keep this list identical to the bench/results block in .gitignore.
# A refused file belongs in ~/mojolearn-evidence/<run>/ and R2 (tools/box_persist_run.sh uploads a box job's
# results dir; python3 tools/evidence_r2.py push <dir> uploads a laptop dir). A new kind of record that must
# be tracked is added to this allowlist and to .gitignore in one reviewed change; there is no override switch.
#
# Allowlist (writer in brackets):
#   bench/results/bench_board/**            live boards        [tools/main_board_ingest.py]
#   bench/results/board_best_ever/*.json    best-ever boards   [tools/board_best_ever.py]
#   bench/results/board-archive/**          archived boards    [tools/board_best_ever.py]
#   bench/results/opponent-archive/**       stored opponents   [tools/main_board_ingest.py]
#   bench/results/paper_compare/*           paper compare      [tools/paper_compare.py, measure_candidate.py]
#   bench/results/dataset_store/manifest.tsv dataset manifest  [dataset store tools]
#   bench/results/identity_break/** (no *.log) identity columns, verify inputs
#                                           [tools/admit_identity_columns.sh, tools/retain_run_summary.py]
#   bench/results/reference_admissions/**   release tools
#   bench/results/release_evidence/**       release tools
#   bench/results/release_verification/**  release tools
#   bench/results/ARCHIVE_INDEX.md          record of removed evidence
#   bench/results/r2-index.tsv              [tools/box_evidence_r2.sh]
#   bench/results/CELLS_INDEX.tsv           record index the matrix and census read [tools/record_index.py]
set -u
local_sha=${1:?local sha}
remote_sha=${2:-0000000000000000000000000000000000000000}
remote=${3:-origin}
zero=0000000000000000000000000000000000000000

allowed() {
    case "$1" in
        bench/results/*.log|bench/results/*/*.log) return 1 ;;
    esac
    case "$1" in
        bench/results/bench_board/*|bench/results/board-archive/*|bench/results/opponent-archive/*) return 0 ;;
        bench/results/identity_break/*|bench/results/reference_admissions/*) return 0 ;;
        bench/results/release_evidence/*|bench/results/release_verification/*) return 0 ;;
        bench/results/board_best_ever/*/*|bench/results/paper_compare/*/*) return 1 ;;
        bench/results/board_best_ever/*.json|bench/results/paper_compare/*) return 0 ;;
        bench/results/dataset_store/manifest.tsv|bench/results/ARCHIVE_INDEX.md|bench/results/r2-index.tsv) return 0 ;;
        bench/results/CELLS_INDEX.tsv) return 0 ;;
        bench/results/*) return 1 ;;
        docs/handoffs/README.md) return 0 ;;
        docs/handoffs/*) return 1 ;;
    esac
    return 0
}

exclude=(--not "--remotes=$remote")
if [ "$remote_sha" != "$zero" ] && git cat-file -e "$remote_sha^{commit}" 2>/dev/null; then
    exclude=("^$remote_sha" "${exclude[@]}")
fi

# paths added by the new commits, restricted to the two guarded trees
added=$(git log --format= --name-only --diff-filter=A --no-renames "$local_sha" "${exclude[@]}" \
            -- bench/results docs/handoffs 2>/dev/null | sort -u) || exit 2
[ -n "$added" ] || exit 0
# only paths the pushed tree still holds (added then removed inside the push is fine)
present=$(git ls-tree -r --name-only "$local_sha" -- bench/results docs/handoffs | sort -u)
offending=$(comm -12 <(printf '%s\n' "$added") <(printf '%s\n' "$present") | while IFS= read -r p; do
    [ -n "$p" ] && ! allowed "$p" && printf '%s\n' "$p"
done)
[ -n "$offending" ] || exit 0

count=$(printf '%s\n' "$offending" | wc -l | tr -d ' ')
echo "pre-push: REFUSED: this push adds $count file(s) that do not belong in the tree:" >&2
printf '%s\n' "$offending" | head -n 20 | while IFS= read -r p; do
    case "$p" in
        docs/handoffs/*) echo "  $p  -> handoffs live in ~/mojolearn-evidence/ (and R2), not docs/handoffs/" >&2 ;;
        *.log)           echo "  $p  -> logs belong in ~/mojolearn-evidence/<run>/ and R2 (tools/box_persist_run.sh)" >&2 ;;
        *)               echo "  $p  -> run output belongs in ~/mojolearn-evidence/<run>/ and R2 (tools/box_persist_run.sh)" >&2 ;;
    esac
done
[ "$count" -gt 20 ] && echo "  ... and $((count - 20)) more" >&2
echo "  git rm --cached the files (keep them in ~/mojolearn-evidence/), commit, push again. Allowed paths:" >&2
echo "  tools/hooks/check_tree_additions.sh header; a new tracked kind is added there and in .gitignore." >&2
exit 1
