#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Python-free kernel directories and RUN modules. Exit non-zero, listing the
# files, when
#   - a Mojo file outside bindings/, providers/, experiments/, bench/ and
#     tests/ imports Python, or
#   - a file listed in tools/python_free_doors.txt (paths or globs; RUN modules
#     and entry tables, wherever they live, bindings/ included), or any file
#     matching `*_run.mojo`, imports Python.
# "Imports Python" is `from python ...`, `from std.python ...`, `import
# python`, a `providers.python` module, or naming the `PythonObject` type.
# A kernel directory whose files import no Python cannot run Python. This is an
# import grep, not a source-count gate: one match is a failure, and there is no
# baseline (AGENTS.md "The Python source guards are retired" forbids count
# gates). It takes milliseconds; bindings/merged_build.sh and
# tools/development_build.py run it before any compile, and the pre-push hook
# runs it over the pushed tree.
#
# Usage: tools/python_free_dirs.sh [<commit>]
#   no argument: the working tree of the checkout this script lives in (tracked
#                and untracked, not ignored, files: what the compiler would
#                read); outside a git checkout (a `git archive` source tree on
#                a box) a plain grep over the same directories.
#   <commit>:    that commit's tree, as the pre-push hook does (run from inside
#                the repository; the hook runs a temporary copy of the script).
set -u -o pipefail
rev=${1:-}
pattern='^[[:space:]]*(from|import)[[:space:]]+((std\.)?python|providers\.python)([.[:space:]]|$)|PythonObject'
doors_file=tools/python_free_doors.txt
kernel_excludes=(bindings providers experiments bench tests)

if [ -z "$rev" ]; then
    cd "$(dirname "$0")/.." || exit 2
fi
in_git=0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && in_git=1
if [ -n "$rev" ] && [ $in_git = 0 ]; then
    echo "python_free_dirs: a commit argument needs a git checkout" >&2
    exit 2
fi
[ $in_git = 1 ] && [ -z "$rev" ] && cd "$(git rev-parse --show-toplevel)"

# The doors list: from the commit when one is given, else from the tree.
if [ -n "$rev" ]; then
    doors_text=$(git show "$rev:$doors_file" 2>/dev/null) || {
        echo "python_free_dirs: $rev has no $doors_file" >&2; exit 2; }
else
    [ -f "$doors_file" ] || { echo "python_free_dirs: $doors_file is missing" >&2; exit 2; }
    doors_text=$(cat "$doors_file")
fi
doors=('*_run.mojo')
while IFS= read -r line; do
    line=${line%%#*}
    line=$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    [ -n "$line" ] && doors+=("$line")
done <<< "$doors_text"

# A literal door (no glob) that does not exist is a stale list entry.
missing=()
for door in "${doors[@]}"; do
    case "$door" in *[*?[]*) continue ;; esac
    if [ -n "$rev" ]; then
        git cat-file -e "$rev:$door" 2>/dev/null || missing+=("$door")
    else
        [ -f "$door" ] || missing+=("$door")
    fi
done

# grep_paths <pathspec...>: the files among them that import Python.
grep_status=0
if [ $in_git = 1 ]; then
    grep_paths() {
        local out status
        if [ -n "$rev" ]; then
            out=$(git grep -l -E "$pattern" "$rev" -- "$@")
            status=$?
            out=$(printf '%s' "$out" | sed "s#^$rev:##")
        else
            out=$(git grep -l --untracked -E "$pattern" -- "$@")
            status=$?
        fi
        [ $status -gt 1 ] && grep_status=$status
        [ -n "$out" ] && printf '%s\n' "$out"
        return 0
    }
    kernel_spec=('*.mojo')
    for d in "${kernel_excludes[@]}"; do kernel_spec+=(":(exclude)$d/"); done
    door_spec=()
    for door in "${doors[@]}"; do
        # a slash anchors at the root; a bare name matches in any directory
        case "$door" in */*) door_spec+=(":(glob)$door") ;; *) door_spec+=(":(glob)**/$door") ;; esac
    done
else
    # No git: find + grep with the same meaning (`-path` for a door with a
    # slash, `-name` for a bare name).
    grep_paths() {
        local -a names=() prune=() hit=()
        local item
        for item in "$@"; do
            case "$item" in
                ':(exclude)'*) item=${item#':(exclude)'}; prune+=(-path "./${item%/}" -prune -o) ;;
                ':(glob)'*) item=${item#':(glob)'}
                    case "$item" in */*) names+=(-path "./$item" -o) ;; *) names+=(-name "$item" -o) ;; esac ;;
                *) names+=(-name "$item" -o) ;;
            esac
        done
        unset 'names[${#names[@]}-1]'
        while IFS= read -r -d '' f; do
            grep -q -E "$pattern" "$f" && hit+=("${f#./}")
        done < <(find . -path ./.git -prune -o -path ./.pixi -prune -o ${prune[@]+"${prune[@]}"} -type f \( "${names[@]}" \) -print0)
        [ ${#hit[@]} -gt 0 ] && printf '%s\n' "${hit[@]}" | sort
        return 0
    }
    kernel_spec=('*.mojo')
    for d in "${kernel_excludes[@]}"; do kernel_spec+=(":(exclude)$d/"); done
    door_spec=()
    for door in "${doors[@]}"; do door_spec+=(":(glob)$door"); done
fi

kernel_hits=$(grep_paths "${kernel_spec[@]}")
door_hits=$(grep_paths "${door_spec[@]}")
if [ $grep_status -ne 0 ]; then
    echo "python_free_dirs: git grep failed (status $grep_status)" >&2
    exit 2
fi
bad=0
if [ -n "$kernel_hits" ]; then
    echo "python_free_dirs: these Mojo files outside bindings/, providers/, experiments/, bench/ and tests/ import Python;" >&2
    echo "move their UNPACK/PACK into bindings/ (or the boundary test into tests/):" >&2
    printf '  %s\n' $kernel_hits >&2
    bad=1
fi
if [ -n "$door_hits" ]; then
    echo "python_free_dirs: these RUN modules or entry tables ($doors_file, or *_run.mojo) import Python;" >&2
    echo "RUN uses typed Mojo values; keep Python in the UNPACK/PACK binding:" >&2
    printf '  %s\n' $door_hits >&2
    bad=1
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo "python_free_dirs: $doors_file lists files that do not exist (renamed or removed? update the list):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    bad=1
fi
[ $bad = 0 ] || exit 1
echo "python_free_dirs: no kernel-directory Mojo file, listed door or *_run.mojo imports Python"
