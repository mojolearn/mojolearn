#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Python-free kernel directories and RUN modules. Exit non-zero, listing the
# files, when
#   - a Mojo file outside bindings/, providers/, experiments/, bench/ and
#     tests/ imports Python, or
#   - a file listed in tools/python_free_doors.txt (paths or globs; RUN modules
#     and entry tables, wherever they live, bindings/ included), or any file
#     matching `*_run.mojo`, imports Python, or
#   - a Mojo file under bindings/ or providers/python/ imports Python and is
#     not a boundary file: `bindings/_mojolearn*.mojo` and `bindings/entry/*.mojo`
#     (PUBLISH), `bindings/py_*.mojo` (shared CPython primitives), `*_unpack.mojo`
#     (UNPACK), `*_pack.mojo` (PACK), `*_door.mojo` (one public operation's
#     unpack, typed RUN call and pack), or an `allow:` line of
#     tools/python_free_doors.txt (dated, owner named; it only shrinks).
#     Everything else there is checked like a kernel directory, or
#   - a Mojo file that is not a boundary file (outside experiments/, bench/
#     and tests/) imports a boundary module (`bindings.<x>` or
#     `providers.python.<x>` whose file matches a boundary pattern or an
#     `allow:` line): RUN reaches the boundary only through typed values and
#     entry tables, never by import, so no typed file can name a
#     Python-holding struct (every one lives in a boundary file). One narrow
#     case is admitted: a native test suite under checks/native_tests/ may
#     import a PUBLISH module (`bindings/_mojolearn*.mojo`) to publish an
#     image's entry-table row as that module's init does; never a door,
#     unpack, pack or py_* module.
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
# The boundary files that may import Python inside bindings/ and providers/python/.
boundary_allow=('bindings/_mojolearn*.mojo' 'bindings/entry/*.mojo' 'bindings/py_*.mojo'
                '*_unpack.mojo' '*_pack.mojo' '*_door.mojo')
allowed=()
while IFS= read -r line; do
    line=${line%%#*}
    line=$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    case "$line" in
        allow:*) line=${line#allow:}; line=${line#"${line%%[![:space:]]*}"}
                 [ -n "$line" ] && allowed+=("$line") ;;
        *) [ -n "$line" ] && doors+=("$line") ;;
    esac
done <<< "$doors_text"

# A literal door or allow entry (no glob) that does not exist is a stale list entry.
missing=()
for door in "${doors[@]}" ${allowed[@]+"${allowed[@]}"}; do
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

# is_boundary <path>: the path matches a boundary pattern or an allow: line
# (a pattern with a slash matches the whole path, a bare one the file name).
is_boundary() {
    local a
    for a in "${boundary_allow[@]}" ${allowed[@]+"${allowed[@]}"}; do
        case "$a" in
            */*) case "$1" in $a) return 0 ;; esac ;;
            *) case "${1##*/}" in $a) return 0 ;; esac ;;
        esac
    done
    return 1
}

# boundary_hits: the Python-importing Mojo files under bindings/ and
# providers/python/ that match no boundary pattern and no allow: line.
boundary_hits() {
    local f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        is_boundary "$f" || printf '%s\n' "$f"
    done < <(if [ $in_git = 1 ]; then
                 grep_paths ':(glob)bindings/**/*.mojo' ':(glob)providers/python/**/*.mojo'
             else
                 grep_paths ':(glob)bindings/*.mojo' ':(glob)providers/python/*.mojo'
             fi)
}

# import_lines: "<file>:<line>" for every `from|import bindings.<x>` or
# `providers.python.<x>` line of the Mojo files outside experiments/, bench/
# and tests/ (one grep over the tree, as grep_paths reads it).
import_pattern='^[[:space:]]*(from|import)[[:space:]]+(bindings|providers\.python)\.[A-Za-z0-9_]'
import_lines() {
    local out status
    if [ $in_git = 1 ]; then
        if [ -n "$rev" ]; then
            out=$(git grep -E "$import_pattern" "$rev" -- '*.mojo' \
                  ':(exclude)experiments/' ':(exclude)bench/' ':(exclude)tests/')
            status=$?
            out=$(printf '%s' "$out" | sed "s#^$rev:##")
        else
            out=$(git grep -H --untracked -E "$import_pattern" -- '*.mojo' \
                  ':(exclude)experiments/' ':(exclude)bench/' ':(exclude)tests/')
            status=$?
        fi
        [ $status -gt 1 ] && grep_status=$status
    else
        out=$(find . -path ./.git -prune -o -path ./.pixi -prune -o -path ./experiments -prune -o \
                   -path ./bench -prune -o -path ./tests -prune -o -type f -name '*.mojo' -print0 \
              | xargs -0 grep -H -E "$import_pattern" 2>/dev/null | sed 's#^\./##')
    fi
    [ -n "$out" ] && printf '%s\n' "$out"
    return 0
}

# import_hits: "<file> imports <module>" for each non-boundary Mojo file that
# imports a boundary module (the fourth check; see the header).
import_hits() {
    local line f mod src
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        f=${line%%:*}
        is_boundary "$f" && continue
        # The module name, by parameter expansion (no process per line).
        mod=${line#*:}
        mod=${mod#"${mod%%[![:space:]]*}"}
        case "$mod" in from*) mod=${mod#from} ;; import*) mod=${mod#import} ;; esac
        mod=${mod#"${mod%%[![:space:]]*}"}
        mod=${mod%%[!A-Za-z0-9_.]*}
        src=${mod//.//}.mojo
        is_boundary "$src" || continue
        case "$f" in
            checks/native_tests/*) case "$src" in bindings/_mojolearn*.mojo) continue ;; esac ;;
        esac
        printf '%s imports %s\n' "$f" "$mod"
    done < <(import_lines)
}

kernel_hits=$(grep_paths "${kernel_spec[@]}")
door_hits=$(grep_paths "${door_spec[@]}")
bnd_hits=$(boundary_hits)
imp_hits=$(import_hits)
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
if [ -n "$bnd_hits" ]; then
    echo "python_free_dirs: these Mojo files under bindings/ or providers/python/ import Python but are not" >&2
    echo "boundary files (_mojolearn*, entry/*, py_*, *_unpack, *_pack, *_door, or an allow: line in $doors_file);" >&2
    echo "move their RUN work to a Python-free module and rename the rest:" >&2
    printf '  %s\n' $bnd_hits >&2
    bad=1
fi
if [ -n "$imp_hits" ]; then
    echo "python_free_dirs: these non-boundary Mojo files import a boundary module (*_door, *_unpack, *_pack," >&2
    echo "py_*, _mojolearn*, entry/* or an allow: line); RUN takes typed values and entry tables, never a boundary import:" >&2
    printf '%s\n' "$imp_hits" | sed 's/^/  /' >&2
    bad=1
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo "python_free_dirs: $doors_file lists files that do not exist (renamed or removed? update the list):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    bad=1
fi
[ $bad = 0 ] || exit 1
echo "python_free_dirs: no kernel-directory Mojo file, listed door, *_run.mojo or non-boundary binding imports Python, and no non-boundary Mojo file imports a boundary module"
