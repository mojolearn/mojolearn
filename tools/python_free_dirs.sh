#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Python-free kernel directories: list every tracked Mojo file outside
# bindings/, providers/, experiments/, bench/ and tests/ that imports Python
# (`from python ...`, `from std.python ...`, `import python`, a
# `providers.python` module, or that names the `PythonObject` type), and exit
# non-zero if there is any. A kernel directory whose files import no Python
# cannot run Python. This is an import grep, not a source-count gate: one
# match is a failure, and there is no baseline (AGENTS.md "The Python source
# guards are retired" forbids count gates).
#
# Usage: tools/python_free_dirs.sh [<commit>]   (default: the working tree's
# tracked files; with a commit, that commit's tree, as the pre-push hook does)
set -u -o pipefail
rev=${1:-}
pattern='^[[:space:]]*(from|import)[[:space:]]+((std\.)?python|providers\.python)([.[:space:]]|$)|PythonObject'
excludes=(':(exclude)bindings/' ':(exclude)providers/' ':(exclude)experiments/'
          ':(exclude)bench/' ':(exclude)tests/')
if [ -n "$rev" ]; then
    hits=$(git grep -l -E "$pattern" "$rev" -- '*.mojo' "${excludes[@]}" | sed "s#^$rev:##")
else
    hits=$(git grep -l -E "$pattern" -- '*.mojo' "${excludes[@]}")
fi
status=$?
if [ $status -gt 1 ]; then
    echo "python_free_dirs: git grep failed (status $status)" >&2
    exit 2
fi
if [ -n "$hits" ]; then
    echo "python_free_dirs: these Mojo files outside bindings/, providers/, experiments/, bench/ and tests/ import Python;" >&2
    echo "move their UNPACK/PACK into bindings/ (or the boundary test into tests/):" >&2
    printf '  %s\n' $hits >&2
    exit 1
fi
echo "python_free_dirs: no Mojo file outside bindings/, providers/, experiments/, bench/ and tests/ imports Python"
