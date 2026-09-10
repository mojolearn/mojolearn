#!/bin/sh
# Install the repository's hooks into the SHARED hooks directory, so they run
# in every worktree on every branch, including branches that predate the
# hooks. `git rev-parse --git-common-dir` is the main repository's .git even
# when run from a linked worktree.
set -e
here=$(cd "$(dirname "$0")" && pwd)
hooks="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$hooks"
for h in pre-commit pre-push; do
    cp "$here/$h" "$hooks/$h"
    chmod +x "$hooks/$h"
    echo "installed $hooks/$h"
done
