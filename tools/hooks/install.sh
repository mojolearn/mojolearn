#!/bin/sh
# Install the repository's hooks into the SHARED hooks directory, so they run
# in every worktree on every branch, including branches that predate the
# hooks. `git rev-parse --git-common-dir` is the main repository's .git even
# when run from a linked worktree.
#
# The installed copies never update themselves: rerun this script after any
# change to tools/hooks/pre-push or tools/hooks/no_host_routes.py. On
# 2026-10-08 the installed copies predated the --branch mode (2026-10-07), so
# every non-main push was judged on its whole tree. Since 2026-10-09 pre-push
# also skips the host-route fence for archive refs (refs/heads/archive/*,
# refs/tags/archive-*, refs/tags/archive/*) and says so in one line; the size
# fence still runs for every push.
set -e
here=$(cd "$(dirname "$0")" && pwd)
hooks="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$hooks"
for h in pre-commit pre-push; do
    cp "$here/$h" "$hooks/$h"
    chmod +x "$hooks/$h"
    echo "installed $hooks/$h"
done
# pre-push runs this beside itself, so branches that predate it are checked too
cp "$here/no_host_routes.py" "$hooks/no_host_routes.py"
chmod +x "$hooks/no_host_routes.py"
echo "installed $hooks/no_host_routes.py"
