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
# the rebuild fence (2026-10-17): refuses a lane change to a module most bindings import
cp "$here/rebuild_fence.py" "$hooks/rebuild_fence.py"
chmod +x "$hooks/rebuild_fence.py"
# the no-Python-in-kernels check (2026-10-21): a kernel .mojo file must not import Python
cp "$here/no_python_in_kernels.py" "$hooks/no_python_in_kernels.py"
chmod +x "$hooks/no_python_in_kernels.py"
# the no-compute-in-shells check (2026-10-21): a shell .mojo file loops over no data and launches nothing
cp "$here/no_compute_in_shells.py" "$hooks/no_compute_in_shells.py"
chmod +x "$hooks/no_compute_in_shells.py"
echo "installed $hooks/no_host_routes.py"
