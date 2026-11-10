#!/bin/sh
# Install the repository's hooks into the SHARED hooks directory, so they run
# in every worktree on every branch, including branches that predate the
# hooks. `git rev-parse --git-common-dir` is the main repository's .git even
# when run from a linked worktree.
#
# The installed copies never update themselves: rerun this script after any
# change to the installed hook sources. On
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
# Rebuild scope uses only the import graph, with no runtime classification.
cp "$here/mojo_import_graph.py" "$hooks/mojo_import_graph.py"
# the rebuild fence (2026-10-17): refuses a lane change to a module most bindings import
cp "$here/rebuild_fence.py" "$hooks/rebuild_fence.py"
chmod +x "$hooks/rebuild_fence.py"
# Owner 2026-11-10: retire the misleading Python source guard, including
# its previously installed copy in the shared hooks directory.
rm -f "$hooks/no_python_in_kernels.py" "$hooks/no_compute_in_shells.py" "$hooks/no_host_routes.py"
# the no-runtime-import_module check (2026-11-06): no new Python.import_module on a bindings/providers call path
cp "$here/no_runtime_import_module.py" "$hooks/no_runtime_import_module.py"
chmod +x "$hooks/no_runtime_import_module.py"
echo "installed $hooks/mojo_import_graph.py"
