#!/bin/sh
# CPU-only build of the base-binding host helpers (the CPU training lane,
# phase 1, 2026-09-13). The flags live in bindings/build_host_family.sh
# (folded 2026-09-14); this file names the family. Env:
# MOJOLEARN_CORE_HOST_OUTDIR, MOJOLEARN_HOST_OUTDIR,
# MOJOLEARN_BUILD_EXTRA_DEFINES, MOJOLEARN_BUILD_JOBS, as documented there.
exec sh "$(dirname -- "$0")/build_host_family.sh" core
