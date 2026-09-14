#!/bin/sh
# CPU-only build of the trees binding: the ExtraTrees classifier and regressor
# fits and predict (et-clf, et-reg; the CPU training lane, phase 1,
# 2026-09-14). The flags live in bindings/build_host_family.sh; this file
# names the family. Env: MOJOLEARN_TREES_HOST_OUTDIR, MOJOLEARN_HOST_OUTDIR,
# MOJOLEARN_BUILD_EXTRA_DEFINES, MOJOLEARN_BUILD_JOBS, as documented there.
exec sh "$(dirname -- "$0")/build_host_family.sh" trees
