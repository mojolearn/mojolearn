#!/bin/sh
# CPU-only build of the svm binding: SVC and the isolation forest (iforest),
# the CPU training lane, phase 1. The flags live in
# bindings/build_host_family.sh (folded 2026-09-14); this file names the
# family. Env: MOJOLEARN_SVM_HOST_OUTDIR, MOJOLEARN_HOST_OUTDIR,
# MOJOLEARN_BUILD_EXTRA_DEFINES, MOJOLEARN_BUILD_JOBS, as documented there.
exec sh "$(dirname -- "$0")/build_host_family.sh" svm
