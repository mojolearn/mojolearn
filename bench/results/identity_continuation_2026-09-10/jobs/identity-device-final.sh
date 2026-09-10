#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/final-continuation
if [ "${IDENTITY_JOB_ACTIVE:-0}" != 1 ]; then
 export IDENTITY_JOB_ACTIVE=1
 exec pixi run bash /root/jobs/identity-device-final.sh
fi
out=/root/evidence/identity-device-final
mkdir -p "$out"
mojo --version > "$out/compiler.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$out/gpu.txt"
sha256sum decomposition/impl/linalg/detail/svd_full.mojo > "$out/source-sha256.txt"
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . decomposition/checks/svd_wide_check.mojo > "$out/pca-wide.log" 2>&1
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . decomposition/svd_full_main.mojo > "$out/pca-tall.log" 2>&1
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . ivf/checks/ivf_large_k_check.mojo > "$out/ivf-large.log" 2>&1
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . ivf/checks/ivf_check.mojo > "$out/ivf-full.log" 2>&1
for arm in native simulated64; do
 flags=()
 if [ "$arm" = simulated64 ]; then flags=(-D MOJOLEARN_COLUMN_AMD=1 -D MOJOLEARN_KNN_IDENTICAL_TREE_SELECT=1); fi
 mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${flags[@]}" -I . neighbors/checks/fused_logical32_check.mojo > "$out/knn-fused-$arm.log" 2>&1
 mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${flags[@]}" -I . neighbors/checks/knn_identity_check.mojo > "$out/knn-identity-$arm.log" 2>&1
done
mojo build --target-accelerator=gfx942 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_REQUIRE_CDNA_TARGET=1 -I . neighbors/checks/fused_logical32_check.mojo -o "$out/fused-cdna" > "$out/cdna-build.log" 2>&1
printf 'PASS\n' > "$out/verdict.txt"
