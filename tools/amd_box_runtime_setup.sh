#!/bin/bash
# AMD box runtime for the GPU-only race campaign (MI325X, Ubuntu 24.04): ROCm 7.2.3 HIP runtime, libraries
# and the HIP SDK (mamba-ssm and causal-conv1d build against it). Used by the 2026-10-14 rerun; committed so the
# next run calls the repo copy, not an evidence copy.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key -o /tmp/mojolearn-rocm-signing.key
gpg --batch --yes --dearmor -o /etc/apt/keyrings/mojolearn-rocm.gpg /tmp/mojolearn-rocm-signing.key
cat > /etc/apt/sources.list.d/mojolearn-rocm723.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/mojolearn-rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.3 noble main
EOF
apt-get update
apt-cache policy rocm-hip-libraries rocm-hip-libraries7.2.3 hip-runtime-amd7.2.3
apt-get install -y --no-remove rocm-hip-libraries7.2.3 rocm-hip-runtime7.2.3 rocm-hip-sdk7.2.3
