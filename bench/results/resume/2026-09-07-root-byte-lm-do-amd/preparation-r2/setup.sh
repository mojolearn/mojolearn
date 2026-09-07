#!/usr/bin/env bash
# Root-only disposable DigitalOcean AMD image setup, under amd_serial_guard.
set -euo pipefail
[[ $(uname -s) == Linux && -e /dev/kfd && -d /root/mojolearn ]] || exit 2
/usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3,12)'
export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
if ! /usr/bin/python3 -c 'import pip,venv,ensurepip' >/dev/null 2>&1; then
  timeout -k 10 90 apt-get update
  timeout -k 10 90 apt-get install -y --no-install-recommends python3-pip python3-venv
fi
# System-site installation is intentional on this disposable droplet: the
# fixed campaign creates a venv with --system-site-packages. Nothing runs locally.
/usr/bin/python3 -m pip install --break-system-packages --disable-pip-version-check \
  --no-input --no-cache-dir --only-binary=:all: \
  --report /root/byte-lm-do-output/dependency-install.json \
  numpy==1.26.4 pytest==8.3.5 \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl' \
  'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl'
/usr/bin/python3 -m pip freeze > /root/byte-lm-do-output/setup-dependencies.txt
