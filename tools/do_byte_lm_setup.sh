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
# FAIRNESS FIX (2026-10-13 audit): torch was pinned to 2.6.0+rocm6.4.1 (repo.radeon.com cp312
# wheels), an outdated release; on the same MI325X host torch 2.13.0+rocm7.1 ran the board's torch
# arms 1.7-9.4x faster (2026-09-30). The pin is now the current release from the PyTorch ROCm 7.1
# index (tools/bench_board.py AMD_TORCH_ROCM). Results raced before this change were wrong for the
# torch opponent; apply extra scrutiny.
/usr/bin/python3 -m pip install --break-system-packages --disable-pip-version-check \
  --no-input --no-cache-dir --only-binary=:all: \
  --report /root/byte-lm-do-output/dependency-install.json \
  --extra-index-url https://download.pytorch.org/whl/rocm7.1 \
  numpy==1.26.4 pytest==8.3.5 'torch==2.13.0+rocm7.1'
/usr/bin/python3 -m pip freeze > /root/byte-lm-do-output/setup-dependencies.txt
