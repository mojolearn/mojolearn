#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /root/classical-pass
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH
exec python3 tools/classical_pass_run.py --vendor "${1:?vendor}" >> /root/classical-pass/driver.log 2>&1
