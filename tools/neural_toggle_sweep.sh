#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /root/neural-toggle-sweep
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH
exec python3 tools/neural_toggle_sweep.py --vendor "${1:?vendor}" >> /root/neural-toggle-sweep/driver.log 2>&1
