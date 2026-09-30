#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /root/priority-pass
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH
exec python3 tools/priority_pass_run.py --vendor "${1:-nvidia}" >> /root/priority-pass/driver.log 2>&1
