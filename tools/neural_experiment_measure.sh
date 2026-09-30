#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p /root/neural-experiment-results-v2
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH
exec python3 tools/neural_experiment_measure.py --vendor "${1:?vendor}" >> /root/neural-experiment-results-v2/driver.log 2>&1
