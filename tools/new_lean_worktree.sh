#!/bin/bash
# Keep recent benchmark runs and pinned baselines; omit only dated old runs.
set -euo pipefail
helper_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$helper_dir/lean_benchmarks.py" "$@"
