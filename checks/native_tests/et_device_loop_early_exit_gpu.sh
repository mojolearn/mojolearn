#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The ET early-exit device loop grows the host queue's forest, bit for bit
# (lane fix-wave2-leftovers, 2026-11-14). The route is a compile-time choice,
# so each arm is its own binary; the two arms build and run in parallel and
# must print one ET_FOREST_DIGEST. Logs: ${ET_EE_LOG_DIR:-.et-early-exit-logs}.
set -euo pipefail
cd "$(dirname "$0")/../.."
log_dir="${ET_EE_LOG_DIR:-.et-early-exit-logs}"
mkdir -p "$log_dir"
src=checks/native_tests/et_device_loop_early_exit_gpu.mojo
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$src" > "$log_dir/host.log" 2>&1 &
p1=$!
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_IDN_ET_DEVICE_LOOP_EARLY_EXIT=1 -I . "$src" > "$log_dir/early_exit.log" 2>&1 &
p2=$!
rc=0
for p in "$p1" "$p2"; do
    wait "$p" || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "et_device_loop_early_exit: an arm failed; see $log_dir/*.log" >&2
    exit 1
fi
d1=$(grep -m 1 '^ET_FOREST_DIGEST' "$log_dir/host.log" || true)
d2=$(grep -m 1 '^ET_FOREST_DIGEST' "$log_dir/early_exit.log" || true)
if [ -z "$d1" ] || [ "$d1" != "$d2" ]; then
    echo "et_device_loop_early_exit: digests differ: host [$d1] early-exit [$d2]" >&2
    exit 1
fi
echo "et_device_loop_early_exit: OK, one digest over two arms ($d1)"
