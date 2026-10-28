#!/bin/bash
# Lane overhead-mamba-bwd self-merge gate on one CPU pod (Andrew 2026-10-28:
# compile only). Clears shipped host binaries, then the integration pod smoke
# builds every host family through the binary cache (including the lane's one
# affected root, _mojolearn_tokenizer_host, CPU only; no define), imports
# mojolearn and probes tier_dir. The same-bits fixture
# (tools/tests/overhead_mamba_bwd_fixture.py, pod_fixture.sh) is RUN OWED to
# the orchestrator's end-of-round fixture pass. Exit codes land in lane.rc.
set -u
OUT=${LEG_OUT:?}
rm -f python/mojolearn/host/*.so
bash experiments/overhead-mamba-bwd/pod_smoke_5b.sh > "$OUT/smoke.out" 2>&1
echo "smoke_rc=$?" >> "$OUT/lane.rc"
cat "$OUT/lane.rc"
