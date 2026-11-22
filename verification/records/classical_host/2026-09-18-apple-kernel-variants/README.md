# Six kernel saved-model records, Apple M4 / independent CPU replay

All six kernel variants, all nine harness fixtures: **54 GPU model recordings,
54 successful CPU saved-model replays**, each also compared against the prior
matching Apple inference column. Captured from Python/tool commit `786163408`.
The local slot completed in 5.55 seconds, one numerical worker, threads=1.

The GPU kernel and CPU estimators bindings are retained binaries from the prior
kernel-identity implementation, not freshly rebuilt from this tool commit.
Their exact SHA256 values and reused source support are retained in receipt.json
and the original `identity_break/2026-09-18-cpu-kernel-variants` evidence.
The recipe references external native artifacts, intentionally not committed.

This validates the newly enabled recording/replay probes and supplies real Apple
saved-model evidence. NVIDIA/AMD recordings, current-artifact qualification and
installed-wheel replay remain owed. No ordinary reference hold was removed.
