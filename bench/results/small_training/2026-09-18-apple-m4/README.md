# Small training pilot: measured CPU/Apple captures

**Current captures: provenance-v2/.** Initial captures below exposed a shared
verifier bug: `.git` files in linked worktrees were overlooked and device.commit
was null. Those initial files are preserved unchanged as historical evidence.
The lookup is repaired and the pilot now also uses the harness's explicit
commit witness. All four refreshed captures name source c47301552. The Metal
device record additionally preserves WORKING TREE DIRTY while the temporary
native-directory symlink was present; CPU records have no dirty marker. The
profile/harness digests agree across the records. Their raw logs, exact commands, per-process timing JSON and shared-slot
timing are retained under provenance-v2/.

Refreshed durations: CPU 0.731 s, CPU replay 0.686 s, Metal 2.414 s, Ridge fault
0.697 s. The whole serial session took 4.593 s. Again all 180 parts match across
CPU/replay and CPU/Metal; all 60 Ridge parts move under the native fault. These
warm runs are not comparable to a cold-start benchmark. Qualification is still
owed, and the initial timing observations below remain historical.

Profile: small-training-v1. Three lanes (Ridge, StandardScaler, MinMaxScaler),
15 cases, two independent fits per cell, four compared parts per cell (train,
inference, saved model, batch invariance). Maximum 257 rows and 17 columns;
each feature matrix occupies at most 17,476 bytes. This is input-array size,
not a measurement or guarantee of total process/GPU memory.

| Capture | Source commit | Slot process duration | Result |
| --- | --- | ---: | --- |
| cpu.json | d38ac81b4 | 1.043 s | 45 cells captured |
| cpu-replay.json | d38ac81b4 | 0.722 s | all 180 parts match CPU |
| metal.json | d38ac81b4 | 4.010 s | all 180 parts match CPU |
| cpu-ridge-sabotage.json | 9bc577ce5 | 0.920 s | all 60 Ridge parts differ; other lanes unchanged |

Process durations above are the shared slot's observed run_seconds, excluding
queue wait (under 0.006 seconds each). The JSON elapsed_seconds field measures
the capture loop after import; it is a different measurement. These are single
observations, not a benchmark distribution or an estimate for all algorithms.
The tool checkpoints each completed cell by atomic replacement.

CPU is Apple M4. Native files were reused from the frozen release worktree's
python/mojolearn/host and identical directories; their actual hashes, package
path and CPU/GPU provenance are in the captures. This is a source-profile
experiment with retained native artifacts, not qualification of a new wheel.
The GPU directory was temporarily symlinked into the dedicated profile worktree
for the Metal capture, then removed. No native rebuild was needed.

All captures used tools/mac_slot.py --timeout 120 --wait-timeout 60 metal,
nice -n 19, the release worktree's .pixi/envs/test/bin/python, and
tools/capture_small_training.py --require-backend cpu|metal --out NEW_DIRECTORY.
The capture tool pins supported CPU thread settings to one before imports.
MOJOLEARN_HOST_DIR pointed to the release host directory for clean captures.

The Ridge fault capture used a separate external directory containing symlinks
to the clean host files, except _mojolearn_estimators_host.so, which pointed at
the retained cpu-kernel-identity/sabotage binary. MOJOLEARN_HOST_ALLOW_SABOTAGE=1
was explicit. Its JSON host.families readback reports sabotage=true on that
binding. All 15 Ridge train/infer/model/batch cells moved under this native
fault; the untouched scaler families stayed equal. Scaler-native fault controls
remain owed; a Ridge fault does not certify those families' controls.

The Metal process emitted repeated "Context leak detected, CoreAnalytics
returned false" diagnostics and exited 0 with stable numerical captures. They
were visible in the tool output; no claim of memory/leak qualification follows
from the numerical match. System memory pressure was normal after the run.

comparisons.json and ridge-fault-comparison.json were produced by
mojolearn._verify_small.compare_captures, which requires complete cells, actual
repeats, matching profile/source/input witnesses and native binding digests.
The CPU replay is same-backend evidence; the CPU/Metal comparison is explicitly
cross-backend. Both remain NUMERICAL_MATCH_UNQUALIFIED. No reference was admitted.

Next: scaler fault controls, NVIDIA/AMD captures under the same profile,
installed-wheel replay, and per-family expansion with measured size budgets.
The full historical verifier table and public route count are unchanged.
