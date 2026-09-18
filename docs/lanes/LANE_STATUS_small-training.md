# Small training verification pilot

Worktree: ~/mojolearn-wt/next-wheel-coverage. Changes are checkpointed on
lane/next-wheel-coverage and merge to main after source/evidence checks.

User direction: routine verification should be inexpensive and resource-aware;
keep boundary cases, separate inference/training/release certification, retain
evidence in stages, and avoid repeating the entire matrix for every edit.

Implemented an isolated small-training-v1 pilot in _verify_small.py and the
maintainer capture tool tools/capture_small_training.py. It covers Ridge,
StandardScaler and MinMaxScaler only, not all 236 routes. Fifteen cases include
31/32/33, 63/64/65 and 255/256/257 rows, 17 columns on the largest odd case,
ties, duplicate/constant columns, wide dynamic range, denormals/FTZ, negatives.
Legacy fixture('odd') ignores requested dimensions and allocates 12345x17, so
the pilot deliberately uses explicit base dimensions for row-boundary cases.

Measured on Apple M4: CPU captures took 1.043 and 0.722 seconds, local Metal
capture 4.010 seconds end-to-end under the shared native slot. All 180
train/infer/model/batch hashes matched across CPU replay and CPU/Metal.
An actual native estimator sabotage binary moved all 60 Ridge parts across
15 fixtures; untouched scaler outputs stayed equal. Raw captures, comparisons,
binary hashes, commands, timing scope and Metal diagnostics are retained under
bench/results/small_training/2026-09-18-apple-m4/README.md.

The profile is a maintainer pilot, not a new public default or a shipped
reference. Its maximum input shape is enforced, each completed cell is saved
atomically, and incomplete/unstable/refused/inapplicable numerical evidence
cannot compare as a match. Matching captures explicitly remain UNQUALIFIED.
No full-size historical fixture or reference was changed. The source tests
check dimensions, actual pathological values, repeatability, input/source
compatibility and rejection of incomplete/invalid evidence without fitting.

Remaining: native scaler controls; independent NVIDIA/AMD; installed-wheel
qualification and explicit reference admission; then additional families with
reviewed small inputs. Do not extrapolate three-lane timings to all algorithms
or label all input sizes/hardware configurations proven. GPU memory/leak
behavior was not qualified by these short numerical runs.

Continue honoring the Apple full-certification hold. These brief local captures
did not start another certification workflow or rental. Multi-GPU CV hardware
qualification and the frozen 0.8.7 UMAP/release gates remain separate work.

Provenance follow-up: _verify._git_commit only recognized .git directories,
missing linked worktrees' .git files. Two source-identity tests cover the repair
and preservation of the dirty-tree marker. The pilot also records the harness's
explicit commit witness. Original null-commit captures remain unmodified;
provenance-v2/ contains fresh four-arm captures at source c47301552, raw
logs and exact per-process/slot times. CPU 0.731/0.686 s, Metal 2.414 s, native
Ridge fault 0.697 s; 4.593 s for the whole serial session. Results again match
180 parts and catch 60 Ridge fault differences. Current tests inspect those
refreshed native records; all 33 profile/resource/provenance tests passed.
The Metal record preserves a dirty-tree marker while the temporary native
directory symlink was present; it is not presented as a clean-checkout release
qualification. CPU records are clean and profile/harness hashes match throughout.
