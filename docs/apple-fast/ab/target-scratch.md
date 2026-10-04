# TargetEncoder device scratch experiment

Base origin/main 973fdf16b. Branch lane/apple-fast-target-scratch.
Define `MOJOLEARN_TARGET_SCRATCH`, opt-in FAST + Apple only.
Build binding: `bindings/build_x_prep.sh`.

TargetEncoder currently calls `_codes` but never consumes its unknown counts.
That adds a serial walk of every row per column (`count_neg_unit`) to both
fit and transform. The candidate skips only that unused computation, keeps
category codes in device scratch, and moves the bucket starts/row indices
from downloaded host arena to device scratch. It explicitly clears bucket
rows so unused tails for omitted categories retain their zero indices.
All encoding arithmetic, bucket ordering, unknown-category semantics and
fitted/output words remain unchanged. This does not revive the old TE_GLOBAL
or TE_ENC parallel-reduction flags.

Run one M3 arm each for target-encoder taxi and istella. No opponent reruns.
Quality in both compiled arms (choose distinct OUT paths):

```
PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_VENDOR=apple \
 ~/board-0834/cache/venv/bin/python tools/target_scratch_quality.py dump OUT.npz
```

Then `... tools/target_scratch_quality.py compare main.npz candidate.npz`.
The checker records all public fitted encodings and transform/cross-fit output
values for continuous/binary/multiclass targets, auto/fixed smoothing, unknown
queries, and explicit categories that omit training values. It independently
checks fixed-smoothing predictions against the mathematical expression. A/B
must be exact because no arithmetic has changed. No local builds/tests run.

## Current-main integration and quality gate (2026-10-04)

Continuation branch `lane/apple-fast-target-current` merges current main
`bd7839d0d` into original `bbda5d847` (original baseline `973fdf16b`). This
reuses the existing experiment; no alternative kernel/reduction is added.
Follow [EXPERIMENT_PROCESS.md](../EXPERIMENT_PROCESS.md).

Source review: the only merge conflict was adjacent optional binding
exports. Both current-main `x_prep_label_present` and candidate
`x_prep_target_scratch` exports are preserved. Main's label-direct default
therefore remains enabled in A and B. Python changes merged cleanly:
`_codes(device_only=False)` preserves all other encoder callers; only
TargetEncoder opts into omitted count_neg and scratch allocation. Its
unknown counts are never consumed. Scratch bucket-row tails remain
explicitly zeroed before gather, preserving original initialized-arena
semantics. All encoding kernels/arithmetic are unchanged. Candidate gate
remains explicitly FAST + Apple and opt-in. Review `git diff bd7839d0d..HEAD`
for the isolated final delta.

**Rebuild both x_prep arms.** Current main changed this binding since the old
candidate; old `bbda5d847` binaries would omit the accepted label default.
A has no experiment define; B adds only `-D MOJOLEARN_TARGET_SCRATCH`.
Neither arm sets LABEL_DIRECT_OFF or old TE_GLOBAL/TE_ENC switches.

Manager compile command (not run by this agent):

```
bash ~/mojolearn-evidence/apple-fast/sync/compile_arms_m2.sh lane/apple-fast-target-current x_prep MOJOLEARN_TARGET_SCRATCH
```

Resolve final branch full SHA as `SOURCE`, synchronize that exact M3
checkout and transfer verified A/B binaries/manifest to
`~/mq/verified-arms/SOURCE/x_prep/`. Ensure current-main FAST dependencies
and an installed x_prep exist using manager `ensure_so` before quality.
No helper below builds bindings. Use fresh tags; preserve any prior results.

M3 quality CMD, before any timing:

```
~/board-0834/cache/venv/bin/python tools/target_scratch_pair.py quality SOURCE gap26-target-current-quality
```

The helper validates source, manifest, defines and SHA256 hashes, refuses
artifact reuse, installs each arm into a fresh checker process, and restores
the original binding on exit. The checker verifies FAST Metal and records
actual loaded-binary hash plus scratch_enabled (must be false for A, true
for B). Its 108 arrays cover cross-fit output, transform after fit_transform,
transform after separate fit, fitted encodings and target means over all
12 existing fixture combinations. Fixed-smoothing outputs also pass the
independent closed-form category-mean oracle. No arithmetic changed, so
A/B must have identical dtype, shape and bytes; do not relax after failure.
PASS.json is written only after both oracle passes and the paired comparison.

Only after PASS, serialize these M3 timing CMDs, one scored run per arm:

```
~/board-0834/cache/venv/bin/python tools/target_scratch_pair.py timing SOURCE gap26-target-current-quality gap26-target-current-taxi taxi
~/board-0834/cache/venv/bin/python tools/target_scratch_pair.py timing SOURCE gap26-target-current-quality gap26-target-current-istella istella
```

Timing rechecks exact checkout/source and receipt binary hashes, then calls
manager `verified_arms.py` -> `afc_ab_def.sh`. Their nonempty race refusal
prevents duplicate scored arms. Failed/missing/stale quality exits before
any timing. No opponent runs. Source or binary changes invalidate receipts;
rebuild/recheck as appropriate. Logs are under
`~/mq/out/gap26-target-current-quality-quality/`; read bounded
`TARGET-SCRATCH-*`, `AFC-AB` and `AFC-DEF-SUMMARY` lines only.

Verdict remains OPEN: no compile, quality or M3 speed result exists for this
integration. Agent ran no builds, tests, SSH, queue operations or main merge.
