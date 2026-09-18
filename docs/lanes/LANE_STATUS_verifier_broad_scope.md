# Broad verifier execution checkpoint — 2026-09-17

User requested broader execution coverage before later measurement, including
withheld CPU routes, parallel variants, saved-model gates and invariance checks.
Worktree: ~/mojolearn-wt/verifier-broad-scope; branch fix/verifier-broad-scope.
Standing workflow: isolated worktree, commit, push and merge verified batches
to main. External artifacts: ~/mojolearn-evidence/verifier-broad-scope/.

## Implemented

- `verify --include-pending`: runs all declared CPU routes, including withheld
  references and supported logical-shard drivers. Current hashes and local
  invariance results survive missing/stale references as OWED. Actual mismatch
  remains DIVERGENT and refusal remains REFUSED. Full broad runs explicitly
  retain unsupported GPU drivers as scope gaps. No admission flag is relaxed.
- Four additional existing CPU shard operations are declared and exercised:
  par-scaler-minmax, par-queries-nn, par-forest-et-clf, par-forest-reg.
- 196 CPU execution routes in total: 179 ordinary lanes plus 17 logical-shard
  drivers. The remaining 33 parallel drivers need GPU bindings; CPU replay
  does not stand in for physical multi-GPU communication.
- `verify --models-only`: loads all four bundled GPU-trained models through
  the CPU saved-model door even on a GPU install, compares model bytes and
  batch results, honors repeats, and does not train for a hidden self-test.
  HostForest/HostGBDT appendix entries expose this installed command and assets.
- Default checks: training, held-out inference, save/reload, batch invariance,
  step/full where applicable. `--batch-checks`: gradient accumulation,
  batch-size, ragged and sampler/replay. Explicit N/A reasons remain visible.
  Cross-device comparison, GPU/CPU cross-check and self-test remain separate
  named operations. Native faults and physical multi-GPU require their own runs.

## Validation and honest remaining scope

246 focused tests pass. Installed wheel source 79eda7622 runs outside the
checkout, using the 32 fresh native binaries from 7f5b786ae unchanged.
All 17 pending CPU lanes and all 17 CPU shard drivers were run on base twice:
zero DIVERGENT/REFUSED; OWED/INCOMPLETE remain where appropriate. Four saved
models pass eight reference comparisons with repeated loaded-model probes.
Pending Transformer extended local properties pass, but references stay OWED.
Full reports, exact wheel receipt, scripts and test logs:
bench/results/verifier-broad-scope/2026-09-17/.

Default qualification counts intentionally stay 162 available / 17 withheld /
50 parallel exclusions. The 20 appendix entries mapped to withheld lanes are
now easy to execute together, not falsely promoted. HostForest and HostGBDT
already had representative portable probes in --all; this work gives them a
clear standalone command and coverage status, not a new blanket certificate.

Next: measure current independent reference columns for the 17 held lanes;
record all-nine and extended-property/native-control evidence for newly exposed
CPU shard routes; qualify the remaining physical multi-GPU pairs; broaden saved
model fixtures beyond the four representatives. Update references only through
strict witnessed admission. The prior 246-gap audit is dated against 714fcddcb;
its missing physical-device and native inference/decode controls remain owed.
PyPI remains 0.8.5; this is an unpublished development wheel. No rentals or
local jobs remain active from this batch. Resource cap remains one local worker
plus at most two cloud vCPUs, all numerical threads one; no unrelated fanout.
