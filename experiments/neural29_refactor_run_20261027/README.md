# Prepared neural measurement plan — not launched

Owner gate updated 2026-10-27: before any new measurement, neural execution,
including tape/Program/Sequential, must actually avoid Python computation,
callbacks and internal CPython dispatch and retain stable metadata in native
state. Folder moves alone do not meet this gate. The run then needs one focused
Mac bit check and the actual NVIDIA/AMD target builds. There is **no separate
NVIDIA/AMD validation campaign** and no release prerequisite. Use incremental
development artifacts with truthful source/binary provenance.

`manifest.json` selects exactly 29 GPU neural recipes per vendor, identical mode,
full saved recipe shapes, one excluded warmup and one scored round. `recipes.json`
contains every dimension, work-changing setting, dataset, arm and clock. Full
means the recipe's configured workload: HIGGS fit recipes use 1,000,000 training
and 100,000 held-out rows, one epoch; recurrent rows form length-16 sequences.
The generation recipes use their pinned checkpoints, one 128-token prompt and
128 new tokens. Other algorithms use their recorded full shape. No hidden small
substitute or new tuning is introduced.

## Existing boxes and fresh directories

Both boxes remain owned and running. The prior source runs were stopped by root;
this lane performs no SSH, launch, compilation, rental, lease or teardown action.

| Vendor | Existing transport | Fresh run directory |
| --- | --- | --- |
| AMD | `root@162.243.78.145` | `/root/measure-neural29-refactor-amd-20261027` |
| NVIDIA | `ssh -p 30772 root@38.80.152.75` | `/root/measure-neural29-refactor-nvidia-20261027` |

Local templates and new seed stores are under
`~/mojolearn-evidence/neural29-refactor-run-plan-20261027/prepared/`.
The templates deliberately contain unresolved frozen-source/interpreter/manifest
fields. They are not launch-ready. Use separate vendor controllers so the GPUs
can run in parallel, with recipes serial within each GPU. `require_existing_box`
prevents accidental provisioning; `release_after_archive=false` keeps both boxes.

Both candidate switches are explicit in `runtime_env`:
`MOJOLEARN_NEURAL_NATIVE_DISPATCH=1` and
`MOJOLEARN_GENERATION_NATIVE_SESSION=1`. The remote worker now applies these
config values before starting children and records their actual environment.
Normal resident routing is preserved; this plan does not force every recipe onto
a nonresident route merely to exercise an optional refactor. Coverage of each
actual timed route must be established by the preceding native-operation audit.

`neural_smoke=false`: no separate smoke/validation campaign. The existing
measurement's own settings/input/scorer preflight remains; accepted recipes then
fit/run. Refusals are logged and skipped, crashes/timeouts and missing time or
quality are logged as failures/incomplete, and `--keep-going` proceeds through
independent recipes. Retention failures still pause before the next recipe so
completed evidence is not lost. Every completion goes to Mac and R2 through the
existing controller. Final board ingestion uses board tools after review; the
prepared controller does not automatically commit or push boards.

## Reuse evidence

New stores were reconstructed from the 29 retained per-recipe boards on each
vendor; the old stores/boards were never overwritten. Candidates passed finite
recorded time/quality, current work-changing settings, complete parameter keys,
verified source/artifact receipts and vendor/full-shape/round-count checks.

| Vendor | Eligible opponent cells | Recipes with eligible cells | Input-file keys | Core-wheel keys |
| --- | ---: | ---: | ---: | ---: |
| AMD | 72 | 23 | 20 | 52 |
| NVIDIA | 119 | 27 | 30 | 89 |

`opponent-reuse.json` lists each exact arm/key/evidence reference and missing arms.
These are eligible retained candidates, **not guaranteed hits**. During normal
measurement preflight the store requires fresh matching effective parameters,
library/version, actual device, settings, input identity and timing rounds. New,
failed, incomplete or mismatching opponents run; no selection uses timing ratios.
Some recipes have no supported real opponent, which stays explicit.

Two unresolved reuse details must remain visible:

1. Non-HIGGS neural store keys currently identify data using the core wheel hash.
   A newly packaged runtime can invalidate 52 AMD/89 NVIDIA otherwise valid cells
   even when actual staged inputs did not change. Do not edit those old keys or
   claim data equivalence from a version or source label. Any transfer requires
   proof of the actual input/checkpoint/generator bytes and preserved original
   evidence; otherwise the existing key check schedules that opponent again.
2. Stored absolute quality is retained. Historical `*_vs_ours`, token agreement
   and divergence annotations refer to the old own arm. They must not be shown
   as freshly compared to the candidate without refreshing the reference from
   retained outputs (or rerunning an opponent whose needed outputs are absent).
   Poor quality never rejects an otherwise valid comparison.

## Remaining launch integration

Root must fill the source freeze and installed candidate interpreter only after
the new native-operation gate and focused Mac check. Use a fresh runtime/cache;
reuse pinned datasets/checkpoint bytes separately. Preserve baseline wheel and
replacement-group receipts and actual loaded binary hashes for development
artifacts. The campaign currently accepts PyPI or candidate-wheel manifests;
it does **not** yet directly admit a `DEVELOPMENT_PROVENANCE.json` overlay. Exact
development-runtime manifest admission is pending. Do not bypass that check or
label altered baseline files as unchanged PyPI bytes. No release is required.

After those inputs are concrete, each filled controller config is consumed by
`tools/run_full_do_measurements.py`. That is a later root-owned launch action,
not an action performed by this preparation script. Its source bundle must
contain the selected source commit and this worker's runtime-switch support.
Preserve original stopped-campaign evidence and complete the ongoing archive
before sharing I/O bandwidth with a new run.

`prepare.py` reproducibly reads only retained local boards, writes new stores and
emits the plan. It never executes a measurement. Complete preparation/test logs
are in `~/mojolearn-evidence/neural29-refactor-run-plan-20261027/`.
