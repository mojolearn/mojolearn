# Explicit full-input TSVD variants

`tools/six_lane_prepare_full_tsvd.py` prepares a separately named `tsvd-full-v1`
input and recipe proposal. It preserves the original capped races and evidence.
It never imports MojoLearn or NumPy, constructs a model, compiles, launches a
queue, updates a board or qualifies a result. Runtime kernels and switches are
unchanged. Source and small retained JSON metadata are sufficient for `plan`.

## Input and split contract

The original TSVD preparation in `bench_board_more.py` loads the regression
population, selects raw numeric features, cleans sentinels, and strides to at
most one million rows. It does not standardize. The full variant projects the
exact `X.npy` member of an accepted full raw `big-<dataset>.npz` into a fresh
archive containing only `X`. It keeps values, order, dtype, shape and NPY header
bytes. It neither includes the original test split nor labels or KMeans seeds.

Taxi uses all regression trips and the original TAXI_NUMERIC feature columns.
The classification/expanded-raw Taxi credit-card population is different and
cannot substitute for this input. Standardized regression arrays cannot
substitute for raw X either.

| Dataset | Stored X | Expanded fit X | Expanded query Xq |
|---|---:|---:|---:|
| Taxi | 5,250,086 × 11 | 4,725,078 × 11 | 525,008 × 11 |
| Istella | 2,043,304 × 220 | 1,838,974 × 220 | 204,330 × 220 |

Expanded runners preserve their original split `cut=max(N-N//10,1)`, using
`X[:cut]` and `X[cut:]`. An input archive already containing Xq would bypass
that split; the projector deliberately emits only X. Randomized SVD uses fit X
and does not transform the query portion. The more:tsvd runner instead fits
every stored row, with no held-out inference. Full corpus coverage is not a
claim that expanded fit uses the held-out rows.

The plan derives constructor records from committed source without importing
estimators. It retains original settings, modes, configurations, implementation
IDs and matrix references. On the reviewed source the existing complete-proposed
matrix supplies 12 NVIDIA-native recipes (more:tsvd plus five expanded lanes,
each on two datasets), twelve matching AMD IDENTICAL proposals, and two Apple
FAST recipes (randomized-svd). AMD execution still requires accepted native gfx
artifacts; a registered proposal does not supply a binary. No Apple
more:tsvd or other vendor race is invented when absent from that configuration.

## Planning without dataset reads

Run from the new reviewed main freeze **after** integrating this helper. Keep
plans and projection outputs outside the source checkout. The source-facts
document is the retained list of dataset/input-files/metadata records for the
complete raw big inputs. The deployment inputs are the owning peer's already
accepted metadata; this helper copies their references/content without reading
or revalidating binaries.

```sh
python3 tools/six_lane_prepare_full_tsvd.py plan \
  --source-sha "$TSVD_REVIEWED_SHA" \
  --source-facts /evidence/full-input-facts.json \
  --nvidia-deployments /evidence/native-complete-proposed-artifacts.json \
  --apple-deployments /evidence/apple-artifacts.json \
  --output /evidence/tsvd-full-v1-plan.json
```

Planning opens only source and JSON metadata. It does not open or stat the
referenced NPZ files. Expected source NPZ/JSON and typed-array hashes come from
retained metadata. New output hashes remain null until projection; copying an
expected value into an unexecuted output-hash field is not evidence.

The plan includes the required numerical bindings (`_mojolearn_estimators` for
more:tsvd; `_mojolearn_x_decomp` for expanded lanes). Retain each complete
accepted A/B package and its core/validation dependencies. The plan does not
establish new AMD or default/PTX availability; those remain pending unless the
owner supplies new accepted artifacts. No compilation is requested.

## Later projection, under the existing lease

The campaign root reviews the helper and plan first. Native projection waits
until its current expanded/GMM/PLS chain finishes. Apple projection waits until
the queued CPU14 quality run finishes; do not interrupt or delay that budget.
Acquiring a briefly free lock between cells is not authorization to change
this schedule. Use a new reviewed main freeze, preserving all active freezes.

Canonical existing locks:

- NVIDIA: `/root/six-lane-full-ab-20261006/device-measurement.lock`
- Apple: `/Users/ec2-user/mojolearn-full-f867b50e8/gpu.lock`

Example for the later NVIDIA data-only phase:

```sh
python3 tools/six_lane_prepare_full_tsvd.py project \
  --plan /evidence/tsvd-full-v1-plan.json --dataset taxi --vendor nvidia \
  --input-directory /root/six-lane-full-ab-20261006/data \
  --lock-file /root/six-lane-full-ab-20261006/device-measurement.lock \
  --output /evidence/tsvd-full-v1/nvidia/taxi
```

Use separate fresh output directories for each dataset/vendor attempt. The
tool refuses an existing output directory, a changed/dirty source freeze, a
different projector, a noncanonical/missing/busy lock, changed source hashes,
unexpected array format/shape, or incomplete source coverage. It hashes the
exact source archives before copying, checks source file identity/size/mtime
for concurrent changes, and verifies the streamed X bytes against the retained
typed-array hash. It uses ZIP_STORED with a fixed member timestamp; it never
loads a complete numerical array or performs numerical computation.

Each projection retains:

- Fresh `tsvd-<dataset>.npz` and metadata JSON, source hashes and original loader
  provenance, exact X dtype/shape/hash, new archive/metadata hashes and split.
- `variant-recipes.json`, with distinct `@input=tsvd-full-v1` IDs and original
  workload/cell links, exact settings, scopes and binding requirements.
- `projection-receipt.json` with success/failure, provenance and zero model or
  compiler executions. Failed/partial artifacts are preserved in that attempt
  directory and never admitted; retry into a new directory.

The array hash uses the established
`sha256(str(dtype) + str(shape_tuple) + C-order bytes)` encoding. The recipe's
dataset hash covers canonical JSON of the saved array descriptors; source and
NPZ/JSON hashes remain separate. Hashing happens in offline preparation, outside
any future declared model-operation timing.

## Review and remaining integration

Projection is not execution admission. Recipe proposals intentionally retain
`changes_frozen_race: true` and `execution_authorized: false`. The master refuses to run them as the historical 1M-row race; the explicit
registration below is required for the distinct variant.
The campaign root must review and register the distinct full variant with its
own matrix/evidence identity and accepted A/B deployments. Do not simply clear
the guard or rename an old measurement. This projector does not modify the master,
stagers, opponent roster, existing quality gates or historical boards.

Whole-operation timing must include loading/preparation, the original split,
construction, full prescribed fit, separate inference where present,
synchronization and consumed outputs. Quality/hashes remain outside that timer.
Full matching opponent evidence is still required; old capped results cannot
supply full-input quality or ratios.

Capture limitations remain explicit: more:tsvd returns components rather than
complete fitted state; randomized SVD omits U and singular values and has no
fitted model owner. Other expanded lanes retain full query transformations,
but complete public fitted-state contracts remain pending. Input completeness
does not establish output/model-state identity or candidate-specific runtime
reach. Apple FAST uses task-quality rules; it does not acquire an IDENTICAL
cross-vendor bitwise requirement.

Validation for this change is source parsing, whitespace checks and planning
from retained real metadata only. No fixture arrays, data projection, estimator
execution, compilation, quality evaluation or identity comparison was performed.
The data-copy path and future harness execution remain unverified.

## Explicit execution registration

`tools/six_lane_full_variants.py` now registers distinct matrix cells for only
these 12 NVIDIA complete-proposed and two Apple FAST complete-proposed variants.
The original cell keys and workload IDs remain unchanged. Each new cell retains
its original links and receives the deterministic `@input=tsvd-full-v1` identity.
No other configuration, lane, vendor, input variant or changed race is admitted.

After the owner reviews this registration and projection completes under the
canonical lock, materialize the separate queue with accepted deployments:

```sh
python3 tools/six_lane_register_full_tsvd.py \
  --projection /evidence/tsvd-full-v1/nvidia/taxi \
  --projection /evidence/tsvd-full-v1/nvidia/istella \
  --vendor nvidia --target-track nvidia-native \
  --deployments /evidence/accepted-workload-deployments.json \
  --output /evidence/registered-tsvd-full-v1
```

Deployment entries use the existing materializer schema and the distinct
variant workload IDs. Keep required loaded core/validation and numerical
bindings explicitly scoped to each workload, with their original receipts.
The command writes an **unauthorized** queue; it never runs a model or a build.
The owning controller records the already granted measurement authorization
before launching under its canonical device lock.

`changes_frozen_race: true` stays present throughout. Materialization, queue
creation and every worker independently check the explicit registration,
original/variant cell identities, retained proposal and projection receipt
hashes, original full raw regression metadata, X-only projection and typed-array
hash, exact original split, source-derived settings, output paths, inference
boundary and capture limitations. A changed seed, shape, constructor, input
roster, original cell relabeling, erased change marker or unknown variant remains
an error. Ordinary original-race admission is unchanged.

These additional cells provide full-input timing coverage only. Randomized SVD
still omits returned U/singular-value identity, and complete fitted model-state
coverage remains pending as recorded in each proposal and worker. Registration
neither qualifies output identity nor promotes a candidate or synthesizes an
opponent comparison. Apple scheduling still waits for CPU14 and failed-only
quality retries before any input projection or execution.

## Native AMD full variants

The same twelve full input recipes are explicitly registered for AMD IDENTICAL
`I.X.complete-proposed`, with distinct AMD cell keys and original AMD matrix
links. They retain the same raw corpus, split, constructor settings and capture
limitations as NVIDIA. AMD projection uses the owning Linux worker's canonical
`/root/six-lane-full-ab-20261006/device-measurement.lock`. Pass `--vendor amd` to
projection and registration, `--amd-deployments` to planning when available,
and the actual native target track (for example `amd-gfx942`) to registration.
The existing materializer still requires the exact supported native gfx target,
compiler/source closure, arm flags, and deployed artifact hashes. No portable or
generic AMD mode is admitted; registration is not evidence of compilation.

A new reviewed harness freeze and plan are required. Existing numerical source
closures are unchanged, so already accepted matching compile receipts may be
reused; no NVIDIA/Apple rebuild or completed-cell rerun is implied. A missing
AMD artifact remains blocked until its independently authorized compile repair
provides the accepted native receipt. Same-arm cross-vendor identity and task
quality remain separate requirements, including the existing output/state gaps.
