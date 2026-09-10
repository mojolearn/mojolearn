# Forest inference measurement contract

`bench/speed/forest_inference_ab.py` compares transient and resident
`parallel_groves` calls on the same fitted RF or ET. `--include-sequential` adds
the existing host inference reference when its cost is useful. It supports FAST Metal and
IDENTICAL CUDA, verified through compiled binding readbacks. Training is GPU;
`sequential` is an optional host inference reference, not a CPU training arm.
No default is changed by this harness.

Use HIGGS for binary vector probabilities, Covtype for seven-class vectors and
Year for scalar regression. Large real datasets are required for performance
decisions. Report actual prediction rows and model complexity, feature/class
counts and memory pressure; there is no universal row threshold. Synthetic and
smoke runs are correctness/diagnostic evidence only. Dataset loading has no
synthetic fallback. A prediction-row cap takes a held-out prefix and refuses
requests larger than the split; it never fabricates scale by repeating rows.

The first public call is retained as round zero, including lazy model setup.
Subsequent measured rounds rotate arm order and may reuse resident device state.
Public timing includes input packing, native dispatch, input transfer and host
output return; training and dataset preparation are excluded. Same-engine hashes must repeat and transient/resident grove outputs must match
exactly. Transient mode invokes the public method with only its shared dispatcher
overridden to the original stateless GPU pointer ABI; it retains public input
checking and output construction. The resident arm uses the ordinary public
method and its device-model cache. Optional sequential inference may differ in
association: record maximum absolute error and explicit tolerances rather than
claiming sequential-bit equivalence. Compiled `forest_vector_groves(outputs)`
readback is required and retained alongside mode/vendor and source/binary hashes. At least five measured calls and a
max/min spread at most 1.10 in each arm are required by the timing gate.
`complete=true` alone does not establish stable timing or a speed claim.

`--calls-per-sample N` optionally measures a block of N repeated public calls
and reports both the full block duration and normalized milliseconds per call.
The first-call warmup remains one call. Every output is retained until the timer
stops, then checked for shape/finiteness and hashed; MojoLearn outputs must all
match their engine and the other grove arms. This measures repeated-call
throughput, not single-call latency, and temporarily retains N output arrays.
JSON schema 2 records that scope, count and per-call hashes. Preserve the prior
single-call source/results before using one bounded batched diagnostic to assess
jitter; do not repeat until a favorable timing appears.

Optional `--staged-rounds` runs separately instrumented public calls after the
primary timings. It records RF's exposed input-check/packing boundary where
available, shared `_predict_forest` dispatch and Python remainder. ET input
checking is part of that remainder. Dispatch includes cache checks, native
validation, allocations, transfers, traversal and output; it is not kernel-only
time. Cold preparation is captured by the primary first call, not these warm
instrumented calls. Byte counts describe host arrays, not peak GPU memory;
retain external device/host telemetry for memory conclusions.

Optional `--borrowed-buffers` adds a third grove arm through
`forest_predict_resident_into_gpu`. It changes only the shared resident-function
selector, preserving input checks and the device-model cache. It requires the
new wrapper hook and binding function and must match the other grove arms bit
for bit. The list-returning resident reference explicitly selects
`forest_predict_resident_gpu`, even if the public default later changes. This
keeps a future A/B from accidentally timing the same ABI twice.

## Exact-size device workspace comparison

`--reuse-io` adds `parallel_groves_reuse`, selecting the shared
`forest_predict_resident_reuse_gpu` entrypoint. It also includes
`parallel_groves_borrowed` explicitly, so the baseline still allocates each call
if public defaults change later. All arms use the same fitted forest and fixed
reduction graph, with full output hashes checked after timing. A retained pair
is allocated on the first nonempty reuse call and replaced when row count
changes. Byte counts include calculated workspace residency; collect telemetry
for peak memory. The retained pair stays alive during subsequent baseline arms,
so this comparison does not measure each arm's isolated peak memory footprint.

The September 10 follow-up predeclares eight single-call samples per arm and
one eight-calls-per-sample companion, covering RF/HIGGS and ET/HIGGS, Year,
Covtype at 100 trees / depth 16. HIGGS uses 1M training and the original 500k
held-out rows. Only RF includes independently trained cuML GPU inference.
Use the existing spread gate per arm and distinguish a stable pair from an
unstable complete grid. No timing retries are scheduled. Small native/public
checks establish candidate reach and correctness before large-data timing.

## Commands

Run each command under the existing build and benchmark locks, with no compiler
or competing GPU work. Start with a small smoke to verify the artifact, then
run the large real-data cell. Use a fresh output path for every attempt.

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda \
PYTHONPATH=python python bench/speed/forest_inference_ab.py \
  --lane rf --dataset higgs --rows 1000000 --rounds 5 --staged-rounds 2 \
  --cuml-context --output /path/to/new-results/rf-higgs.json

MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SPEED_EXPECTED_VENDOR=metal \
PYTHONPATH=python python bench/speed/forest_inference_ab.py \
  --lane et --dataset year --rows 500000 --rounds 5 --staged-rounds 2 \
  --output /path/to/new-results/et-year.json
```

Change `--lane` to cover both forests and `--dataset covtype` for multiclass.
Explicit `--trees`/`--depth` overrides are recorded diagnostic configurations,
not the shipped workload. No cuML arm is offered for ET or Metal. CUDA RF's
optional cuML context fits a different forest and uses its cached GPU inference
model; this is not a same-model algorithm comparison. The pinned cuML
`265b9da6` `randomforest_common.pyx:665–693` initializes/caches nvForest, and
`randomforestclassifier.py:396–409` converts public inputs to device arrays.
The upstream nvForest pin and dispatch are documented in
[the source audit](GPU_FOREST_INFERENCE_NEXT.md).

## Prior evidence and rental setup

The [September 10 H100 comparison](../../bench/results/tree_tuning_h100_2026-09-10/README.md)
used HIGGS 1M, 500k prediction rows and 100 depth-16 trees. The parallel-groves
arm failed timing stability. Preserve its raw evidence; it is not a qualified
speed baseline or proof of an identity overhead. New resident-cache and kernel
work requires fresh source/binary hashes and public timings.

A dedicated session must use its own newly created pod and an on-pod API
teardown watchdog (`tools/runpod_guard.sh arm`, then `check`) before work.
Do not use or alter the protected training pod. Refuse an unarmed session and
terminate only the newly owned pod if setup fails. A 60-minute H100 session at
the September 10 observed $3.49/hour is about $3.49 compute, excluding storage;
check the live offer before creation. Parent orchestration owns budgets and
teardown; this benchmark never rents or extends a session.

Set `GBM_BENCH_DATA` to the prepared cache root. HIGGS uses
`higgs/higgs_speed.npz` and Year `year/year_speed.npz`; existing local files are
about 1.276GB and 188MB respectively. Transfer verified caches before the run
instead of re-downloading HIGGS's 2.6GB gzip. Covtype uses sklearn's dataset
cache and should be prepared separately. Record cache/data hashes. A fresh pod
must install the pinned project toolchain and build its own CUDA bindings;
Metal binaries cannot be reused. cuML is optional and its installation time
counts against the lease. Preserve setup/build failures, telemetry and outputs,
then terminate via the API and verify absence.

## Local FAST Metal evidence, September 10

The [RF HIGGS public run](../../bench/results/forest_groves_2026-09-10/metal/rf-higgs-resident-borrowed.json)
uses 1M training rows, the original 500k held-out rows, 100 depth-16 trees and
six measured calls per arm under both locks. Compiled vector dispatch was true.
All three grove arms and the separately instrumented calls matched output bits.
Medians were 589.89ms transient, 550.73ms resident list and 512.08ms resident
borrowed buffers. Spreads were 1.128, 1.180 and 1.076 respectively, so the full
comparison failed the declared stability gate. These are exploratory timings,
not an accepted speed ratio or a basis for a default change.

The separate staged calls spent almost all measured time inside shared
dispatch. That boundary includes native validation and transfers as well as
traversal, so this does not isolate a kernel bottleneck. The harness itself
passed host-mocked RF/ET scalar/vector execution, restoration and ABI-forwarding
checks; these mocks establish no GPU correctness or performance claim.

The [ET Year scalar companion](../../bench/results/forest_groves_2026-09-10/metal/et-year-resident-borrowed.json)
uses the shared loader's 463,811/51,534 train/test split with 90 features,
100 depth-16 trees and six measured calls. All grove paths and staged calls
matched bits; compiled vector dispatch was false for scalar leaves. Medians
were 76.02/43.82/42.46ms for transient/resident-list/borrowed, with spreads
1.183/1.021/1.010. The transient arm failed stability, so the complete comparison
is exploratory. This companion complements the HIGGS workload; it is not broad
large-data qualification. The initial cache-hit failure from an unnecessary
pandas import is retained; moving that import into the zip-decode branch
preserved the cached arrays and split.
