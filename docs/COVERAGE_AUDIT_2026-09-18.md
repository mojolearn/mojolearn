# Coverage audit, 2026-09-18

**No: implementation, CPU/GPU identity, native sabotage, installed verification,
publication and physical multi-GPU qualification are not all complete.**
This audit reads main at `77f3a6077`, the frozen release handoff, the separate CV
branch, retained records, actual wheel ZIPs and live PyPI/GitHub status. It does
not run new numerical qualification. Counts below describe different scopes.

## Current source inventory

| Scope | Count | Meaning |
| --- | ---: | --- |
| Registered harness routes | 236 | Variants/properties/drivers, not unique algorithms |
| Default public CPU verification routes | 163 | Selectable and reference-backed; not a current-wheel four-vendor certificate |
| Ordinary CPU routes withheld | 23 | Implemented, but not admitted to default verification |
| Parallel routes excluded from default CPU verification | 50 | GPU distribution requires its own hardware evidence |
| Parallel routes with a logical CPU replay | 18 | Does not prove physical GPU distribution |
| Parallel routes requiring GPU execution | 32 | A distributed CPU implementation is not required by the plan |
| Ordinary routes lacking a declared CPU verifier implementation | 0 | Within this registered inventory only |
| Historical appendix entries | 246 | Old catalog of algorithm/variant/API labels, not the harness denominator |
| Current harness routes absent from that appendix mapping | 26 | Catalog maintenance is owed |

> **DERIVED SINCE 2026-09-19.** That 26 was counted by hand and has already
> moved: `python3 tools/appendix_delta.py` reads `identity_break.LANES` by
> import and reports **31** today, the five additions being `linalg-qr`,
> `linalg-eigh`, `linalg-svdvals`, `par-ivf` and `language-model-config`.
> The companion `appendix-delta.csv` carries a REASON per lane, because
> most are weight-format or kernel variants of an algorithm the appendix
> already maps, and one is a configuration object -- counting them as new
> algorithms is the arithmetic the paragraph below refuses. **246 stays
> frozen**: it records what was published, and a row added to it rewrites
> a published claim.

The 246-entry catalog maps 244 entries to harness routes. HostForest and HostGBDT
are the other two: representative bundled saved models and dedicated gates
cover them. They are not two unimplemented algorithms.

> **CHECKED AND PART-CORRECTED SINCE 2026-09-19** (lane/laneless-public-classes).
> The gate is real: `tools/forest_host_gate.py check` over the eight Apple
> fixtures re-ran on the M4 at this commit and read IDENTICAL on every
> `predict` and `predict_proba` SHA-256, and 24 fixture directories across
> three vendors are committed under `bench/results/forest_host/`. TWO THINGS
> THE SENTENCE ABOVE COVERED THAT THE GATE DOES NOT. (1) The gate calls
> `host_model`; `host_predict` and `host_predict_proba`, which the package
> exports, are called by no gate and no lane. (2) The gate REFUSES a
> `parallel_groves` archive by name ("the host engine is sequential"), which
> stopped being true on lane/forest-groves-cpu-and-speed (2026-09-17) when
> `core/forest_host_groves.mojo` and `forest_host_groves_prepare/predict/
> release` landed, so the groves HOST engine had no recorded evidence of any
> kind. The `saved-model-host-infer` lane of `tools/identity_break.py` now
> runs all three and is watched failing under the forest family's own
> define, 9 of 9 fixtures
> (`bench/results/identity_break/2026-09-19_laneless-public-classes/`).
> The gate's own groves refusal is still owed a fix; it is a gate bug, not
> an engine bug. Nor can we add 26 to 246
and claim 272 unique algorithms: the inventories overlap and count different
things. The checked-in audit inventory lists all 26 additions, all 236 routes,
per-property declarations and reference-class counts.

All 163 public routes have reference entries for the ordinary train/infer/model
parts, including explicit N/A where applicable. This does **not** imply that all
have current agreeing GPU columns. On numerical training entries alone, 41
public routes per GPU class lack that class on at least one applicable fixture
(the three sets differ). Some are intentionally host-only utilities; others
need refreshed independent GPU evidence. The JSON records the exact sets via
`reference_support.train.agreeing_device_classes`. CPU numerical training
references are complete for the public set. A CPU-produced expected answer is
not itself CPU-versus-GPU proof.

## The 23 ordinary holds

| Reason | Routes |
| --- | --- |
| Only one device class supports the admitted reference | `mamba3`, `transformer`, `transformer-window`, `samba`, `samba-untied-dropout-accum` |
| Qualification pending | `gbdt-query-rmse`, `gmm-sample`, `gmm-random-init-sample`, `gp-normalize-y`, `gp-sample-y`, `gp-sample-y-normalize`, `gp-optimize`, `gp-optimize-restarts`, `gpc`, `gpc-multiclass`, `ivf-extend`, `svc-poly` |
| No admitted reference | `kernel-ridge-poly`, `kernel-ridge-sigmoid`, `kernel-ridge-laplacian`, `nystroem-poly`, `nystroem-sigmoid`, `nystroem-laplacian` |

The six kernel variants now have CPU implementations and targeted CPU/Apple
evidence. Independent GPU saved-model recordings remain explicitly owed in
`SAVED_MODEL_INFERENCE_OWED`; they are not in the frozen 0.8.7 native artifact.

“Reference qualification” means admitting matching input/revision/protocol and
repeat witnesses with the independent hardware/property evidence required for
that route, then validating installed replay. `--include-pending` exposes a
diagnostic run; it does not turn missing evidence into certification or promote
the route. See [the completion plan](CPU_IDENTITY_AND_MULTI_GPU_PLAN.md).

## Correctness gaps beyond the route count

The latest [Apple handoff](lanes/HANDOFF_2026-09-18_evening.md) and
[seam evidence](lanes/LANE_STATUS_apple-seam-repair.md) explicitly reject a claim
of universal Apple identity. GEMM and kNN were repaired, but the static audit
identifies approximately:

- **390 other device FMA sites** with the same round/flush boundary exposure,
  across Mamba, optimizer/training, forecasting, transformer, GP and other families.
- **20 no-flush FMA sites** where Apple's subnormal behavior differs. These need
  a deliberate numerical-contract choice, followed by implementation and reference
  qualification; flushing all vendors would move NVIDIA/AMD reference bits.
- **230 plain multiply/divide sites** whose exposure is untested. These are
  potential sites, not 230 demonstrated failures; a small probe should precede
  broad repair work.
- A demonstrated overflow fixture with **different NaN payloads on all three
  vendors**.

These are approximate call-site counts, not additional algorithms or failed
verifier routes. Existing fixtures can pass while missing these boundary inputs.
Boundary checks remain necessary even when ordinary fixtures become smaller.
Apple attention replay is also inert under its default schedule; changing that
schedule needs qualification. The repaired GEMM path adds roughly 16–23% to the
measured Apple LM step; performance work remains separate from correctness.

Whole loaded CausalLM composition still needs a complete verifier covering
checkpoint/tensor mapping, prefill, stateful decode, reset, logits, low-bit
variants and reload. Block-level identity does not establish that composition.
The handoff also flags an unresolved device-context lifetime/deadlock lead
(about 127 similar release sites, not a confirmed census) and an AMD zdot corner
without replay. Those must not be silently labeled covered by route totals.

## Sabotage and other properties

I regenerated the historical evidence scan read-only against current main.
The bundled `_verification_evidence_data.py` is stale relative to the harness.

| Observed qualifying historical native build control | Routes |
| --- | ---: |
| Default ordinary CPU set | 163 / 163 |
| Withheld ordinary set | 23 / 23 |
| Parallel set | 14 / 50 |

These counts mean **at least one numerical part was observed to move** under a
native control, with the generator's matching-context checks. They do not mean
every fixture, every property, every vendor or the current wheel has been
controlled. A refusal is not counted as moved arithmetic. The audit JSON lists
observed parts per route; the summary names all 36 parallel routes for which no
qualifying native pair was found. Missing indexed evidence is not proof no test
exists elsewhere. The normal user's `--self-test` perturbs input and validates
the comparator; it is distinct from a fault injected into compiled arithmetic.

Batch invariance is present. Among the default 163 routes:

| Numerical property declaration | Applicable routes | Selection |
| --- | ---: | --- |
| Batch invariance | 145 | Default full lane checks |
| Full sequence versus incremental steps | 11 | Default full lane checks |
| Gradient batch behavior | 11 | `--batch-checks` |
| Batch-size scaling | 24 | `--batch-checks` |
| Ragged sequence behavior | 15 | `--batch-checks` |
| Sampler/trainer replay pair | 15 | `--batch-checks` |

Every applicable declaration above has all nine reference fixtures in the
current table. N/A cases are separate. Reference presence is not universal
cross-vendor agreement or a fresh execution. `--all` alone does not select all
the optional properties. Save/reload, held-out inference and repeated training
outputs are also ordinary lane checks.

## CPU inference and verifier cost

Public CPU saved-model inference is substantially broader than public CPU
training. Private CPU verifier fit paths do not promise general public `.fit()`
support or production CPU speed. The six kernel saved-model recording debts and
whole-model CausalLM proof remain as above. `--inference` on main is an alias for
the representative bundled-model suite, not every inference API/input.

Main now separates `--training` and `--inference`, uses conservative one-thread
defaults, and allows explicit CPU thread requests. Supported thread controls
are not a hard RAM/CPU quota. GPU resources are managed by device/workload
selection, not a request for “five GPU cores.”

The fast training profile is still a **three-route pilot**: Ridge, StandardScaler
and MinMaxScaler, 15 small/boundary cases, two repeats, 180 applicable comparisons.
Retained CPU replay/Apple comparisons match; Ridge native faults were detected.
Scaler-specific faults, NVIDIA/AMD, installed-wheel qualification, coverage of
the other routes and public profile admission are still owed. It is not a
replacement default. See [pilot evidence](../bench/results/small_training/2026-09-18-apple-m4/README.md).

Completed CI stages now upload artifacts, and local column shards have strict
resume support. Automatic restoration across workflow runs and stage skipping
are not implemented. A killed stage can still lose unuploaded work. Parallel
architecture scheduling is configured for three jobs with bounded per-job
shards, but an already-running old workflow keeps its old configuration.

## Multi-GPU: implementation is not complete qualification

There are 50 parallel routes spanning forests/boosting, clustering, Gram-based
linear algebra, solvers, neighbors, graphs, scaling, forecasting fit, neural
training and other families. Historical scan results: **39/50** have qualifying
same-context single-versus-requested-multiple-device comparisons; none of the
indexed paired parts differ. However, recorded device requests/binding digests
do not independently attest physical GPU execution or certify the current wheel.

The 11 without a qualifying pair in that scan are:
`par-forest-reg`, `par-forest-et-clf`, `par-boosting-clf`, `par-boosting-reg`,
`par-gram-ols`, `par-gram-pca`, `par-gram-tsvd`, `par-cd-elasticnet`,
`par-svm-svr`, `par-scaler-minmax`, `par-queries-nn`.

| Useful missing capability | Remaining work |
| --- | --- |
| Generic loaded-LM inference across GPUs | Model/layer and KV-state placement; capacity and exact decode proof |
| Distributed IVF storage/search | Sharded index residency and exact candidate merge; query splitting alone does not expand index capacity |
| GaussianProcessClassifier distribution | Independent classes for throughput; matrix partitioning for larger individual fits |
| ARIMA/Holt-Winters distributed prediction | Forecast scheduling across independent series; existing fit sharding is not this |
| Multi-GPU cross-validation | Implemented only on `lane/multigpu-cv`; 91 software tests passed, 10 optional skips; NVIDIA/AMD numerical runs and actual execution traces owed |
| General distributed GEMM | A general partitioned operation with measured transfer/capacity value; existing distributed Gram operations do not establish it |

The CV feature remains deliberately off main and out of the release wheel.
All new distributed claims need actual distinct-GPU execution, one/two/reversed
device comparisons, uneven partitions, transport/ownership controls and measured
residency before claiming increased capacity. Parallel CPU implementations of
the 32 GPU-only drivers are not a missing requirement.

## What PyPI actually exposes

Live [PyPI metadata](https://pypi.org/pypi/mojolearn/json) reports **0.8.5** as
latest and no 0.8.7 files. I downloaded both published wheels, verified their
SHA-256 values against PyPI, and inspected ZIP contents without executing them.

| Surface | Published 0.8.5, both wheels | Inspected local 0.8.7 candidates | Current main |
| --- | --- | --- | --- |
| Host native bindings | Byte-LM only | 32 bindings | Expanded host surface |
| Broad verifier/table and coverage inventory | Absent | Present | Present, with newer routes |
| `--batch-checks`, `--include-pending`, `--models-only` | Absent | Present | Present |
| Parallel Python modules | None | Seven modules | Present; CV still separate |
| `--training`, `--inference`, `--cpu-threads` | Absent | Absent | Present |
| New six kernel CPU variants | Not the new implementation | Not the new implementation | Implemented; qualification pending |

**Important CLI ambiguity:** 0.8.5 has `verify --all`, but its help says to list
every diverging stage of the pinned check. It is not the newer broad-algorithm
verifier. Repository documentation must not imply today's commands are available
to someone who simply installs the currently published version.

Inspected local artifacts:
Linux `linux-wheel-final-v2/mojolearn-0.8.7-py3-none-manylinux_2_35_x86_64.whl`
and the retained macOS 0.8.7 candidate under `mojolearn-evidence/next-wheel-coverage`.
Their contents are not equivalent to current main or later release-tool repairs.
Public 0.8.5 offers macOS ARM64 and Linux x86-64 wheels; an ARM64 Linux CPU CI
column is not itself an ARM64 Linux published wheel. No Windows wheel appears
in the inspected latest-release files.

## Release status and recommended closure order

[Run 35350125464](https://github.com/mojolearn/mojolearn/actions/runs/35350125464)
is for old source `aff968968`, not the repaired release branch. At audit time:
wheel build passed; ARM64 CPU job failed classical saved-model and native-fault
stages; Apple CPU job was cancelled after its old timeout and had a classical
gate failure; x86 CPU job remained running with its classical gate failed.
Alpha admission was skipped. This is not a passing release certificate.

The release handoff names nine stale UMAP saved-model expectations, fresh
NVIDIA/AMD capture and explicit supersession still owed, plus full CPU and
installed-artifact requalification after proof-tool repairs. Targeted fixes and
prepared orchestration are not replacement all-platform receipts. Existing
earlier GPU receipts belong to their exact artifact/tool scope.

Recommended order:

1. Resolve the numerical boundary contract and add tiny separating probes. Do
   not claim universal Apple identity based solely on existing fixture hashes.
2. Finish exact frozen-release blockers and qualify the exact bytes to publish.
   Keep newly added native features out of that frozen branch unless the release
   scope is explicitly reopened and its native artifacts rebuilt/requalified.
3. Close the 23 holds with independent current property witnesses and installed
   replay; complete whole-CausalLM inference proof and the six saved-model debts.
4. Refresh shipped evidence/catalog/docs, make the small verifier profile broad
   and independently qualified, and complete cross-run checkpoint restoration.
5. Qualify existing physical multi-GPU routes and their native/transport controls;
   then finish the six useful capabilities above in the order workloads justify.

One run cannot permanently certify future native binaries, fixture revisions or
wheel packaging. Cache and reuse evidence with matching provenance; rerun the
parts affected by changes and retain the older receipts.

## Reproducible audit evidence

[Machine-readable inventory](../bench/results/coverage_audit/2026-09-18/inventory.json)
contains all 236 routes, current references/properties and refreshed historical
control summaries. Adjacent JSON files record wheel contents/hashes and exact
missing parallel lists. The full raw scans and downloaded wheels are retained
outside Git at `/Users/andrewhendel/mojolearn-evidence/coverage-audit-2026-09-18`;
the committed inventory pins the raw JSON hashes. The refreshed historical scan
was generated with `tools/verification_evidence.py` without `--write`, preserving
the shipped snapshot. No full Apple sweep, rental, native rebuild or publication
was performed for this review.
