# Tree IDENTICAL source implementation, 2026-10-06

**NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.**

This lane covers only T01–T45, their sub-arms and listed interactions, and the
tree cards C45–C51. Every new candidate is opt-in and OFF in an ordinary build.
The untracked ideas document is preserved. No compilation, candidate import,
test, lint, manifest validation, identity/quality check, measurement, remote job,
board update, merge or push is part of this source-only task.

Per-ID source records live in `forest/`, `boosting/`, and `inference/`. They link
the new switches to production callers and state the supported scope and gaps.
Source integration means a caller invokes the selected behavior in source. It
does not mean that a compiler accepted the code or a runtime reached the branch.
All runtime reach and all vendor qualification remain unproven.

[NEW experiment inventory](NEW_TREE_AB_INVENTORY.md) lists every named arm and
interaction. [OLD / HISTORICAL inventory](HISTORICAL_TREE_AB_INVENTORY.md) is a
separate reference to prior tree work, including retired/rejected controls.
Both have JSON companions. [The final source handoff](HANDOFF.md) links the
production integration and remaining prerequisites.

## Discovery, controls and interactions

[`tools/trees_identical_ideas.py`](../../tools/trees_identical_ideas.py) reads the
per-ID records without importing estimator code. The existing
[`tools/trees_identical_ab.sh`](../../tools/trees_identical_ab.sh) dispatches
`ideas-list`, `ideas-show`, `ideas-plan` and `ideas-matrix` to it before any pod
setup or hardware discovery. These entry points were **written, not run**.
The explicit future `ideas-run` command invokes an already-frozen public driver
and retains its complete output beside the result as `.log` and `.driver.json`;
it never builds or submits a remote job. It was also **not run**.

`Txx:subarm` selects one named sub-arm; comma-separated selections compose
independent candidates. An optional `@A` or `@B` suffix selects each component's
arm explicitly, including references that require a define (for example
`T31@B,T35@A`). `X01` through `X10` name the ten listed tree interactions.
`XC45_51` is the full classical tree interaction, sharing the same implementations.
The matrix command emits every off/on subset, including complete A and incumbent
B. Conflicting defines refuse rather than silently taking the last value.
Matrix cells keep an explicit A/B assignment for every member, so omitting an
enabled member cannot accidentally restore a different historical default.
X05 uses T21's grouped two-wave replica sub-arm; the four-wave alternative is
selectable explicitly. X06 includes all three compatible T28 controls; PairLogit
and YetiRank use their own T29 sub-arm and saved recipe.
Each source configuration needs its own frozen/attested binding set; settings
and binaries must never change inside a scored operation.
The planner emits the affected GPU and host builder argv/environment pairs,
including each builder's actual define input. No builder was invoked.

A is the candidate. B is the explicitly recorded reference; where the idea
deliberately reopens a historical layout comparison, the record names it. An
ordinary build retains incumbent defaults. Reusing an already-selected incumbent
path can make an A/B cell coincide; those cells are not a new optimization win.

## Full-operation harness

The existing public driver
[`bench/speed/forest_speed_arm.py`](../../bench/speed/forest_speed_arm.py) accepts
`--trees-experiment`, `--trees-arm`, `--trees-recipe`, `--trees-artifact`,
`--trees-recipe-facts` and `--trees-result`. It dispatches to
[`tools/trees_identical_workload.py`](../../tools/trees_identical_workload.py).
The new route preserves the existing lane settings and seeds, uses no opponents,
and requires one excluded warmup and one scored sample on the full saved recipe.
It has not been executed.

The whole-operation span includes public input buffer preparation, model
construction, fit, mandatory synchronization, and requested output consumption.
The same sample separately records preparation, fit, cold inference (including
explanation preparation when requested), and repeated inference using the fitted
model. It retains the previous output after reuse. File decoding and task fixture
construction are the explicit CPU-only input step before estimator runtime.
The historical fit-only board timing path retains its own meaning.

[`recipes.json`](recipes.json) maps RF, ET, GB, OT, IF and AUX to saved dataset
loaders and lane settings. It records missing weighted/multioutput, auxiliary,
and other full-data recipe facts explicitly. In particular, the saved nonranking
Istella cache truncates its test split; an absent `--rows` argument does not prove
full original-dataset coverage. No smaller substitute closes a missing cell.

Future execution requires dataset byte hashes, split hashes, actual train/test
dimensions, classes, exact estimator settings and seed, intrinsic cap inventory,
full-coverage basis, frozen source/compiler/binary/defines and hardware provenance.
The facts must match the loaded data. The artifact lists absolute native library
paths/hashes and repository-relative source/harness hashes; the worker checks its
actual loaded extensions. A compiled define is not proof that a conditional
candidate branch ran. Result receipts keep reach, identity, quality and board
admission pending, preserve failures, and do not invent opponent ratios.

Future GPU invocations also require `MOJOLEARN_SPEED_EXPECTED_VENDOR` to identify
the frozen worker (`cuda`, `hip`, or `metal`), as required by the existing driver.
No vendor is inferred from a controller. An empty `defines` list is a valid
incumbent B artifact. RF importance recipes carry the explicit saved
`fit_options: {"feature_importances": true}` request; combined OOB/importance
recipes consume both outputs from one fit. ExtraTrees importance recipes remain
source-blocked. Dedicated DecisionTree/DART/AdaBoost extensions require retained
full constructor settings and seeds; no reduced fixture or guessed constructor
is substituted. Single-tree RF recipes require the explicit best-split route.

Host and Apple must implement/match each numerical version. NVIDIA and AMD alone
vote on later IDENTICAL performance decisions. T14/T26/T34 interaction X10 also
requires accepted individual host/all-column numerical evidence before execution.

## Classical overlap map

| Card | Shared tree ideas and scope |
| --- | --- |
| C45 | T05/T07 forest histogram scoring/selection; C45_GBDT fuses bounded symmetric and non-symmetric histogram prefixes with score selection |
| C46 | T03 exact sibling histograms; no floating subtraction claim |
| C47 | T02 forest tasks; T16/T17 and C47_GBDT boosting frontier; C47_IF live extrema tasks |
| C48 | Distinct stable-partition control for forests; T19 boosting and C48_IF partition/bookkeeping |
| C49 | T12/T13 bounded forest tasks/shared representations; T41 IF tasks |
| C50 | T31/T33/T35 exact inference layout/traversal, C50_IF packed nodes, and C50_GB packed resident GB plans; T34 is a separate version-changing experiment |
| C51 | T44 exact SHAP structural reuse/batching; T45 has a separate numerical contract |

These are attribution links, not claims that every estimator named on a broad C
card is covered by every linked T arm. The per-ID gaps are authoritative. There
is one shared implementation per overlapping behavior; no classical-worktree
files are changed.
