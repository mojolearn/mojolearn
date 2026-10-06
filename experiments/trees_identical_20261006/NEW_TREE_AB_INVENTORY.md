# NEW tree A/B experiment inventory

**NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.**

Branch: `ideas/trees-identical-20261006`. Revision: the Git commit containing this inventory; no frozen binary exists.

This source snapshot covers T01–T45, C45–C51 attribution, 52 scoped implementation records, 84 named controls and 11 listed interactions. Every new candidate is default OFF. A record or source caller is not demonstrated runtime reach. Full-dataset facts and every qualification result remain pending.

The separate [OLD / HISTORICAL inventory](HISTORICAL_TREE_AB_INVENTORY.md) is reference only and adds no old implementation tasks. The [JSON companion](NEW_TREE_AB_INVENTORY.json) contains all recipe IDs and full per-arm records.

## Selection and full-workload integration

`tools/trees_identical_ab.sh` exposes `ideas-list`, `ideas-show`, `ideas-plan`, `ideas-matrix` and future `ideas-run`. The underlying selector is [trees_identical_ideas.py](../../tools/trees_identical_ideas.py); the actual driver is [forest_speed_arm.py](../../bench/speed/forest_speed_arm.py), through [trees_identical_workload.py](../../tools/trees_identical_workload.py). All were authored, not executed.

`ID:subarm` names an arm; `@A` or `@B` independently selects its side. Comma-separated controls compose compatible candidates. The `default_subarm` field only chooses a representative when that candidate is explicitly selected; it does not enable a production default. Empty B definitions preserve the incumbent, while explicit B layout controls are retained where the idea specifies them.

[recipes.json](recipes.json) maps the affected full-data loaders/settings and output consumers. Whole-operation boundaries include preparation, constructor, fit, synchronization and consumed outputs; fit, cold and repeated inference are reported separately. Missing dataset hashes/versions, dimensions, saved constructors/settings, caps and output recipes stay pending. No reduced fixture qualifies a full workload.

## New cards and named sub-arms

### T01 — Stream exact histogram replicas

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T01.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T01:replicas-2-rows-128` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=2`<br>`MOJOLEARN_TREES_T01_ROWS=128` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-2-rows-256` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=2`<br>`MOJOLEARN_TREES_T01_ROWS=256` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-2-rows-512` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=2`<br>`MOJOLEARN_TREES_T01_ROWS=512` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-2-rows-1024` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=2`<br>`MOJOLEARN_TREES_T01_ROWS=1024` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-4-rows-128` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=4`<br>`MOJOLEARN_TREES_T01_ROWS=128` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-4-rows-256` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=4`<br>`MOJOLEARN_TREES_T01_ROWS=256` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-4-rows-512` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=4`<br>`MOJOLEARN_TREES_T01_ROWS=512` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-4-rows-1024` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=4`<br>`MOJOLEARN_TREES_T01_ROWS=1024` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-8-rows-128` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=8`<br>`MOJOLEARN_TREES_T01_ROWS=128` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-8-rows-256` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=8`<br>`MOJOLEARN_TREES_T01_ROWS=256` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-8-rows-512` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=8`<br>`MOJOLEARN_TREES_T01_ROWS=512` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T01:replicas-8-rows-1024` | `MOJOLEARN_TREES_T01=1`<br>`MOJOLEARN_TREES_T01_REPLICAS=8`<br>`MOJOLEARN_TREES_T01_ROWS=1024` | Omit this candidate’s defines (incumbent) | RF; DT |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo).

**Full-workload mapping:** 105 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C47.

**Remaining gaps, conditional scope and no-difference cases:**

- Replicated RF shared histograms retain existing representability/shared-memory fallbacks for unsupported weighted or wide-output accumulators. ET random-threshold statistics are not RF quantile-bin histograms.

### T02 — Cost based histogram task sizing

**Source status:** SOURCE_INTEGRATED. [Per-ID integration record](forest/T02.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T02:cost` | `MOJOLEARN_TREES_T02=1` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/tree_identical_ideas.mojo](../../ensemble/tree_identical_ideas.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/kernels/et_loop_kernels.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/kernels/et_loop_kernels.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C47.

**Remaining gaps, conditional scope and no-difference cases:**

- ET coarsens search-task row spans from sampled-feature accumulator footprint; partition uses its unchanged fixed geometry. RF changes histogram row descriptors. Neither policy is a measured winner.

### T03 — Sibling histogram subtraction

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T03.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RFClassifier; DTClassifier.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T03:retained` | `MOJOLEARN_TREES_T03=1` | Omit this candidate’s defines (incumbent) | RFClassifier; DTClassifier |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo](../../ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo).

**Full-workload mapping:** 4 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C46.

**Remaining gaps, conditional scope and no-difference cases:**

- Retained exact unweighted classifier parent histograms apply only when all features have matching quantile bins; sampled-feature intersections and weighted/regression sufficient statistics need an explicit compatible cache representation.
- ET draws new node thresholds, so parent threshold counts cannot be subtracted into child threshold counts; raw-value/threshold sufficient statistics would be a separate algorithm and are not falsely aliased.
- Dedicated best-split DecisionTree full-data constructor facts remain pending; inherited RF code does not establish a measured standalone-tree cell.

### T04 — Sparse histogram clearing

**Source status:** SOURCE_INTEGRATED. [Per-ID integration record](forest/T04.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T04:live` | `MOJOLEARN_TREES_T04=1` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- RF clears each consumed quantile-bin prefix including padded node slots. ET initializes the current active-node/report prefixes and initializes newly live prefixes before each use. Failure/refit/stale-capacity behavior remains unexecuted.

### T05 — Fuse small node histogram and split evaluation

**Source status:** SOURCE_INTEGRATED. [Per-ID integration record](forest/T05.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T05:fused` | `MOJOLEARN_TREES_T05=1` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:search_batch_enqueue](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:search_batch_regression_enqueue](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:small_node_raw_split_kernel](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:node_feature_range_kernel](../../extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:node_feature_range_tiled_kernel](../../extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C45.

**Remaining gaps, conditional scope and no-difference cases:**

- RF per-node/feature histogram must fit 2048 shared bin slots; every selected row is still visited. Large row counts may regress despite fitting scratch.
- ET A admits raw-value nodes with at most 128 rows, one row per lane in a portable 128-thread block, and at most the existing 32 supported classes. Larger nodes and range-only rescue surveys retain B. This bounds per-thread work and shared state; it is not a reduced dataset recipe.
- Existing optional ET quantile-code regression retains B because its snapped threshold contract is distinct from raw-value range fusion. Weighted ET objectives remain a pre-existing unsupported public/native boundary. Missing/NaN refusal, bootstrap multiplicities and estimator settings are unchanged.
- The common setup, empty-cell conversion and canonical feature reducer remain separate launches. Small-node range and statistic row scans are skipped by those incumbent passes; the fused block forms statistics in shared memory and writes its final candidate. No workload, identity, quality or timing execution establishes a benefit.

### T06 — Cache entropy count transforms

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T06.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RFClassifier; DTClassifier; ExtraTreesClassifier.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T06:cached-parent` | `MOJOLEARN_TREES_T06=1` | Omit this candidate’s defines (incumbent) | RFClassifier; DTClassifier; ExtraTreesClassifier |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/objectives.mojo](../../ensemble/decisiontree/batched_levelalgo/objectives.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/objectives.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/objectives.mojo).

**Full-workload mapping:** 6 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- RF caches parent class terms only for unweighted entropy with at most 64 classes. Weighted l+r Float32 rounding is candidate-dependent and retains B under the S contract. Larger class counts retain B.
- ET existing supported entropy class cap is 32. A single node thread scores feature candidates in order using one cached parent vector; occupancy/performance remain unmeasured. ET weighted objectives remain an existing unsupported public/native boundary.
- Dedicated best-split DecisionTree full-data constructor facts remain pending; inherited RF code does not establish a measured standalone-tree cell.

### T07 — Parallel canonical split selection

**Source status:** SOURCE_INTEGRATED. [Per-ID integration record](forest/T07.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T07:parallel` | `MOJOLEARN_TREES_T07=1` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C45.

**Remaining gaps, conditional scope and no-difference cases:**

- RF parallel merge consumes distinct-feature winners; within-feature equal-gain plateau order remains the incumbent scan. ET varies the existing canonical reduce block to 128 lanes and retains exact rational split keys. No split identity has been executed.

### T08 — Bit exact bin packing and access layout

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T08.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T08:u8-row` | `MOJOLEARN_TREES_T08=1`<br>`MOJOLEARN_TREES_T08_BITS=8`<br>`MOJOLEARN_TREES_T08_LAYOUT=1` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T08:u8-column` | `MOJOLEARN_TREES_T08=1`<br>`MOJOLEARN_TREES_T08_BITS=8`<br>`MOJOLEARN_TREES_T08_LAYOUT=2` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T08:u8-tile32` | `MOJOLEARN_TREES_T08=1`<br>`MOJOLEARN_TREES_T08_BITS=8`<br>`MOJOLEARN_TREES_T08_LAYOUT=3` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T08:u16-row` | `MOJOLEARN_TREES_T08=1`<br>`MOJOLEARN_TREES_T08_BITS=16`<br>`MOJOLEARN_TREES_T08_LAYOUT=1` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T08:u16-column` | `MOJOLEARN_TREES_T08=1`<br>`MOJOLEARN_TREES_T08_BITS=16`<br>`MOJOLEARN_TREES_T08_LAYOUT=2` | Omit this candidate’s defines (incumbent) | RF; DT |
| `T08:u16-tile32` | `MOJOLEARN_TREES_T08=1`<br>`MOJOLEARN_TREES_T08_BITS=16`<br>`MOJOLEARN_TREES_T08_LAYOUT=3` | Omit this candidate’s defines (incumbent) | RF; DT |

**Production callers:** [ensemble/randomforest.mojo](../../ensemble/randomforest.mojo); [ensemble/decisiontree/batched_levelalgo/dataset.mojo](../../ensemble/decisiontree/batched_levelalgo/dataset.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo).

**Full-workload mapping:** 105 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Runtime supports exact UInt8/UInt16 row-major, column-major and tile32 storage. Current estimator bin-count limit is unchanged (1024); UInt16 does not expand estimator settings.
- ET raw-value threshold comparisons are not representable by RF quantile IDs without a numerical algorithm change. Existing ET approximate binning is not enabled. RF NaN refusal remains unchanged, so no missing sentinel is invented.

### T09 — Bootstrap locality with unchanged random draws

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T09.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T09:sorted` | `MOJOLEARN_TREES_T09=1` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/randomforest.mojo](../../ensemble/randomforest.mojo); [ensemble/bootstrap_sort.mojo](../../ensemble/bootstrap_sort.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Only exact integer/fixed-point histogram consumers reorder samples. RF weighted drawing retains exact draw multiplicities; ET public weighted training remains unsupported.
- ET sort buffers are fit-stage owned and explicitly synchronized before release. Sorting overhead belongs in the whole fit. Weighted regression public host coverage is pre-existing unsupported.

### T10 — Fuse bootstrap sampling and gather

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T10.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T10:fused` | `MOJOLEARN_TREES_T10=1` | Omit this candidate’s defines (incumbent) | RF; DT |

**Production callers:** [ensemble/randomforest.mojo](../../ensemble/randomforest.mojo).

**Full-workload mapping:** 105 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Compatible unweighted bootstrap only; T09 disables fused gather after sorting. NVIDIA incumbent already fuses and A has no additional change there. ET has no sampled-label staging to fuse.

### T11 — Keep tree frontiers and status on device

**Source status:** SOURCE_INTEGRATED. [Per-ID integration record](forest/T11.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T11:levels-2` | `MOJOLEARN_TREES_T11=1`<br>`MOJOLEARN_TREES_T11_LEVELS=2` | Omit this candidate’s defines (incumbent) | RF; DT; ET |
| `T11:levels-8` | `MOJOLEARN_TREES_T11=1`<br>`MOJOLEARN_TREES_T11_LEVELS=8` | Omit this candidate’s defines (incumbent) | RF; DT; ET |
| `T11:levels-16` | `MOJOLEARN_TREES_T11=1`<br>`MOJOLEARN_TREES_T11_LEVELS=16` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C48.

**Remaining gaps, conditional scope and no-difference cases:**

- RF and ET incumbent routes already use a device frontier. A changes the bounded ordinary-launch level sequence length to 2, 8 or 16. Each source branch retains capacity/status propagation; no device execution has demonstrated reach.

### T12 — Multiple trees sharing a frontier workspace

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T12.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T12:bounded` | `MOJOLEARN_TREES_T12=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [ensemble/randomforest.mojo](../../ensemble/randomforest.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C49.

**Remaining gaps, conditional scope and no-difference cases:**

- ET A adjusts its actual merged-tree frontier group from row and queue bytes. RF A schedules independent builders from an explicit histogram/row byte estimate, so RF still uses separate Builder arenas rather than one packed frontier.
- The estimate bounds the accounted arenas, not every allocation or measured peak memory. No new unsupported streams or device launch mechanisms are used.

### T13 — Cache exact training representations within one fit

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T13.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T13:capacity` | `MOJOLEARN_TREES_T13=1` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/randomforest.mojo](../../ensemble/randomforest.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C49.

**Remaining gaps, conditional scope and no-difference cases:**

- RF A reserves fit-owned node/range/leaf capacity with a 16MiB default cap; ET A reserves its group-owned node/queue capacity with the same byte budget. This implements reusable capacity, not a new persistent cache across mutable fits.
- Bins/quantiles and slot staging are already shared by incumbent RF; ET raw threshold representation stays unchanged. Cold/repeated use and failed/refit lifetime remain unexecuted.

### T14 — Canonical balanced regression moments

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T14.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RFRegressor; DTRegressor; ExtraTreesRegressor.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T14:versioned` | `MOJOLEARN_TREES_T14=1` | Omit this candidate’s defines (incumbent) | RFRegressor; DTRegressor; ExtraTreesRegressor |
| `T14:exact` | `MOJOLEARN_TREES_T14=1`<br>`MOJOLEARN_TREES_T14_EXACT=1` | Omit this candidate’s defines (incumbent) | RFRegressor; DTRegressor; ExtraTreesRegressor |

**Production callers:** [ensemble/tree_moments.mojo](../../ensemble/tree_moments.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [ensemble/decisiontree/batched_levelalgo/objectives.mojo](../../ensemble/decisiontree/batched_levelalgo/objectives.mojo); [ensemble/host/rf_oracle.mojo](../../ensemble/host/rf_oracle.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo).

**Full-workload mapping:** 87 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- V1 leaf sums use fixed 128 logical row-ID partials, incumbent integer quantization, incumbent RF divide/ET reciprocal dequantization, portable binary64 pair tree and final explicit Float32 FTZ output on host and all devices. MSE uses the shared explicitly rounded balanced left/right impurity graph.
- RF deviance split objectives keep incumbent score formulas; their leaf output uses the shared V1 moments. ET keeps its exact rational split comparator while changing rounded MSE metric/min-impurity value consistently in host and GPU.
- Weighted RF regression public host fit and weighted ET objectives are existing unsupported boundaries. Their full-dataset routes cannot be claimed. Native multioutput contracts remain limited; wrapper multioutput recipes require explicit saved facts.

### T15 — Device OOB aggregation and feature importance

**Source status:** PARTIAL_SOURCE_INTEGRATION. [Per-ID integration record](forest/T15.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RFClassifier; RFRegressor.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T15:register` | `MOJOLEARN_TREES_T15=1` | Omit this candidate’s defines (incumbent) | RFClassifier; RFRegressor |

**Production callers:** [ensemble/randomforest.mojo](../../ensemble/randomforest.mojo); [ensemble/oob_device.mojo](../../ensemble/oob_device.mojo); [ensemble/importance_device.mojo](../../ensemble/importance_device.mojo); [ensemble/host/oob.mojo](../../ensemble/host/oob.mojo); [ensemble/host/importance.mojo](../../ensemble/host/importance.mojo); [ensemble/host/rf_oracle.mojo](../../ensemble/host/rf_oracle.mojo); [bindings/_mojolearn_rf.mojo](../../bindings/_mojolearn_rf.mojo); [bindings/_mojolearn_rf_host.mojo](../../bindings/_mojolearn_rf_host.mojo); [python/mojolearn/randomforest.py](../../python/mojolearn/randomforest.py); [xtrees/exact_oob_sum.mojo](../../xtrees/exact_oob_sum.mojo).

**Full-workload mapping:** 21 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- RF requested OOB and impurity importance are native fit outputs. A importance consumes each live finished device tree before its builder is reused; normalized per-tree vectors fold in tree order. B calls the incumbent Mojo host importance fold. Host oracle retains gain/count metadata only on a requested importance fit.
- RF OOB A is per-row/output register accumulation; B already retains bootstrap masks and device OOB model. The card does not falsely claim those incumbent dependencies as new.
- ET OOB needs retained bootstrap membership, canonical output fold and public auxiliary export. ET impurity importance needs fitted split-gain/count retention plus a defined estimator normalization contract. No ET auxiliary recipe is claimed.
- RF native regression is one output; wrapper multioutput OOB/importance and weighted regression are not silently covered. Zero-voter policy and force-finite score logic match existing native source but are unexecuted.

### T16 — Resident lossguide frontier

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T16.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-lossguide.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T16:T16` | `MOJOLEARN_TREES_T16=1` | Omit this candidate’s defines (incumbent) | Retain exact scored-node partition-stat snapshots using the existing resident-frontier implementation, including its bounded node arena. |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/doc_parallel_boosting.mojo](../../gbdt/methods/doc_parallel_boosting.mojo).

**Full-workload mapping:** 5 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C47.

**Remaining gaps, conditional scope and no-difference cases:**

- Existing I17 scope/timings remain historical; the new full-workload T16 selector is unmeasured.
- Host scheduling already holds frontier state; placement arm does not alter its graph.

### T17 — Exact best-first pending batch

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T17.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-lossguide.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T17:T17` | `MOJOLEARN_TREES_T17=1` | Omit this candidate’s defines (incumbent) | Select 64 pending exact-batch expansions instead of incumbent width 32; existing sequential priority replay, folded-node guard and signed addressing bound remain. |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/doc_parallel_boosting.mojo](../../gbdt/methods/doc_parallel_boosting.mojo).

**Full-workload mapping:** 5 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C47.

**Remaining gaps, conditional scope and no-difference cases:**

- Do not attribute incumbent LG_EXACT_ID to this new width arm.
- Host executes sequential equivalent priority semantics; new schedule identity has not been demonstrated.

### T18 — Search partition inheritance across matching permutations

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T18.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-depthwise; gbdt-lossguide; gbdt-categorical.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T18:T18` | `MOJOLEARN_TREES_T18=1` | Omit this candidate’s defines (incumbent) | Extend exact final partition handoff to the learning permutation in multi-permutation sequential and batched estimation; other permutations still rebuild. |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/doc_parallel_boosting.mojo](../../gbdt/methods/doc_parallel_boosting.mojo).

**Full-workload mapping:** 12 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- A/B coincide for already-inherited single-permutation fits.
- Folded-back lossguide leaves retain existing final_ready refusal because their concatenated row order differs.

### T19 — Stable fused partition and deferred copy

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T19.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-depthwise; gbdt-lossguide.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T19:T19` | `MOJOLEARN_TREES_T19=1` | Omit this candidate’s defines (incumbent) | New IDENTICAL route uses stable integer flags/count/scan/scatter and row-index-only storage; it skips FAST histogram-derived stats and preserves the canonical partition-stat reductions. Separate defer subarm reuses the existing exact copy-lifetime schedule. |
| `T19:T19_DEFER` | `MOJOLEARN_TREES_T19_DEFER=1` | Omit this candidate’s defines (incumbent) | Deferred-copy lifetime only; no partition arithmetic change |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/greedy_subsets_searcher/kernel/split_chain_fused.mojo](../../gbdt/methods/greedy_subsets_searcher/kernel/split_chain_fused.mojo).

**Full-workload mapping:** 10 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C48.

**Remaining gaps, conditional scope and no-difference cases:**

- Fused-chain arm is Depthwise only; Lossguide uses the deferred-copy subarm.
- Device arbitration/deferred level waits remain off because that path must also upload IDENTICAL winning-cell payloads safely.

### T20 — Retained parent histogram lifetime

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T20.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-depthwise; gbdt-lossguide.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T20:T20` | `MOJOLEARN_TREES_T20=1` | Omit this candidate’s defines (incumbent) | Enable existing deferred parent-copy/dirty-slot histogram schedule on NVIDIA/AMD IDENTICAL; exact sibling-subtraction implementation remains the incumbent. |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo).

**Full-workload mapping:** 10 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C46.

**Remaining gaps, conditional scope and no-difference cases:**

- Apple already takes this schedule: A/B may coincide and must be marked no-difference.
- No claim of newly implementing subtraction where it was already present.
- Additional symmetric/ordered missing histogram routes not changed by this control.

### T21 — Per-feature-group width and streamed exact histogram replicas

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T21.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-symmetric; gbdt-depthwise; gbdt-lossguide.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T21:T21` | `MOJOLEARN_TREES_T21=1` | Omit this candidate’s defines (incumbent) | Wire existing per-group representable one-byte widths into non-symmetric histogram launchers; preserve integer addends, borders, dither and split order. |
| `T21:T21_STREAM` | `MOJOLEARN_TREES_T21_STREAM=1` | Omit this candidate’s defines (incumbent) | 2 sequential waves of the original exact two-stat 8-bit histogram replicas. Logical replica IDs, stripes, dither keys, active counts and integer addends are unchanged; each replica executes once. Independent of per-group width. Narrow 5/6/7-bit ladder kernels and other histogram families stay incumbent. |
| `T21:T21_STREAM4` | `MOJOLEARN_TREES_T21_STREAM4=1` | Omit this candidate’s defines (incumbent) | 4 sequential waves of the original exact two-stat 8-bit histogram replicas. Logical replica IDs, stripes, dither keys, active counts and integer addends are unchanged; each replica executes once. Independent of per-group width. Narrow 5/6/7-bit ladder kernels and other histogram families stay incumbent. |
| `T21:T21_GROUP_STREAM2` | `MOJOLEARN_TREES_T21=1`<br>`MOJOLEARN_TREES_T21_STREAM=1` | Omit this candidate’s defines (incumbent) | Complete combined T21 grouping + exact replica streaming configuration. Grouping applies representable widths; streaming applies only resulting exact 8-bit two-stat groups. Other groups retain their existing exact kernels. |
| `T21:T21_GROUP_STREAM4` | `MOJOLEARN_TREES_T21=1`<br>`MOJOLEARN_TREES_T21_STREAM4=1` | Omit this candidate’s defines (incumbent) | Complete combined T21 grouping + exact replica streaming configuration. Grouping applies representable widths; streaming applies only resulting exact 8-bit two-stat groups. Other groups retain their existing exact kernels. |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:launch_hist2_8bit](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:launch_hist2_width_group](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo); [gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo](../../gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo).

**Full-workload mapping:** 15 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Existing symmetric width grouping remains the incumbent; the new streaming controls cover its reachable exact 8-bit launcher only.
- Streaming for the narrower 5/6/7-bit ladder, multi-stat, non-one-byte and ordered pointwise histogram families remains unimplemented.
- Wider-than-supported compressed bins remain pending; no lossy packing added.
- T21_STREAM4 takes precedence if both stream controls are explicitly selected; independent arms and interactions should select a single stream count.

### T22 — Document-keyed ordered storage

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T22.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-ordered; gbdt-categorical.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T22:T22` | `MOJOLEARN_TREES_T22=1` | Omit this candidate’s defines (incumbent) | Select fold-position compressed storage while passing original document IDs to dither; fold-index cache lives within the fit pool. |

**Production callers:** [gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo](../../gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo); [gbdt/methods/ordered_fast_switches.mojo](../../gbdt/methods/ordered_fast_switches.mojo); [gbdt/methods/ordered_boosting.mojo](../../gbdt/methods/ordered_boosting.mojo).

**Full-workload mapping:** 6 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Only multi-fold ordered search activates; plain search keeps incumbent.
- Host numerical function remains document-keyed and unchanged; cross-column identity unverified.

### T23 — Device one-step preparation fusion

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T23.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-symmetric; gbdt-symmetric-1000; gbdt-depthwise; gbdt-lossguide; gbdt-ordered; gbdt-categorical; gbdt-multiclass; gbdt-rank-yetirank; gbdt-rank-pairlogit.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T23:T23` | `MOJOLEARN_TREES_T23=1` | Omit this candidate’s defines (incumbent) | For exactly one requested step fuse three +0 workspace fills and the optional exact weight copy into one leaf pass. Reuse incumbent device walker/objective/regularization arithmetic. |

**Production callers:** [gbdt/methods/leaves_estimation/device_walker.mojo](../../gbdt/methods/leaves_estimation/device_walker.mojo); [gbdt/methods/doc_parallel_boosting.mojo](../../gbdt/methods/doc_parallel_boosting.mojo); [gbdt/methods/ordered_boosting.mojo](../../gbdt/methods/ordered_boosting.mojo).

**Full-workload mapping:** 30 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Single-dimensional device-walker routes only; blocked-Hessian multiclass keeps its current route.
- Incumbent device one-step solver was already enabled; candidate changes preparation, never requested iteration count.

### T24 — Device noise combine and scale fusion

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T24.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-ordered; gbdt-categorical.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T24:T24` | `MOJOLEARN_TREES_T24=1` | Omit this candidate’s defines (incumbent) | Combine exact existing lane reductions and soft-float64 noise/scale statements in one kernel; thread zero publishes and consumes the sums. |

**Production callers:** [gbdt/methods/ordered_boosting.mojo](../../gbdt/methods/ordered_boosting.mojo).

**Full-workload mapping:** 6 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Requires ordered score noise and the split-lane path; random_strength=0 has no candidate work.
- Incumbent non-symmetric device-scale routes are not new work.

### T25 — Apply-prefix leaf index reuse

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T25.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-ordered; gbdt-categorical.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T25:T25` | `MOJOLEARN_TREES_T25=1` | Omit this candidate’s defines (incumbent) | Grow cached permutation leaf coverage to each apply prefix while retaining estimation-prefix exclusion; completion reads permutation-order leaf IDs directly. |

**Production callers:** [gbdt/methods/ordered_boosting.mojo](../../gbdt/methods/ordered_boosting.mojo); [gbdt/methods/dynamic_boosting.mojo](../../gbdt/methods/dynamic_boosting.mojo).

**Full-workload mapping:** 6 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Batched ordered estimation only; nonbatch and other mappings use incumbent.
- Extra apply-prefix preparation is inside fit and may cost more than saved gathers.

### T26 — Canonical no-sort RMSE leaf reduction V1

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T26.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-depthwise; gbdt-lossguide.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T26:T26` | `MOJOLEARN_TREES_T26=1` | Omit this candidate’s defines (incumbent) | Original-row logical chunks of 256; row-order per-leaf partials then adjacent balanced merge, using shared software-rounded Float32/FTZ units. Host and GPU callers bypass partition/sort/gather and apply the same new-version leaf estimates. |

**Production callers:** [gbdt/methods/doc_parallel_boosting.mojo](../../gbdt/methods/doc_parallel_boosting.mojo); [gbdt/host/gbdt_oracle_depthwise.mojo](../../gbdt/host/gbdt_oracle_depthwise.mojo); [gbdt/methods/leaves_estimation/tree_t26_units.mojo](../../gbdt/methods/leaves_estimation/tree_t26_units.mojo); [gbdt/methods/leaves_estimation/tree_t26_device.mojo](../../gbdt/methods/leaves_estimation/tree_t26_device.mojo).

**Full-workload mapping:** 10 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Only unweighted RMSE, one permutation, one-dimensional Newton with exactly one requested estimation iteration.
- Ordered, symmetric, multiclass, weighted and multi-step contracts remain pending; those routes explicitly retain B.
- 64 MiB scratch budget streams leaf batches; a single leaf exceeding that budget is refused.
- Quality and new-version identity remain entirely unverified; no A-vs-B bit-equality requirement.

### T27 — Objective and row-state fusion

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T27.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-symmetric; gbdt-symmetric-1000; gbdt-depthwise; gbdt-lossguide; gbdt-ordered; gbdt-categorical; gbdt-multiclass; gbdt-rank-yetirank; gbdt-rank-pairlogit.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T27:T27` | `MOJOLEARN_TREES_T27=1` | Omit this candidate’s defines (incumbent) | Use the existing exact move-plus-objective bodies so cursor movement and gradient/Hessian/status production share a row pass. |

**Production callers:** [gbdt/methods/leaves_estimation/pointwise_oracle.mojo](../../gbdt/methods/leaves_estimation/pointwise_oracle.mojo); [gbdt/targets/kernel/pointwise_targets.mojo](../../gbdt/targets/kernel/pointwise_targets.mojo); [gbdt/methods/doc_parallel_boosting.mojo](../../gbdt/methods/doc_parallel_boosting.mojo); [gbdt/methods/ordered_boosting.mojo](../../gbdt/methods/ordered_boosting.mojo).

**Full-workload mapping:** 30 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C45.

**Remaining gaps, conditional scope and no-difference cases:**

- Single-dimensional pointwise losses only; grouped ranking/multiclass fusion needs a separate canonical graph.
- Reuses existing 2030 implementation as dependency without treating old experiment backlog as new tasks.

### T28 — CTR scratch, stable segment and compatible prefix-statistic reuse

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T28.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-categorical; gbdt-ordered.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T28:T28` | `MOJOLEARN_TREES_T28=1` | Omit this candidate’s defines (incumbent) | New IDENTICAL selector reuses fit-owned categorical prep/scratch; independent sort arm derives feature-frequency counts from the existing stable last-permutation category segments. |
| `T28:T28_SORT` | `MOJOLEARN_TREES_T28_SORT=1` | Omit this candidate’s defines (incumbent) | Stable last-permutation category segments reused for frequency counts |
| `T28:T28_PREFIX` | `MOJOLEARN_TREES_T28_PREFIX=1` | Omit this candidate’s defines (incumbent) | Retain exact exclusive-prefix denominators, flagged sample weights and gathered target across compatible simple-CTR config groups in the same stable category/permutation generation. Invalidate on every bins build, target/order upload and frequency pass; numerator and priors remain per config. Enables fit-owned prep as its required lifetime. No additional row-sized cache allocation. |
| `T28:T28_ALL` | `MOJOLEARN_TREES_T28=1`<br>`MOJOLEARN_TREES_T28_SORT=1`<br>`MOJOLEARN_TREES_T28_PREFIX=1` | Omit this candidate’s defines (incumbent) | Complete supported T28 simple-CTR configuration: fit scratch, frequency segment reuse and compatible exclusive-prefix sufficient statistics. |

**Production callers:** [gbdt/ctrs/fast_prep.mojo](../../gbdt/ctrs/fast_prep.mojo); [gbdt/train.mojo](../../gbdt/train.mojo); [gbdt/train.mojo:train](../../gbdt/train.mojo); [gbdt/ctrs/fast_prep.mojo:fast_dependent_ctrs](../../gbdt/ctrs/fast_prep.mojo); [gbdt/ctrs/fast_prep.mojo:CtrPrepFast.borders_ctrs](../../gbdt/ctrs/fast_prep.mojo).

**Full-workload mapping:** 6 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Public tree-tensor CTR training is refused at gbdt/options/catboost_options.mojo:TCatFeatureParams.check when max_tensor_complexity != 1; it requires tree_ctr_datasets_visitor integration. gbdt/train.mojo:TrainedModel.tensor_ctr_registry states public combination training does not yet populate the registry. An isolated cache in internal tensor helpers would not reach public training. This is an existing representation/caller prerequisite, not an implemented or claimed T28 cache.
- Simple-CTR outputs already live for the fit and are reused by all trees. New T28_PREFIX reuse saves rebuilds between compatible config groups in that prep; a single group has no prefix-reuse benefit.
- Inputs/permutation/targets belong to one fit; no cross-fit mutable-input cache. Every feature/permutation rebuild and target replacement invalidates prefix scratch.
- Strict ordered-prefix exclusions, target-border/bucket numerators and requested priors are retained.

### T29 — Canonical query scheduling and shared generated-PairLogit fold

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T29.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-rank-pairlogit; gbdt-rank-yetirank.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T29:T29` | `MOJOLEARN_TREES_T29=1` | Omit this candidate’s defines (incumbent) | PairLogit launches separate short/long query work classes at one shared tile, retaining logical group IDs and original per-query arithmetic; YetiRank subarm reuses exact block task schedule on Apple IDENTICAL. |
| `T29:T29_YETI` | `MOJOLEARN_TREES_T29_YETI=1` | Omit this candidate’s defines (incumbent) | Exact YetiRank block schedule; new on Apple IDENTICAL, existing on NVIDIA/AMD |
| `T29:T29_VERSIONED` | `MOJOLEARN_TREES_T29_VERSIONED=1` | Omit this candidate’s defines (incumbent) | Generated-pair PairLogit V1 shared host/all-GPU partner and group fold; search derivatives/curvature, estimation including search reuse, group magnitudes and final learn loss. GROUP_VERSIONED additionally selects the independent short/long launch classification. Explicit supplied pairs and other objectives keep the incumbent. |
| `T29:T29_GROUP_VERSIONED` | `MOJOLEARN_TREES_T29=1`<br>`MOJOLEARN_TREES_T29_VERSIONED=1` | Omit this candidate’s defines (incumbent) | Generated-pair PairLogit V1 shared host/all-GPU partner and group fold; search derivatives/curvature, estimation including search reuse, group magnitudes and final learn loss. GROUP_VERSIONED additionally selects the independent short/long launch classification. Explicit supplied pairs and other objectives keep the incumbent. |

**Production callers:** [gbdt/targets/kernel/pair_logit_group.mojo](../../gbdt/targets/kernel/pair_logit_group.mojo); [gbdt/targets/kernel/yeti_rank.mojo](../../gbdt/targets/kernel/yeti_rank.mojo); [gbdt/methods/doc_parallel_boosting.mojo](../../gbdt/methods/doc_parallel_boosting.mojo); [gbdt/targets/kernel/pair_logit.mojo:_launch_pair_logit_group_layout](../../gbdt/targets/kernel/pair_logit.mojo); [gbdt/targets/kernel/pair_logit_group.mojo:launch_pair_logit_group](../../gbdt/targets/kernel/pair_logit_group.mojo); [gbdt/targets/kernel/tree_t29_pair.mojo:pair_logit_group_versioned_kernel](../../gbdt/targets/kernel/tree_t29_pair.mojo); [gbdt/targets/tree_t29_units.mojo:t29_pair_row](../../gbdt/targets/tree_t29_units.mojo); [gbdt/host/gbdt_oracle_pair.mojo:_group_values_t29](../../gbdt/host/gbdt_oracle_pair.mojo); [gbdt/host/gbdt_oracle_pair.mojo:_group_search_pass](../../gbdt/host/gbdt_oracle_pair.mojo); [gbdt/host/gbdt_oracle_pair.mojo:pair_logit_eval](../../gbdt/host/gbdt_oracle_pair.mojo); [gbdt/host/gbdt_oracle_pair.mojo:pair_logit_value](../../gbdt/host/gbdt_oracle_pair.mojo); [gbdt/methods/leaves_estimation/pointwise_oracle.mojo:BinOptimizedOracle](../../gbdt/methods/leaves_estimation/pointwise_oracle.mojo).

**Full-workload mapping:** 2 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Explicit supplied PairLogit pairs keep incumbent path.
- PairLogit classes filter existing descriptors, not an independent compacted descriptor cache.
- YetiRank NVIDIA/AMD may already use the block schedule; mark no-difference rather than speedup.
- T29_VERSIONED requires the existing generated-pair group layout (IDN_PAIRLOGIT_GROUP). An explicit pair list or legacy group-layout OFF/master-OFF selects the incumbent instead; no generated pairs replace a supplied list.
- The host public ranking fitter supports its incumbent SymmetricTree/Cosine scope in gbdt/host/gbdt_oracle_losses.mojo. This candidate does not introduce a missing host ranking policy/score implementation.
- YetiRank keeps its existing S schedule: permutation sampling and position-derived pair weights form a separate graph in gbdt/targets/kernel/yeti_rank.mojo and gbdt/host/gbdt_oracle_yeti.mojo. The PairLogit V1 helper is not substituted for that objective.

### T30 — DART and tree AdaBoost state reuse

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/T30.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** dart; dart-reg; adaboost; adaboost-reg.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T30:T30` | `MOJOLEARN_TREES_T30=1` | Omit this candidate’s defines (incumbent) | DART stores exact unscaled per-tree row predictions in its fit session during the existing add pass; later dropout folds read them in original tree order. AdaBoost subarm caches exact SAMME miss flags/R2 loss values within a round on host and GPU. |
| `T30:T30_ADABOOST` | `MOJOLEARN_TREES_T30_ADABOOST=1` | Omit this candidate’s defines (incumbent) | DecisionTree-base SAMME/R2 round-state cache |

**Production callers:** [xtrees/dart_units.mojo](../../xtrees/dart_units.mojo); [xtrees/dart_device.mojo](../../xtrees/dart_device.mojo); [xtrees/dart_host.mojo](../../xtrees/dart_host.mojo); [xtrees/ops_device_boost.mojo](../../xtrees/ops_device_boost.mojo); [xtrees/ops.mojo](../../xtrees/ops.mojo); [xtrees/api.mojo](../../xtrees/api.mojo); [python/mojolearn/_expansion_trees.py](../../python/mojolearn/_expansion_trees.py).

**Full-workload mapping:** 10 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- AdaBoost cache explicitly selected only for public DecisionTreeClassifier/Regressor base objects; custom non-tree estimators use B.
- AdaBoost cross-round resident weight sessions remain pending; no pointer-only cache over mutable input.
- DART memory grows by 4*rows*classes*iterations bytes and is released at session close; complete fit boundary includes it.
- Saved full-data DART/AdaBoost constructor recipes must be resolved from retained settings, not invented.

### T31 — Pack resident inference nodes

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T31.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T31:T31` | `MOJOLEARN_TREES_T31_PACKED_A=1` | `MOJOLEARN_TREES_T31_PACKED_B=1` | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py).

**Full-workload mapping:** 201 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C50.

### T32 — Share input rows across tree walkers

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T32.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T32:T32` | `MOJOLEARN_TREES_T32_SHARED_ROWS=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py).

**Full-workload mapping:** 22 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Ordinary ordered RF/ET predictions without T35/T36 do not dispatch this standalone schedule.

### T33 — Map rows and tree groups by cost

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T33.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T33:T33` | `MOJOLEARN_TREES_T33_COST_SCHEDULE=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py).

**Full-workload mapping:** 22 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C50.

**Remaining gaps, conditional scope and no-difference cases:**

- Old dispatch is preserved only in B; required neighboring-shape/non-board A/B is not run.
- Ordinary ordered RF/ET predictions without T35/T36 do not dispatch this standalone schedule.

### T34 — Canonical versioned forest prediction reduction

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T34.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T34:T34` | `MOJOLEARN_TREES_T34_CHUNK_FOLD=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py); [core/forest_experiments.mojo:forest_chunk_sum](../../core/forest_experiments.mojo); [core/forest_host_predict.mojo](../../core/forest_host_predict.mojo); [core/forest_host_groves.mojo](../../core/forest_host_groves.mojo).

**Full-workload mapping:** 201 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Pooled multi-device reduction is refused under this candidate; IF scoring retains its separate existing contract.

### T35 — Traverse once for all outputs

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T35.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T35:T35` | `MOJOLEARN_TREES_T35_LEAF_REUSE=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py).

**Full-workload mapping:** 201 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C50.

**Remaining gaps, conditional scope and no-difference cases:**

- Scratch holds at least one complete row if one row exceeds the nominal byte budget.

### T36 — Fuse input finiteness and prediction preparation

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T36.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T36:T36` | `MOJOLEARN_TREES_T36_FINITE_STAGE=1` | Omit this candidate’s defines (incumbent) | RF; ET |
| `T36:finite-only-on-leaf-prep` | `MOJOLEARN_TREES_T35_LEAF_REUSE=1`<br>`MOJOLEARN_TREES_T36_FINITE_STAGE=1` | `MOJOLEARN_TREES_T35_LEAF_REUSE=1` | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py).

**Full-workload mapping:** 201 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** T35.

**Remaining gaps, conditional scope and no-difference cases:**

- Raw DMA upload cannot execute arithmetic; fusion is at device leaf preparation, not the host DMA. IF already fuses validation with FTZ/transpose and adds no separate T36 implementation.

### T37 — Reuse resident model and output workspaces

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T37.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T37:T37` | `MOJOLEARN_TREES_T37_WORKSPACE=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py).

**Full-workload mapping:** 201 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- No mutable X address cache. Larger/smaller incompatible demands release prior buffers.

### T38 — Fuse forest score normalization and classification

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T38.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T38:T38` | `MOJOLEARN_TREES_T38_FUSED_LABELS=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [core/forest_inference.mojo:launch_forest_inference](../../core/forest_inference.mojo); [core/forest_inference_model.mojo:ResidentForest.predict_into](../../core/forest_inference_model.mojo); [python/mojolearn/_forest_protocol.py](../../python/mojolearn/_forest_protocol.py); [bindings/forest_inference_binding.mojo:forest_identical_fused_labels_binding](../../bindings/forest_inference_binding.mojo).

**Full-workload mapping:** 24 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Selected candidate requires one device; host retains existing same-statement labels route.

### T39 — Share traversal for apply and staged outputs

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T39.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T39:T39` | `MOJOLEARN_TREES_T39_AUXILIARY_WALK=1` | Omit this candidate’s defines (incumbent) | RF; ET |

**Production callers:** [core/forest_auxiliary_units.mojo](../../core/forest_auxiliary_units.mojo); [core/forest_auxiliary_device.mojo](../../core/forest_auxiliary_device.mojo); [xtrees/api.mojo:forest_auxiliary_binding](../../xtrees/api.mojo); [python/mojolearn/_forest_protocol.py:ForestProtocol](../../python/mojolearn/_forest_protocol.py).

**Full-workload mapping:** 57 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Eager prefix volume is rows×trees×outputs and remains caller-owned; GB/DART prefix semantics are not claimed.

### T40 — Isolation forest row major preparation

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T40.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** IF.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T40:T40` | `MOJOLEARN_TREES_T40_ROWMAJOR=1` | Omit this candidate’s defines (incumbent) | IF |

**Production callers:** [isolation_forest/impl/isolation_forest.mojo:IsolationForest.fit](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/estimator.mojo](../../isolation_forest/estimator.mojo); [python/mojolearn/_iforest_impl.py](../../python/mojolearn/_iforest_impl.py); [isolation_forest/impl/isolation_tree_builder.mojo:IF_FAST_ROWMAJOR](../../isolation_forest/impl/isolation_tree_builder.mojo).

**Full-workload mapping:** 4 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

### T41 — Isolation forest bounded tree batching

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T41.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** IF.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T41:T41` | `MOJOLEARN_TREES_T41_TREE_BATCH=1` | Omit this candidate’s defines (incumbent) | IF |

**Production callers:** [isolation_forest/impl/isolation_forest.mojo:IsolationForest.fit](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/estimator.mojo](../../isolation_forest/estimator.mojo); [python/mojolearn/_iforest_impl.py](../../python/mojolearn/_iforest_impl.py).

**Full-workload mapping:** 4 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C49.

**Remaining gaps, conditional scope and no-difference cases:**

- Diagnostic trace keeps full scratch so its archive remains available. No unsupported free-memory query is used; one tree is minimum capacity.

### T42 — Cache isolation path corrections and score preparation

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T42.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** IF.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T42:T42` | `MOJOLEARN_TREES_T42_CORRECTION_CACHE=1` | Omit this candidate’s defines (incumbent) | IF |

**Production callers:** [isolation_forest/impl/isolation_forest.mojo:IsolationForest.fit](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/estimator.mojo](../../isolation_forest/estimator.mojo); [python/mojolearn/_iforest_impl.py](../../python/mojolearn/_iforest_impl.py); [isolation_forest/impl/isolation_tree_builder.mojo:if_correction_table_kernel](../../isolation_forest/impl/isolation_tree_builder.mojo).

**Full-workload mapping:** 4 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Host existing leaf corrections remain the same arithmetic; no new score approximation or numeric version.

### T43 — Deterministic isolation contamination selection

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T43.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** IF.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T43:T43` | `MOJOLEARN_TREES_T43_RADIX_SELECT=1` | Omit this candidate’s defines (incumbent) | IF |

**Production callers:** [isolation_forest/impl/isolation_forest.mojo:IsolationForest.fit](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/estimator.mojo](../../isolation_forest/estimator.mojo); [python/mojolearn/_iforest_impl.py](../../python/mojolearn/_iforest_impl.py); [isolation_forest/impl/isolation_forest.mojo:contamination_offset](../../isolation_forest/impl/isolation_forest.mojo).

**Full-workload mapping:** 3 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Explicit contamination required to reach offset selection; no-contamination auto path is inapplicable.

### T44 — Resident TreeSHAP metadata and path reuse

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T44.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** AUX.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T44:T44` | `MOJOLEARN_TREES_T44_METADATA_CACHE=1` | Omit this candidate’s defines (incumbent) | AUX |

**Production callers:** [xtrees/shap.mojo](../../xtrees/shap.mojo); [xtrees/shap_host.mojo](../../xtrees/shap_host.mojo); [xtrees/shap_device.mojo](../../xtrees/shap_device.mojo); [xtrees/api.mojo](../../xtrees/api.mojo); [python/mojolearn/_expansion_trees.py:TreeExplainer](../../python/mojolearn/_expansion_trees.py).

**Full-workload mapping:** 19 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C51.

**Remaining gaps, conditional scope and no-difference cases:**

- Supported explanation outputs remain standard exact SHAP; no interaction API exists in this source.

### T45 — Canonical TreeSHAP path unwinding

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/T45.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** AUX.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `T45:T45` | `MOJOLEARN_TREES_T45_LINEAR_UNWIND=1` | Omit this candidate’s defines (incumbent) | AUX |

**Production callers:** [xtrees/shap.mojo](../../xtrees/shap.mojo); [xtrees/shap_host.mojo](../../xtrees/shap_host.mojo); [xtrees/shap_device.mojo](../../xtrees/shap_device.mojo); [xtrees/api.mojo](../../xtrees/api.mojo); [python/mojolearn/_expansion_trees.py:TreeExplainer](../../python/mojolearn/_expansion_trees.py).

**Full-workload mapping:** 19 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Remaining gaps, conditional scope and no-difference cases:**

- Linear in one unwind; total per-leaf feature work still includes one unwind per feature. Additivity/explanation correctness remains unverified.

### C45_GBDT — Bounded histogram prefix, sibling derivation and split score fusion

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/C45_GBDT.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-symmetric; gbdt-symmetric-1000; gbdt-depthwise; gbdt-lossguide; gbdt-multiclass; gbdt-categorical; gbdt-ordered.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `C45_GBDT:C45_GBDT` | `MOJOLEARN_TREES_C45_GBDT=1` | Omit this candidate’s defines (incumbent) | Non-symmetric root in one 256-lane scorer, plus ordinary and synchronized symmetric root/descendant levels with incumbent one-block score grid (128 lanes, 1 KiB shared score/tie scratch) and at most 16 KiB live histogram cells. Symmetric prefix scan and optional sibling subtraction execute before the actual incumbent symmetric score/noise/tie kernel in the same block. Partition-stat fold and parent-cache prefixes are retained. Traced or wider levels retain B. |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:run_tree_layout_traced](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:TSynchronizedSymmetricLevelState.enqueue_pre_score](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo); [gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:TSynchronizedSymmetricLevelState.enqueue_score](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo); [gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo:fused_symmetric_scan_score_kernel](../../gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo); [gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo:fused_root_scan_score_kernel](../../gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo).

**Full-workload mapping:** 28 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C45; T05; T07.

**Remaining gaps, conditional scope and no-difference cases:**

- Non-symmetric descendant frontier fusion and wider root histograms are outside the bounded arm; symmetric levels beyond the existing one-block score grid or 16 KiB live working set retain the incumbent.
- Parent-cache prefixes remain required by sibling subtraction. The default symmetric incumbent already combines scan with partition-stat phase one, so a net launch-count or performance benefit is not asserted.
- Traced paths retain B to preserve intermediate stage capture; stage-time labels differ on the fused arm, so only the programmed whole-operation boundary can be used for candidate measurements.
- All source routes and resource eligibility remain unexecuted. Saved full-dataset mapping does not establish that any recipe reaches an eligible level; full recipe facts and coverage remain pending.
- No host GPU-training emulation or unsupported vendor compiler workaround is introduced; source is shared across supported GPU backends and the S arithmetic contract retains the host incumbent.

### C47_GBDT — Pending best-first work by live histogram bytes

**Source status:** source_integrated_uncompiled_scoped. [Per-ID integration record](boosting/C47_GBDT.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** gbdt-lossguide.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `C47_GBDT:C47_GBDT` | `MOJOLEARN_TREES_C47_GBDT=1` | Omit this candidate’s defines (incumbent) | Lossguide pending exact expansions bounded by an 8 MiB live-histogram working set; stable node IDs and exact priority replay unchanged. |

**Production callers:** [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo); [gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo](../../gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo).

**Full-workload mapping:** 5 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C47; T16; T17.

**Remaining gaps, conditional scope and no-difference cases:**

- Depthwise descriptor compaction and row-work buckets remain pending.
- Narrow frontiers may match incumbent capacity; report no-difference explicitly.

### C47_IF — IF extrema tasks by live node rows

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/C47_IF.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** IF.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `C47_IF:C47_IF` | `MOJOLEARN_TREES_C47_IF_LIVE_TASKS=1` | Omit this candidate’s defines (incumbent) | IF |

**Production callers:** [isolation_forest/impl/isolation_forest.mojo:IsolationForest.fit](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/estimator.mojo](../../isolation_forest/estimator.mojo); [python/mojolearn/_iforest_impl.py](../../python/mojolearn/_iforest_impl.py); [isolation_forest/impl/isolation_tree_builder.mojo:_block_min_max](../../isolation_forest/impl/isolation_tree_builder.mojo).

**Full-workload mapping:** 4 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C47.

### C48 — Stable tree partition fused with child bookkeeping

**Source status:** SOURCE_INTEGRATED. [Per-ID integration record](forest/C48.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** RF; DT; ET.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `C48:fused` | `MOJOLEARN_TREES_C48=1` | Omit this candidate’s defines (incumbent) | RF; DT; ET |

**Production callers:** [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo); [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo); [extratrees/impl/decisiontree/batched_levelalgo/builder.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/builder.mojo); [extratrees/impl/decisiontree/batched_levelalgo/kernels/partition_multiblock.mojo](../../extratrees/impl/decisiontree/batched_levelalgo/kernels/partition_multiblock.mojo).

**Full-workload mapping:** 159 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** T11; T19; T41.

**Remaining gaps, conditional scope and no-difference cases:**

- RF single-device global histogram child count equals local child count; one publish kernel replaces reset/count/publish before unchanged stable scan/scatter. Distributed/sharded local partition statistics are not claimed.
- ET scatter computes integer per-block left prefixes directly from retained count blocks, eliminating the separate scan launch. Stable row order, seeds and leaf ranges remain source-preserved but unexecuted.

### C48_IF — IF stable partition and child split publication

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/C48_IF.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** IF.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `C48_IF:C48_IF` | `MOJOLEARN_TREES_C48_IF_PARTITION=1` | Omit this candidate’s defines (incumbent) | IF |

**Production callers:** [isolation_forest/impl/isolation_forest.mojo:IsolationForest.fit](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/estimator.mojo](../../isolation_forest/estimator.mojo); [python/mojolearn/_iforest_impl.py](../../python/mojolearn/_iforest_impl.py); [isolation_forest/impl/isolation_tree_builder.mojo:_block_stable_partition](../../isolation_forest/impl/isolation_tree_builder.mojo).

**Full-workload mapping:** 4 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C48.

### C50_GB — GB exact packed and bounded inference

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/C50_GB.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** GB.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `C50_GB:C50_GB` | `MOJOLEARN_TREES_C50_GB_PACKED=1` | Omit this candidate’s defines (incumbent) | GB |

**Production callers:** [gbdt/resident_model.mojo:_apply](../../gbdt/resident_model.mojo); [gbdt/models/add_non_symmetric_tree_doc_parallel.mojo:add_non_symmetric_trees_packed](../../gbdt/models/add_non_symmetric_tree_doc_parallel.mojo); [core/gbdt_host_predict.mojo:gbdt_host_predict](../../core/gbdt_host_predict.mojo); [bindings/_mojolearn_gbdt.mojo:gbdt_resident_predict_binding](../../bindings/_mojolearn_gbdt.mojo); [gbdt/resident_model.mojo:ResidentGbdtModel._apply](../../gbdt/resident_model.mojo).

**Full-workload mapping:** 30 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C50.

**Remaining gaps, conditional scope and no-difference cases:**

- Standalone non-resident oblivious legacy prediction does not use the resident model arm; public resident prediction and host archive inference are wired.
- Packing/preparation cost is inside inference boundary; no speed improvement is established.
- Categorical, ordered and ranking resident models retain existing preprocessing/archive guards. Unsupported combined simple-CTR/tensor-CTR or missing archive-table configurations are not newly supported.

### C50_IF — IF resident exact packed scoring nodes

**Source status:** source_integrated_unverified. [Per-ID integration record](inference/C50_IF.json). Runtime reach: **NOT DEMONSTRATED**.

**Supported source scope:** IF.

| Selectable sub-arm | A candidate defines | B reference defines | Scope |
| --- | --- | --- | --- |
| `C50_IF:C50_IF` | `MOJOLEARN_TREES_C50_IF_PACKED=1` | Omit this candidate’s defines (incumbent) | IF |

**Production callers:** [isolation_forest/impl/isolation_forest.mojo:IsolationForest.fit](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/estimator.mojo](../../isolation_forest/estimator.mojo); [python/mojolearn/_iforest_impl.py](../../python/mojolearn/_iforest_impl.py); [isolation_forest/impl/isolation_forest.mojo:if_pack_nodes_kernel](../../isolation_forest/impl/isolation_forest.mojo); [isolation_forest/impl/isolation_forest.mojo:if_packed_paths_kernel](../../isolation_forest/impl/isolation_forest.mojo).

**Full-workload mapping:** 4 source recipe references, listed in the per-ID record and JSON companion; every fact set remains pending.

**Attribution:** C50; T31.

**Remaining gaps, conditional scope and no-difference cases:**

- One-device packed IF required; multi-device packed implementation is explicitly refused.

## C45–C51 shared attribution

| Classical card | Shared implementations | Scope |
| --- | --- | --- |
| [C45](overlaps/C45.json) | `T05`, `T07`, `C45_GBDT` | Forest T05/T07 raw or quantile node scoring/selection, plus C45_GBDT bounded symmetric levels and non-symmetric root prefix/score fusion. Wider/traced or otherwise guarded-out routes retain B; source integration is not observed runtime reach. |
| [C46](overlaps/C46.json) | `T03` | Exact bounded sibling subtraction only; T20 reuses boosting incumbent histogram-lifetime machinery but does not extend this RF/ET accumulator claim. |
| [C47](overlaps/C47.json) | `T02`, `T16`, `T17`, `C47_GBDT`, `C47_IF` | Forest task geometry, exact boosting pending-work budget, and IF live extrema reduction share attribution but use estimator-specific representations. |
| [C48](overlaps/C48.json) | `C48`, `T19`, `C48_IF` | Own forest stable-partition control plus T19 boosting and IF stable partition. No exact standalone T ID duplicates forest C48. |
| [C49](overlaps/C49.json) | `T12`, `T13`, `T41` | T12/T13 forest concurrent immutable representations and T41 IF bounded tasks; preserve stable tree/node/draw IDs. |
| [C50](overlaps/C50.json) | `T31`, `T33`, `T35`, `C50_IF`, `C50_GB` | Exact forest T31/T33/T35 and IF C50_IF layouts, plus C50_GB packed resident oblivious/non-symmetric GB plans. T34 remains a separately selected numerical version. |
| [C51](overlaps/C51.json) | `T44` | T44 exact structural/path metadata reuse. T45 numerical path-unwind formulation remains separately selected. |

The classical worktree was read only. Shared behavior is implemented once; overlap attribution does not duplicate measured cells or broaden supported objective scope.

## Listed interactions

| Interaction | Complete source selection | Qualification / scope |
| --- | --- | --- |
| `X01` | `T01,T02,T04,T05` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X02` | `T03,T08,T09,T10` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X03` | `T11,T12,T13,T15` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X04` | `T16,T17,T18,T19,T20` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X05` | `T21:T21_GROUP_STREAM2,T22,T23,T24,T25` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X06` | `T26,T27,T28:T28_ALL,T29` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X07` | `T31,T32,T33,T35,T36,T37,T38` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X08` | `T40,T41,T42,T43` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X09` | `T44,T45` | future frozen source and complete retained recipe facts; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `X10` | `T14,T26,T34` | accepted individual host/NVIDIA/AMD/Apple numerical identity and quality; Separate full affected-estimator recipes; a branch-inapplicable estimator does not demonstrate that component |
| `XC45_51` | `C45,C46,C47,C48,C49,C50,C51` | ;  |

Each interaction has complete A and complete B plus independently assigned off/on subsets in the programmed matrix. The matrix was not executed. X05 uses the two-wave grouped T21 arm; four-wave and other named sub-arms are explicitly selectable. X06 includes T28_ALL; PairLogit scheduling/versioned fold and YetiRank scheduling use their distinct T29 controls and recipe cells. X10 requires accepted individual T14/T26/T34 all-column evidence before future execution. See [interactions.json](interactions.json).

## Remaining evidence and source work

Every card remains NOT COMPILED, NOT TESTED, IDENTITY NOT VERIFIED, QUALITY NOT VERIFIED and NOT MEASURED. Per-ID scope gaps above are retained rather than presented as compiler support or completed runtime coverage. See [HANDOFF.md](HANDOFF.md) for the production/harness map and material limitations. No result or board cell was admitted; historical status in the separate inventory is not fresh evidence.
