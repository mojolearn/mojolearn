# Apple FAST experiment coverage

All F01–F20 cards use **FAST on Apple**. They do not promise IDENTICAL bits, and no candidate is promoted by this work. Kernel changes are opt-in; existing FAST defaults are compared with independent rollback arms where applicable. A successful source build does not establish device correctness, task quality, speed, or opponent admission.

Each `Fxx/manifest.json` declares its actual source, baseline/candidate compiler definitions, caller, dependencies and variants. `build_pair.py` builds both arms once on the retained Apple build queue, under the compile semaphore, with an exact required source SHA. `pair.py` runs isolated, attested packages on the existing M3 Ultra and retains both complete logs and captures. Every requested output is read before its caller timer ends. Test-only NumPy constructs inputs and independent references; new product arithmetic remains compiled Mojo.

| ID | Implemented experiment and actual caller | Quality/reach/lifecycle checks |
| --- | --- | --- |
| F01 | Scoped G1 tall/Gram and PCA geometry through public fitted PCA projection/inverse, odd dimensions and neighboring widths. | Actual route counts; independent fitted singular/noise estimates, reconstruction; fit, cold transform, repeat and inverse measured separately. |
| F02 | Independent SDK NN/NT and true alias-Gram entrances; existing decomp/PCA split adapter; new strided fused LU subtract and lower-triangle Cholesky subtract; existing active-batched MCD adapter. Variants `sdk`, `pca`, `cholesky`, `mcd`, plus default LU. | Positive route counters; FP64 SDK product references and GEMV control; actual LU factor/solve residuals; Cholesky factor/solve residuals and unchanged upper-triangle semantics; fitted PCA/LLE and MCD/elliptic consumer quality. No generic extra subtraction pass; MCD remains one batch launch with inactive gates preserved. |
| F03 | Real byte-LM resident sessions versus stateless calls, with two vocabulary sizes. Same source/definitions; runtime retention is the switch. | Fixed training order/loss, checkpoint/reload handle refresh, refused token recovery, independent live peer model, close/export/re-admit; cold and repeated calls separated. |
| F04 | Direct source/output transport for independent resample arrays, with one grouped completion and buffers retained until completion. | Actual native return/reach; exact drawn rows, asymmetric array widths, old-output lifetime, negative-count refusal and recovery; transfer/output byte counts. Exact sampling is an API requirement rather than an IDENTICAL digest gate. |
| F05 | Existing narrow softmax GEMM candidate through complete multiclass `qn_fit`, optimizer/line-search work and prediction. | Positive candidate count, fixed optimizer settings and seeded conditioning spread; heldout logloss and misclassification, iterations/fit time and consumed predictions. |
| F06 | New bounded independent MCD batch launches, alongside incumbent split-policy fallback, through MinCovDet and EllipticEnvelope. | Native bounded-launch counts, contamination and nearly singular columns, fitted location/covariance/support quality, full fit output completion. |
| F07 | Separate FLASH and grouped-query attention candidates through a complete TransformerBlock, causal tails and real GQA ratios. | Independent full-block FP64 reference and actual flash/grouped hit counts; inference admission stays separate from backward/training. |
| F08 | Separate backward no-sync, backward fusion and parameter-view candidates through actual byte-LM training. | Fixed data/hyperparameters, heldout loss, returned-loss completion, live peer isolation, refusal recovery and checkpoint/reload/view refresh. No combined candidate arm. |
| F09 | Existing bounded vocabulary-tile head versus full logits plus CE, through native linear/CE/backward and compiled SGD. | Tail tokens, large vocabularies and extreme logits; stable independent loss and hidden/weight gradient references; fixed short training loss; explicitly requested full logits still materialize. |
| F10 | Separate SSD MMA, Mamba-1 input fusion/chunk scan, Mamba-3 elementwise fusion and arena arms through full Mamba callers. | Committed FP64 corpus references across lengths/state widths; carried-state prefix plus token decode drift; actual zero-state backward directional reference and compiled SGD training task. No mixed-angle/state/fusion arm or automatic combination. |
| F11 | New selective compensated centered PCA covariance product; separate FAST TSQR norm/grid candidates; actual least-squares and randomized range-finder callers. | Conditioned coefficient/residual references, rank/singular state, independent PCA singular/noise/reconstruction errors, unchanged input, downstream LLE neighbor preservation; compensation is limited to covariance rather than every product. |
| F12 | Separate partition reuse, leaf inputs and arena arms through full boosting fits; categorical and ranking fixtures; independent depthwise and exact-best-first lossguide scheduling controls. | Fixed seeds, tree budget, data/sampling settings and zero score noise for scheduling controls; heldout task quality, completed iterations, cold fit/first prediction and retained prediction. |
| F13 | New FAST-only admission of shared forest rows; separate new adjacent-row-per-tree SHAP schedule without pipeline overlap. | Shallow/deep, skewed and multiclass forests; heldout accuracy; actual TreeExplainer additivity, pair-route counts, preparation cost, old-output lifetime and repeated queries. |
| F14 | New exact streaming query group of four; separate root-owned approximate IVF balanced-task variant `ann`. | Exact exhaustive neighbor certification remains distinct from approximate recall; ANN fixes probes/filters/index hash and includes full-probe control, candidate count and actual hit checks. Index build/cold and warmed query time remain separate. |
| F15 | Separate mini-batch label/stopping candidates; new FAST-only direct DBSCAN label output; separate HDBSCAN core-distance, linkage, selection and output-completion candidates. | Full inertia/cluster-pair quality, unchanged starts/seeds/stopping settings, iteration/repeat diagnostics; DBSCAN/HDBSCAN diagnostic phase reruns have separate logs and never substitute for uninstrumented full-fit timing. |
| F16 | Gemm/platform-owned production compensated Kalman tail and retained-low-word gradient adapter; paired builder compiles and attests its native independent checker before actual AutoARIMA. | Native every-gradient-component FP64 captured-stage oracle runs before any measured caller; production initialization/Jones/covariance/h/scale and refusal preservation; separate positive likelihood and gradient reach counts, selected orders/optimizer status and heldout forecasts. |
| F17 | Existing bounded stacked time-series evaluation versus batch-gradient rollback through actual multi-series AutoARIMA search/fit/forecast. | Same model search settings/data, independent seeded stationary/near-unit/integrated DGP, forecast error, selected order/criteria and optimizer status. No private likelihood timing substituted for full search. |
| F18 | Separate direct KDE prepare and immutable resident-fit snapshot, through public fitted scoring. | Independent Gaussian logsumexp reference after every bandwidth invalidation; actual immutable source-mutation reference, two live models, output lifetime, nonfinite refusal/refit recovery. Snapshots are byte transport rather than new host numerical work. |
| F19 | New coalesced tiled row gather and a separate device-generated permutation arm: stable UInt64 key/position merge passes followed by caller-owned gather. | Draws/sort/gather on device for the admitted route, replacement and nonreplacement exact semantics, narrow/wide tails, positive native admission, output read and byte accounting, oversampling refusal. |
| F20 | Separate multi-tensor optimizer, fused status scan and new FAST LayerNorm fold admission, through actual task/layer callers. | Fixed two-head compiled training, loss and refusal rollback; cancellation/large-magnitude/tail normalization, independent input/affine gradients and compiled downstream classifier loss. |

## Qualification and original-card limits

No GPU qualification or speed result is claimed by this document. Local checks comprise Python syntax/manifest review and compile-only builds; the complete retained logs include unsuccessful attempts and their corrections. Explicit Metal target builds passed for the shared SDK adapter, SHAP, permutation gather, and compensated PCA/direct DBSCAN output. Publication hooks passed without bypassing or expanding the host-loop baseline. These checks are not source-frozen artifact receipts and cannot stand in for M3 runs.

The following portions remain separate work or explicit capability boundaries:

- F01: no calibrated geometry cost selector, broad estimator promotion, or measured scratch/crossover rule has been installed. Its fitted PCA scope and F02's independent adapters are experiments, not a global selector.
- F02: host counters prove dispatch and batch-launch reach; a device count of active MCD candidate slots and all caller peak scratch/wait metrics still need measurement. Similar tile names do not certify batched coverage.
- F03/F08/F10: physical peak GPU retained/intermediate memory has not been measured. Mamba supported zero-state backward is exercised; stateful/decode backward and final-state cotangents remain outside the public API.
- F07: no new sliding-window/ragged/packed backend is claimed. The fixture covers the supported causal/full-block contract; F08 separately covers training.
- F12: no new document-ID coalesced permutation storage or categorical device winner-residency kernel has been invented. Existing supported scheduling/storage candidates now have the broader actual-caller fixtures; categorical scratch and waits still require device evidence.
- F13: no new compact node conversion is claimed. Shared rows and SHAP row pairing are independently scoped mechanisms; a compact-layout/conversion experiment remains separate.
- F14: root owns the ANN source and `ann_caller.py`; they must be integrated before the lane's ANN recipe runs. The exact caller never adopts approximate search semantics.
- F15: speculative batching of independent initializations is not added; seeded incumbent initialization/selection is held fixed. New graph algorithms are not claimed from output transport or phase timing. Graph-phase dominance and scratch measurements remain runtime evidence.
- F16: root must integrate the production adapter, counter binding and native checker from the gemm/platform lane. Admission is narrow scalar supported order/state/resource limits, with incumbent fallback elsewhere. Full covariance/seasonal/exogenous compensated scan requires its own mathematical oracle; source readiness is not a waived covariance gate.
- F17: spatial/seasonal search and independent first-refusal/candidate-card semantics still need wider workload qualification; the fixture does not certify every optimizer option.
- F19: weighted `sample_weight` draws remain explicitly refused by the existing public API. They cannot be claimed implemented without a pinned weighted draw definition. No NumPy/CPU runtime fallback is used to qualify the candidate permutation/gather path.

## Frozen queue use

Build exactly one selected variant on the retained M3 build queue, for example:

```sh
python3 experiments/performance_ideas/apple_fast/build_pair.py \
  --idea F14 --variant ann --source-sha "$FROZEN_SHA" \
  --output "$HOME/performance-idea-arms/$FROZEN_SHA/F14-ann"
```

The source checkout must already match the reviewed SHA, be clean, and include every implementation/caller dependency. The existing queue starts the command on the lane tree; the helper refuses source drift and any other build machine. Do not run a second baseline build: `paired_build: true` means one invocation prepares both separately compiled arms. Defines use only MOJOLEARN_MOJO_BUILD_FLAGS and MOJOLEARN_BUILD_EXTRA_DEFINES is explicitly empty; build jobs are one, kernel smoke is disabled only during build, and native binary validation remains the repository builder's responsibility.

Root stages an artifact directory only after `manifest.status == OK` and every arm, prerequisite and native-check file hash matches. M3 runs `pair.py --idea F14 --variant ann --arms <staged-arms> --output <new-capture-dir>` on the same frozen source. Existing retained M3 queue and staging only; no lane SSH or new machines. F16 additionally runs its attested independent native checker before public captures.

M3 orchestration validates the environment/hardware without importing mojolearn; only isolated captures load the product. Use the existing test environment (`pixi run -e test python`) or an already supported oracle environment for M3. Python assertions must remain enabled. ANN candidate-count equality is an explicit cross-arm semantic field gate. M3's selected Python must have NumPy available; these harnesses do not install it or import PyTorch. Source-package isolation must retain the attested core and required secondary bindings (for example training for Mamba/normalization, x_decomp for LLE, RF for SHAP). A missing dependency, non-FAST/non-Metal binding, caller exception, wrong variant, hash/source mismatch or native-oracle failure stops the run and retains its exit/log. `pair.py` quality PASS only admits the captured caller comparison; repeated timing, peak device memory and strongest-opponent admission remain separate gates.

## Static caller and queued package audit

Reviewed the F01–F20 public fixture calls against the Python source on this experiment branch. This is a source audit and Python syntax check; it does not establish GPU execution, task quality or performance. The common fixture reads every returned scalar/array before ending its timer. Constructor `numeric_mode` keywords supplied through `NumericModeMixin` were checked against its wrapper rather than only the underlying constructor signature.

| Cards | API and result contract inspected |
| --- | --- |
| F01, F02, F11 | PCA fitted singular values/noise/components and transform/inverse shapes; LU tuple and solve; Cholesky `L_`, `info_`, `logdet_`; `lstsq` four-result tuple; randomized SVD three-result tuple; LLE fitted embedding |
| F03, F08 | LanguageModelConfig positional fields and registry offsets; trainer train_step dict, evaluate float, state export/load, close/reopen, logits shape |
| F04, F19 | Resample multiple arrays produce a list, wrapped as a tuple for complete consumption; one array produces one array; indices, permutation and weighted refusal |
| F05 | Native qn_fit and decision-function argument layouts and existing softmax quality case identifiers |
| F06 | MCD/elliptic fitted location, covariance and support arrays and random_state/numeric_mode constructor parameters |
| F07 | Transformer weights/head parameters, allocate_state/forward, full block Float64 oracle and GQA K/V expansion |
| F09, F20 | CE float or loss/gradient pair; linear backward input/weight pair; chunked head loss/input/weight triple; SGD parameter-list constructor and gradient-list step; LayerNorm backward and affine gradient fields |
| F10 | Corpus input/reference layouts, Mamba weight constructors and dt_limit availability, carried forward/step, named zero-state backward dictionary, actual SGD update arguments; every family has length-four training cases |
| F12 | GradientBoosting loss, grow_policy, categorical/one-hot constructor parameters and fit group_id; prediction and loss_curve layouts |
| F13 | RandomForest fitted prediction/probability; TreeExplainer background and expected_value; multiclass SHAP `(rows, features, classes)` fold |
| F14 | Exact NearestNeighbors fit/kneighbors distance/index pair; approximate ANN remains a separate root-owned fixture and admission |
| F15 | MiniBatchKMeans labels/centers/n_iter_; DBSCAN brute memory budget; HDBSCAN fitted labels and Boruvka rounds |
| F16, F17 | AutoARIMA endog/search/fit/forecast; selected order/ic and fitted ARIMA params/n_iter/retcode/llf; search calls the TSA stationarity binding |
| F18 | KDE fit, mutable bandwidth and score_samples, immutable fitted source and recovery paths |

The audit found and corrected missing FAST TSA dependencies in both AutoARIMA cards. Every paired build additionally compiles the exact-source IDENTICAL core with no experiment defines as the sole `input_transport_helpers` dependency: public `_buffer._native` uses this tier even when the estimator is FAST. The pair installer admits only this explicit role at `identical/_mojolearn.so` and the card's enumerated FAST prerequisites, checking source, mode, Apple target, defines and binary hashes. The called estimator binding must still report FAST/Metal and match its A/B hash. Earlier artifacts lacking this helper cannot qualify the corrected source.

Every card exposes its paired quality capture as `validation_argv`; a build makes both arms once (`paired_build: true`). Queue qualification must use an existing Python environment with NumPy, such as `pixi run -e test python`, and hardware-only pair admission occurs before importing the isolated product. The paired builder pins both the Apple vendor and target column and supplies experiment defines only through `MOJOLEARN_MOJO_BUILD_FLAGS` and explicitly clears `MOJOLEARN_BUILD_EXTRA_DEFINES`, because builders that read both reject duplicate defines. No artifact from another source SHA, arbitrary installed library or unrelated-source rebuild can satisfy the source-frozen pair.


## Final frozen compile campaign and additional subarms

The user authorized compilation on the retained M3 and requested one full
campaign after integration. `compile_campaign.py --source-sha <frozen-SHA>
--output <new-evidence-directory> --execute --jobs 4` creates four pristine
source worktrees, installs the locked SDK normally in each, and shares the
normal package cache and existing four-slot compiler semaphore. It deduplicates
exact binding/tier/define configurations across every card arm and prerequisite,
including independent native checks. Each worker builds sequentially; outputs,
full logs, exit codes, source/define/tier/vendor and hashes remain per job.
`--job jobNNN` permits only failed or previously unbuilt configurations to be
selected. A compiler pass makes no device quality, speed, opponent or promotion
claim. No kernel smoke or Python product import runs during this campaign.

F12 now includes explicit Ordered tasks, four fixed permutations, off-grid
regression targets and nonzero fixed-seed score noise. The promoted storage
bundle is retained; an independent rollback arm compares its whole-fit result.
The new `MOJOLEARN_ORD_DOC_ID_STORAGE` candidate is FAST Apple and default off:
it passes the existing retained document-ID table through every pointwise
histogram launch and scalar/vector loop while keeping compressed-index loads in
fold-position order. Fixed-point quantizers receive original document IDs.
Actual submission counts establish reach; a native actual PointHist8 checker
uses independent host quantization for unaligned head/tail, scalar/vector and
empty cases before public caller admission. Existing winner records and
score-before state already stay on the device through all levels, with one
production completion drain per tree; the Ordered fixture exercises that route.
Public device quality, full-fit speed and peak scratch remain pending.

F13 adds `packed-layout`: rollback `MOJOLEARN_FOREST_SEPARATE_NODES` versus the
promoted packed conversion. It asserts the public binding's resident layout,
records cold conversion/first prediction separately from retained prediction,
and requires old caller outputs to survive both reuse and a fixed-seed refit.
The promoted four-word nodes and compact leaf conversion remain unchanged.

F15 adds independent `center-accumulation` and `scratch-reuse` controls for the
promoted compact ascending-row center sum and cross-fit device-input pool.
Fixtures attest the compiled policy, report cold/repeated complete fits and
require old center/label outputs to survive pool reuse. These promoted defaults
remain unchanged. Batched independent starts remain a mathematical/capability
prerequisite: current winner selection sums distances in one ascending Float64
chain, retaining the first strict minimum (`common.mojo:sum_f64`,
`minibatch_ptr.mojo`). A different reduction or Float32/compensated comparison
must not be silently substituted for that policy. No new CPU data loop or
approximate winner policy was introduced; a safe supported device implementation
or independently admitted replacement oracle is still required.
