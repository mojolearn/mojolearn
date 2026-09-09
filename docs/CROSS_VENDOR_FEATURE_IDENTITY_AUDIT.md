# Public feature identity audit — retained evidence, September 7

This is a source/document/artifact audit, **not a new comparison**. No tests,
models, compilers, validators, measurements, or cloud calls were executed.
The inventory follows `python/mojolearn/__init__.py`, its exported submodules,
[SUPPORT_MATRIX](../SUPPORT_MATRIX.md), [IDENTITY_PATHS](../IDENTITY_PATHS.md),
and [DERIVATION_MAP](../DERIVATION_MAP.tsv). A derivation entry describes source
implementation, not successful execution. This table covers public families and
their materially distinct paths, not every Cartesian product of parameters.

**No current all-feature, all-three-device certificate was located.** In
particular the historical **180/209 NVIDIA/AMD cells must not become an
all-three-vendor total**, and neither old cells nor refusal counts include new
training trajectories. A stage trace stores named hashes of stages; it is not
the same artifact as retaining all intermediate FP32 bytes. A successful final
prediction hash, import, reference-tolerance test, or same-device repeat is a
different scope again. FAST and DETERMINISTIC do not inherit IDENTICAL coverage.

Root supplement: subsequent same-session OS readback identifies this local host as **Apple M4**. [Saved hardware witness](../bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple-hardware-supplement.json) binds the retained Metal binary hash. This supplements, rather than rewrites, the capture runtime metadata.

## Evidence anchors and device/source boundaries

All paths are repository-relative; the links below locate the retained records.
“Historical matrix” in the table means these located records need exact cell
membership/source/device inspection before promotion to any current result.

| Anchor | Source, actual devices and scope |
|---|---|
| H13 | Historical E2/E2U at `a0a0eeee9a5b774ef62d91fc003a245e0c90af6f`: [Apple leg](../bench/results/e1/2026-08-28_130918-MacBook-Air-1-terrabyte/), [NVIDIA leg](../bench/results/e1/2026-08-28_131651-runpod-nvidia/), [AMD leg](../bench/results/e1/2026-08-28_173933-mojolearn-e2-amd/). `e2_cells.json`, `e2u/e2u_cells.json`, `.cell.json` and `.card` files bind individual cases. Sibling `mlsys/results/cross-vendor-identity.json` identifies the historical hardware as Apple M4 base10-core, NVIDIA H10080GB, AMD MI325X, but its round/provenance fields span revisions. Do not infer M4 from a directory hostname alone or transfer that device assertion to other Apple dates. |
| I60 | [Installed gap closure](../bench/results/resume/2026-09-06-installed-gap-closure/README.md): frozen `eb835021dcd79a59a7e8f78c754a75db3c1fea83`, NVIDIA sm89 and AMD gfx942. All45 native builds; AMD24/24 installed jobs, NVIDIA19/24 before scoped fixes. Individual `.installed.json` plus logs distinguish binary readback, final smoke hashes and tolerance checks. These are not a paired all-feature stage-byte comparison. |
| A60 | [Apple candidate](../bench/results/resume/2026-09-06-feature-finish/README.md): fifteen interpreter/mode jobs, Mamba102 and Transformer44 checks per job. Apple-specific native build/public surface evidence, not automatic current-source or cross-vendor qualification. Use that capture's hardware witness to establish chip generation. |
| FNV | [NVIDIA feature followup](../bench/results/resume/2026-09-06-root-feature-nvidia/README.md), run3 `4a271ae6d719c79e2e871f4776b74c9a12f380ee`, RTX4090/driver580.159.04. FAST and IDENTICAL Mamba102 checks and backward reference/dumps pass; DETERMINISTIC not run there. Old FAST key-state failure is superseded for this fixture only. |
| OMK | [Ordered/Mamba/kNN continuation](../bench/results/resume/2026-09-06-ordered-mamba-knn/README.md): `6dd44ac5` ordered/kNN, NVIDIA4090 and AMDMI325X;130 ordered records and5440 selected index/distance pairs per each of four kNN arms. Separate corrected Mamba profile at `395d9421`, explicit compositional-reference contract. |
| TS | [Detailed time-series/spectral/GP evidence queue](MISSING_VENDOR_EVIDENCE_QUEUE.md): exact artifact paths, source splits and shapes. Includes Apple/AMD `221aa141` Holt-Winters/spectral, NVIDIA historical filter at `fe038d...`, and explicit missing full NVIDIA cells. |
| T1 | Historical one-block training: sibling `mlsys/results/neural-training-three-vendor-2026-09-03.json` points to historical `bench/results/checkpoint_2026-09-03/README.md`, commit `5e5bd5e0`; AppleM4, NVIDIAH100, AMDMI325X. B2/L8/DM32/V64, eight composed steps, embedding/loss/optimizer/train cards and checkpoint-file equality. Original document now resolved via historical git according to sibling provenance; independently re-admit exact card files before treating this as current evidence. It explicitly does **not** establish foreign resume or independently correct all nonlinear gradients. |
| G1 | [Independent single-block gradient gate](../bench/results/resume/2026-09-06-root-training-nvidia/README.md), `201e5ebcf58b388573426a2df77b1d5161fd156a`, NVIDIA4090, B2/L8/DM32/V64,13376 parameters, all11 gradients and loss against FP64 with effective controls. Tolerance correctness; not three-vendor equality. |
| MLP | [MLP pair](../bench/results/resume/2026-09-07-root-mlp-amd/README.md), `8f6ed4112a3c140326011f908e8caf14c0dee4af`, NVIDIA4090 versus RunPodAMDMI300X. Fixed8→16→3 ReLU,195 parameters,48 examples,16 AdamW steps;21024 retained scalar cells (logits/loss/input+parameter gradients/weights/moments/flags/counters). Same-device step8 resume on each GPU; no foreign MLP resume or Metal training. |
| LM3 | [Three-vendor continuous byte LM](../bench/results/resume/2026-09-07-root-byte-lm-three-vendor/README.md) and `comparison.json`: NVIDIA4090 CUDA, DOMI325X VF HIP, AppleMetal, same258-file inventory. Linux source `eac39c367beeddb8ba4792551d154673e654ce21`; matching Metal source `45cc2d9dcd2279646f9a9716fdfab9d598320535` per common expanded manifest. B2/L32/DM32/H4/KV2/FF64/V256,2blocks,34944parameters,128steps. Raw parameters/gradients/moments/flags/counters/losses/tokens/heldout/checkpoint match. Apple runtime directly records Darwin25.5.0, arm64, macOS26.5.2; chip model was **not found in the runtime witness inspected**, so this audit labels this column AppleMetal, not independently proven M4. NVIDIA/AMD first-step FP64 gates pass; no MetalFP64 oracle. |
| LMR | [Historical bidirectional byte-LM resume](../bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md), both comparison JSONs and bidirectional-admission.json: NVIDIA↔AMD foreign step64 checkpoint to128, full raw states and effective missing-moments controls. Earlier inventory needs [transitive source supplement](BYTE_LM_TRANSITIVE_PROVENANCE_AUDIT.md); the fresh expanded258-file continuous round does **not** itself repeat foreign resume. Metal resume remains open. |

The LM3 Apple run used explicitly user-authorized `macos-root-user-tiny-v1`,
not the original4GiB-reserve guard. Its receipt retains that distinction; do not
rewrite earlier blocked attempts as successful or claim the stricter guard ran.

## Exhaustive public-family/path queue

“Historical card” is a locator/status from the ledger, **not a fresh raw-byte
comparison performed here**. Where exact shape/source/device membership was not
resolved in this bounded audit, the row says so and remains open for a current
three-device claim. “All3” below includes AppleMetal only where chip generation
is not established; an explicit **M4** claim needs an additional hardware witness.

| Public family / distinct feature | Located evidence and gap | Small next root run |
|---|---|---|
| GradientBoosting symmetric numeric RMSE | H13 historical stage cards; FNV matched external numeric fixture1024×8,16trees/depth4 is NVIDIA accuracy/timing, not all3 equality | Freeze small nonconstant numeric fit, retain tree arrays, prediction bytes and all stage cards on3 devices |
| Binary Logloss | H13 named `gbdt_logloss_*` cells/cards; current exact membership unverified | Small imbalanced binary fixture, probabilities and leaf/model bytes |
| MultiClass softmax | H13 `gbdt_multiclass.cell.json`:20trees/depth6,borders128,lr.3,seed7; card and final prediction/probability/model hashes; row count absent in that cell | Retain exact input shape plus logits/probabilities across3 |
| MultiClassOneVsAll | Recent public/boundary checks; current all3 path proof not found | Small3-class OVA with explicit gradients/leaves/probabilities |
| MAE / Quantile / Expectile / Huber / Poisson and other exposed losses | Do not infer from RMSE; per-loss current3-device card membership not found | One nontrivial small fixture per actually accepted loss; retain effective boundary cases |
| Depthwise / Lossguide / max-leaves growth | H13 named cells; semantics differ from symmetric trees and sklearn max_leaf_nodes | Tiny unbalanced splits with leaf counts and stage bytes |
| Bootstrap / weights / random-subspace / border variants | H13 named RMSE/logloss configuration cards; parameter variants require individual membership | Small deterministic weighted data; all accepted variants individually |
| Categorical one-hot / CTR / ordered target statistics | H13 categorical cells; OMK weightedCTR pair. Full CatBoost categorical parity unproven | Small repeated categories, unseen value, weights, explicit CTR tables and final bytes |
| OrderedRMSE numeric single permutation | OMK CUDA/HIP130records; A60 public checks separate; no admitted same-source3way record located | One permutation, zero-mass prefix, weighted multi-tree predictions, fresh3way raw capture |
| ExperimentalTwoLevelFeatureFreq | Experimental public surface; no distinct current3way evidence located | Tiny fixture plus explicit experimental scope |
| Ranking / multi-target / generalized Ordered / categorical combinations | Broad features remain unsupported or incomplete; not inherited from exposed losses | First exact API admission/refusal audit; do not count refusal as implemented identity |
| RandomForestClassifier / Regressor | H13 historical RF cards | Tiny binary/multiclass and regression, bootstrap and prediction bytes |
| ExtraTreesClassifier / Regressor | H13 ET cards including `et_reg` variants | Same small data with nonconstant random splits |
| IsolationForest | Historical fixture card per support ledger; I60 smoke | Small outlier data, scores/path lengths and model bytes |
| KMeans | H13 historical classical matrix | Small ties/empty-cluster-sensitive fixture; centers/labels/inertia |
| DBSCAN Euclidean brute / RBC | Historical distinct arms; IDENTITY_PATHS batch fixture1020points,6border/14noise demonstrates nonvacuity | Tiny border/noise case each accepted algorithm, batching varied |
| DBSCAN Manhattan | Source implementation exists; historical refused ID may use defaultRBC while Manhattan requiresbrute; no current3way card found | Explicit `algorithm='brute'`, Manhattan and tied borders |
| AgglomerativeClustering | Historical classical card; current exact linkage/shape coverage unresolved | Separate accepted linkage/connectivity paths |
| SpectralClustering | TS Apple/AMD stage cards n144/d4 graph and n144/d3/c3 cluster; NVIDIA installed final smoke only | Full small NVIDIA native card then fresh3way pairing |
| KernelDensity | Historical classical card; metric/kernel coverage unresolved | Tiny per-accepted-kernel query log-density arrays |
| NearestNeighbors Euclidean | Historical kNN cards; OMK layout four-arm CUDA/HIP bit equality and older Apple hashes | Shared tiny query/train input, full selected distances/indices |
| kNN Manhattan / cosine | Current sources support metrics; dedicated current3way membership not found | Zero vectors for cosine, ties, metric-specific distance outputs |
| KNeighborsClassifier uniform / distance | Uniform historical family coverage; distance-vote zero/tie paths not separately qualified here | Duplicate zero-distance neighbors, class ties and predict_proba |
| KNeighborsRegressor uniform / distance | Same distinction | Duplicate points, nonconstant multi-target values where supported |
| RadiusNeighbors | Historical family surface; exact radius/boundary scope unresolved | Equal-radius boundary and empty results, all returned arrays |
| kNN IDENTICAL k257..1024 | New rank-capacity source port; current3way evidence not located | n≥257, k257 then1024, ties spanning capacity; oldk256 control |
| kNN KD-tree/index algorithms | Refused; do not label brute-force port as index parity | Retain refusal only unless implementation lands |
| PCA randomized/truncated path | Historical matrix; current solver-specific shape scope unresolved | Small tall matrix, projected data/variance/singular values |
| PCA full solver | New native public export; distinct release/source from old refused cell | Small tall fullSVD, meaningful reconstruction and all metadata |
| PCA whitening | New public/native path; historical PCA certificate insufficient | Nondegenerate scales plus degenerate/rank-deficient behavior |
| TruncatedSVD | Historical matrix | Tiny sparse/dense accepted profiles separately, raw decomposition outputs |
| LinearRegression / Ridge | Historical classical cards | Tiny full-rank and singular-conditioned cases, coefficients/predictions |
| LogisticRegression L2 / none | Historical matrix | Binary and multiclass accepted strategies separately |
| LogisticRegression L1 | Existing Python→QN→OWLQN path; old refusal entry not current certificate | Small effective sparsity fixture with coefficient zeros and probabilities |
| Lasso / ElasticNet | Historical classical cards | Small correlated design and effective regularization |
| SVC | Historical native card; current kernel/options membership unresolved | Linear/RBF and each accepted kernel separately, decision values |
| SVR | Public exposure postdates older SVC card; regression half not inherited | Tiny epsilon tube with active/inactive residuals |
| ARIMA filter | TS filter n_obs24,batch6,salt7; historical3leg source differences explicit | Same frozen filter input/state across3, stage bytes |
| ARIMA fitter / predict / forecast / criteria | I60 CUDA/HIP allmodes functional reference tests length512,batch6,AR1/MA1/ARMA11; not paired fitted-state identity | Small accepted fitter fixture, full optimizer state and prediction/forecast bytes |
| ExponentialSmoothing / Holt-Winters | TS Apple/AMD full card n20,batch3,frequency5; NVIDIA smoke only | NVIDIA full additive+effective multiplicative fixture then pair3 |
| kpss_test / select_d | Time-series helper paths; dedicated current3way membership not found | Short stationary/trending series and exact statistics/decisions |
| GaussianProcessRegressor | TS Apple/AMD cards; NVIDIA binding readback/FAST timings do not close IDENTICAL | Small fit/posterior/uncertainty native gate on NVIDIA then3way |
| GP ConstantKernel / RBF / Matern / WhiteKernel compositions | TS examples train2×1,16×1,4×2,ARD12×3,test6,Matern8×2,test4; vendor source mismatch recorded | Each supported composition and scalar/ARD gradients separately |
| linalg.matmul / public matmul | Historical frozen-profile3vendor sweep, not arbitrary dimensions | Tiny rectangular/transpose/zero-edge profiles and raw results |
| metrics accuracy_score / r2_score | Historical metric-family cards; option-level3way scope unresolved | Weighted/unweighted, constant targets, accepted multioutput cases |
| metrics rand_score / adjusted_rand_score | Historical metric-family cards | Nontrivial contingency including singleton labels |
| metrics entropy / mutual_info_score | Historical metric-family cards | Unequal class masses and provided contingency if supported |
| metrics homogeneity / completeness / v_measure / combined | Historical metric-family cards | Asymmetric contingency and nondefault beta |
| metrics kl_divergence | Historical metric-family coverage unresolved perpath | Nonuniform distributions and declared zero handling |
| metrics silhouette_score / silhouette_samples | Historical metric-family coverage unresolved permetric/batching | Tiny separated+overlapping clusters, samples and reduction |
| metrics trustworthiness | Historical metric-family cards; UMAP quality metric not proof of all options | Small exact neighbor-rank fixture with ties |
| UMAP dense fit/fit_transform | Historical identity lane; current fit changes require scope-specific cards; I60 surface and FNV native controls | Small dense fit retaining graph/embedding and stage state |
| UMAP transform | I60/Apple surface+expanded quality checks; current3way raw transformer membership needs retained comparison | Small heldout batch, repeated/split query raw outputs |
| UMAP precomputed CSR / sparse graph | September public sparse records located; no fullcurrent3way card established here | Fixed CSR indptr/indices/weights, graph reuse and invalidCSR refusals |
| UMAP learning_rate / repulsion / negative_sample_rate | FNV FAST/IDENTICAL effective controls, not general feature parity | Small each-control sensitivity case, then3way identical |
| Mamba1 forward / state continuation | Historical3vendor17-stage block record, H13 samea0a0eee; public current fixture separate | B2/L4/D8 plus split/decode raw state |
| Mamba1 backward | Historical718495cd AppleM4/NVIDIA4090 baseline comparison; corrected laterCUDA/HIP records; A60Apple checks separate | Current same-source B2/L4/D8, arbitrarycotangent, fullrawgradient3way |
| Mamba2 forward / continuation | Historical26-stage3vendor block record with per-vendor source revisions, not one sharedcommit | B2/L4/D32, B1/L257/D64 chunk boundary separately |
| Mamba2 backward | CorrectedCUDA/HIP baseline gradientpair; publicFNV native/ref checks | Current all3 independentcotangent profile; continuation-state VJP unsupported |
| Mamba3 forward / continuation | Historical3vendor block cards, per-column revisions; FASTrepairFNV validated; DET failed the installed 0.7.0 column at the same seam and is repaired by DEVIATION 2300 (stable log1p now covers every unpinned tier); the fixed commit's installed requalification is the evidence | B2/L4/D32 k_last/theta plusL65 boundary, unchanged tolerances |
| Mamba3 backward | Old718495cd agreement superseded by missingchain-rule correction; correctedCUDA/HIP54baseline/21longpublic tensors plus86diagnostics/9operands; AppleL65 independent intermediate previouslyRED | Fresh correctedApple then all3; retain all independent references/control outcomes |
| Mamba1/2/3 full model training / optimizer resume | Block VJPs are not a composed training/checkpoint API; fulltrainedMamba result not found | Define supported composed state/data contract before any claim |
| TransformerBlock forward / state | Historical30-stage3vendor record; currentI60 NVIDIAfix/A60surface separate | FixedDM32 tinysequence, rawattention/block state3way |
| Transformer backward / embedding composed training | T1 historical3vendortrainingcards; G1 newerNVIDIA independent11gradient gate | Same-source raw gradients and independent reference across3 |
| SGD / Adam / AdamW public optimizers | T1 native primitive cards; AdamW integratedMLP/LM3raw3way; standaloneSGD/Adam exposedlater | One meaningful nonzero-momentum/bias-correction update peroptimizer3way |
| clip_grad_norm_ / cross_entropy | T1 native cards, G1/LM3 meanCE integration; option coverage not inherited | Active clipping, reduction variants and accepted target/smoothing boundaries |
| SmallMLPTrainer forward/backward/AdamW | MLP NVIDIA/AMD16-step fullrawtrajectory; MetalOPEN | Same8→16→3/48rows16steps onAppleM4 with chipwitness |
| SmallMLPTrainer checkpoint/resume | MLP same-device step8 oneachGPU only | ForeignCUDA→HIP→Metal checkpoint transfer and effective lost-momentcontrol |
| SmallByteLanguageModelTrainer forward/backward/AdamW | LM3 bounded128-step all3continuousraw equality; AppleM4 chipwitness stillneedslocating | Rebind exactApple chipwitness; currentwheel availability and newrelease mathsource separate |
| ByteLM checkpoint bytes / foreign resume | LM3 finalcheckpoint bytes all3; LMR historicalbidirectionalCUDA/HIP continuation; expandedroundresume/MetalresumeOPEN | ActualMetal incomingstep64 resume to128 plus controls on same258sources |
| Model serialization / numeric-mode preservation | PublicA60/I60 scoped checks; serialization cannot inherit math-stage proof | Perfamily save/load currentmode, realfile crossvendor inference where supported |

## Admission order for new small runs

Root should first resolve hardware/model/source witnesses from existing raw
records, especially Apple M4 versus unspecified AppleMetal. Next freeze a common
source, input bytes and accepted parameter profile for each open row; run one
small case at a time on NVIDIA, AMD and Apple, with the currently authorized
vendor-specific guard policy and two-core/thread limits. Keep original failures.
Retain exact build/binary identities, independent reference/control results,
per-stage arrays where feasible, final output bytes, and successful guard exit
including teardown. Recompute comparisons only in the root thread.

Prioritize corrected Mamba3 Apple backward, separate MLP Metal training, actual
Metal LM resume, new PCA full/whitening and large-k/weighted/metric kNN paths,
then NVIDIA GP/spectral/Holt-Winters and per-loss tree gaps. A named refusal
closes a refusal contract only; a passed smoke closes a smoke only. Never loosen
tolerances or relabel a historical hash-card result as a current full-byte proof.
