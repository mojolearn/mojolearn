# Measurement decisions

## 2026-10-14 owner decision: parameters decide fairness; quality is always shown

This supersedes "quality before time" below. A race is fair when every work-changing parameter is set
explicitly and identically on both arms (not each library's default), proven by read-back. Quality never
decides whether a race is shown: both arms' quality is always recorded and shown beside the time, and it is
fine for mojolearn to be better. The margins below remain only as annotations
(`quality_comparison`); no verdict, exclusion or rerace mark rests on them. Quality is how we judge changes
to our own algorithms: speed is good, but never at the expense of quality.

## 2026-10-13 owner clarification: quality before time

This supersedes the same-day rule excluding different executed iteration counts.
Different solvers may converge in different numbers of iterations. If they solve
the same task with comparable measured quality, compare their time. Do not force
extra iterations, pad a clock, or truncate a converged solver to make counts match.
Keep requested limits, tolerances, actual counts (or explicit unavailability),
random seeds, raw parameters and actual library calls in the evidence.

This does not permit changing the dataset, train/evaluation split, model,
objective, requested output, or timed operation. Preflight still compares actual
settings, with source-reviewed semantic exceptions for different library APIs.
Equal iteration caps never prove equal executed work or equal quality. Fixed
training-step throughput measurements still perform the declared step count;
they are distinct from time to a quality target.

`tools/board_quality.py` implements `mojolearn.quality-parity.v6`. Its engineering
margins are declared by metric, independent of dataset, vendor and timing winner:

| Metric | Admission margin |
| --- | --- |
| Accuracy, AUC, R², recall, trustworthiness, explained variance, silhouette, modularity | Absolute difference ≤ 0.01 score units |
| RMSE, forecast/masked RMSE, log loss, inertia, perplexity, final training loss, reconstruction error | Difference / max(abs(values)) ≤ 5%; two zeros agree |
| Mean log likelihood, mean predictive log density, mean NLL | Absolute difference ≤ 0.05 nats per observation |
| GEMM relative error against FP64 | Both errors ≤ 0.01 (permits native fast precision) |
| Linear algebra relative residual / Gram / orthogonality error | Both errors ≤ 0.0001 |
| Deterministic transformations against a common scikit-learn reference | Both relative Frobenius errors ≤ 0.0001; nonfinite outputs fail |
| Neural layer outputs against the same eager FP32 torch reference | Both relative Frobenius errors ≤ 0.01 (permits native fast precision) |
| Optimizer updates against the same eager FP32 torch reference | Both relative Frobenius update errors ≤ 0.0001; compare final minus initial parameters |

The v2 additions were declared before the resumed GPU campaign on 2026-10-14.
All v1 limits remain unchanged. The reference arm's zero distance is recorded
explicitly. Scikit-learn reference work is an untimed test oracle, executed
after the GPU workers stop; its source, parameters, input hashes and output
bytes are retained. It is never added as a timed CPU opponent. An optimizer
must pass both the parameter-vector and update-distance checks; a missing
update cannot hide behind a large initial parameter vector.

These are reproducible screening margins, **not statistical equivalence tests**.
One scored run cannot establish stability over random starts or quantify timing
noise. All recorded comparable metrics must pass. Missing, nonfinite or unknown
metrics need review; counts, finite values, output shapes, and iteration counts
alone do not prove quality. An existing supported human audit can remain usable
for a metric without an automatic rule, with that historical basis stated. A
measured failure under a supported metric always overrides that old approval.
Audit entries retain each metric, both values, margin, observed difference and
verdict. New metric rules require a documented decision before timing; never
adjust a margin because a particular row wins or loses.

Re-audit saved evidence first. Restore a pairing only when its other requirements
also pass. Preserve the old audit, raw observations, failures and exclusions.
Rerun only missing, invalid or insufficient evidence; a policy change does not
require rerunning supported measurements. The current campaign covers NVIDIA
and AMD GPU opponents; Apple is explicitly out of the execution scope.

The separate publication hold on faster-than-opponent rows (except Apple fast)
remains in force. Quality admission does not remove that hold. The generated
paper and website consume the same per-mode, per-phase admitted opponents.

## Retention and guarded execution

Comparison-clock policy `mojolearn.comparison-clocks.v2` enforces the owner's
declared boundary at admission. Torch GPU arms with device-resident inputs use
kernel/kernel; arms that already upload inside the clock, including neural
torch twins, use whole/whole. Other libraries, including CuPy and GPyTorch,
require whole/whole. A displayed alternate common clock remains diagnostic
and cannot admit a pairing. Earlier audits incorrectly accepted this fallback.
Keep those measured kernel times; do not invent their missing copy durations
or re-score a valid opponent. New CuPy linear-algebra adapters separately
record their actual synchronized input upload before the kernel clock.

The v6 correction recognizes classical PCA inference's already recorded
`transform_max_rel_err_own_fp64`: maximum absolute projection error divided by
the maximum absolute FP64 projection of the same fitted model and query rows.
Both arms must satisfy the existing linear-algebra reference bound of `1e-4`.
This repairs an unrecognized metric name without changing any existing margin.
Re-audit retained scored outputs and clocks; neither arm needs new timing.
It does not admit an unsupported inference implementation or an unknown clock.

The v5 addition is declared before the next linear-algebra measurements on
2026-10-14. QR retains both complete scored factors. Each arm must pass the
existing full-input reconstruction and R-Gram bounds, plus
`||Q.T @ Q - I||_F / ||I||_F <= 1e-4`. Previously, retaining only R neither
preserved Q nor validated it. All previous numeric margins remain unchanged.
SVD and randomized SVD also retain every scored factor, even when a quality
metric uses a declared sample or projection. Solve outputs are retained without
truncation. These are benchmark evidence repairs, not product algorithm changes.

The v4 additions are declared before the repaired classical reruns on
2026-10-14. All existing numeric margins remain unchanged. HDBSCAN records
`sklearn.metrics.silhouette_score` on every consumed row, using Euclidean
distance and including noise as its own label group under sklearn's label
convention. No rows are sampled or removed; the existing absolute silhouette
margin is 0.01. ARI, cluster counts and noise fractions remain diagnostics.
When a density result has no intrinsic score (for example, one cluster), exact
partition equivalence and an identical noise mask on the same positive number
of rows establish output-quality parity. Partial partition agreement alone
neither passes nor proves a quality failure; it still needs an intrinsic score.

Classical output arrays must survive the quality pass and be retained in the
verified archive. Their archive member, byte hash and scored-output digest are
recorded per arm and phase. A logs-only archive cannot satisfy this rule. The
old classical cleanup deleted these arrays before sealing; affected new
measurements remain excluded until their output evidence is repaired. HDBSCAN
shape metadata records the consumed slice, not the larger staged parent block.

The v3 additions, declared before repaired dropout timing on 2026-10-14, retain
every v2 margin. Dropout2d uses each library's own RNG. Each saved output must
zero whole channels or scale their input by `1 / (1 - p)`, with relative contract
error at most `1e-6`. The dropped fraction among identifiable, nonzero input
channels must lie within six binomial standard deviations of the declared `p`.
Both arms must pass. This is a single-sample engineering check, not a proof of
RNG independence or statistical equivalence. It replaces the broken scorer that
discarded Torch's dropout outputs and then tried to compare random masks.
Original errors remain recorded; missing opponent output bytes require a new
measurement. Margins are fixed without using timing ratios.

New paired measurements retain the frozen harness source and actual loaded
library/binary bytes; device identity, software/driver versions, resource policy;
seed, raw and normalized parameters, fitted readback where available; dataset
identity and actual dimensions; exact harness command and observed library call;
clock boundaries, copy costs, warmups, samples, quality, and failure status.
`board.json` includes these records and SHA-256 references to complete artifacts
verified by download from R2. A failed quality scorer must propagate into every
affected cell, never disappear during extraction of per-arm results.

Every rented batch has a 60-minute provider billing deadman and a process
supervisor. Independent watchers retain progress and artifacts throughout the
run. Verify required artifact bytes off-machine before early teardown. Reuse
valid completed cells across batches; do not claim full-board completion from
successful selected recipes.
