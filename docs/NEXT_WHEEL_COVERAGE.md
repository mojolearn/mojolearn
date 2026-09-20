# Next-wheel public surface and verification audit

Audited 2026-09-18 from main `c7442abed` in `lane/next-wheel-coverage`.
The separately frozen 0.8.7 release is `aff968968`; these changes do not alter
its qualified wheel. The 0.8.7 release workflow is run 35350125464.
Publication is not claimed here.

## What the counts mean

- 246 historical appendix algorithm/variant entries: preserved as published.
- 18 registered lanes outside that appendix, listed below.
- 284 public export names, including aliases, constants, state and helpers.
- 229 callable API entries after named non-algorithm exclusions, resolving to
  183 implementation symbols. Neither number is a count of distinct algorithms.
- 228 registered lanes on this main snapshot. The frozen release has 229:
  main removed the explicit `kmeans-cosine` refusal lane, not a working cosine
  implementation.
  - RECOUNTED 2026-09-19: main carries **236**. The 228 above is correct AT ITS
    OWN COMMIT `c7442abed` and is kept so the audit stays reproducible; eight
    lanes landed after it, none removed. They are the kernel-family variants
    `kernel-ridge-laplacian`, `kernel-ridge-poly`, `kernel-ridge-sigmoid`,
    `nystroem-laplacian`, `nystroem-poly`, `nystroem-sigmoid`, and the two
    tokenizer lanes `bpe-vocabulary` and `tokenized-corpus`. Any count here
    ages the moment a lane lands: read it from `identity_break.LANES` by
    import, as `tools/lane_select.py` does, never from this file.

The old scanner missed every API under the nested `models` package. It now
reads package and child-module export lists without importing them, resolves
relative re-exports, and prevents a same-named tokenizer elsewhere from lending
its evidence to `models.Tokenizer`. State/configuration objects remain visible
exports but are not classified as algorithms.

## Actual wheel comparison

Downloaded the published 0.8.5 manylinux wheel from PyPI and verified its SHA-256:
`9c415b87a3cbbb4777cc34093d882e738bab4264a1dba92a95b515b3a41e62b3`.
Compared with the qualified 0.8.7 Linux candidate:
`d20e02f9665c46049dd4655304daae0f8eb897db3d90584295f0485d4c69ce4a`.

| Artifact | Public export names | CPU host bindings | Source export names absent |
|---|---:|---:|---:|
| Published 0.8.5 | 141 | 1 | 143 |
| Candidate 0.8.7 | 284 | 32 | 0 |

The candidate already packages every public name found by this scanner on
current main. Examples absent from 0.8.5 include HDBSCAN, GaussianMixture,
GaussianProcessClassifier, IVFIndex, Cholesky, Embedding, KernelRidge,
Nystroem, RBFSampler, the parallel/pooled/offloaded trainers, low-bit APIs,
CPU block inference and `models.CausalLM`/checkpoint/tokenizer APIs.
Names include aliases; matching exports does not imply matching implementation
bytes, complete parameter support or numerical qualification.

Reproduce the comparison without importing either wheel:

```sh
python3 tools/wheel_api_audit.py published.whl candidate.whl --output wheel-api.json
```

Native-only code is a separate issue: six QN one-target objectives are still
recorded in `glm/NOT_IMPLEMENTED.tsv` as lacking a Python surface, and no public
`SpectralEmbedding` class exists. These are implementation/API/measurement tasks,
not names that should be exported before their contracts and gates exist.
The model loader is packaged, but the matrix now exposes its missing whole-model
identity lane instead of hiding it behind the single name `models`.

## Subsequent tokenizer rename on main

While this audit ran, `5bde47f20` renamed `GPT2Tokenizer` to `BpeTokenizer`,
retaining the old spelling as a deprecated alias. The merged tree now has
286 public names, 231 filtered callable API entries and the same 183
implementation symbols. The frozen candidate lacks the new root
`BpeTokenizer` and `tokenizer.BpeTokenizer` names; it already contains the
tokenizer implementation under the old name. No existing wheel export is
missing from the merged source. This is a naming addition for the following
wheel, not two additional algorithms. The source change's own tokenizer and
BPE training identity/sabotage evidence is in `LANE_STATUS_tokenized-corpus.md`.

The scanner now includes explicitly declared deprecated imports even when
they are outside a submodule's `__all__`, so compatibility aliases cannot be
mistaken for removed capabilities. The latest comparison is retained as
`wheel-export-audit-after-tokenizer.json`. All 49 targeted tests passed after
the concurrent main changes were integrated.

## Reference qualification and pending execution

A reference is an expected digest from a recorded calculation, with source,
binary, input, repeat and property-protocol witnesses. Qualification requires
current fixtures, repeated stable results, appropriate independent device
records, negative controls and installed-wheel replay. A CPU route existing
or matching an old single-vendor digest does not meet that entire contract.

`--include-pending` executes declared CPU routes that are excluded from default
selection. It does not suppress mismatches, invent missing answers, or promote
routes. `OWED` means evidence is missing; `REFUSED` means an operation did not
run; neither is a pass.

The new `--coverage --json` field `reference_support` makes matching numerical
fixture counts visible per property and CPU/Apple/NVIDIA/AMD class. It excludes
stale, conflicted, superseded/disagreeing and N/A values from numerical counts.

### 17 ordinary routes awaiting qualification

`gbdt-query-rmse`, `gmm-random-init-sample`, `gmm-sample`, `gp-normalize-y`, `gp-optimize`, `gp-optimize-restarts`, `gp-sample-y`, `gp-sample-y-normalize`, `gpc`, `gpc-multiclass`, `ivf-extend`, `mamba3`, `samba`, `samba-untied-dropout-accum`, `svc-poly`, `transformer`, `transformer-window`.

Five neural routes (`mamba3`, `transformer`, `transformer-window`, `samba`,
`samba-untied-dropout-accum`) have fresh matching Apple/NVIDIA/AMD train,
inference, model-where-applicable, batch and sampler/replay records.
A real scoped-admission attempt against current main still refuses all five:
`batchgrad`, `batchscale`, `ragged` and `stepfull` records are missing.
The narrower frozen 0.8.7 table lacks only `stepfull`; do not mistake admission
against that older table for preserving all of current main's properties.

The other twelve need the missing NVIDIA/AMD witnesses plus complete property
records and replay, as their `PUBLIC_REFERENCE_CANDIDATES` comments specify.
Do not remove the holds until those conditions actually pass.

Capture all required properties together on the actual candidate artifacts:
`--repeats 2`, with all nine fixtures, an explicit backend, and failure on
refused stages. Since 2026-09-20 every part is collected by default, so the
four old flags (`--batch-grad`, `--batch-scale`, `--ragged`, `--step-full`)
are accepted and inert and nothing has to be remembered; `--partial-column`
is the only way to collect less, and a column that used it is stamped
`partial_column` and refused as a record. Build scoped references without dropping an
existing property; compare vendor classes and replay the resulting candidate
on CPU before promoting the lane. Repeat final installed-wheel qualification
after the table and public selection change.

### 17 CPU logical-shard drivers

`par-arima`, `par-forest`, `par-forest-et`, `par-forest-et-clf`, `par-forest-reg`, `par-holtwinters`, `par-mlp`, `par-queries-kde`, `par-queries-knn`, `par-queries-nn`, `par-queries-radius`, `par-reference-knn`, `par-reference-knn-reg`, `par-samba`, `par-samba-clip`, `par-scaler`, `par-scaler-minmax`.

These execute Python shard splitting and ordered assembly using native CPU
operations. They need current complete references and installed replays for
default admission. They do not prove inter-GPU transfers or synchronization.

### 33 parallel drivers requiring GPU execution

`par-boosting`, `par-boosting-clf`, `par-boosting-pointwise`, `par-boosting-reg`, `par-byte-lm`, `par-byte-lm-model-pool`, `par-byte-lm-offload`, `par-cd`, `par-cd-elasticnet`, `par-cholesky`, `par-dbscan`, `par-feature-freq`, `par-forest-pool`, `par-gmm`, `par-gp`, `par-gram`, `par-gram-ols`, `par-gram-pca`, `par-gram-tsvd`, `par-graph-agglomerative`, `par-graph-spectral`, `par-graph-umap`, `par-hdbscan`, `par-iforest`, `par-kernel-ridge`, `par-kmeans`, `par-logistic`, `par-nystroem`, `par-ordered-rmse`, `par-rbf-sampler`, `par-resample`, `par-svm`, `par-svm-svr`.

Most use cooperative operations whose partitioning lives inside GPU bindings;
others require worker/model-pool/offload operations without a declared CPU
route. `_parallel_pool.py` deliberately refuses them. Admitting their operation
names alone, or running a serial fit and calling it parallel, would miss the
very partition/assembly behavior the CPU verifier needs to check.

Two independent pieces of work remain: implement CPU equivalents of those
partition/reduction protocols, and qualify physical multi-GPU behavior on real
hardware. For the latter, collect one-device and two-device records from the
same wheel and commit, verify actual device identities and worker placement,
then compare every applicable cell/property. `MOJOLEARN_PAR_DEVICES=0,1` alone
is not proof that two physical GPUs executed. Do this for NVIDIA and AMD;
a one-GPU Apple recording cannot substitute for either multi-GPU measurement.

## Registered additions outside the appendix

`gemm-bf16`, `gemm-int8`, `gp-optimize`, `gp-optimize-restarts`, `mamba1-bf16w`, `mamba1-int8w`, `mamba2-bf16w`, `mamba2-int8w`, `mamba3-bf16w`, `mamba3-int8w`, `metrics-homogeneity-completeness`, `mlp-bf16w`, `mlp-int8w`, `ordered-gradient-sum`, `samba-bf16w`, `samba-int8w`, `transformer-bf16w`, `transformer-int8w`.

## Evidence and validation

External evidence is under `~/mojolearn-evidence/next-wheel-coverage/`:
`wheel-export-audit.json`, `pypi-metadata.json`, `source-matrix.json` and
`pending-reference-admission.json`. The latter retains the actual scoped
admission refusals, not a proposed approval.

The source scanner, wheel comparison and reference-count regressions passed
48 targeted tests. The first test invocation lacked native bindings and failed
during import; the successful run used the existing host-binding directory
only to import the package. These tests establish inventory/reporting behavior,
not new numerical qualification. No pending route was promoted by this audit.

## Row-sharded RBF sampler CPU route (next-wheel follow-up)

`par-rbf-sampler` now has a declared CPU logical-shard route. Its existing
`transform_rbf_sampler` driver splits query rows in Python, sends each shard
with identical fitted random weights and offsets, and rejoins results in input
order. The kernel-methods host binding already implements that per-shard
transform; the missing piece was admitting `rbf_sampler_rows` in the
non-cooperative CPU worker pool. Cooperative kernel-method operations remain
outside this route.

The installed verifier can select it explicitly with
`verify --include-pending --lanes par-rbf-sampler`. Default public selection
is unchanged. On this newer main snapshot the implementation inventory becomes
18 CPU logical-shard drivers and 32 parallel drivers requiring GPU execution;
the earlier 17/33 inventory above describes the prior audit snapshot. This is
an execution-route addition, not new physical multi-GPU certification or a
claim that the frozen 0.8.7 wheel contains this change.

Regression coverage checks pending selection, cooperative refusal, row order,
uneven and single-row shards, retained model bytes, and a deliberately reordered
worker response detected by the identity comparator. Numerical execution is
queued behind the release build; no fresh numerical qualification is claimed
until that queued run completes. The existing manifest, coverage and export
checks passed (209 tests across targeted invocations).
