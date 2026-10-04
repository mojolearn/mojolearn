# Apple FAST candidate recovery and queue checkpoint

Manager checkpoint, 2026-10-04. M2 compiles; the existing M3 serial runner
executes quality and timing. No new machines or runners were created.

## Submitted measurements

The following four jobs passed metadata/artifact/quality-receipt preflight and
were inserted after queue position 1571. The admission window was released.

- `g1-pca-tall-transform-t-v1`
- `g1-pca-tall-inverse-t-v1`
- `g5-pca-tall-transform-t-v1`
- `g5-pca-tall-inverse-t-v1`

G1 harness `0849df9ac1ed518ffa7577ae5cf2bdc72579052a` uses compiled source
`30e4562c2129569ed03d93d878ec6a903ea51691`; G5 harness
`768632c271f544d188430fc16803751eded4cfff` uses compiled source
`495c30c33a8805a1944b45f9bc911446f7e89ed8`.
These are scoped actual-caller measurements with diagnostic counters, not
PCA-fit board measurements or permission to promote a broad GEMM default.

## Recovered branch leads

| Candidate | Evidence | Next action |
| --- | --- | --- |
| GPU resample gather, `origin/lane/apple-fast-resample@50b96e795`, `MOJOLEARN_RESAMPLE_FAST_GATHER` | Absent current main. Historical `resample-rs-gather-taxi` failed Mojo parsing and produced no timing. This is a device gather, distinct from held host `RESAMPLE_FAST_ROW_GATHER`. | Recover minimal device implementation on current main in a private branch; preserve current index generation, empty/zero-width cases and fallback contracts. Exact output/lifetime quality before a fresh timing tag. |
| SVGP split reductions, `lane/apple-fast-w2-svgp@146898d2c`, `MOJOLEARN_SVGP_FAST_BSPLIT` | Kernel already present opt-in on main. No matching BSPLIT entry found in retained M3 queue/results. | Readiness review committed as `b1487e2e278ab5eb5349fd493d1c3e8ce602670a`; verified prerequisites and pinned submission integration still owed. Lower priority: SVGP already beats its opponent. |
| Scoped decomp/PCA GEMM, `28f06e1923cfa164fd068c31b0f986ff0126d306` | Compile repair for prior failed probe; no runtime acceptance. | M2 A/B compilation started; source/harness readiness review in parallel. Quality before caller timing. |

The separate `experiment/apple-kernel-lanes-20261004` catalog is already
tracked in `CONSOLIDATED_CANDIDATE_QUEUE_2026-10-04.md`; do not reimport or
rerun its completed experiments. Old branches and OPEN ledger text alone
are not proof of missing measurements. In particular, SHAP pipe, minibatch
label regrouping, LU double buffering and the host resample gather already
have outcomes; VAR fused remains parked pending a parallel implementation.

The wider request audit inspected 274 local candidate branches and found
624 unique historical request lines. 151 tags were absent from the retained
queue/results snapshot, predominantly neural work outside this classical/tree
board effort. These are audit leads, not queue-ready jobs. Raw audit files
are retained under `~/mojolearn-evidence/apple-fast/branch-audit-20261004/`.

No recovered kernel has been merged into main or enabled by this checkpoint.
Board times change only after accepted application-level evidence.

## First harvested results

All four admitted PCA caller timings completed and passed the unchanged output
quality gate. Cold call plus first full output copy, one sample per arm:

| Candidate / operation | A ms | B ms | Decision |
| --- | ---: | ---: | --- |
| G1 PCA transform | 32.633750 | 29.887000 | Measured lead; no board/default admission |
| G1 PCA inverse | 34.177417 | 36.050709 | Slower; retain incumbent |
| G5 PCA transform | 32.698166 | 30.608875 | Measured lead; no board/default admission |
| G5 PCA inverse | 35.538000 | 36.727542 | Slower; retain incumbent |

All pairs have equal A/B independent-oracle error metrics. Diagnostic counters
are enabled. These cold caller samples do not establish a noise distribution or
validate broad shape eligibility. Preserve the existing board fit timings.

Compensated Kalman reference `arima-k3-df-reference-v1` passed its pinned
preflight and was queued after position1578; admission window released. This
is a reference-only accuracy investigation, not a GPU performance claim.
Scoped GEMM r2 `28f06e1923cfa164fd068c31b0f986ff0126d306` compiled both
A and all-profile B on M2 successfully; M3 quality admission is next.
