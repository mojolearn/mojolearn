# GEMM outcomes behind inline candidate comments — 2026-10-04

Read-only inspection of actual M3 files under `~/mq/out/`; no new run.
Numeric status below comes from metrics/capture reports, not process exit codes.
All candidates discussed remain default OFF. No board timing cell changes.

## Separate shared G1/G5 PCA caller route

| Source / tag | Actual call A -> B ms | Numerical result | Decision |
|---|---:|---|---|
| G1 `30e4562c`, `g1-pca-tall-transform-t-v1` | 32.633750 -> 29.887000 | PASS, independent output errors equal | Narrow caller lead; no broad default/board admission |
| G1 `30e4562c`, `g1-pca-tall-inverse-t-v1` | 34.177417 -> 36.050709 | PASS, independent output errors equal | Slower in this one sample; do not promote inverse |
| G5 `495c30c3`, `g5-pca-tall-transform-t-v1` | 32.698166 -> 30.608875 | PASS, independent output errors equal | Narrow caller lead; no broad default/board admission |
| G5 `495c30c3`, `g5-pca-tall-inverse-t-v1` | 35.538000 -> 36.727542 | PASS, independent output errors equal | Slower in this one sample; do not promote inverse |

These are frozen A-fitted PCA models: training513x220,64components, query32769.
Diagnostic counters verify core NT reach. One cold call plus full first output
copy, no warmup; all reports explicitly say `board_evidence:false`. No variance
estimate or threshold invented from one sample. A report status PASS here is
numerical PASS, not automatically a speed winner. No fit/optimizer change.
Shared G1/G5 code is preserved on historical candidate branches; it was not
restored to main merely to add comments. Current scoped AFN code is different.

Report SHA256s (same row order, `TAG-timing/report.json`):

- `06a926a52cd77061c6a207dd863886b24b7bd81e0892bf671ee3c23c2e91afdb`
- `8f2e1c181688aa5b326b5fb04810f6053c95f4c3ed3efcb38e042fc1ceb45eed`
- `1dc7e616ff9f780cb93237c1cecbbaf2c27ecd2fec2fb4ebf8550198f8b239cc`
- `f39f98aa92e9e92c7313ffb29fe5cc7bcee1fd5ec84037ff5e2087454de237e0`

## Scoped AFN mechanism and actual PCA fit

`scoped-r2-all-q-v1-quality/report.json` is valid JSON, status PASS,
source28f06e19/harness5776a5d1, all28cases and no failures. Bound5e-6 for
relative Frobenius error; ZERO allowance for relative and max-absolute
regression. Includes G1 tall/dense/Gram, G2 narrow and split/PCA lifecycle
controls. It does not provide fitted-estimator or timing admission.
Report hash `23f525d9279ed2a943cb5836ccee077e94ee2358771ef50328d28040010d113a`.

`scoped-pca-fit-istella-q-v1-quality/report.json` is truncated INVALID JSON.
Its preserved fields explicitly say HOLD and show the following failures:

- singular relative error:1.964589033302299e-5 ->1.9646424561163682e-5;
- noise relative error:.20861880621005235 ->.20864345966060238;
- noise max-absolute error:438.73138701320477 ->438.78323393319715.

Compiled source201fe736; bound5e-6 and zero degradation allowance. Some other
fields improve, which does not cancel these failures. Original partial report
hash `a16202281b80b3c55117e76ccd30b3e6c9f449e8a66c47d1db65c85cf5cc8458`.
No fabricated complete verdict or numerical PASS; no fit timing admitted.
Manager reconstruction helper4c69e438 exists to rejudge preserved captures,
not replay GPU work. Complete serialization remains owed at this audit.

## Resident catalog and new softmax caller

`resident-catalog-t-v1-timing/report.json`, sourcefa390736, has66records;
selected G1/G2 leads match `RESIDENT_GEMM_RESULTS_2026-10-04.md`. Its report
hash is `4bcd5379070cd0c8dd4c3b231053837fa0a9340e12fe20dab95069b346e130a9`.
G1 tall6.826875->3.524209ms and G2 narrow1.203750->.865625ms compare against
SDK resident-input calls, not the decomp AFN incumbent. Narrow G2 does not
prove KMeans/softmax or arbitrary widths improve. G5 dense/square losses and
G1/G2 square losses prevent any universal-winner inference.

Softmax G2 source4bfc1424a is SOURCE-READY/UNBUILT, with no actual softmax
quality/timing result. Its algorithm flag remains opt-in. The shared-kernel
mechanism PASS does not certify optimizer results. Actual reach, full quality
cases and M2 builds remain owed; see `ab/softmax-g2-narrow.md`.
