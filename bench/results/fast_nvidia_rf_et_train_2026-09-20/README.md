# FAST NVIDIA RF/Extra Trees training screen — 2026-09-20

- Source: `5e33d2db3`
- RunPod: `mvxstbyfc7zubc`, NVIDIA L40S, CUDA target `sm_89`
- Mode: FAST only; IDENTICAL/DETERMINISTIC were not changed.
- Data: R2-pinned Taxi and Istella caches; public Python estimator fit, 1,000,000 training rows.

## Results

Taxi RF (1,000,000 x 16, 100 trees, depth 16) produced model/prediction hash
`dadf56da766bfb3c`, logloss `0.525910`, and AUC `0.617154` in every arm.
After the first cold sample, the baseline median was 908.525 ms and the
two-column histogram arm median was 901.736 ms: only 0.75%, so it is rejected.

Istella RF (1,000,000 x 220) produced hash `491748ce45376578`, logloss
`0.145560`, and AUC `0.964538` in every arm. Absolute timings drifted sharply
(baseline 2633.701 then 1880.069 ms; candidate 2657.123 then 2641.958 ms), so
this comparison is invalid and the candidate is rejected.

The standalone 1,000,000 x 28 column sweep also preserved full model hash
`10220206359389784804`, but its canary spread was `1.104277` and arm ordering
reversed between forward/reverse passes. It is invalid, not a win.

Taxi Extra Trees (1,000,000 x 16, 100 trees, depth 16) took 717.325 ms and
produced hash `9d151e0449a610ab`, logloss `0.527541`, AUC `0.608084`. Its measured
steady core stages were 181 ms stage/sampler, 151 ms score, 164 ms partition,
89 ms range; no exact, general >=10% candidate was identified.

No source change or commit is recommended from this screen. Raw logs and
build logs are retained beside this file.
