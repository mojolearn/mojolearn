# Every public estimator on three GPUs (2026-09-13, 46 lanes, 414 cells)

`tools/identity_break.py` at 321ba494 (46 lanes, every public estimator; the 18 lanes added
that day are svr, arima, gp, umap, radius, standard-scaler, minmax-scaler, gbdt-ordered-rmse,
gbdt-feature-freq, mlp, byte-lm, byte-lm-host-infer, byte-lm-host-train, mamba1, mamba2,
mamba3, transformer, samba), nine hostile fixtures, each cell fitted twice, three columns
(train, infer, model). Bindings built from the same native source as 0.8.4 on every box.

| column | box | cells |
|---|---|---|
| apple-m4 | Apple M4, this Mac, Metal (two-core cap, under the 0.8.4 Mac wheel build's load) | 414: stable 413, moved 1, refused 0 |
| nvidia-h100-sm_90a | RunPod H100, sm_90a (`bench/results/e1g/2026-09-13_210408-nvidia-h100-identity-46-lanes`) | 414: stable 387, refused 27 |
| amd-mi325x-gfx942 | DigitalOcean MI325X, gfx942 (`bench/results/e1g/2026-09-13_210415-amd-mi325x-do-identity-46-lanes`) | 414: stable 396, refused 18 |

`diff.apple-nvidia-amd.txt`: `summary: DIVERGENT=8, IDENTICAL=387, MOVED=1, ONE-COLUMN=18` and
`summary (infer/model): DIVERGENT=16, IDENTICAL=423, MOVED=2, N/A=369, ONE-COLUMN=18`.

What is identical. 43 of the 46 lanes on every fixture where every box ran them, including all
28 lanes of the morning's run and 15 of the 18 new ones (svr, arima, gp, umap, radius, both
scalers, gbdt-ordered-rmse, mlp, mamba1, mamba2, mamba3, transformer, samba, and byte-lm on
Apple and AMD).

What diverges. ONE lane, `gbdt-feature-freq` (`ExperimentalTwoLevelFeatureFreq`,
`python/mojolearn/ensemble.py:1632`), differs on 8 of 9 fixtures between every pair of vendors,
and its `base` cell is the single MOVED (two fits in one process on the Mac differed, once, never
reproduced in thirteen later runs). It is the first cross-vendor divergence the tool has found
on a shipped surface; its prior evidence was AMD against NVIDIA only. Diagnosis lane
`lane/feature-freq-divergence`, brief `docs/lanes/BRIEF_feature_freq_divergence_2026-09-13.md`.

What did not run. The two CPU byte-LM lanes refused on both GPU boxes because the leg body did
not build `bindings/build_byte_lm_host.sh` (18 ONE-COLUMN cells; the seven-runner CPU gate
covers that binding). The `byte-lm` GPU lane refused on the H100 because
`bindings/build_byte_lm.sh` failed on that box (9 cells); see the leg's `build_byte_lm.log`.
