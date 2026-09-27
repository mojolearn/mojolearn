# sequence: progress

Pass 1 (code first). Pod: dev_pod `sequence`, NVIDIA A40. Gate per algorithm:
builds, sanity vs reference, `tools/algos_lane_check.sh <lane>` AGREE (CPU == NVIDIA).

Shared machinery (first commit): `sequence/` — `ops.mojo` (one scalar body per
output element, IDENTICAL seams), `exec.mojo` (`Exec` trait, `HostExec`),
`exec_device.mojo` (`DeviceExec`, one GPU thread per element), `recurrent.mojo`
(RNN/LSTM/GRU BPTT + optimizers), `pyapi.mojo` (the address contract both
bindings export). Python: `python/mojolearn/_x_sequence_rnn.py`.

| # | algorithm | lane | commit | AGREE line | sanity |
|---|---|---|---|---|---|
| 1 | LSTM (LSTMRegressor, LSTMClassifier) | sequence-lstm | bb6a93893 | CLEAN: sequence-lstm: AGREE: compared infer 9, train 9 (cuda column vs CPU column) | torch nn.LSTM, 5 full-batch steps, float64 torch: max param diff SGD 4e-8, Adam(2 layers) 1e-6, CE Adam 8e-7, RMSprop(centered, momentum) 2e-5, Adagrad 2e-7, AdamW 1e-6 |
| 2 | GRU (GRURegressor, GRUClassifier) | sequence-gru | bbd4a08e0 | CLEAN: sequence-gru: AGREE: compared infer 9, train 9 | torch nn.GRU: SGD 5e-8, Adam 2 layers 1e-6, CE Adam 8e-7, RMSprop centered 2e-5 |
| 3 | RMSprop (torch.optim-shaped, in-place float32 params) | sequence-rmsprop | 3b3648bc1 | CLEAN: sequence-rmsprop: AGREE: compared train 9 | torch.optim.RMSprop float64, 8 steps: default 2.5e-7, centered+momentum+wd 5.1e-7, alpha 0.9 3.7e-7 |
| 4 | Adagrad (torch.optim-shaped) | sequence-adagrad | c7ff7aee1 | CLEAN: sequence-adagrad: AGREE: compared train 9 | torch.optim.Adagrad float64, 8 steps: default 1.7e-7, lr_decay+wd+init acc 1.3e-7 |
| 5 | AutoARIMA (cuML auto_arima.pyx search over mojolearn.ARIMA + select_d) | sequence-autoarima | e47ca54e3 | CLEAN: sequence-autoarima: AGREE: compared train 9 (runs _mojolearn_arima, _mojolearn_tsa and their hosts) | 4 simulated series (AR1, MA1, random walk, ARMA(2,1)), p,q in 0..2, aic: same orders and AICs as statsmodels ARIMA (to 3e-3), forecasts to 1e-3 |
| 6 | STL (statsmodels STL, batched, one series per GPU thread) | sequence-stl | d5cbc28c3 | CLEAN: sequence-stl: AGREE: compared train 9 | statsmodels STL float64 on 150 obs with outliers: default 3e-5, robust 4e-5, deg-0 + jumps 8e-6, period 7 robust deg 0 1e-6 (season, trend, weights) |
| 7 | VAR (statsmodels VAR, OLS per equation, fixed lag) | sequence-var | 812d79dff | CLEAN: sequence-var: AGREE: compared train 9 | statsmodels VAR on a simulated 3-variable VAR(2): params 9e-6, sigma_u 7e-7, 5-step forecast 2e-5 (p=1,2 with constant, p=3 without) |
| 8 | MLPClassifier / MLPRegressor (sklearn-shaped, own trainer) | sequence-mlp | 70349dafc | CLEAN: sequence-mlp: AGREE: compared infer 9, train 9 | sklearn MLP (shuffle=False, same random_state): loss curves and predictions within 3e-6 for Adam+tanh+L2 regressor, relu 3-class, logistic binary, SGD invscaling, SGD adaptive (both stop at epoch 57) |
| 9 | RNN (RNNRegressor, RNNClassifier; tanh / relu) | sequence-rnn | 8035fcbbf | CLEAN: sequence-rnn: AGREE: compared infer 9, train 9 | torch nn.RNN float64, 5 steps: tanh SGD 5e-8, relu Adam 2 layers 2e-6, tanh CE Adam 8e-7, relu CE AdamW 2 layers 1e-6 |
| 10 | Lion (torch.optim-shaped; also `optimizer="lion"` in the recurrent estimators) | sequence-lion | d1f97820a | CLEAN: sequence-lion: AGREE: compared train 9 | float64 restatement of lion-pytorch's step, 6 steps, defaults and betas (0.95, 0.98) + wd 0.1: within rtol 2e-5 (tests/test_x_sequence_optim.py, GPU and CPU) |
| 11 | Adafactor (torch 2.5 rule; 1-D and 2-D tensors) | sequence-adafactor | cc6ca19b8 | CLEAN: sequence-adafactor: AGREE: compared train 9 | torch.optim.Adafactor (2.5.1, float32), 10 steps on a 6x5 matrix + a vector: defaults 7e-9, lr/beta2_decay/d/wd moved 1.2e-7 |
| 12 | LAMB (timm Lamb statement: global clip, trust ratio, trust_clip, always_adapt) | sequence-lamb | 7f722502c | CLEAN: sequence-lamb: AGREE: compared train 9 | timm.optim.Lamb (float32), 8 steps: defaults 1.2e-7, no-decay always_adapt trust_clip no-clip 3.6e-7, wd 0.1 clip 0.5 1.2e-7 |
| 13 | Adamax (torch.optim-shaped; also `optimizer="adamax"` in the recurrent estimators) | sequence-adamax | 315b89a64 | CLEAN: sequence-adamax: AGREE: compared train 9 (and the six optimizer-using lanes re-AGREE after the host-scalar refactor) | torch.optim.Adamax float64, 8 steps: defaults 2.4e-7, betas/wd moved 2.9e-7 |
| 14 | NAdam (torch.optim-shaped; also `optimizer="nadam"`) | sequence-nadam | d47e0a82c | CLEAN: sequence-nadam: AGREE: compared train 9 | torch.optim.NAdam float64, 8 steps: defaults 5.2e-7, decoupled wd + momentum_decay 0.01 6.8e-7 |
| 15 | LR schedulers StepLR, ExponentialLR, OneCycleLR (exact rational, float32 once; `lr_schedule=` on the lane's optimizers and recurrent estimators) | sequence-lr-schedulers | 0e01f3930 | CLEAN: sequence-lr-schedulers: AGREE: compared train 9 | torch.optim.lr_scheduler (float64), 40 steps: StepLR 1.5e-8, ExponentialLR 5.3e-8, OneCycle cos 4.8e-8, linear three-phase 5.4e-8 relative |
| 16 | LayerNorm (module + layer_norm_forward / layer_norm_backward) | sequence-layernorm | 7b732166d | CLEAN: sequence-layernorm: AGREE: compared infer 9, train 9 | torch F.layer_norm float64 with weight and bias: y 3.5e-7, dx 2.1e-7, dw 8.7e-6, db 3.0e-6; (2, 5) normalized shape 2.8e-7 |
| 17 | Theta forecasters (Theta, OptimizedTheta, DynamicTheta, DynamicOptimizedTheta, AutoTheta; statsforecast) with statsforecast's Nelder-Mead (`sequence/nm.mojo`, reusable) | sequence-theta | 462d18391 | CLEAN: sequence-theta: AGREE: compared train 9 | statsforecast 2.1.1 on 4 series (seasonal, random walk, trend, additive seasonal), 12-step forecasts: max relative diff 7.5e-7 except OTM/DOTM on the trend series 6.7e-5 / 1.7e-3 (float32 Nelder-Mead path) |
| 18 | Croston (CrostonClassic, CrostonOptimized, CrostonSBA; statsforecast) | sequence-croston | 5612c9178 | CLEAN: sequence-croston: AGREE: compared train 9 | statsforecast 2.1.1 on 6 intermittent series (one all-zero, one with a negative event): classic 4.6e-8, SBA 6.7e-8, optimized 2.1e-4 relative (float32 golden section) |
| 19 | Damped-trend ETS (ETS for ANN/AAN/MNN/MAN and damped, DampedETS; statsforecast) | sequence-ets | 6b0a36994 | CLEAN: sequence-ets: AGREE: compared train 9 | statsforecast AutoETS on 4 series, 10-step forecasts: AAdN <= 8.6e-4, MAdN <= 1.2e-3, AAN <= 6.1e-4, ANN <= 1.8e-6, MNN <= 3.7e-5 relative |
| 20 | GARCH(p, o, q), constant or zero mean, normal (arch package statement; Nelder-Mead instead of SLSQP) | sequence-garch | 5baf7a3f3 | CLEAN: sequence-garch: AGREE: compared train 9 | arch 8.0 on 3 simulated series (n=1000): GARCH(1,1) params within 1e-3, loglik within 0.004, 5-step variance forecast <= 1.3e-3 relative; GJR zero-mean loglik within 0.03 |
| 21 | Prophet-style forecaster (ProphetForecaster: linear trend with changepoints, Fourier seasonality, holiday regressors, additive / multiplicative, MAP by our L-BFGS) | sequence-prophet | e63e27fa8 | CLEAN: sequence-prophet: AGREE: compared train 9 | prophet 1.4.0 (Stan) on 800 daily points with a trend break, weekly + yearly seasonality and a holiday regressor: max diff / max y 2.2e-4 history, 3.2e-4 30-day future (additive); 9.3e-4 / 4.2e-4 (multiplicative) |
| 22 | Mixture-of-experts block (MoEBlock, Mixtral sparse block forward; top-k with the lower-index tie rule) | sequence-moe | (the commit adding this row) | CLEAN: sequence-moe: AGREE: compared infer 9, train 9 | float64 torch restatement of MixtralSparseMoeBlock (HF layout) on 50 tokens, 8 experts, top 2: max diff 9.6e-10 (max y 3.3e-3), same experts, logits 1.2e-6 |

Pod setup note: the ARIMA path needs `python/mojolearn/.libs/libMojolearnMath.so`; build it on a fresh pod with `packaging/portable_math/stage.py`'s `build()`.

PASS 1 COMPLETE (main table and every Addition merged).

Merge-gate notes: test_host_surface passes; tools/test_lane_select.py has one
failure that is not this lane's (core/forest_host_predict.mojo answers 81
lanes, not 80: all trees/gbdt lanes, none sequence-*), as of the Prophet merge.

## PASS 2 (proof)

Seams: 17, DEVIATIONS 5500-5516 (`sequence/README.md`), IDENTITY_PATHS rows
150-159. Host oracles `sequence/checks/oracle.mojo` (from the reference
semantics, each with its `alt` spelling); driver `sequence/checks/seams_check.mojo`
(fixture must separate pinned vs alt, then device == oracle and host ==
oracle, stage `sequence.<seam>` on the card). Arms: `sequence/checks/sabotage/seam_55xx_*.patch`,
listed in `tools/identity_lanes/sequence.checks`. `DeviceExec` now takes ONE
process-lifetime DeviceContext (`exec_device.mojo::sequence_ctx`, `_Global`
per numeric tier); torch.lerp is one body (`ops.mojo::lerp`) for Adamax,
NAdam and Adafactor.

- NVIDIA A40 (fixed lane check, after 02b63f107), `--pass 2` over all 22
  lanes: every one of the 17 arms PASS / FAIL (exit 1) / PASS after reversal;
  all 22 lanes CLEAN AGREE (CPU == cuda). DONE, never re-run.
- End-to-end CPU-column sabotage for the stewards:
  `sequence/checks/sabotage/e2e_host_download_bit.patch` (21 lanes: HostExec
  download flips the low bit) and `e2e_arima_host_param_bit.patch`
  (sequence-autoarima: the ARIMA host binding's params). Both proven on the
  A40: AGREE, DISAGREE under the patch on every lane, AGREE after reversal.
- Merge gate on the A40: test_host_surface 196 passed; tools/test_lane_select.py OK, 0 failures.
- Steward requests at d5a849bb5 (m2pro + do-amd; m3ultra spooled):
  1790537359123-sequence-d5a849bb57 (21 lanes), 1790537368020-sequence-d5a849bb57
  (autoarima). do-amd PASS on both.

### The family's earlier algorithms (LANE CHARTER: ARIMA, ExponentialSmoothing, KPSS)

Their own drivers and oracles (`arima/checks/arima_check.mojo`,
`fit_check.mojo`, `holtwinters/checks/hw_check.mojo`,
`tsa/checks/stationarity_check.mojo`) are now listed in
`tools/identity_lanes/sequence.checks` with 15 source arms 5520-5535
(`sequence/README.md` maps each to its DEVIATION / SEAMS.tsv row).
Unarmable seams, written down there: `F = Z P Z'` and the Jones inverse's
`fma(sign, prod, x)` (no separating spelling).

- NVIDIA A40, `--pass 2` over arima, arima-011, arima-seasonal-c, arima-exog,
  arima-exog-seasonal, holtwinters, holtwinters-multiplicative, kpss,
  sequence-autoarima: all 32 arms of sequence.checks PASS / FAIL / PASS;
  all 9 lanes CLEAN AGREE. DONE, never re-run.
- E2E steward sabotages: `e2e_arima_host_param_bit.patch` (the five arima
  lanes), `e2e_tsa_host_bits.patch` (holtwinters x2, kpss).

### Apple build fix (m2pro FAIL on d5a849bb5, both requests: the Metal build)

`seq_kernel` had 33 arguments (Metal binds each to a buffer slot, 31 max):
the 12 ints and 8 floats now travel packed two per Int64 (bit-exact, ints
checked to fit Int32). Theta, ETS and GARCH then segfaulted Apple's air-lld
(`LazyLinker::LinkDefinition`): their helpers (and nm.mojo's) are now
`@always_inline`. Bisected on the M2 Pro (`~/seqdbg`, compile only): all 55
operations and seams_check build for Metal. A40 re-check after the fix
(e71c8f66a): all 32 arms bite, all 22 sequence lanes AGREE. DONE.
One batched steward request per hour covers the family; its end-to-end
sabotage is `sequence/checks/sabotage/e2e_family_host_bits.patch` (the three
e2e patches in one).

Batched steward request (all 30 family lanes, e2e_family_host_bits.patch) at
3607aa10e: 1790537359124-sequence-3607aa10ee on m2pro, m3ultra, m4pro-a, do-amd.

NEXT (a fresh session starts here), per the LANE CHARTER (one phase per session):
1. Read `python3 tools/apple_steward.py status | grep sequence-3607aa10ee`.
   PASS on every Mac and do-amd closes PHASE 1. A FAIL is a fix commit at
   the root (fetch the log with `tools/cloudmac.sh ssh <mac> ...` from
   ~/mojolearn-evidence/apple-steward/done/<id>/check/lane_check.log; the
   M2 Pro has a compile-only scratch tree ~/seqdbg with bisect.sh/probe.sh),
   re-proven on the A40, merged, one batched resubmission.
2. PHASE 2, option parity, whole family including ARIMA, ExponentialSmoothing
   and KPSS: seasonal ETS FIRST (the bench race needs it), MoE backward,
   forecaster prediction intervals, then every NOT IMPLEMENTED row of
   sequence/, arima/, holtwinters/, tsa/ NOT_IMPLEMENTED.tsv. Each option:
   AGREE, a sabotage for a numeric change, existing bits unchanged; merge each.
3. Then phase 3 FAST speed, 4 IDENTICAL speed, 5 CPU speed.
