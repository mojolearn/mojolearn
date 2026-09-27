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
| 20 | GARCH(p, o, q), constant or zero mean, normal (arch package statement; Nelder-Mead instead of SLSQP) | sequence-garch | (the commit adding this row) | CLEAN: sequence-garch: AGREE: compared train 9 | arch 8.0 on 3 simulated series (n=1000): GARCH(1,1) params within 1e-3, loglik within 0.004, 5-step variance forecast <= 1.3e-3 relative; GJR zero-mean loglik within 0.03 |

Pod setup note: the ARIMA path needs `python/mojolearn/.libs/libMojolearnMath.so`; build it on a fresh pod with `packaging/portable_math/stage.py`'s `build()`.

Main table done. Next: the Additions in order (Prophet-style, MoE block).
