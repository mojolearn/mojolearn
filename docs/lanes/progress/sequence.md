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
| 5 | AutoARIMA (cuML auto_arima.pyx search over mojolearn.ARIMA + select_d) | sequence-autoarima | (the commit adding this row) | CLEAN: sequence-autoarima: AGREE: compared train 9 (runs _mojolearn_arima, _mojolearn_tsa and their hosts) | 4 simulated series (AR1, MA1, random walk, ARMA(2,1)), p,q in 0..2, aic: same orders and AICs as statsmodels ARIMA (to 3e-3), forecasts to 1e-3 |

Pod setup note: the ARIMA path needs `python/mojolearn/.libs/libMojolearnMath.so`; build it on a fresh pod with `packaging/portable_math/stage.py`'s `build()`.

Next:
AutoARIMA, STL, VAR, then the Additions in order.
