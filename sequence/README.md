# sequence: the neural sequence, optimizer and time-series lane

RNN, LSTM, GRU, MLP, LayerNorm, the mixture-of-experts block; the optimizers
(RMSprop, Adagrad, Lion, Adafactor, LAMB, Adamax, NAdam and the LR schedulers);
AutoARIMA, STL, VAR, Theta, Croston, damped ETS, GARCH and the Prophet-style
forecaster (`python/mojolearn/_x_sequence_*.py`).

## How it is built

Every floating-point operation is one element body in `ops.mojo` (and the
per-family files it dispatches to): the work of one output cell, one row or
one whole series. `exec.mojo::HostExec` loops over those bodies on the CPU;
`exec_device.mojo::DeviceExec` runs the same bodies one GPU thread per
element, on ONE process-lifetime DeviceContext. Both bindings export the
address contract of `pyapi.mojo`, so the two columns run the same statements.

## Seams (IDENTITY_PATHS.md rows 150-159)

Each seam's host oracle is in `checks/oracle.mojo`, written from the reference
semantics, not from this directory; `checks/seams_check.mojo` requires the
fixture to separate the pinned spelling from the alternative, then device ==
oracle and host == oracle bit for bit. One sabotage arm per seam,
`checks/sabotage/seam_55xx_*.patch`, listed in `tools/identity_lanes/sequence.checks`.

| DEVIATION | seam | move (pinned) | alternative the arm writes |
|---|---|---|---|
| 5500 | GEMM (`op_gemm`) | k ascending, one fma per term | k descending |
| 5501 | column sums (`op_colsum`) | rows ascending | rows descending |
| 5502 | LSTM cell state | `fma(f, c_prev, i*g)` | `fma(i, g, f*c_prev)` |
| 5503 | GRU update | `fma(z, h_prev - n, n)` | `(1 - z) n + z h_prev` |
| 5504 | BPTT dW (`recurrent.mojo::backward`) | one fold over (step, batch) rows ascending after the sweep | per step, steps descending |
| 5505 | softmax cross entropy | max first, exp-sum classes ascending | classes descending |
| 5506 | Adam denominator | `sqrt(v) / sqrt(1 - b2^t) + eps` (torch) | `sqrt(v / (1 - b2^t)) + eps` |
| 5507 | torch.lerp (Adamax, NAdam, Adafactor) | two branches at w = 0.5 | `s + w (e - s)` always |
| 5508 | MLP log loss | p clipped to [eps32, 1 - eps32] (sklearn) | no clip |
| 5509 | MLP epoch shuffle | splitmix64 Fisher-Yates, i descending, j by rejection | i ascending |
| 5510 | STL robustness weights | `3 (r[m0] + r[m1])` (statsmodels) | `3 r[m0] + 3 r[m1]` |
| 5511 | VAR column scaling | an exact power of two | `1 / max|x|` |
| 5512 | VAR Cholesky solve | inner sums k ascending; first bad pivot's column as status | k descending |
| 5513 | Nelder-Mead simplex order | stable, ties keep the lower index | ties to the higher index |
| 5514 | MoE top-k routing | strict `>`, ties to the lower expert | ties to the higher expert |
| 5515 | LayerNorm statistics | mean columns ascending, then centred squares | mean columns descending |
| 5516 | SES recursion (Croston) | `alpha x + (1 - alpha) f`, one fma | `f + alpha (x - f)` |
