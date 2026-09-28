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

Two Apple constraints shape the device side (found on the M2 Pro steward,
2026-09-27): `seq_kernel` passes its twelve integers and eight floats
packed two to an Int64 word (bit-exact; integers checked to fit Int32),
because Metal binds each kernel argument to its own buffer slot and has 31;
and every helper the Nelder-Mead forecasters (Theta, ETS, GARCH) call is
`@always_inline`, because Apple's `air-lld` segfaults
(`LazyLinker::LinkDefinition`) linking those as separate functions.

## Seams (IDENTITY_PATHS.md rows 150-159 and 199; 5517-5518 in row 159, 5536-5540 in row 154, 5541-5543 in row 199)

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
| 5517 | ETS seasonal update (`ets.mojo::seas_update`) | `old_s + gamma (t - old_s)`, one fma (statsforecast Update) | `(1 - gamma) old_s + gamma t` |
| 5518 | ETS initial seasons: the decomposition's centred moving average (`ets.mojo::_moving_average`) | taps ascending, half-weight ends as `0.5 x`, one division by m | every tap times its weight `w / m` |
| 5536 | RMSprop centered variance (`op_opt` OPT_RMSPROP) | `v - ga*ga`, two roundings (torch's `addcmul(ga, ga, value=-1)`), a negative difference taken as 0 | `fma(-ga, ga, v)` |
| 5537 | Adagrad accumulator (`op_opt` OPT_ADAGRAD) | `fma(g, g, sum)` | `sum + g*g` (product rounded) |
| 5538 | Lion momentum (`op_opt` OPT_LION) | `fma(b2, m, (1 - b2) g)` | `b2 m + (1 - b2) g` (both products rounded) |
| 5539 | LAMB trust ratio (`adafactor.mojo::op_lamb_ratio`) | `sqrt(sum p^2) / sqrt(sum u^2)` (timm's two norms) | `sqrt(sum p^2 / sum u^2)` |
| 5540 | LR schedulers (`python/mojolearn/_x_sequence_sched.py`; driver `checks/sched_check.py`) | exact rational, ONE rounding to float32 | the float64 closed form, then rounded (double rounding at a float32 midpoint) |
| 5541 | Theta level (`theta.mojo::theta_run`) | `alpha y + (1 - alpha) level`, one fma | `level + alpha (y - level)` |
| 5542 | GARCH variance recursion (`garch.mojo::garch_sigma2`) | omega, alpha, gamma, beta terms in order, each one fma | the ARCH term's product rounded, then the add |
| 5543 | Prophet Fourier argument (`prophet.mojo::op_prophet_features`) | `(2 pi i) frac` | `2 pi (i frac)` |

## ARIMA, ExponentialSmoothing (Holt-Winters) and KPSS

These are the family's earlier algorithms (`arima/`, `holtwinters/`, `tsa/`),
with their own DEVIATIONS (ARIMA 670, 673-679, 687 and `arima/SEAMS.tsv`;
Holt-Winters 660-665, 697-699, 930, 2717 and IDENTITY_PATHS row 57; KPSS 671-672), their own host
oracles and check drivers, and their own identity cards
(`arima.identical.card`, `arima.fit.identical.card`, the Holt-Winters and
KPSS cards). Pass 2 lists those drivers in `tools/identity_lanes/sequence.checks`
with one source sabotage per seam; each moves the DEVICE spelling only, so
the driver's device-vs-oracle gate must fail:

| arm | seam (DEVIATION) | driver | the edit |
|---|---|---|---|
| 5520 | Kalman `MM_l` fold (SEAMS.tsv `MM_l inner`, 670) | `arima/checks/arima_check.mojo` | k descending |
| 5521 | Kalman `Mv_l` fold (SEAMS.tsv `Mv_l inner`) | `arima_check.mojo` | j descending |
| 5522 | Jones transform contraction (SEAMS.tsv, the audit's arm d) | `arima_check.mojo` | `fma(sign*a, x, t)` instead of `fma(sign, round(a*x), t)` |
| 5523 | likelihood `s2` (SEAMS.tsv `likelihood s2`) | `arima_check.mojo` | `vs * (vs / F)` instead of `(vs*vs) / F` |
| 5524 | state update `alpha = tmp + K vs` (SEAMS.tsv `alpha update`) | `arima_check.mojo` | unfused |
| 5525 | x0 Householder QR column norm (678) | `arima/checks/fit_check.mojo` | rows descending |
| 5527 | HW level / trend / season mix (698, the one flush-and-fuse rule) | `holtwinters/checks/hw_check.mojo` | the other product fused |
| 5528 | HW decompose conv1d order (660) | `hw_check.mojo` | filter sum rotated by launch geometry |
| 5529 | HW zero search direction guard (662) | `hw_check.mojo` | guard off |
| 5530 | HW signed-zero clamp (663) | `hw_check.mojo` | `>=` lower test |
| 5531 | HW line-search acceptance tie (row 57, `hw_optim.cuh`) | `hw_check.mojo` | `>=` |
| 5532 | HW stop-criterion order (row 57, `hw_optim.cuh:524-526`) | `hw_check.mojo` | swapped |
| 5533 | HW line-search limit keeps the best trial (2717) | `hw_check.mojo` | keeps the last |
| 5534 | KPSS sum of squares (671) | `tsa/checks/stationarity_check.mojo` | unfused |
| 5535 | KPSS long-run variance accumulator (671) | `stationarity_check.mojo` | unfused |

Seams with no separating spelling are not armed: `F = Z P Z'` (Z is a 0/1
selection vector, so every association rounds the same) and the Jones
inverse's `fma(sign, prod, x)` (sign is +-1, so fused and unfused agree).
