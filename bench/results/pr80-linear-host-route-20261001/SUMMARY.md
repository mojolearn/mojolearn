# PR #80 x_linear host route for CV / Bayesian / ARD / ridge-clf (lane/neural-pass76, head d85590eb5)

Same tree, 0.8.34 wheel with the lane's bindings overlaid. Arms: lane default (host route) vs device route
(`MOJOLEARN_X_LINEAR_HOST_ALGOS=glm,isotonic`). Quality (r2 / rmse / accuracy) is identical between routes on every race.

| Lane / dataset | MI325X host | MI325X device | L40S/EPYC host | L40S/EPYC device | sklearn (AMD / NV) |
|---|---|---|---|---|---|
| lasso-cv taxi | 704 | 5318 | 1075 | 2262 | 230 / 593 |
| lasso-cv istella | 35562 | 143765 | 49915 | 71982 | 42070 / 83442 |
| enet-cv taxi | 692 | 5274 | 1062 | 2254 | 266 / 1926 |
| enet-cv istella | 35485 | 148922 | 48619 | 73511 | 45600 / 115486 |
| ridge-clf taxi | 192 | 633 | 357 | 317 | 63 / 110 |
| ridge-clf istella | 5813 | 64705 | 8089 | 14766 | 4484 / 10445 |
| bayesian-ridge taxi | 168 | 1098 | 285 | 314 | 156 / 648 |
| bayesian-ridge istella | 127359 | 186307 | **190239** | **71243** | - / 8032 |
| ard taxi | 22.6 | 210 | **35.8** | **32.6** | - / 29.2 |
| ard istella | 13042 | 71705 | **18120** | **15030** | - / 9329 |

Median ms. MI325X: DigitalOcean, R2 `measurements/2026-10-01/pr80-amd.tar.gz` (includes tree and venv). L40S/EPYC: RunPod nvc3, 13-thread quota, R2 `measurements/2026-10-01/pr80-nvc3.tar.gz`.

The host route wins every AMD race. On NVIDIA it loses bayesian-ridge istella (2.7x), ARD (10-21%) and ridge-clf taxi (+12%).
NOT MERGED: needs bayesian-ridge and ARD kept on the device for cuda/nvidia (as `_glm_host` does), or a recheck after #86/#87 (parallel host Gram, Cholesky, par_rows).
