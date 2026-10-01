# PR #73: GLM Newton-decrement stop + host route, per-vendor (lane/neural-pass68 08d1fdca6), merged 01304cfb2

Probe: board reg-taxi block through the board's lane_arrays, board params (alpha 1e-4, max_iter 100, tol 1e-4), branch bindings
(x_linear + x_linear_host on the CPU column) overlaid on a clean 0.8.33 venv. Two processes x two rounds per arm.

Bits: identical on every route and vendor (L40S box, MI325X, M4 from the writer): poisson n_iter 11 r2 0.035964 coef0 -0.000689295;
gamma n_iter 5 coef0 -5.8418e-05; tweedie n_iter 3 coef0 -0.0017189.

| family | 0.8.33 board NVIDIA | #73 NVIDIA default | 0.8.33 board AMD | #73 AMD default | #73 AMD device | sklearn (NVIDIA box / AMD box) |
|---|---|---|---|---|---|---|
| poisson | 49,086 ms | 2,060-2,089 (host) | 78,359 | 1,398-1,411 (host) | - | 3,480-4,108 / 1,241-1,434 |
| gamma | 946 | 886-1,038 (device) | 1,744 | 615-624 (host) | 1,726-2,271 | 3,291-3,361 / 931-1,076 |
| tweedie | 686 | 632-812 (device) | 1,220 | 495-505 (host) | 1,237-1,484 | 2,957-3,011 / 983-1,005 |

Route: host for every GLM family except gamma/tweedie on a "cuda"/"nvidia" vendor read-back (the device wins there: tweedie
632 warm vs host 764-774). MOJOLEARN_X_LINEAR_GLM_HOST=1/0 forces host/device; MOJOLEARN_X_LINEAR_DEVICE=1 disables every host route.
First head (6bbb4dd34) compared against "nvidia" while the read-back is "cuda"; fixed in 08d1fdca6.

Bulk: R2 measurements/2026-10-01/glm73-nvidia.tar.gz, glm73-amd.tar.gz (see r2-index.tsv).
