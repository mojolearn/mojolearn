# lane/infer-speed-trees, 2026-09-17: RTX 4090 pod evidence

Branch `lane/infer-speed-trees` from `main` at `e3213a59a`. One RunPod pod,
`hhb4bs9mhcv6lt`, NVIDIA GeForce RTX 4090 (driver 580.159.04, CUDA 13.0),
AMD Ryzen 9 7950X (16 cores, 32 threads), Mojo 1.0.0 (ed45d567), $0.74 per
hour, created 13:42:35Z and reaped 16:54:28Z (HTTP 404 verified), about
$2.37; a first pod with a 570 driver (Mojo refuses drivers below 580) ran
about three minutes before it, about $0.04. Total about $2.41.

Full logs and every per-process JSON are outside the repo under
`~/mojolearn-evidence/infer-speed-trees/leg_out/` (`identity/`, `speed/`,
`speed-rerun/`, the build logs and `.so` digests). This directory holds the
two summaries only.

## Speed (`speed_summary.json`)

Taxi, the last 1,000,000 rows of the table in temporal order, one path per
process, BEFORE (`e3213a59a`) and AFTER (`411a69a96`) binaries in
alternating processes over the same input bytes, one warmup then 5 timed
calls (`speed/`) or 15 (`speed-rerun/`), spread gate 1.10 per process. The
ratio is quoted only from processes that passed the gate, and only where
every process of both arms produced the same output SHA-256. Threads:
`MOJOLEARN_CPU_THREADS` unset, one per physical core (16).

| model | path | before ms | after ms | ratio | status |
|---|---|---|---|---|---|
| RF regressor 100 x depth 16 | GPU class `predict` (sequential engine) | 38864 | 7570 | 5.13 | qualified, 2 + 2 stable processes |
| RF regressor 100 x depth 16 | `host_model().predict` | 35116 | 7393 | 4.75 | qualified, 3 + 4 stable processes |
| ET regressor 100 x depth 16 | GPU class `predict` (sequential engine) | 15257 | 4572 | 3.34 | qualified, 2 + 1 stable processes (the other AFTER processes read 4518 to 5322 ms at spreads 1.12 to 1.27) |
| ET regressor 100 x depth 16 | `host_model().predict` | 14827 | 4477 | 3.31 | qualified, 1 + 2 stable processes |
| GBDT Logloss 1000 x depth 6 | `host_model().predict_proba` | 22883 | 1383 | 16.5 | qualified, 2 + 2 stable processes |
| GBDT Logloss 1000 x depth 6 | `host_model().predict` | 24134 and 24110 | 1339 and 1372 | 17.8 | four stable processes in `leg_out_speed.log` (16:06 to 16:09 UTC); their JSONs were overwritten by the 100-iteration model's records of the same path (a harness naming defect fixed in `e08ce80cb`, the rerun names files by model) |
| GBDT Logloss 100 x depth 6 | `host_model().predict` | 2733 | 199 | not qualified | AFTER spreads 1.11 to 1.40 at 200 ms; every AFTER round is below every BEFORE round |
| GBDT Logloss 1000 x depth 6 | GPU class `predict_proba` | 828 and 838 (5 rounds, stable), 873 and 877 (15 rounds, spreads 1.16 to 1.19) | 129 to 137 | not qualified | AFTER spreads 1.35 to 1.40; every AFTER round is below every BEFORE round |
| GBDT Logloss 100 x depth 6 | GPU class `predict_proba` | 784 | 86 | not qualified | AFTER spreads 1.48 to 1.54 at 85 ms; every AFTER round is below every BEFORE round |
| GBDT Logloss 1000 x depth 6 | GPU class `predict` | 99 | 109 | not qualified | both arms spread 1.38 to 1.72 at 100 ms; this path is unchanged by the lane |
| GBDT Logloss 1000 x depth 6 | model text parse alone (`gbdt_model_dim`) | 34.8 and 33.7 | 33.8 and 33.9 | 1.0 | diagnostic: the parse is about a third of a 100 ms GPU `predict` call for a 1000-tree model, unchanged by the lane |

The GPU class `predict` of a forest runs the `sequential` engine, which is
the host walk; the GPU is idle on that path before and after. The `predict`
of a GBDT on the GPU class is the device path and is not changed here; its
`predict_proba` gain is the DEVIATION 2333 Python comprehension retired
(DEVIATION 2902).

## Identity (`identity_diff_summary.json`)

`tools/identity_break.py --lanes <33 rf, et and gbdt lanes> --fixtures
base,ties,odd,dupes,wide --repeats 2`, six columns on the pod, diffed with
`--diff`. The CUDA columns skip `rf-clf-balanced-parallel` and
`et-reg-bootstrap-parallel` (their batch part hung the UNMODIFIED main tree
at `rf-clf-balanced-parallel/base repeat=1/batch`, 90 minutes in
`futex_wait` at 0 percent CPU and GPU on the one-GPU box; the CPU columns
carry them). The CUDA columns were built without the metrics GPU binding,
so `rf-score-weighted` and `gbdt-adapter-score-weighted` read REFUSED there
and were run as a separate CUDA part after that binding was built (`-sw`).

| diff | infer and model cells | batch cells | moved |
|---|---|---|---|
| before-cuda vs after-cuda | IDENTICAL 280, N/A 10, NOT-COMPARED 20 | IDENTICAL 145, NOT-COMPARED 10 | 0 |
| before-cuda-sw vs after-cuda-sw | IDENTICAL 10 (train parts), N/A 20 | N/A 10 | 0 |
| before-cpu vs after-cpu | IDENTICAL 210, N/A 20, NOT-COMPARED 20, ONE-COLUMN 20, REFUSED 60 | IDENTICAL 135, N/A 10, NOT-COMPARED 10, ONE-COLUMN 10 | 0 |
| after-cuda vs sabotage-cuda | DIVERGENT 55, IDENTICAL 225 | DIVERGENT 55, IDENTICAL 90 | 55 (every rf, et and Logloss-proba lane) |
| after-cuda-sw vs sabotage-cuda-sw | | | 5 (`rf-score-weighted`, its four regression parts) |
| after-cpu vs sabotage-cpu | DIVERGENT 105, IDENTICAL 105 | DIVERGENT 125, IDENTICAL 10 | 135 (every rf, et and gbdt lane the column runs) |
| after-cuda vs after-cpu | IDENTICAL 210, ONE-COLUMN 70, REFUSED 20 | IDENTICAL 125, ONE-COLUMN 30 | 0 |

The ONE-COLUMN cells of the CPU diff are the ten `gbdt-symmetric` and
`gbdt-pointwise-l2-bayesian-eval` cells the AFTER CPU column REFUSED: the
CPU training GBDT family (`_mojolearn_gbdt_host`) of that build did not yet
export `gbdt_sigmoid_pair`, and a CPU-only install's binding proxy raises
ImportError by name for a missing entry point. Both are fixed in
`e08ce80cb` (the family carries the pair; the Python layer treats that
ImportError as absence) and are OWED a rerun of the CPU column. The 60
REFUSED cells common to both CPU columns are the adapter archives, the
multi-output losses, the CTR-table lanes and the two two-device lanes,
refused by name on main as before.

On the CUDA sabotage arm the gbdt lanes other than the two Logloss-proba
lanes read IDENTICAL, as they must: on the GPU class the GBDT `predict`
path is the device path this lane did not touch, and only `predict_proba`
of a Logloss model reaches `gbdt_sigmoid_pair`.
