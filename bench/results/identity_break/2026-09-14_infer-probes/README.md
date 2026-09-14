# The infer column on the two forecasters (2026-09-14, Apple column only; NVIDIA and AMD OWED)

`tools/identity_break.py` grew an infer probe on `holtwinters` and `arima`, the two of the nine
`n/a` lanes whose public class has an out-of-sample method. Neither takes new rows, so the held-out
axis is time: the infer column hashes the forecast beyond the fitted series at
`FORECAST_HORIZON = 512` steps (the fitted length, as the held-out row counts mirror the training
row counts elsewhere), through both public entries the estimator documents as the same answer
(`forecast(h)` and `forecast(h, index=0)` for Holt-Winters; `forecast(h)` and
`predict(n_obs, n_obs + h)` for ARIMA), asserted byte-equal by `_same_bytes` so a difference reads
REFUSED with its name, never a quiet hash. The train column of both lanes is unchanged (its 24-step
forecast and, for ARIMA, `ar_`, `mu_`, `sigma2_`), and the JSONs here carry the same train hashes as
`../2026-09-14_46-lanes/`.

The other seven `n/a` lanes were audited against their classes the same day and stay `n/a`:
`KMeans` has `fit` and `fit_predict` only; `DBSCAN` and `AgglomerativeClustering` have `fit` and
`fit_predict` only; `SpectralClustering.predict` raises `NotImplementedError` by design; `gemm-pinned`
and `metrics` are functions. `iforest` already carried its probe (`score_samples`, `predict` on the
held-out rows, since 2026-09-13) and is IDENTICAL x3 in `../2026-09-14_46-lanes/`; it has no
save/load, so no model column.

## The Apple column (this Mac, M4, two-core cap, commit 3cd3afd1c, bindings from the main checkout)

`apple-m4.txt` is the run; `apple-m4.json` the column (lanes holtwinters, arima, iforest, kmeans):

    cells=36 stable=36 moved=0 refused=0
    infer: stable=27 n/a=9
    model: n/a=36

Run twice in one process, as the tool does; the run was repeated in a second process (the
byte-equality assertions added between the two) and every one of the 18 new hashes was the same.
The probe can fail: on the `base` fit, `forecast(h)` against `predict(n_obs - 1, n_obs + h - 1)`
raised `differ: 62 bytes of 8192` (ARIMA) and `forecast(h)` against `forecast(h + 1)[1:]` raised
`differ: 1553 bytes of 2048` (Holt-Winters); the documented pairs passed; `forecast(24)` is the
prefix of `forecast(512)`.

`diff.probes-vs-46-lanes.txt` is this JSON against the three committed 2026-09-14 columns:

    summary: IDENTICAL=414
    summary (infer/model): IDENTICAL=459, N/A=351, ONE-COLUMN=18

The 18 ONE-COLUMN cells are exactly `holtwinters/*` and `arima/*` infer, which the older JSONs record
as `n/a:forecast`. The three committed files diffed among themselves still read
`summary: IDENTICAL=414` and `summary (infer/model): IDENTICAL=459, N/A=369`.

The new probe is STABLE on the M4 and THREE-VENDOR OWED. Nothing here is a cross-vendor claim.

## The OWED legs (do not run from a worktree; the drivers archive HEAD)

With this branch's head checked out (or merged), `SHA=$(git rev-parse HEAD)`. The body is the
tracked `tools/identity_three_columns_leg.sh`, which runs EVERY lane, so each leg also refreshes
the other 44; the wrapper bakes the commit witness because the runners ship a git archive with no
`.git`:

    W=/tmp/identity_wrap.sh
    printf '# wrapper body: bake the commit witness, then the tracked body\necho %s > /root/mojolearn/commit.txt\nexec sh tools/identity_three_columns_leg.sh\n' "$SHA" > "$W"

NVIDIA (RunPod H100, `--payload gemm` runs the GEMM gates and then the wrapper as the extra body):

    export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key
    MOJOLEARN_GEMM_LEG_GPU_NVIDIA="NVIDIA H100 80GB HBM3" \
    MOJOLEARN_GEMM_LEG_EXTRA="$W" \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-identity-infer-probes \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent

AMD (Hot Aisle MI300X, gfx942, the 22.04 ROCm container; the commit also travels as env):

    MOJOLEARN_GEMM_LEG_EXTRA="$W" \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-identity-infer-probes \
    MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_COMMIT=$SHA" \
    MOJOLEARN_HOTAISLE_LANE=identity-infer-probes \
    bash tools/hotaisle_leg.sh amd --rent --minutes 60 --skip-gates

Each leg's column lands in `<out>/remote/identity/identity_break.<label>.json` with `gate.txt`
beside it. Copy the two JSONs here as `nvidia-h100-sm_90a.json` and `amd-mi300x-gfx942.json`, then

    PYTHONPATH=$PWD/python pixi run python tools/identity_break.py --diff \
        bench/results/identity_break/2026-09-14_infer-probes/apple-m4.json \
        bench/results/identity_break/2026-09-14_infer-probes/nvidia-h100-sm_90a.json \
        bench/results/identity_break/2026-09-14_infer-probes/amd-mi300x-gfx942.json \
        --require-columns 3 --lanes holtwinters,arima

The claim to make after that, and only after, is `IDENTICAL x3` on 18 infer cells; the Apple
column must be re-recorded at the legs' commit if any `.mojo` file under `holtwinters/`, `arima/`
or `bindings/` moved between 3cd3afd1c and it.
