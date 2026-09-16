# lane/arima-exog: exogenous regressors on ARIMA

RESUMABLE FROM NOTHING. This file assumes the reader holds no context, no
scratchpad and no built binaries. The Mac was restarted after the work below;
`/private/tmp/claude-501/...` and every `.so` in it are gone. Everything that
matters is on the branch.

Branch `lane/arima-exog`, pushed. Head when this was written: `9805b5df3`.
Worktree (recreate it anywhere): `git worktree add -b lane/arima-exog <dir> origin/lane/arima-exog`.

## What the lane does

`ARIMA.fit(y, exog)`, `forecast(steps, exog)`, `predict(start, end, exog)`:
regression with ARIMA errors, following cuML (`arima.pyx:342-362`, `:688-723`;
`batched_arima.cu:117-157`, `:854-931`; `batched_kalman.cu:921-972`, `:181`,
`:286`; `arima_helpers.cuh:73-119`, `:262-292`). `beta` packs after `mu`, the
regressors are differenced beside `y`, their future values are differenced
against their past, `estimate_x0` regresses `y` on them before the ARMA start
values, and `x_t beta` is the observation intercept the filter adds to every
prediction. DEVIATIONS 994 (EXOG_MAX = 17), 995 (the closed gemms' fold is
ours: serial ascending fma from 0), 996 (the `(batch, n_obs, n_exog)` layout),
997 (a non-finite regressor refused by name), 998 (saved format
`mojolearn-arima-2`; `mojolearn-arima-1` unchanged byte for byte).

Every entry point without `_x` is the `n_exog = 0` door the checks and the
card call, handing the `_x` entry placeholders nothing reads.

## STATE: what is PROVEN and what is OWED

PROVEN and pushed:
- Both host bindings and the Metal binding compile with the exog arms.
- `test_arima_exog.py`: 7 source checks + the runtime check pass on Metal
  (8 passed); the runtime check passes again against the host bindings on a
  staged CPU-only package, which is where `mojolearn.host_model` is covered.
- Identity, Metal, base fixture, both new lanes: `cells=2 stable=2 moved=0
  refused=0`, infer/model/batch stable, `reload` equal to `infer`
  (`bench/results/identity_break/2026-09-15_arima-exog/metal/metal.new-base.json`).
  THE SEASONAL LANE'S CELL LANDED; it is `arima-exog-seasonal/base`.
- EXISTING LANES UNCHANGED: arima, arima-011, arima-seasonal-c on the base
  fixture read `IDENTICAL x4` against the three GPU columns of the 166-lane
  record (`metal/diff.old-base.txt`, `summary: IDENTICAL=27`). The three
  `model` rows read ONE-COLUMN only because those columns predate
  `ARIMA.save`.
- Merge gates on this tree: `docs_facts --check` OK (13 facts),
  `packaging/wheel_ci.py pins .` 56 build scripts,
  `packaging/wheel_ci.py inventory python/mojolearn` 84 modules.

OWED:
1. NOTHING. The recording finished: `classical_host_gate.py record` of BOTH
   lanes on all nine fixtures each (18 fixture directories, every
   `expected.json` parses), committed under
   `bench/results/classical_host/2026-09-15-apple-m4-arima-exog` and listed in
   `FORECAST_RECORDED` in `python/mojolearn/host_surface.py`.
2. The CPU column, both sabotage columns and the owed check (one CPU pod).
3. The NVIDIA column (one small pod). No AMD box this lane.
4. statsmodels agreement (runs on the CPU pod; the script is committed).
5. DONE: the recording is in `FORECAST_RECORDED`, and that entry is backed by
   what the test actually asserts. `test_recordings_and_columns_exist`
   (`python/mojolearn/tests/test_host_surface.py`) asserts ONE thing about
   every name in `CLASSICAL_RECORDED + FORECAST_RECORDED +
   INFERENCE_ONLY_RECORDED + SEARCH_LOOKUP_RECORDED + CLASSICAL_GPU_COLUMNS`:
   `(ROOT / rel).exists()`. It says nothing about GPU columns. GPU columns are
   a DIFFERENT test, `test_training_gpu_columns_exist`, over
   `TRAINING_GPU_COLUMNS`, which this lane does not touch. The directory is
   committed and holds 18 fixtures whose `expected.json` all parse, so the
   registration does not outrun its evidence. If that assertion ever grows a
   GPU-column requirement, this entry must come out until the NVIDIA and AMD
   columns land.

## Resume, exactly

Build (nothing is prebuilt after a restart; `pixi run` from the worktree):

    sh bindings/build_arima.sh                  # Metal _mojolearn_arima.so -> python/mojolearn/identical/
    MOJOLEARN_ARIMA_HOST_OUTDIR=<dir> sh bindings/build_arima_host.sh      # reference host (fits)
    MOJOLEARN_FORECAST_HOST_OUTDIR=<dir> sh bindings/build_forecast_host.sh # shipped host (predicts)

The identity column of the two lanes on Metal (the M4's GPU is degraded by a
command-queue leak; ONE Metal job at a time, base fixture first):

    python3 tools/identity_break.py --lanes arima-exog,arima-exog-seasonal \
        --fixtures base --repeats 1 --json <out>/metal.new-base.json

The recording the CPU pod's gate reads is DONE and committed (both lanes,
nine fixtures each, 9m07s on the M4). Redo it only if a fixture changes (~1m25s per fixture-pair on a healthy M4, far
slower on a leaking one, which is why it was stopped):

    python3 tools/classical_host_gate.py record \
        bench/results/classical_host/2026-09-15-apple-m4-arima-exog \
        --lanes arima-exog-seasonal

Until it exists, the pod command's gate steps cover the `arima-exog` half
only; `$REC` in cpu_cmd.sh is that same directory.

The existing lanes' spot check (must stay IDENTICAL x4):

    python3 tools/identity_break.py --lanes arima,arima-011,arima-seasonal-c \
        --fixtures base --repeats 1 --json <out>/metal.old-base.json
    R=bench/results/identity_break/2026-09-14_166-lanes
    python3 tools/identity_break.py --diff $R/apple-m4.json \
        $R/nvidia-h100-sm_90a.json $R/amd-mi325x-gfx942.json <out>/metal.old-base.json \
        --lanes arima,arima-011,arima-seasonal-c

The CPU comparison, the two sabotage columns, the owed check, the installed
wheel gate and the statsmodels agreement, ONE pod, command committed at
`bench/results/identity_break/2026-09-15_arima-exog/cpu_cmd.sh`:

    bash tools/runpod_cpu_leg.sh --lane arima-exog --build arima,forecast \
        --sabotage-build arima,forecast \
        --cmd-file bench/results/identity_break/2026-09-15_arima-exog/cpu_cmd.sh --rent

NOTE ON OWED: `--require-columns 4 --owed-json` against the 166-lane record
FAILS locally for a brand-new lane and says why ("not OWED: no CPU column
hashes it"). OWED is the verdict only once the CPU column exists, which is
why that diff lives in the pod command, as lane/inference-holtwinters did it.

The NVIDIA column (one small pod; the leg defaults to an RTX 4090, so name
the H100 to match the record's `nvidia-h100-sm_90a` column):

    MOJOLEARN_GEMM_LEG_PAYLOAD=phase8 MOJOLEARN_GEMM_LEG_E1_PHASES=9 \
    MOJOLEARN_GEMM_LEG_P9_BINDINGS="build_arima.sh" \
    MOJOLEARN_GEMM_LEG_P9_LANES=arima-exog,arima-exog-seasonal \
    MOJOLEARN_GEMM_LEG_P9_TIERS=identical MOJOLEARN_GEMM_LEG_P9_BREAK=1 \
    MOJOLEARN_GEMM_LEG_GPU_NVIDIA="NVIDIA H100 PCIe" \
        tools/gemm_remote_leg.sh nvidia --payload phase8 --rent
    # the column comes home as stability/identity_break.identical.json

Evidence goes in `bench/results/identity_break/2026-09-15_arima-exog/`
(README.md is there with the verdict table; fill the OWED rows).

## The CPU pod

Results live OUTSIDE /private/tmp, under `~/mojolearn-evidence/arima-exog-cpu/`,
and the pod is DELETED and the delete verified (`teardown.txt`). The NVIDIA and
AMD cells of these two lanes are OWED to the next release record; no NVIDIA pod
was rented for this lane.

## Merge

    git merge origin/main
    python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins .
    python3 packaging/wheel_ci.py inventory python/mojolearn
    git push origin HEAD:main      # main only, 0.8.7; never wait on CI

Then remove the worktree.
