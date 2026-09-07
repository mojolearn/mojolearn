# Missing vendor evidence queue — read-only audit, 2026-09-06

This inventory was authored by reading retained files and source. No validators, tests, builds, models, or measurements were executed. OPEN means the requested full evidence was not located in this checkout; it does not mean the implementation cannot run. Only root may execute the queued remote NVIDIA/AMD work. Historical Apple evidence below is archival. The user subsequently allowed necessary MacBook runs with bounded CPU and memory usage; prefer remote NVIDIA/AMD, and keep all execution with root.

## Scope and queue

| Cell | Located evidence | Status of missing scope |
|---|---|---|
| ARIMA fitted public API, NVIDIA and AMD | September 6 installed FAST, DETERMINISTIC and IDENTICAL functional gates; NVIDIA IDENTICAL has 88 checks, zero failures | Bounded functional vendor cells located. Full native fitter cross-vendor identity remains OPEN. |
| ARIMA Kalman filter, three vendors | Historical `arima.identical.card` and successful filter checks | Filter evidence must not be promoted to the optimizer/fitter. Fresh-source fitter cards, iteration/retcode/objective/state comparison remain OPEN. |
| Holt-Winters, NVIDIA full IDENTICAL | August 31 partial native check log without terminal pass/card; later installed smoke succeeds | Full native identity cell OPEN. Smoke and FAST timings are separately located. |
| Spectral, NVIDIA full IDENTICAL | Later installed smoke succeeds; historical FAST timings exist | Full native identity cell OPEN. Apple/AMD native cards exist. |
| Gaussian process, NVIDIA full IDENTICAL | Historical FAST timing fixture; installed `_mojolearn_gp` binary/vendor/mode witness | Full native identity cell OPEN. Loading a binding does not prove GP fit/predict; the cited installed smoke has no GP numerical lane. |

## Exact retained paths and provenance

All paths below are relative to the repository root. Directory aliases are explicitly defined so individual artifacts can be located without guessing dates.

### Installed public fitting and smoke, September 6

- `NV = bench/results/resume/2026-09-06-installed-gap-closure/nvidia-installed/candidate/qualification-normalized`
- `AMD = bench/results/e1/2026-09-06_091452-mojolearn-e2-amd/diag/candidate/qualification-normalized`
- Provenance narrative: [installed gap-closure README](../bench/results/resume/2026-09-06-installed-gap-closure/README.md). Frozen native/package commit `eb835021dcd79a59a7e8f78c754a75db3c1fea83`; controller changes are separately described there. NVIDIA is CUDA/sm_89, AMD HIP/gfx942.
- `NV/arima-identical.log`, `NV/arima-fast.log`, `NV/arima-deterministic.log`, and the same files under `AMD`: fitted public ARIMA, recovery length 512, six-series batches, planted AR(1), MA(1), ARMA(1,1), criteria/prediction/forecast and batch-composition checks. `NV/arima-identical.log` explicitly reports 88 checks, zero failures. Corresponding `arima-<mode>.installed.json` retains wheel and installed binary SHA256 plus mode/vendor witnesses; `results.tsv` records each ARIMA job exit 0. This is functional/reference-tolerance and local batch-bit evidence, not paired cross-vendor fitted-state byte comparison.
- NVIDIA IDENTICAL installed wheel SHA256 `805970b2bc44a002194cee66cbe378996f7e4a483aa3b6ca18a4f6da06060663`; ARIMA binary SHA256 `a756ad0830053daefab807dab00c2159b52396fea50e52a11ee1f92006f52e3d`. AMD normalized wheel SHA256 `7c5f9af825cbcbd74a293adc75ad15670a30a179d3a9a8a8cfd993cd9476f7be`; native source inventory SHA256 `ade965b90496132596d8dda79860a87f472193c14129093c5c3a72f273b8159f` is reported by the provenance README.
- `NV/smoke-identical.json` and `AMD/smoke-identical.json` contain successful `holtwinters` and `spectral` entries, empty failure lists, and explicit loaded vendor architecture/mode; corresponding FAST/DETERMINISTIC smoke files and `results.tsv` also exist. Smoke stores final hashes, not native stage cards or fixture dimensions. The current recipe in `tools/repeat_run_stability.py` uses spectral clustering on up to 2000 four-dimensional points (four clusters), and additive Holt-Winters on up to 512 observations, frequency 12, forecast 24; inspect the frozen recipe before asserting exact historical extents.
- The NVIDIA installed manifest also records `_mojolearn_gp` mode 1 and SHA256 `0b10dcae06a5a389cde4c63086548302cee5cd0080d72b428cbede8f57c079aa`. This is binary readback, not a GP fit result. `smoke-identical.json` has no GP entry in `lanes`.

The ARIMA logs end with the obsolete sentence that the fit “has never left one Apple M4.” Their own CUDA/HIP installed witnesses, actual recovery output and zero statuses supersede that sentence for this bounded public test. Conversely, the NVIDIA `results.tsv` includes unrelated Mamba/Transformer failures; do not label the entire NVIDIA qualification green because ARIMA passed.

### Historical native identity lanes

- `APPLE31 = bench/results/e1/2026-08-31_180957-MacBook-Air-1-terrabyte`
- `AMD31 = bench/results/e1/2026-08-31_221142-mojolearn-e2-amd`
- Both `commit.txt` files name `221aa141accb7d1de49d3b64c77e266c79d60c30`. Each directory has `lanes/holtwinters.identical.{log,card}` and `lanes/spectral.identical.{log,card}`. AMD logs end `ALL OK` / `ALL PASSED` respectively. Holt-Winters card starts with `n=20`, batch 3, frequency 5, start periods 2, additive, epsilon .00224, trace iterations 64; the gate additionally exercises multiplicative seasons, optimization, decomposition, forecasts and packed-buffer semantics. Spectral cards include graph stages for n=144,d=4,k=10 and k=52, n=30,d=2,k=5, and clustering n=144,d=3,clusters=3. The [spectral contract](../spectral/IDENTICAL_SPECTRAL_CONTRACT.md) records the historical same-commit Apple/AMD 18-check, 171-line card equality claim. This audit did not rerun its comparator.
- `NV31 = bench/results/e1/2026-08-31_150758-runpod-nvidia`, commit `fe038d0073fd01aef23c39dc936cb43402ffdab5`. Its `lanes/holtwinters.identical.log` reaches optimizer/forecast/signed-zero checks but ends without a terminal all-pass or card. No NVIDIA spectral identity card was located. The spectral contract describes that queue stopping during Holt-Winters. Later smoke does not fill the full native-card gap.
- `NV31/lanes/arima.identical.{log,card}` records successful **filter** checks, n_obs=24, batch=6, salt=7. Matching-scope Apple paths exist under `bench/results/e1/2026-08-31_150607-MacBook-Air-1-terrabyte/lanes/arima.identical.{log,card}` (same `fe038d...` commit); AMD paths exist at `AMD31/lanes/arima.identical.{log,card}` (different `221aa...` commit). Preserve those source differences; this audit has not recertified three-vendor equality.
- `AMD28 = bench/results/e1/2026-08-28_203552-mojolearn-e2-amd`, commit `26eb8ba6d0e9509bc082a031b76b847f8a32381f`: `lanes/gp.identical.log` ends `ALL PASSED`; `lanes/gp.identical.card` retains kernel/factorization/solve/posterior/uncertainty stages. Fixtures include n_train/d pairs 2/1, 16/1, 4/2, 12/3 (ARD, six test points), and 8/2 (Matern, four test points), with explicit ridge bits. The log records 55 kernel-case/fixture combinations, 5324 cellwise checks, float64 references, launch/batch invariance, and runtime sabotage outcomes.
- Historical Apple GP log/card: `bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/gp.identical.{log,card}`, commit `241aed689be0e950fde4eb143ab096915cbc1ae5`. Later Apple GP artifacts also exist under `bench/results/e1/2026-09-04_034529-MacBook-Air-1-terrabyte/lanes/`. These are located evidence, not a freshly verified same-source cross-vendor certificate.

### Fitter diagnostics and FAST evidence are separate

- [ARIMA fitter FAST log](../bench/results/resume/2026-09-05-arima-fit/fast.log): compiler paths identify the historical Apple workspace; header `FIT_N_OBS=512 SALT=7`, six planted series per case; ends `ALL ARIMA FIT CHECKS PASSED [FAST] (no card: set MOJOLEARN_IDENTITY_TRACE)`. It exercises initialization, finite-difference gradients, L-BFGS rules, recovery/minimizer and batch invariance. No adjacent source/vendor receipt was located, and this is neither NVIDIA evidence nor an IDENTICAL fitter card.
- `FASTNV = bench/results/e1g/2026-08-28_040832-nvidia-speed-classical`, commit `65d5e91903546c18d0ca599aaf68ff18604e948a`: `remote/logs/classical.holtwinters.ours.log`, `classical.spectral.ours.log`, `classical.gp.ours.log` have NVIDIA H100 FAST headers, five retained timing rounds and respectively shapes `7x72f12`, `48x4c3`, `ard.12x3s6`. Corresponding `remote/dump/{holtwinters,spectral,gp}.fixture` and `classical.<lane>.vendor.log` files exist. These establish historical FAST execution on NVIDIA, not IDENTICAL full-gate success; no new performance claim or comparison was computed here.

## Root-only next evidence work

Keep full native NVIDIA Holt-Winters, spectral and GP IDENTICAL cells OPEN until complete logs, zero guard exits, mode/vendor readback, exact source inventory and matching stage artifacts exist. Keep ARIMA native fitter cross-vendor identity OPEN while acknowledging the located installed CUDA/HIP functional fits. Root should freeze the exact source/fixture set, run one bounded remote job at a time, retain failures and effective controls, and compare only matching scopes. Changing a README or finding an import manifest cannot close these identity cells.
