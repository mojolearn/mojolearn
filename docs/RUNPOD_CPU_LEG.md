# RunPod CPU leg (`tools/runpod_cpu_leg.sh`)

Heavy CPU work leaves the shared Mac and runs on one RunPod CPU pod per lane.
That covers host binding builds, identity_break CPU columns, sabotage builds
and pytest modules. The Mac keeps Metal checks and one-core quick checks. No
GPU is rented, so Andrew's release-only GPU rule is untouched.

## One command

```sh
bash tools/runpod_cpu_leg.sh --lane my-lane \
  --build core,estimators --sabotage-build core,estimators --envs default,test \
  --cmd 'python3 tools/identity_break.py --lanes ols,ridge,kmeans --fixtures base --repeats 2 --json "$LEG_OUT/cpu-x86.json"'
```

That is a dry run. It builds the source tarball, renders the create request
and the box script, prints the cache keys, lists live `mojolearn-cpu-*` pods
and says which R2 objects already exist. It creates nothing. Add `--rent` to
create the pod.

- `$LEG_OUT` is the directory that comes back, under `--out` (default
  `<worktree>/bench/results/runpod_cpu/<stamp>-<lane>/remote/leg_out`).
- The command runs with bash in `/root/mojolearn`, with the `default` pixi env
  first on `PATH`, `PYTHONPATH=python`, `MOJOLEARN_COMMIT` set to the shipped
  HEAD and `MOJOLEARN_NUMERIC_MODE=identical`.
- `--build` families land in `python/mojolearn/host/`. `--sabotage-build`
  families are built with `--sabotage-defines` (default
  `-D MOJOLEARN_HOST_SABOTAGE=1`) into `python/mojolearn/host-sabotage/`. Load
  them with `MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage
  MOJOLEARN_HOST_ALLOW_SABOTAGE=1`.
- `--vcpu` (default 8), `--flavors` (default `cpu3c,cpu5c`), `--lease` minutes
  (default 60), `--include PATH` to ship a tracked path under `bench/results`.
  THE HOURLY RATE SCALES WITH `--vcpu`. The $0.24/hr measured below is the
  8 vCPU price, not a fixed price for a CPU pod: at `--vcpu 16` the same pod
  bills $0.48/hr (measured 2026-09-16, lane/sabotage-evidence, pod
  `mojolearn-cpu-sabevid-20260916-095653`). Read the rate off the create
  response, which the leg prints and records in `create_response.json`, rather
  than assuming the figure in this file.
- `bash tools/runpod_cpu_leg.sh list` shows live pods. `bash
  tools/runpod_cpu_leg.sh reap POD_ID` deletes one `mojolearn-cpu-*` pod and
  verifies it is gone.

## What a rented run does

1. Pre-flight. It refuses if a pod with the same lane tag is live, or if the cap of
   `mojolearn-cpu-*` pods is already live (default 8, `--max-pods N` or
   `MOJOLEARN_RUNPOD_CPU_MAX_PODS`, at most 12; raised from 2 on Sep 15 when a dozen lanes
   queued behind the proof's limit).
2. A Mac dead-man is armed BEFORE the create. It deletes the pod by id, or by
   name, after the lease plus the ready timeout plus ten minutes.
3. Create. The cost per hour is printed from the create response.
4. The on-pod watchdog (`tools/runpod_guard.sh arm`) is armed and read back.
   Its pid must be alive and its token must answer a GET with 200. At the
   lease the pod deletes itself through the API, even if the Mac is gone.
5. The worktree's tracked files ship as a gzip tarball, and its sha256 is
   checked on the box. Uncommitted edits to tracked files ship too, and the
   dirty count is recorded in `leg.txt`. `bench/results`, `mamba/corpus` and
   `bench/oracle_*` stay home unless `--include` names them.
6. Presigned R2 URLs reach the box inside a script on ssh stdin. They never
   appear in argv, and no credential leaves the Mac.
7. On the box, detached: restore or install pixi (and the envs with `--envcache`), build through
   the binding cache, run the command, then upload whatever missed.
8. Fetch `$LEG_OUT`, promote binding uploads to their content address, DELETE
   the pod and ask the API until the pod is gone. The dead-man is cancelled
   only after that verified delete. `timings.tsv` splits create, arm, stage,
   env, build, run, fetch and teardown, and adds the spend.

## What is cached in R2 (bucket `mojolearn-data`)

Every object is content addressed. A PUT URL is minted only for an object that
does not exist, so nothing is overwritten in place. Datasets and corpora keep
using `tools/stage_from_r2.sh`, which lands every `gbm-bench/` key under
`/root/datasets/gbm-bench` where `tools/speed_gbdt_arm.py` reads it
(`GBM_BENCH_DATA`), so no rented pod fetches a dataset from its origin. The
pins are in `bench/results/dataset_store/manifest.tsv`.

This is the rule for every remote run, not only this leg: see
[REMOTE_DATA_R2.md](REMOTE_DATA_R2.md).

| dataset key | bytes | decoded form |
|---|---|---|
| `gbm-bench/taxi/taxi_speed.npz` | 419,757,252 | taxi, the regression and binary tasks |
| `gbm-bench/istella/istella_speed.npz` | 2,248,281,826 | Istella-S, train and test features and grades |
| `gbm-bench/istella/istella_rank.npz` | 624,022,440 | Istella-S query ids and the ranking test half |
| `gbm-bench/istella/istella-s-letor.tar.gz` | 472,129,615 | Istella-S source tarball, so the decode is reproducible |
| `gbm-bench/higgs/higgs_speed.npz` | 1,276,000,490 | HIGGS, 11,000,000 x 28 float32 `x`, float32 `y` |
| `gbm-bench/covtype/covtype_speed.npz` | 127,823,140 | Covertype, 581,012 x 54 float32 `x`, int32 `target` (1..7; `covtype` and `covtype2` derive from it) |
| `gbm-bench/year/year_speed.npz` | 187,586,070 | YearPredictionMSD, 515,345 x 90 float32 `x`, float32 `y` |

| object | key | restore check |
|---|---|---|
| `runpod-cpu/v1/pixi-bin/<version>/linux-64/pixi.tar` | the pinned pixi version (0.77.0) | sha256 sidecar `.sha` |
| `runpod-cpu/v1/pixi-env/linux-64/<env>/<key>.tar` (only with `--envcache`) | sha256 of the env name, platform, `pixi.lock` sha256, `pixi.toml` without task tables, the box prefix `/root/mojolearn`, pixi version and image (`tools/runpod_cpu_cache.py keys`) | sha256 sidecar, then `pixi install --locked` must accept it |
| `bincache/v1/none/runpod-cpu-runpod-base-1.3.1-ubuntu2204/<key>.tar.gz` | `tools/bincache.py` fields: the binding's import closure and scripts, the toolchain in `pixi.lock`, numeric mode, every build `MOJOLEARN_*` variable (so `MOJOLEARN_LINUX_CPU=x86-64-v3`, `MOJOLEARN_BUILD_JOBS` and the defines), device arch `none`, image, OS and glibc, repo path | archive manifest must hash to the key the box computed |
| `bincache/sabotage-v1/none/.../<key>.tar.gz` | the same fields plus `variant=sabotage` | as above |

The upload order is the object first and the sidecar last, so a half-written
upload is never restored.

**Sabotage never serves production.** A build whose defines mention SABOTAGE
is refused by the cache unless the runner sets `MOJOLEARN_BINCACHE_NEGATIVE=1`,
which only `--sabotage-build` does. A negative control carries
`variant=sabotage` in its key fields, looks up only `sget` rows (the
`bincache/sabotage-v1` listing) and uploads only there. A production build
reads only `get` rows. An archive of the other variant fails the key check on
placement, and the Mac's promote step refuses a row whose `sabotage=` flag
disagrees with its fields or its destination
(`tools/test_bincache.py`, the negative-control test).

**The Mac stays off.** Nothing on the Mac stages a URL map, so
`tools/bincache.py build` is a plain `sh <script>` there.

The env cache is OPT-IN (`--envcache`). Measured on 2026-09-15, restoring the
562 MB default env took 37 s, while a cold `pixi install --locked` of default
and test took 12 s, because RunPod reaches the conda channels fast. The pixi
binary (79 MB) is always cached, because it replaces the GitHub fetch that
failed three legs between Sep 8 and 12.

Not cached: git history, secrets, the identity records the repo carries, and
the Mojo compiler cache.

## Measured (2026-09-15, 8 vCPU, $0.24/hr)

| create to first result | cold | warm |
|---|---|---|
| create (POST to ssh up) | 88 s | 67 s |
| stage | 14 s | 42 s |
| env | 16 s | 46 s |
| build (core + estimators, production and sabotage) | 152 s | 3 s |
| run | 25 s | 10 s |
| total | 302 s | 180 s |

The warm pod's bindings were four cache hits with the same sha256 as the cold
build, and its identity column matched the one-core M4 column and the three
committed GPU columns. Record: `bench/results/runpod_cpu/2026-09-15_proof/README.md`.

The image must ship glibc 2.35 or newer. On `runpod/base:0.6.3-cpu` (glibc
2.31) the envs install and mojo does not load.
