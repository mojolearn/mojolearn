# RunPod CPU leg proof (2026-09-15)

Lane `lane/runpod-cpu-leg`. Runner `tools/runpod_cpu_leg.sh`, caches
`tools/runpod_cpu_cache.py` and `tools/bincache.py`. There were three CPU pods,
run one at a time, all `cpu3c`/`cpu5c` with 8 vCPU at $0.24/hr. No GPU was
rented.

Every pod ran `proof_cmd.sh` with `--build core,estimators --sabotage-build
core,estimators --envs default,test`. That is the readback, then
`identity_break.py --lanes ols,ridge,kmeans --fixtures base --repeats 2` on the
production host bindings, then the same run on the sabotage set
(`MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage`). `compare.sh` holds the
Mac-side diffs, exactly as run.

## Pods

| dir | pod | image | outcome | billed | spend |
|---|---|---|---|---|---|
| `cold-glibc231-failed/` | nhdwjzo42lm3r8 | runpod/base:0.6.3-cpu | the envs installed, then mojo would not load (`GLIBC_2.35 not found`, image glibc 2.31); every build failed | 415 s | $0.0277 |
| `cold/` | sv1hmmshyj8e23 | runpod/base:1.3.1-ubuntu2204 | cold: every cache missed, built and uploaded | 428 s | $0.0285 |
| `warm/` | byg7nqtw65c4j6 | runpod/base:1.3.1-ubuntu2204 | same keys: pixi, the default env and all four bindings restored | 253 s | $0.0169 |

Total spend is $0.073. Each pod's `teardown.txt` says `terminated_verified=1`:
the pod GET returned HTTP 404 and the pod was absent from the listing. Each
dead-man was cancelled only after that. `runpod_cpu_leg.sh list` was empty
after the warm pod.

The first pod's two envs had been uploaded under that image's key, and they
were deleted from R2. Since then the box uploads an env only after
`mojo --version` answers. On the `cold/` pod a poll bug (a remote `[ -f DONE ]
&&` exiting 1) cut the upload short after the default env, so the test env
first uploaded on `warm/`. That is fixed at 7b76dd4a4, and no partial object
exists (object first, sidecar last).

## Timings, seconds (Mac clock, box phases from `phases.tsv`)

| phase | cold | warm |
|---|---|---|
| create (POST to ssh up) | 88 | 67 |
| arm the on-pod watchdog | 6 | 10 |
| stage (17 MB source + URL maps) | 14 | 42 |
| env (pixi binary + default and test) | 16 | 46 |
| build (4 host bindings) | 152 | 3 |
| run (readback + 2 identity runs) | 25 | 10 |
| **create to first result** | **302** | **180** |
| cache upload (misses only) | cut short | 48 (test env) |
| fetch / teardown | 1 / 1 | 1 / 1 |

The binding cache is the saving: 152 s of compiles became 3 s of four hits.
The env cache is not. The cold locked install took 8 s plus 4 s, while the
warm restore of a 562 MB default env took 37 s. That is why `--envcache` is
now opt-in. The pixi binary (79 MB) stays cached, because it replaces the
GitHub fetch that failed three legs on Sep 8 to 12. The stage row varies with
the Mac uplink, and the cache does not touch it.

## (b) CPU identity across ARM and x86

- M4, one core (`m4-one-core/`, nice 19, thread variables 1): core and
  estimators host bindings built in the lane worktree, `vendor cpu-apple-m4`.
- The cold pod (AMD EPYC 7702P) and the warm pod (AMD EPYC 7713) both read
  `cpu_identity_gate_check.py column` OK and readback OK.
- Pod against M4 (`*/diff_pod_m4.txt`): train IDENTICAL=3, infer/model
  IDENTICAL=5 N/A=1 (kmeans has no save), batch IDENTICAL=3, on both pods.
- Pod plus M4 against the three committed GPU columns of
  `2026-09-14_166-lanes` (`*/diff_pod_m4_gpu3.txt`): kmeans/base, ols/base and
  ridge/base train read IDENTICAL x5. The ols and ridge infer, model and batch
  cells read IDENTICAL x5. kmeans infer and batch read IDENTICAL x2, because
  the recorded GPU cells are `n/a:no-predict`. No row is DIVERGENT or MOVED.
  The other fixtures read IDENTICAL x3 with the CPU columns `(not run)`,
  because this proof ran `base` only.
- The warm pod's four `.so` sha256 equal the cold pod's, and warm against cold
  JSON reads IDENTICAL on every cell.

## (d) Sabotage served from the cache reads DIVERGENT

On the warm pod, `leg_out/bincache/provenance.tsv` reads `hit` for the two
production bindings and `negative-hit` for the two sabotage bindings. The
sabotage objects came from `bincache/sabotage-v1/`, and no build script ran.
The production and sabotage `.so` digests differ (`so_sha256.txt`).
Production against sabotage (`warm/diff_pod_sabotage.txt`) reads train
DIVERGENT=3, infer/model DIVERGENT=5 N/A=1 and batch DIVERGENT=3. The cold pod
reads the same from freshly built sabotage bindings.

## R2 objects after the proof

| object | bytes |
|---|---|
| `runpod-cpu/v1/pixi-bin/0.77.0/linux-64/pixi.tar` | 78,512,480 |
| `runpod-cpu/v1/pixi-env/linux-64/default/74724bfe....tar` (opt-in) | 561,987,622 |
| `runpod-cpu/v1/pixi-env/linux-64/test/a68b3049....tar` (opt-in) | 559,044,165 |
| `bincache/v1/none/runpod-cpu-runpod-base-1.3.1-ubuntu2204/` 2 bindings | 379,145 |
| `bincache/sabotage-v1/none/runpod-cpu-runpod-base-1.3.1-ubuntu2204/` 2 bindings | 379,637 |

No record file holds a presigned URL, an R2 credential or the RunPod key.
The check was a grep for `X-Amz-Signature`, `X-Amz-Credential`, `R2_SECRET`,
`Authorization: Bearer` and the key's prefix, printed as no match.
