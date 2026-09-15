# 0.8.6 release: state at the Mac restart (2026-09-15)

Written before Andrew's approved restart of the release Mac, which clears the degraded
Metal GPU (a Metal command-queue leak made fits about 20x slow, per
`~/mojolearn-evidence/gbdt-metal-slowdown-2026-09-15`). The restart ends every session,
agent and local driver; everything under `/private/tmp` is gone afterwards. This file is
what a fresh session needs to resume. Publishing needs Andrew's separate explicit "ship";
nothing here has been published.

## Branch and commits

- `release/0.8.6` at the commit that adds this file (with the macOS runtime-environment
  evidence). Builds and records use
  **db9047b9f**; commits after it change no native-inventory file and no wheel file.
- Freeze 274a9d161 at main b8c9e477e, then these picks (fix commits only, never a main merge):
  - 65a9e9302 release build preflight loads `_backend.py` as a submodule
  - 70d0940a6 admission `BINDINGS` names the six identical bindings added since 0.8.5
  - CPU gate fix 5dd8bc98e, c88ccb22f, c94cc7d98
  - third-party cleanup 429a8da9a, 40e41063c, 5e2b570f6, bc47909fd, 198248e60
  - NOTICE and README c49b40140, f2806635b, ff56c0bf8
  - Holt-Winters line search f80e5921a, 420a8ec73, 5376ee552 (DEVIATION 2717)
  - post-record allowlist, host binaries against their own proof, wheel content audit,
    macOS repack: 6cb6fa571, aa23e7b3a, 45fd47f2f
  - 9cc557e3d native source rule includes `tokenizer/tools/` (the Unicode table generator)
  - e1d818919 macOS smoke launches the six new identical-only bindings
  - 68906d504 `verify_wheel.sh` passes the identity check child `env=dict(os.environ)`
- Main carries each fix alone: c25fc046c, 81cc87979, 3f8b5bd69 (and the lanes' own merges).

## Artifacts kept across the restart (`~/mojolearn-evidence/release-0.8.6/`)

Nothing release-critical is under `/private/tmp`; the session scratchpad, its worktrees and the
Apple venv are gone after the restart and are not needed.

| path | what |
|---|---|
| `macos-wheel/mojolearn-0.8.6-py3-none-macosx_11_0_arm64.whl` | the macOS release wheel, sha256 eba69f83c9a94556003aeac71c4cf08b74c9c460ab55a4c11cb49d2a78a1d9ef |
| `linux-builds/cuda-sm_90a-db9047b9f/` | H100 sm_90a set, proof, leg evidence, `SHA256SUMS.so.txt` |
| `linux-builds/hip-gfx942-db9047b9f/` | MI300X gfx942 set, proof, leg evidence, `SHA256SUMS.so.txt` |
| `linux-builds/cuda-sm_89-db9047b9f/` | L40S sm_89 set, proof, leg evidence, `SHA256SUMS.so.txt` (see the table below for its result) |
| `linux-builds/hip-gfx942-hotaisle-8core-g/`, `h100-sm_90a-2f53960ca-refused/`, `h100-sm_90a-65a9e9302-verifier-reference/` | superseded builds kept as evidence |
| `apple-record/apple-m4.chunk00.json`, `chunk00.log`, `chunk00.rowtimes.tsv`, `chunk00.remaining.txt` | Apple chunk 00 recorded under the Metal slowdown, its log, per-lane timestamps, the lanes it did not reach |
| `apple-record/lanes.txt`, `lanes.00` to `lanes.06` | the 192 lanes of the wheel's harness and the seven lane groups |
| `scripts/resume_apple_record.sh` | resumes the Apple record after the restart (slowdown rerun and compare, remaining lanes, merge) |
| `scripts/check_sets_against_proofs.py` | checks each fetched set against its own proof and compares host bindings across legs |
| `scripts/record_body.template.sh`, `make_record_body.sh` | the AMD and NVIDIA record leg body from the installed Linux wheel (R2 presigned GET) |
| `scripts/hip_build_body_h.sh`, `amd_idle_body_b.sh` | the gfx942 build body at db9047b9f, the idle GPU 1 body for a 2x MI300X VM |
| `scripts/wheel_vs_changed.py` | shows which changed source files are members of a wheel |
| `scripts/mac_slot.sh` | the session's Mac Metal and CPU slot helper (Metal lock) |
| `logs/macos_build_g.log`, `logs/macos_verify_fixed.log` | the macOS wheel build and its five-interpreter verification |
| `reference/mojolearn-0.8.5-py3-none-macosx_11_0_arm64.whl` | the published 0.8.5 macOS wheel (sha256 matched PyPI), for the audit's comparison |

## Linux build sets (release-linux3 needs sm_89, sm_90a and gfx942 at one commit)

All three at **db9047b9f**, native inventory 1687 files, `source_sha256` a13ea3babe597f2b.
Each directory holds the fetched leg evidence (`remote/release-build/build/sets/...`,
`build-provenance.json`), `SHA256SUMS.so.txt` over every `.so`, and `leg-console.log`.
The copied bytes were checked against each proof's `extensions` and `host_extension`.

| set | box | path under `~/mojolearn-evidence/release-0.8.6/linux-builds/` | proof sha256 | result |
|---|---|---|---|---|
| cuda/sm_90a | RunPod H100 80GB HBM3, pod 24cymfdtasa4sq, deleted (HTTP 404) | `cuda-sm_90a-db9047b9f/` | a88f80fdc2f55238ceb876259e4b521a52cbbbe6ce4e306c942250ffb556ec4c | complete: 29 GPU + 15 host bindings, all four steps 0; leg exit 1 only from the sm_90 vs sm_90a witness false alarm (known issues) |
| hip/gfx942 | Hot Aisle 1x MI300X 8-core, VM 4f97be66, deleted (GET 404) | `hip-gfx942-db9047b9f/` | 5f112dc51f0e736f5b6d68e1961ab00d3154148066a3c1a17dfe6601dec22061 | complete: 29 GPU + 15 host bindings, all four steps 0 |
| cuda/sm_89 | RunPod L40S, pod he898ztyf5p1pg, deleted (HTTP 404) | `cuda-sm_89-db9047b9f/` | 913d15dac32ccc57d0957432aba596c5be7523995aed8c8833061cace8d814f9 | complete: 29 GPU + 15 host bindings, all four steps 0, leg exit 0 |

**Checklist 2b byte compare at db9047b9f: all 15 host bindings are byte-identical on sm_89,
sm_90a and gfx942** (`scripts/check_sets_against_proofs.py`, which also found 44 of 44 digests
matching each set's own proof, 0 missing, 0 mismatched). Pending step 1 below is done.

Host binding digests (the 15 the manifest ships), identical on all three legs:
byte_lm 7cafde6d5ef211df, core e953a666b58340fa, embedding_infer ef6d49a34578b7c7,
estimators f2a0b08141ca5352, forecast 781e27bc33b7d209, forest fd307d8f9f534a42,
gp_infer 78ff616419be9ec8, hdbscan_infer 140830afee2869db, ivf_search 365143c1cf6b1cb3,
linalg 9411e2ad46fb4c3a, metrics 166c85a9fd7baf38, mixture_infer 8367e370928a6dd6,
neural 01f85b24499926b9, svm b84ad447c4d6024e, tokenizer e596851fdc39cb45.

Earlier, superseded builds (kept only as evidence, not packable): `hip-gfx942-hotaisle-8core-g`
(2f53960ca, complete, first Linux run of the tokenizer Unicode generator),
`h100-sm_90a-2f53960ca-refused` (the tokenizer generator was not in the NVIDIA archive,
fixed by 9cc557e3d), `h100-sm_90a-65a9e9302-verifier-reference` (complete 65a9e9302 set used
only to run the verifier count check before each rental).

## macOS wheel

- `~/mojolearn-evidence/release-0.8.6/macos-wheel/mojolearn-0.8.6-py3-none-macosx_11_0_arm64.whl`,
  sha256 `eba69f83c9a94556003aeac71c4cf08b74c9c460ab55a4c11cb49d2a78a1d9ef`, 26,368,494 bytes,
  built at 2f53960ca (commit witness inside the wheel), `MOJOLEARN_PACKAGE_BYTE_LM=1`, one job.
- `verify_wheel.sh` with the fixed harness (68906d504): all 5 interpreters (3.10 to 3.14) pass
  fast, deterministic and identical (`logs/macos_verify_fixed.log`).
- No file changed between 2f53960ca and db9047b9f is in the wheel; its `_identity_break.py`,
  `_identity_trace_diff.py`, NOTICE and `host_surface.py` equal db9047b9f. Keep it; do not
  rebuild it. After the record, `packaging/macos/repack_post_record.py` replaces only
  `host_surface.py` (record lists), `verify_reference/table.json` and the identity column JSONs.

## Apple Metal record (from the installed macOS wheel, under `mac_slot.sh metal`)

From the saved macOS wheel in a clean python3.11 venv, the wheel's own harness copy
(byte-equal to `tools/identity_break.py` at db9047b9f), `--vendor apple-m4`,
`MOJOLEARN_COMMIT` = the wheel's build commit 2f53960ca, all 192 lanes, 9 fixtures,
2 fits per cell, batch part on, one core, in groups of 30 lanes, each group through the
Metal lock.

**Chunk 00** ran on the degraded GPU from 18:16:12 and was stopped after a whole lane,
never mid-lane: the stop waited for gbdt-lossguide's row and for the JSON to hold its 9 cells
with a stable size, then ended the process at 18:54:23; the Metal lock was released.

- Saved: `~/mojolearn-evidence/release-0.8.6/apple-record/apple-m4.chunk00.json`, sha256
  `b4757c8dd47e76f7e353dfa6f524ef562677ed2cde586be3f0933d573f2135e7`, 63 cells, `complete: false`
  (a stopped part), commit 2f53960ca, vendor apple-m4.
- Whole lanes (9 cells each), all 63 cells STABLE on train, infer, model and batch: rf-clf,
  rf-reg, et-clf, et-reg, gbdt-symmetric, gbdt-depthwise, gbdt-lossguide.
- Wall time 2280 s from the lock (18:16:12) to the last lane row (18:54:12).
- **Chunk 00 remaining** (`chunk00.remaining.txt`, 23 lanes): gbdt-rmse, kmeans, knn, knn-clf,
  knn-reg, dbscan, pca, pca-whiten, tsvd, ols, ridge, logistic, lasso, elasticnet, svc, kde,
  agglomerative, spectral, holtwinters, gemm-pinned, metrics, svr, arima.

Chunk 00's lanes: rf-clf, rf-reg, et-clf, et-reg, gbdt-symmetric, gbdt-depthwise,
gbdt-lossguide, gbdt-rmse, kmeans, knn, knn-clf, knn-reg, dbscan, pca, pca-whiten, tsvd, ols,
ridge, logistic, lasso, elasticnet, svc, kde, agglomerative, spectral, holtwinters,
gemm-pinned, metrics, svr, arima. Timing on the degraded GPU: rf-clf 86 s, rf-reg 68 s,
et-clf 25 s, et-reg 33 s, gbdt-symmetric 488 s (about 27 s per fit over 18 fits, including
the infer, model and batch parts), gbdt-depthwise 657 s, gbdt-lossguide 923 s (about 51 s per
fit); the GBDT lanes slowed lane by lane, consistent with the Metal command-queue leak.

**Remaining, not started** (`~/mojolearn-evidence/release-0.8.6/apple-record/lanes.01` to `lanes.06`):

- chunk 01 (30): gp, gpc, gpc-multiclass, umap, radius, standard-scaler, minmax-scaler, gbdt-ordered-rmse, gbdt-feature-freq, mlp, byte-lm, byte-lm-host-infer, byte-lm-host-train, mamba1, mamba2, mamba3, transformer, samba, rf-clf-entropy-log2-noboot, rf-clf-balanced-parallel, rf-reg-poisson, rf-reg-gamma-ig, et-clf-entropy-bestfirst, et-reg-bootstrap-parallel, gbdt-multiclass, gbdt-onevsall, gbdt-parametric-losses, gbdt-lossguide-newtoncosine, gbdt-pointwise-l2-bayesian-eval, gbdt-exact-mae
- chunk 02 (30): gbdt-categorical-ctr, gbdt-nan-modes, gbdt-adapter-clf, gbdt-adapter-reg, gbdt-query-rmse, gbdt-pair-logit, gbdt-adapter-score-weighted, rf-score-weighted, mamba2-dtlimit, transformer-window, byte-lm-resident, byte-lm-host-infer-threaded, samba-untied-dropout-accum, optim-sgd, optim-adam-clip, cross-entropy-arms, kmeans-random, kmeans-array, kmeans-weighted, dbscan-brute-l1, dbscan-weighted, kde-tophat-sqeuclidean, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-cosine-minkowski, kde-weighted, pca-full-whiten, ols-no-intercept, ols-weighted
- chunk 03 (30): ridge-no-intercept, logistic-l1, logistic-multiclass, logistic-elasticnet, logistic-unpenalized-no-intercept, elasticnet-l2end-no-intercept, svc-linear, svc-poly, svr-linear, knn-sqeuclidean, knn-manhattan, knn-chebyshev, knn-cosine, knn-minkowski-p3, knn-rbc, knn-clf-distance, knn-reg-distance, radius-manhattan, radius-chebyshev, radius-minkowski-p3, standard-scaler-no-mean, standard-scaler-no-std, minmax-scaler-clip, spectral-precomputed, holtwinters-multiplicative, kpss, arima-011, arima-seasonal-c, gp-normalize-y, gp-sample-y
- chunk 04 (30): gp-sample-y-normalize, gp-matern12, gp-matern32, gp-matern52-ard, gemm-transposed, metrics-classification, metrics-fowlkes-mallows, tokenizer, cross-val, cholesky, kernel-ridge, nystroem, rbf-sampler, gmm, gmm-random-init, gmm-sample, gmm-random-init-sample, hdbscan, hdbscan-leaf, bootstrap, permutation-test, monte-carlo, training-primitives, ivf, ivf-euclidean, ivf-extend, embedding, embedding-sort, kmeans-sqrt, kmeans-classic-pp
- chunk 05 (30): kmeans-cosine, par-forest, par-forest-et, par-boosting, par-kmeans, par-gram, par-logistic, par-cd, par-svm, par-gp, par-dbscan, par-scaler, par-arima, par-mlp, par-samba, par-byte-lm, par-queries-knn, par-queries-radius, par-queries-kde, par-reference-knn, par-reference-knn-reg, par-graph-agglomerative, par-graph-spectral, par-graph-umap, par-ordered-rmse, par-feature-freq, par-boosting-pointwise, par-holtwinters, par-byte-lm-model-pool, par-byte-lm-offload
- chunk 06 (12): par-samba-clip, iforest, iforest-tuned, par-iforest, par-forest-pool, par-gmm, par-resample, par-hdbscan, par-cholesky, par-kernel-ridge, par-nystroem, par-rbf-sampler

**Slowdown rule for the resuming session.** The finished chunk 00 lanes were written whole
under the exclusive lock, but they were recorded on a degraded GPU. After the restart, RERUN
every one of them and compare it bit for bit with the saved JSON (train hashes, infer, model,
reload and batch parts of every cell). Keep the rerun as the column; a mismatch is a finding
to report, never a reason to pick one run.

**The exact command** (does the rerun and comparison, chunk 00's remaining lanes, chunks 01
to 06, and the merge; a finished part is skipped when rerun):

    bash ~/mojolearn-evidence/release-0.8.6/scripts/resume_apple_record.sh \
      ~/mojolearn-evidence/release-0.8.6/scripts/mac_slot.sh <release/0.8.6 checkout>

(`scripts/mac_slot.sh` is the saved copy of the session's Metal and CPU slot helper; the
lock directory `/tmp/mojolearn-metal-slot` does not survive the restart, which is correct.)

## Linux wheel packed, audited and in R2 (2026-09-15 19:12)

Packed from the three db9047b9f sets and proofs with  at release tip
76d6e8a8a (, , ,
, ).

-  packed: 15 of 15 host bindings carried once,
  byte-identical across the three legs; payload  db9047b9f, 
  empty, 87 extensions, identity COMMIT witness db9047b9f.
- : auditwheel repaired to manylinux_2_35_x86_64, PASSED,  entries 0.
- : 214 files unchanged, receipt .
- **Final Linux wheel** ,
  sha256 , 70,862,796 bytes,
  214 members, 15 host bindings, 4 identity column files.
- Content audit PASSED on the final Linux wheel and the macOS wheel: NOTICE byte-equal to the
  release branch NOTICE with no "used under license", no GPT-2 table, fixture or vocabulary,
  no vendored environment,  imports and declares its bindings.
- In R2 at  (uploaded in 7 s,
  round-trip sha256 matched). Record legs fetch it with a presigned GET minted by
  .

## Pending steps, in order

1. DONE: byte compare of the 15 host bindings across the three Linux sets at db9047b9f
   (checklist 2b), 15 of 15 identical.
2. DONE: all three Linux sets are home; no rental is owed before packing.
3. Pack on the Mac: `packaging/linux/pack_wheel.py --profile release-linux3` with the three
   sets and proofs, then `packaging/linux/audit.sh` (Docker Desktop must be running) and
   `tools/strip_wheel_dir_entries.py`.
4. `tools/release_wheel_content_audit.py <linux wheel> <macos wheel>`: NOTICE equals the
   release branch NOTICE (copyright, Apache line, Modular trademark sentence, Modular
   components section, no "used under license"), no GPT-2 data, no vendored environment,
   `host_surface.py` imports.
5. Put the Linux wheel in R2 (`tools/dataset_store.sh presign-put`, measured 2.3 MB/s).
6. Record legs from the installed Linux wheel, AMD first, then NVIDIA (one device each,
   one identity process per GPU, 60-minute cap per leg, parts merged with
   `identity_break.py --merge`): body template `scripts/record_body.template.sh`, filled by
   `scripts/make_record_body.sh` (presigned GET, `--skip` of lanes already recorded, plus
   `identity --check`, `verify --quick`, the `host_surface.py` import and the
   `sys.executable` probe).
7. Apple chunks 01 to 06 after the restart (command below).
8. Diff: three GPU columns plus a CPU column, `identity_break.py --diff ... --require-columns 4`
   and the batch summaries; every OWED cell must now be recorded (565 distinct owed parts over
   56 lanes in the 2026-09-15 owed files, plus the 27 tokenizer cells); a DIVERGENT cell is a
   finding: print its hash from every column before naming a vendor.
9. Record commit on `release/0.8.6` under `bench/results/identity_break/<release dir>`:
   `TRAINING_GPU_COLUMNS` and the other record lists, `verify --all --emit-reference`,
   CHANGELOG, docs_facts. Merge the record back to main (cherry-pick, not a main merge).
10. Final Linux pack from the recorded proofs at the record commit (allowlist below) and the
    macOS repack; final content audit on both wheels; `host_surface.py` import and
    `verify --quick` from each installed final wheel.
11. STOP before publish. Report readiness against the checklist Finish line.

## Rules in force

- **Recorded bytes are shipped bytes (plan C).** Builds and records stay at db9047b9f. After the
  record only `python/mojolearn/host_surface.py` (only `TRAINING_GPU_COLUMNS`,
  `TRAINING_FIX_COLUMNS`, `TRAINING_FIX_LANES`) and `python/mojolearn/verify_reference/table.json`
  (package data outside the native inventory) may differ from the build proofs; the packer and
  audit enforce it (`POST_RECORD_FILES`, `post_record_differences`), every `.so` and every other
  inventoried file must match its proof. If that is impossible, fall back to a rebuild plus a
  quick identity check on any vendor whose digests differ.
- One AMD box and one NVIDIA box at a time; box order Hot Aisle, DigitalOcean, RunPod; a
  self-deleting lease and an on-box dead-man on every box, delete verified by API; 60-minute cap.
- One identity process per GPU; every Mac GPU process through `mac_slot.sh metal`.
- Only commits the coordinator names enter `release/0.8.6`, as cherry-picks of fix commits.
- Never `git stash`, `git add -A`, force push or rewrite history in the shared checkout; check the
  branch before every commit; never type a full SHA.
- Never dispatch workflows; `release-provenance.yml` is the publish step and is not run.
- Stop before publish.

## Known issues to carry into the report

- `tools/gemm_remote_leg.sh` campaign 7's final admission (line 1832) compares the
  preflight's `device_architecture` (`sm_90`, what an H100 reports) with the requested
  `sm_90a` and prints `Physical/source witness mismatch`, exiting 1, on every Hopper release
  build. It lacks the DEVIATION 2293 rule (`sm_90a` builds exactly the `sm_90` chip) that the
  build preflight applies. The line dates from the 0.6.1 candidate (fcdcabb39) and both 0.8.5
  H100 preflights also read `sm_90`. The H100 build at db9047b9f is complete regardless
  (every step 0, proof complete, preflight, proof and local inventories equal, 1687 files).
  Not fixed on this branch (not a named fix); the packer's own checks are what admit the set.

- `MOJO_PYTHON_LIBRARY` / `PYTHONEXECUTABLE` / `PYTHONPATH=:` set at the C level by the Mojo
  runtime: `bench/results/releases/2026-09-16-macos-0.8.6/runtime-environment-finding.md`
  (to report to Modular later; nobody contacted).
- A possible ~30x GBDT slowdown on Metal: the Metal command-queue leak above.
- The wheel ships no GPT-2 tokenizer tables (Andrew's decision); Holt-Winters inference is not in
  this release (the line-search fix is).

## Costs so far

Hot Aisle from the team balance where the leg logged it, RunPod estimated from runtime at the
list rate (H100 about $2.69/h); every box verified deleted.

| leg | commit | result | cost |
|---|---|---|---|
| Hot Aisle MI300X 13-core | 274a9d161 | refused at preflight | $0.05 |
| Hot Aisle MI300X 13-core (b) | 65a9e9302 | built, refused by the stale verifier; host digests kept | about $1.84 |
| Hot Aisle MI300X 8-core (g) | 2f53960ca | complete (superseded) | about $1.55 |
| Hot Aisle MI300X 8-core (h) | db9047b9f | complete | $1.54 |
| three Hot Aisle waits for stock | none | never created a VM | $0 |
| RunPod H100 autuqm2rlcgzms | 274a9d161 | refused at preflight | about $0.18 |
| RunPod H100 g1gf82jxk0633w | 65a9e9302 | built, refused by the stale verifier | about $1.15 |
| RunPod H100 agqhexa0xrr02g, c9z5jl50savs48, ytgyj8jtr2iwzb | 70d0940a6, 77dd4fd59, 6cb6fa571 | stopped when a named fix moved the branch | about $0.95 |
| RunPod H100 g08kkhe4zptotn | 2f53960ca | refused (tokenizer generator not archived) | about $1.40 |
| RunPod H100 24cymfdtasa4sq | db9047b9f | complete | about $1.40 |
| RunPod L40S he898ztyf5p1pg | db9047b9f | complete (23 minutes) | about $0.35 at an assumed list rate of about $0.86/h |

Hot Aisle balance at the end of the last AMD leg: $39.29.
