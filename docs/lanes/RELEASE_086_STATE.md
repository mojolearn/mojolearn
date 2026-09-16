# 0.8.6 release: resumable state (2026-09-15 evening)

What a session needs to resume the 0.8.6 release, whether or not this one survives.
Publishing needs Andrew's separate explicit "ship"; nothing here has been published.

**The Metal slowdown, corrected at 19:38 EDT.** No restart is required. `ioclasscount
AGXCommandQueue` reads 34 to 41 at rest, not the 6754 first reported: that number came from
`ioreg -l -c AGXCommandQueue`, which lists no nodes of the class, so two different populations
were being subtracted. The real fault is one process crossing the machine's ~512 command-queue
limit WITHIN ITS OWN LIFETIME (the leak lane watched a single `mojo` process hold 1211 of 1243
queues, climb to 2491, and the count fall to 34 about 15 s after it exited). Queues are released
on process exit, so a long identity run degrades and a short one does not. **Mitigation until
the per-call `DeviceContext` fix lands: run each lane group as its OWN process**, which
`scripts/resume_apple_record.sh` already does (one `python` invocation per chunk).

**Health confirmed 2026-09-15 evening, no restart taken.** Like for like on the same
`gbdt_direct.py` at 20,000 rows: SymmetricTree 8.289, 7.735 and 8.288 s against 21.5 to 23.4 s
while degraded; Depthwise 11.848, 14.082 and 13.373 s against 27.0 to 30.4 s. Queue counts held
flat at 34 across eleven fits in two processes. Evidence:
`~/mojolearn-evidence/metal-queue-leak-2026-09-15/gbdt-metal-health.log`, branch
`lane/metal-queue-leak` at be12003b8. What this shows is that the degraded state is GONE; it
does NOT establish a healthy per-fit figure, because no pre-slowdown GBDT Metal timing is
committed anywhere in the repo. Do not quote a "healthy" per-fit number in the record or here.

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

## Artifacts kept outside the scratchpad (`~/mojolearn-evidence/release-0.8.6/`)

Nothing release-critical is under `/private/tmp`, so a restart or a lost session costs nothing:
the session scratchpad, its worktrees and the Apple venv are all rebuildable and are not needed.

| path | what |
|---|---|
| `macos-wheel/mojolearn-0.8.6-py3-none-macosx_11_0_arm64.whl` | the macOS release wheel, sha256 eba69f83c9a94556003aeac71c4cf08b74c9c460ab55a4c11cb49d2a78a1d9ef |
| `linux-builds/cuda-sm_90a-db9047b9f/` | H100 sm_90a set, proof, leg evidence, `SHA256SUMS.so.txt` |
| `linux-builds/hip-gfx942-db9047b9f/` | MI300X gfx942 set, proof, leg evidence, `SHA256SUMS.so.txt` |
| `linux-builds/cuda-sm_89-db9047b9f/` | L40S sm_89 set, proof, leg evidence, `SHA256SUMS.so.txt` (see the table below for its result) |
| `linux-builds/hip-gfx942-hotaisle-8core-g/`, `h100-sm_90a-2f53960ca-refused/`, `h100-sm_90a-65a9e9302-verifier-reference/` | superseded builds kept as evidence |
| `apple-record/apple-m4.chunk00.json`, `chunk00.log`, `chunk00.rowtimes.tsv`, `chunk00.remaining.txt` | Apple chunk 00 recorded under the Metal slowdown, its log, per-lane timestamps, the lanes it did not reach |
| `apple-record/lanes.txt`, `lanes.00` to `lanes.06` | the 192 lanes of the wheel's harness and the seven lane groups |
| `scripts/resume_apple_record.sh` | runs the rest of the Apple record: the slowdown rerun and its bit-for-bit compare, the remaining lanes, then the merge, each chunk its own process |
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

**Chunk 00** ran from 18:16:12 in one process that slowed as it went (the command-queue limit
above) and was stopped after a whole lane,
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
gemm-pinned, metrics, svr, arima. Timing in that one slowing process: rf-clf 86 s, rf-reg 68 s,
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

**Slowdown rule.** The finished chunk 00 lanes were written whole under the exclusive lock,
but they were recorded by a process that had crossed the command-queue limit and slowed lane
by lane (27 s per fit at gbdt-symmetric, 51 s by gbdt-lossguide). RERUN every one of them in a
fresh process and compare it bit for bit with the saved JSON (train hashes, infer, model,
reload and batch parts of every cell). Keep the rerun as the column; a mismatch is a finding
to report, never a reason to pick one run. Run every chunk as its own process for the same
reason, one chunk at a time under the Metal lock.

**The exact command** (does the rerun and comparison, chunk 00's remaining lanes, chunks 01
to 06, and the merge; a finished part is skipped when rerun):

    bash ~/mojolearn-evidence/release-0.8.6/scripts/resume_apple_record.sh \
      ~/mojolearn-evidence/release-0.8.6/scripts/mac_slot.sh <release/0.8.6 checkout>

(`scripts/mac_slot.sh` is the saved copy of the session's Metal and CPU slot helper; the
lock directory `/tmp/mojolearn-metal-slot` is rebuilt on first use, so losing it is harmless.)

## Linux wheel packed, audited and in R2 (2026-09-15 19:12)

Packed from the three db9047b9f sets and proofs with `--profile release-linux3` at release tip
76d6e8a8a. Logs in `~/mojolearn-evidence/release-0.8.6/linux-wheel/`: `pack.log`, `audit.log`,
`strip.log`, `content-audit-both.txt`.

- `dist/mojolearn-0.8.6-py3-none-linux_x86_64.whl` packed: 15 of 15 host bindings carried once,
  byte-identical across the three legs; payload `source_commit` db9047b9f, `post_record_files`
  empty, 87 extensions, identity COMMIT witness db9047b9f.
- `audit.sh`: auditwheel repaired to manylinux_2_35_x86_64, PASSED, `mojolearn.libs` entries 0.
- `strip_wheel_dir_entries.py`: 214 files unchanged, receipt `dist/final/dir-entry-strip.json`.
- **Final Linux wheel** `dist/final/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl`,
  sha256 `7cab1aa3cfcde2f82123ce465410cc1b2d7d971dc4f46cc11084fb78b5cd7ecf`, 70,862,796 bytes,
  214 members, 15 host bindings, 4 identity column files.
- Content audit PASSED on the final Linux wheel and the macOS wheel: NOTICE byte-equal to the
  release branch NOTICE with no "used under license", no GPT-2 table, fixture or vocabulary,
  no vendored environment, `host_surface.py` imports and declares its bindings.
- In R2 at `releases/0.8.6/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl` (uploaded in 7 s
  at about 10 MB/s, round-trip sha256 matched). Record legs fetch it with a presigned GET minted
  by `scripts/make_record_body.sh`, whose key and sha256 are in `linux-wheel/r2-key.txt` and
  `linux-wheel/r2-sha256.txt`.

## Records taken so far (from the installed final Linux wheel)

**AMD leg 1**, Hot Aisle 1x MI300X 8-core, 2026-09-15 19:16 to 19:57, $2.10 (balance $39.24 to
$37.14), VM 34c731dc deleted and verified gone (HTTP 204 then GET 404). Evidence:
`~/mojolearn-evidence/release-0.8.6/records/amd-leg-1/` (column
`remote/identity/identity_break.amd-mi300x-gfx942.json`, sha256 starts f928d4003a578b99).

- On the box, from the R2 wheel (sha256 checked on arrival): `pip` exit 0, import reads
  0.8.6 vendor hip, harness copy equal to the checkout's `tools/identity_break.py`,
  `identity --check` exit 0 (3 columns, 166 lanes, wheel COMMIT witness), `verify --quick`
  VERIFIED (94 IDENTICAL, 0 DIVERGENT, 1 OWED, 13 N/A), `host_surface.py` imports and
  declares 15 wheel bindings.
- Column: vendor amd-mi300x-gfx942, commit db9047b9f, 558 cells, **62 of 192 lanes complete**,
  no partial lane. Verdicts: train 558 STABLE; infer 531 STABLE, 27 N/A; model 459 STABLE,
  99 N/A; batch 522 STABLE, 36 N/A. No MOVED, DIVERGENT or REFUSED cell.
- It stopped at its 2400 s bound (`identity_break_exit=124`), which is a clean partial: the
  harness writes the JSON after every lane. The 130 lanes still owed on AMD are in
  `~/mojolearn-evidence/release-0.8.6/records/amd-remaining-after-leg1.txt`; the next leg
  skips the 62 by passing that column as `<done-json>`.
- **Finding, same as macOS:** the `sys.executable` probe reads
  `changed False child BASE` on Linux too, so the Mojo runtime's C-level `PYTHONEXECUTABLE`
  moves a child process out of its venv there as well. The package's own children pass an
  explicit environment, so only callers that do not are affected
  (`bench/results/releases/2026-09-16-macos-0.8.6/runtime-environment-finding.md`).

Legs rent only from a CLEAN checkout: both runners refuse a dirty tree, so commit state-file
edits before renting.

## In flight at 2026-09-15 20:13 EDT

Four jobs at once, one GPU box per vendor. The CPU pod holds no GPU, so it does not count
against that rule.

| job | box | started | bound | output |
|---|---|---|---|---|
| AMD leg 2 | Hot Aisle 1x MI300X 8-core, VM 20c7ec81 | 20:00 | 3221 s | `bench/results/identity_break/2026-09-16_release-0.8.6/amd-leg-2` |
| NVIDIA leg 1 | DigitalOcean H100 `gpu-h100x1-80gb` nyc2, droplet 600855289 | 20:03 | 3021 s | the same directory, `nvidia-leg-1` |
| Apple chunks | this Mac under the Metal lock, one process per chunk | 20:04 | none | `~/mojolearn-evidence/release-0.8.6/apple-record/` |
| CPU column | RunPod CPU pod zz5ylla3sgkejs, 8 vCPU, $0.24/h, 75 minute lease | 20:13 | 3000 s | `scratchpad/rel086/cpu-column` |

AMD leg 2 skips the 62 lanes leg 1 recorded. NVIDIA leg 1 has all 192 owed. Hot Aisle balance
read $36.65 before AMD leg 2 (floor 500 cents) against 299 cents an hour for the VM, so the
leg cannot exceed it.

**The slowdown rerun is home, and NOTHING MOVED.** The seven lanes chunk 00 recorded while the
Metal queue count was high (rf-clf, rf-reg, et-clf, et-reg, gbdt-symmetric, gbdt-depthwise,
gbdt-lossguide) were rerun in a fresh process from the saved macOS wheel and compared with the
saved JSON cell by cell: **63 cells compared on hashes, infer, model, reload and batch, none
missing, BIT FOR BIT IDENTICAL** (`apple-record/chunk00-rerun-compare.txt`,
`apple-m4.chunk00-rerun.json`). So the degraded state cost time and no bits, and the rerun
stands as the column's copy of those lanes with the slowdown JSON kept beside it as evidence.

### Two measurements of the same degraded window disagree, and nobody has established why

Read both of these before drawing any conclusion about that GPU. Either one alone invites a
confident and possibly wrong answer.

| what was measured | degraded | after | ratio |
|---|---|---|---|
| `gbdt_direct.py` at 20,000 rows, SymmetricTree, the leak lane's own before and after | 21.5 to 23.4 s | 7.7 to 8.3 s | about 2.7x |
| `gbdt_direct.py` at 20,000 rows, Depthwise, the same | 27.0 to 30.4 s | 11.8 to 14.1 s | about 2.2x |
| the seven identity lanes chunk 00 recorded under the slowdown, whole run | 2194 s | 2057 s | 1.06x |

So one workload says the machine was running somewhere near 2.2x to 2.7x slow and is not now,
and the other says the same window cost 6 percent, with the same per-lane shape both times
(the three gbdt lanes dominate each run; degraded they took 488, 657 and 923 s per
`chunk00.rowtimes.tsv`). Both cannot be describing the same effect.

Readings that would explain it, none of which anyone has evidence for, and none worth a box or
a lane to settle: the identity harness may never accumulate enough command queues in one lane
to cross the limit while `gbdt_direct.py` does; the degraded window may have partly cleared by
the time those seven lanes ran; or the two workloads may differ in shape enough that only one
is sensitive to the leak at all.

What IS settled: the rerun of those seven lanes came back **bit for bit identical across all
63 cells** (hashes, infer, model, reload and batch, none missing), so whatever the degraded
window did, it cost time and not one bit, and the rerun stands as the column's copy of those
lanes with the slowdown JSON kept beside it as evidence.

What must NOT be drawn: no claim about the machine's state, in either direction, rests on the
6 percent figure. It is not evidence that the GPU was healthy while those lanes ran, and it is
not evidence that the leak lane's 2.2x to 2.7x is wrong. There is still no pre-slowdown Metal
GBDT timing anywhere, so no healthy per-fit figure is claimed here either.

**The fourth column.** The diff wants a CPU column at the wheel's build commit, and the last
record shipped only the three GPU columns, so this release takes one. It runs on a box with no
GPU over the 159 lanes `python/mojolearn/host_surface.py --covered-lanes` names, and the four
column diff is then scoped to the 154 of those that `--record-covered-lanes` names (the five
left out are embedding, embedding-sort, ivf, ivf-euclidean and kmeans-sqrt; the 38 harness
lanes outside the covered list have no CPU training path at all). The CPU gate scopes its own
diff exactly this way.

*Attempt 1 took the column from the installed release wheel and that cannot work.* Pod
zz5ylla3sgkejs, $0.0187, deleted and verified gone. Everything about the wheel checked out on
the box (sha256 equal to R2, `vendor()` reads cpu, harness byte equal to the checkout's
`tools/identity_break.py`, `identity --check` and `verify --quick` exit 0, 15 wheel bindings
declared), but of 1386 cells only 639 read STABLE and 747 read REFUSED, by name: "no host
binding covers `_mojolearn_rf`" and "`_mojolearn_trees`". The wheel ships the 15 wheel
families; the manifest declares 32, and the CPU gate builds all 32 from source. The `par-*`
lanes refused for the same reason underneath: `_parallel_pool` takes its CPU reference route
only when those host bindings exist. This is not a defect in the wheel, which ships what
`--wheel-families` names; it is the wrong source for this column.

*Attempt 2 builds the column the way the gate does.* All 32 host families compiled from
db9047b9f in a detached worktree (`scratchpad/wt-cpu-column`), 16 vCPU, 120 minute lease, the
159 covered lanes as 16 sharded processes through `tools/cpu_identity_gate_check.py
run-column`, which merges the parts itself. The body refuses unless the package reads back
`vendor() == cpu`, reads every one of the 32 bindings back as the kernel matrix's cpu column
first, and ends with the gate's own `column` judgement of the merged JSON. Script:
`scripts/cpu_column_body.template.sh`. The wheel based scripts
(`scripts/cpu_record_body.template.sh`, `scripts/make_cpu_record_body.sh`) stay as the record
of attempt 1.

Local `release/0.8.6` sat at db9047b9f while this work was pushed to `origin/release/0.8.6`
from the branch `fix/release-post-record-allowlist`. The local branch is now fast-forwarded to
7e23bc670, so the two agree again.

## Pending steps, in order

1. DONE: byte compare of the 15 host bindings across the three Linux sets at db9047b9f
   (checklist 2b), 15 of 15 identical.
2. DONE: all three Linux sets are home; no rental is owed before packing.
3. DONE: packed with `pack_wheel.py --profile release-linux3`, repaired by
   `packaging/linux/audit.sh` (Docker) and stripped by `tools/strip_wheel_dir_entries.py`
   (see the section above).
4. DONE for both wheels with `tools/release_wheel_content_audit.py`; rerun it on the FINAL
   wheels after the record repack, since the record changes package data in both.
5. DONE: the final Linux wheel is in R2 (see the section above).
6. Record legs from the installed Linux wheel, AMD first, then NVIDIA (one device each,
   one identity process per GPU, 60-minute cap per leg, parts merged with
   `identity_break.py --merge`): body template `scripts/record_body.template.sh`, filled by
   `scripts/make_record_body.sh` (presigned GET, `--skip` of lanes already recorded, plus
   `identity --check`, `verify --quick`, the `host_surface.py` import and the
   `sys.executable` probe).
7. Apple chunk 00's rerun and remaining lanes, then chunks 01 to 06 (command below), one chunk
   per process under the Metal lock.
8. Diff: three GPU columns plus a CPU column, `identity_break.py --diff ... --require-columns 4`
   and the batch summaries; every OWED cell must now be recorded; a DIVERGENT cell is a
   finding: print its hash from every column before naming a vendor. The driver is
   `scripts/release_diff.sh` (four column diff scoped to the 154 record-covered lanes with
   `--owed-json`, the three column diff over every lane, the batch rows, the DIVERGENT report
   and the owed check).
   - **The owed arithmetic.** The 33 owed files under
     `bench/results/identity_break/2026-09-15_*` name 583 distinct cell parts over 56 lanes
     (259 model, 126 infer, 109 batch, 89 train), aggregated to
     `scratchpad/rel086/owed_parts_2026-09-15.json`. Of those, 18 are already hashed by the
     166-lane record, which leaves the 565 this file has been quoting. The new record must
     fill all 583; the check prints filled and still empty counts and lists every part left.
   - **The DIVERGENT report reads the column JSONs, never the diff's rows.** A DIVERGENT row
     overwrites the FIRST column's hash with its "parts differ" note (`shown[0]` in
     `identity_break.py`), so parsing that row names the wrong column. Watched failing first on
     the 166-lane record: reading the rows named all three columns as standing alone, while
     reading the JSONs gives one DIVERGENT cell, kmeans-sqrt/wide, with centers, labels and
     scales agreeing on every column and `inertia` ALONE on nvidia (apple, amd and the fourth
     column 52ea06cbbcc24144, nvidia 1a7e4ac5b8c0caaf), which is the known answer.
   - The owed check also had to be watched failing: a cell's `infer`, `model` and `batch` are
     LISTS (whose first entry may be an `n/a:` sentinel) and train hashes live in
     `cell["hashes"]`, so an earlier version read every part as empty and would have reported
     583 of 583 owed however good the record was.
9. Record commit on `release/0.8.6` under `bench/results/identity_break/<release dir>`:
   `TRAINING_GPU_COLUMNS` and the other record lists, `verify --all --emit-reference`,
   CHANGELOG, docs_facts. Merge the record back to main (cherry-pick, not a main merge).
   The exact pieces, so none of this is rediscovered:
   - the record lists live in `python/mojolearn/host_surface.py`: `TRAINING_GPU_COLUMNS` (the
     three column paths, today the 166-lane record), `TRAINING_FIX_COLUMNS` and
     `TRAINING_FIX_LANES` (kmeans-sqrt, embedding, embedding-sort, ivf, ivf-euclidean). These
     three names are also the post-record allowlist the packer admits.
   - the table: `MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify --all
     --emit-reference python/mojolearn/verify_reference/table.json`, run from the release
     checkout. It needs no host binding and no GPU, and it walks
     `bench/results/identity_break` (or the `--records` paths given), so the new record
     directory must be COMMITTED first or the table will not carry it. It refuses on a FAST
     build, and prints a summary plus the table's byte size.
   - the facts: `python3 tools/docs_facts.py --check` fails when a doc disagrees with the
     tree, `--write` rewrites the marked spans (the version span and the CHANGELOG date are
     the ones this release moves).
   - the CHANGELOG's 0.8.6 heading still reads `(unreleased 2026-09-16)` and its opening
     paragraph still calls the identity record OWED. Both are edited when the record lands,
     and the date is set only when the wheels go out.
10. Final Linux pack from the recorded proofs at the record commit (allowlist below) and the
    macOS repack; final content audit on both wheels; `host_surface.py` import and
    `verify --quick` from each installed final wheel.
11. STOP before publish. Report readiness against the checklist Finish line.

## Exact commands to resume

Everything below runs from a clean checkout of `release/0.8.6` (a fresh
`git worktree add --detach <dir> origin/release/0.8.6`, since a new session has no worktree
under `/private/tmp`). `E=~/mojolearn-evidence/release-0.8.6`.

**The wheel the record legs install** (already in R2, nothing to re-upload):

- local: `$E/linux-wheel/dist/final/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl`
- R2 key: `releases/0.8.6/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl` (`$E/linux-wheel/r2-key.txt`)
- sha256: `7cab1aa3cfcde2f82123ce465410cc1b2d7d971dc4f46cc11084fb78b5cd7ecf` (`$E/linux-wheel/r2-sha256.txt`)

**1. Fill a record leg body** (mints a short-lived presigned GET; the body is never committed).
`<done-json>` is the merged column so far, so the leg skips lanes already recorded, or `none`:

    bash $E/scripts/make_record_body.sh <label> \
      $E/linux-wheel/dist/final/mojolearn-0.8.6-py3-none-manylinux_2_35_x86_64.whl \
      "$(cat $E/linux-wheel/r2-key.txt)" 2400 <done-json|none> /tmp/record_body.sh

Labels: `amd-mi300x-gfx942`, `nvidia-h100-sm_90a`. The body installs the wheel into a clean venv,
runs `identity --check`, `verify --quick`, the `host_surface.py` import and the `sys.executable`
probe, then one identity_break process over the lanes not yet recorded.

**2a. NVIDIA record leg (DigitalOcean H100, the simplest body runner; about $4.41/h, 60-minute cap):**

    MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_GEMM_LEG_EXTRA=/tmp/record_body.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/identity_break/2026-09-16_release-0.8.6/nvidia-leg-1 \
      bash tools/do_extra_leg.sh nv --minutes 60 --skip-gates        # add --dry-run first

(The RunPod path would be `gemm_remote_leg.sh nvidia --payload gemm` with
`MOJOLEARN_GEMM_LEG_EXTRA`, but that payload also builds an Apple reference card locally unless
`MOJOLEARN_GEMM_LEG_LOCAL_CARD` names an existing one, so prefer DigitalOcean here.)

**2b. AMD record leg (Hot Aisle; 13core when in stock, else 8core):**

    MOJOLEARN_HOTAISLE_SPEC=8core MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_GEMM_LEG_EXTRA=/tmp/record_body.sh \
    MOJOLEARN_GEMM_LEG_OUT=bench/results/identity_break/2026-09-16_release-0.8.6/amd-leg-N \
    MOJOLEARN_HOTAISLE_LANE=release086-record-amdN \
      bash tools/hotaisle_leg.sh amd --rent --skip-gates             # dry run: drop --rent

Each leg's evidence lands under `<out>/remote/identity/`: the column JSON
`identity_break.<label>.json`, `identity_break.log`, `record.txt`, `identity_check.log`,
`verify_quick.json`, `host_surface_import.txt`, `sys_executable.txt`, `installed_sha256.txt`.
Copy each leg's directory to `$E/records/` as soon as it is home.

**3. Merge a vendor's parts** into one column before the diff:

    python3 tools/identity_break.py --merge <part JSONs> --json <vendor>.json

**4. Apple chunks:** `scripts/resume_apple_record.sh` (above).

**5. Diff, record commit, final pack and audit:** pending steps 8 to 11 below.

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
- Andrew, 2026-09-15 19:50 ET: **no more new lanes.** Finishing 0.8.6 is existing work and
  continues; nothing new opens around it.

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
