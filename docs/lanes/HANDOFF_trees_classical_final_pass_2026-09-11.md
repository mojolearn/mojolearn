# Handoff: trees and classical ML, 2026-09-11 night (0.8.3 shipped, final speed pass wound down)

Written by the orchestrator session bd222cd8 on Andrew's order ("let them
wind down when done... and then create handoff file with next steps"). The
neural session (mojolearn-d1, continuing mojolearn-df) owns transformer/,
gemm/, core/step_phase.mojo and the byte LM, and DEVIATIONS 2640 to 2659.
This file covers trees and classical only. Read ENGINEERING_RULES.md
sections 9 and 10 and bench/OPPONENT_REFERENCE.md before touching a lane.

## 1. State now (2026-09-12, after the merges)

- **Everything is merged into main and pushed.** All seven finish lanes plus
  their follow-ups; section 2 has the commits.
- **Nothing is renting.** Every pod of this session is VERIFIED GONE by the
  RunPod API (HTTP 404, and the account listing shows none of ours):
  forest `8gsem9f3thnhvu` (DELETE 204 then GET 404 at 22:59:22Z), kde
  `ur95zh3h9qbx2p`, svm `4oih8bhjepzlmm`, knn `zwmta1li2twxx2`,
  linear-cluster `1yxsotvvcbxtuu`, pointwise `eqxtzdcpctpnkh`, gbdt
  `dk14p0y15w0ig5`. Never touch `samba-*` (another program) or
  `mojolearn-gemm-nvidia-*` (the neural session).
- **Three legs died before handing back their final numbers** (their agents
  stalled, then the pods' watchdogs fired). What was lost is listed as the
  first item of section 6. Nothing merged depends on recovering them, but one
  merged switch set is unproven for speed, which section 2 marks.
- Every measurement here is our IDENTICAL arm against the opponent's FAST arm
  on the same pod unless a row says otherwise. Ratios are ours divided by the
  opponent's time; after/before ratios below 1 mean we got faster.

## 2. What landed on main

| item | main commit | result |
|---|---|---|
| OLS conditioning, DEVIATIONS 2620, 2621 (tall) and 2622 (wide) | df77d6c1, 26683dba | power-of-two Gram equilibration and a relative eigenvalue cutoff; Istella-S R2 -115.6 to 0.332; `glm/ols_main.mojo` 17/17 on M4, H100 and MI300X |
| NVIDIA SVC above 512 rows, DEVIATION 2623 | f7a10cd9 | kernel-matrix row gives NVIDIA above width 512 the halving-tree schedule; `svm/svc_main.mojo` 44/44 on H100, M4, MI300X; fits at n=400/600/2000 hash 457e29b82bca9df9, 733a383c5699f427, 2b66bc991a9c9ed0 on all three |
| pointwise searcher identity, DEVIATION 2624 | 36ca51fd | document-block multiplier 1 in the ordered tiers; new check passes on H100, MI300X, M4; pointwise and greedy hashes on five 1M synthetic fixtures equal H100 vs MI300X; the opt-in pointwise arm is 1.6x to 2.2x slower |
| 0.8.3 release merged | a824d9da | see section 3 |
| SVM block-solve schedule row, DEVIATIONS 2627, 2628 | 3671a1e5 | opt-in schedules, default unchanged; see section 5 |

### Merged the same night (the final speed pass)

| lane | main commit | result |
|---|---|---|
| knn-finish, DEVIATION 2631 | af194de1 | NVIDIA query tile 4,096: taxi 0.900, Istella-S 0.952, **geomean 0.926**, recall and every bit unchanged. 2667 fused select measured NEGATIVE (geomean 1.523) and stays opt-in; 2668 UMAP live-row NOT flipped (wins taxi 0.9323, LOSES Istella-S 0.9636 against the shipped 0.9737) |
| kde-finish, DEVIATIONS 2625, 2626, 2660 | 44bf06a0 | taxi 0.811, Istella-S 0.311, **geomean 0.502**, log-likelihood identical to the last digit |
| svm-finish, DEVIATIONS 2665, 2666 | aede96e8 | taxi 0.8886, Istella-S 0.8498, **geomean 0.869**, accuracy unchanged. Also fixed a REAL BUG in 48f92b19: the RARY_TREE arity row had no dispatch branch, so that define silently selected the default and three earlier "rary" measurements measured nothing |
| forest-finish, DEVIATIONS 2637, 2638, 2663 | b3d6eeb5 | RF **0.775**, ET **0.929**, iforest **0.108**; 2663 at bw16k **0.880** and bw32k **0.865**. Every quality delta exactly +0.000000. ET is SLOWER on taxi (1.018) and is not claimed there |
| gbdt-finish, DEVIATIONS 2634, 2635, 2636 (2661 opt-in) | 5030ebc3 | identity 36/36 on four sets, sub-byte 16/16 PASS. **SPEED NOW MEASURED** (2026-09-12, pod n2ltmel2optel5): 2634 geomean 0.9603 FLIP and 2635 0.9913 FLIP both stay on; **2636 is 1.0006 NO FLIP and was turned OPT-IN in 31f0142f**; total baseline->all 0.9526. Quality equal in all 24 cells |
| pointwise-speed | 652ccd8f | merged on Andrew's instruction while the lane was still running; its verdicts are owed |
| linear-cluster-istella, DEVIATION 2671 | 0c6c1249 | Jacobi with two barriers per rotation: OLS geomean 0.9610, PCA 0.9435, 0 of 48,400 matrix and 0 of 48,400 eigenvector cells differing. **DEVIATION 2672 RESOLVED as a flip**: the single 1.066 instance was noise; pooled over five race instances it reads taxi 0.8979 and Istella-S 0.9943, geomean 0.9449, digests equal. k-means 2672 and OLS/PCA 2671 both ship on |

Apple M4 gates passed before each merge: check-knn-identity, check-knn,
query_batch_check; kde_check 15/15 and kde_stage_profile; svc_main 44/44 plus
the three fit hashes; check-if, check-forest-resident-layouts and extratrees
device_batched_check (45 cells, sabotage arms move thousands of nodes).
## 3. Release 0.8.3 (published 2026-09-11)

- Contents: 2620 to 2623 only, on top of v0.8.2. Branch release-0.8.3,
  wheel commit f8b65ee2, published-docs commit 0768c753, tags
  alpha-api-0.8.3-20260911 and v0.8.3.
- PyPI: Linux x86-64 (sm_89, sm_90a, gfx942) 20:17Z, run 34643281339,
  sha256 c8c2975fdb70fd51...; macOS arm64 20:35Z, run 34643372856, sha256
  60573c840b4b7b01....
- Finish line: `pip install mojolearn==0.8.3` from PyPI on the Apple M4
  imports 0.8.3 and gives the three SVC hashes above.
- Installed Linux qualification was partial. On gfx942 (DigitalOcean) and
  sm_90a (RunPod H100) every IDENTICAL job passed and all 29 smoke lane
  hashes were equal across the two. Every FAST and DETERMINISTIC job failed
  in `run_installed.py` asking `_backend.binding('_mojolearn')` in a tier
  that DEVIATION 2490 removed. The fix to the job loop (cc117fdf) is on main
  but was not on the release branch, because `tools/linux_surface_qualification.sh`
  is in the native inventory the build proofs bind. sm_89 was never
  qualified installed (no L40S, RTX 4090 or L4 stock on RunPod). An
  installed-wheel SVC check on a RunPod H100 gave the three hashes above.
- Evidence (copied out of the scratchpad):
  `~/mojolearn-evidence/release-0.8.3-2026-09-11/` and
  `~/mojolearn-evidence/release-0.8.2-2026-09-11/`.

## 4. Same-pod H100 baselines from the final pass (source 36ca51fd)

### Gradient boosting (1M rows, 100 trees, depth 6, 5 interleaved rounds)

| policy | dataset | opponent | opponent ms | ours ms | ratio | ours logloss / AUC | opponent logloss / AUC |
|---|---|---|---|---|---|---|---|
| symmetric | taxi | CatBoost GPU | 690.9 | 349.5 | 0.51x | 0.525735 / 0.619460 | 0.525904 / 0.618511 |
| depthwise | taxi | CatBoost GPU | 852.6 | 473.1 | 0.55x | 0.525086 / 0.621421 | 0.525047 / 0.621092 |
| depthwise | taxi | XGBoost GPU | 366.7 | 473.1 | 1.29x | same | 0.525227 / 0.620101 |
| lossguide | taxi | CatBoost GPU | 1151.7 | 1007.6 | 0.87x | 0.525504 / 0.619386 | 0.525110 / 0.620541 |
| lossguide | taxi | XGBoost GPU | 497.2 | 1007.6 | 2.03x | same | 0.525227 / 0.620101 |
| symmetric | Istella-S | CatBoost GPU | 1526.9 | 1373.6 | 0.90x | 0.138653 / 0.966990 | 0.138982 / 0.966853 |
| depthwise | Istella-S | CatBoost GPU | 1676.0 | 1926.9 | 1.15x | 0.126517 / 0.971896 | 0.125819 / 0.972797 |
| depthwise | Istella-S | XGBoost GPU | 1693.8 | 1926.9 | 1.14x | same | 0.125110 / 0.973774 |
| lossguide | Istella-S | CatBoost GPU | 2424.8 | 2428.0 | 1.00x | 0.122045 / 0.975037 | 0.121363 / 0.975309 |
| lossguide | Istella-S | XGBoost GPU | 1883.4 | 2428.0 | 1.29x | same | 0.125110 / 0.973774 |

Model hashes held one value per cell in 5 of 5 rounds (taxi symmetric
90c3558501933f47, depthwise 40c1683b9e0eb151, lossguide 0dd8bcfc3c3a4a1d;
Istella-S 238d3abce0cabf43, 5d053cd086658072, 6182fd2bee4fb941). LightGBM's
pip wheel has no CUDA learner, so it has no row.

### Forests (1M rows, 3 interleaved rounds)

| family | dataset | opponent and device | opponent ms | ours ms | ratio | logloss ours / opponent |
|---|---|---|---|---|---|---|
| RandomForest | taxi | cuML 26.08, GPU | 1980 | 877 | 0.44x | 0.525910 / 0.525800 |
| RandomForest | Istella-S | cuML 26.08, GPU | 3668 | 2113 | 0.58x | 0.145560 / 0.145504 |
| ExtraTrees | taxi | scikit-learn 1.9.1, CPU, 24-core pod quota | 4075 | 2058 | 0.51x | 0.527541 / 0.527011 |
| ExtraTrees | Istella-S | scikit-learn 1.9.1, CPU, 24-core pod quota | 16259 | 6073 | 0.37x | 0.188191 / 0.187901 |
| IsolationForest | taxi | cuML 26.08, GPU | 63 | not measured | owed | - |
| IsolationForest | Istella-S | cuML 26.08, GPU | 1526 | not measured | owed | - |

cuML 26.08 does ship IsolationForest (the Aug 28 85 ms row was it, against
our FAST arm). Our iforest arm was refused by a harness bug, fixed on
lane/forest-speed 759e1aac but not re-run. RF and ET hashes equal the AMD
rows (RF taxi d8f64dae01de00bd, Istella-S 574b24d0d7af51d0; ET taxi
e683f121d11f59dd, Istella-S 40b1c5b03ba40420); identity_break 45/45.

### k-NN and UMAP (cuML 26.8.0)

| workload | cuML ms | ours ms | ratio | quality |
|---|---|---|---|---|
| kNN taxi, 400k index x 4k queries, d11, k10 | 8.19 | 24.79 | 3.03x | recall@10 ours 0.99915, cuML 0.99925 |
| kNN Istella-S, same shape, d220, k10 | 51.11 | 124.71 | 2.44x | recall@10 ours 0.923025, cuML 0.92205 |
| kNN synthetic 400k x 4k x d32, k10 | 10.05 | 23.66 | 2.36x | same neighbors in the same order as cuML |
| kNN synthetic, k15 | 10.22 | 26.21 | 2.57x | same |
| UMAP taxi, 100k rows | 5297 | 3564 | NOT QUOTED | trustworthiness ours 0.906, cuML 0.966; our rounds bimodal |

### Linear models and clustering (taxi 4,000,000 x 11, 5 interleaved rounds)

| family | cuML ms | ours ms (main) | ratio | quality |
|---|---|---|---|---|
| LinearRegression | 23.12 | 263.5 | 11.40x | R2 0.908836 cuML, 0.908837 ours |
| PCA | 20.99 | 28.69 | 1.37x | EVR sum 0.99786 / 0.997861 |
| KMeans | 128.2 | 304.0 | 2.37x | inertia 1.20192e8 cuML, 1.20628e8 ours |

cuML KMeans gave a different centroid digest every round; ours held one.
Istella-S was not measured (download 980 s).

### SVC and KDE (earlier same-pod races, still current on main)

| family | dataset | cuML ms | ours ms | ratio |
|---|---|---|---|---|
| SVC, 10k training rows | taxi | 415 | 857 | 2.06x |
| SVC, 10k training rows | Istella-S | 20.64 | 70.7 | 3.43x |
| KDE, 100k x 2k queries | taxi (d11) | 2.37 | 39.2 | 16.55x |
| KDE, 100k x 2k queries | Istella-S (d220) | 6.79 | 219 | 32.25x |

## 5. Lane branches

All seven finish lanes are MERGED (section 2). `lane/cpu-sweep` (byte LM GPU
logits, verified 1680/1680 equal to the CPU reference on an H100 and an
MI325X) was ALSO merged by the neural session after it renumbered its
DEVIATION 2660 to 2658, which resolved the collision with KDE's 2660. Main
now has exactly one meaning for 2660.

Branches still ahead of main, deliberately not merged here:

| branch | what is on it | why not merged |
|---|---|---|
| lane/amd-gbdt-identity-verify | nothing unique | zero non-merge commits ahead of main; only a stale merge commit |
| lane/trees-hotaisle-run | trees-hotaisle body switch | its own commit says "NOT RUN: lane stopped as a duplicate" |

**Deviation-number collisions are a recurring failure here.** 2624 collided
with `lane/cpu-speed` and 2660 collided with `lane/cpu-sweep`, both byte-LM
lanes from other sessions drawing from the same range; the 2660 one was caught
and renumbered to 2658 before it merged. Always run
`git grep -n "DEVIATION 26[0-9][0-9]"` across main AND every open branch
before taking a number.

## 6. Next steps, in the order I would take them

Rules: runs on RunPod NVIDIA H100 through `tools/trees_leg.sh` (one pod per
lane, watchdog baked in, reap and confirm 404); subagents never run anything
on the Mac; the flip rule of section 9 (geomean of after/before over taxi and
Istella-S below 1, quality not worse on EITHER dataset); same bits proven on
the H100, then gate the Apple M4 and an AMD box before merging.

1. **Re-run the three results that died with their pods.** All three had paid
   their setup cost and were minutes from finishing.
   - **GBDT 2634, 2635, 2636 speed: DONE 2026-09-12.** Re-run on a fresh
     H100 (pod n2ltmel2optel5). 2634 six-cell geomean 0.9603 FLIP and 2635
     0.9913 FLIP stay default-on; **2636 measured 1.0006 and is now OPT-IN**
     behind `-D MOJOLEARN_2636_PARALLEL_STAGING=1` (commit 31f0142f, gated on
     the Apple M4: both arms build to different binaries so the switch still
     flips, sub-byte 16/16 PASS on each and byte-identical between them).
     Quality equal in all 24 cells. Tables in `bench/OPPONENT_REFERENCE.md`.
     DEVIATION 2661 also measured: six-cell geomean 0.9968, identity clean
     (IDENTICAL=36, sub-byte 16/16). It clears the threshold but STAYS OPT-IN,
     because the margin is 0.3 percent with gbdt-symmetric at 1.000 and because
     it was timed with 2636 ON underneath, which is no longer the default.
     Re-measure it against the new default before promoting it.
     The CatBoost and XGBoost opponent cells for this tuple are now DONE too,
     on a second pod (u4elzj1eo486ps) that had to repay the whole setup because
     I reaped the first one ~20 minutes early. Rows in
     bench/OPPONENT_REFERENCE.md: ours beats CatBoost on four of six cells
     (symmetric 0.437x taxi and 0.830x Istella-S on CatBoost's own policy) but
     is 1.030x on Istella-S depthwise and 0.997x parity on Istella-S lossguide,
     and XGBoost is faster wherever it competes, up to 1.995x on taxi lossguide.
     That XGBoost lossguide gap is the biggest single number left on the GBDT
     board. Rule now in ENGINEERING_RULES 11: never cancel a run that is owed.
   - **Pointwise Istella-S** (`ours-ab` and the greedy control). The taxi
     result stands and is strong: the opt-in pointwise arm 1803.3 -> 795.4 ms
     (0.441x, medians of nine interleaved rounds), logloss 0.525925 ->
     0.525668, greedy control flat at 309.3 -> 308.9 ms with an unchanged
     hash. DEVIATION 2670 stays opt-in until Istella-S plus M4 and AMD land.
     DEVIATION 2669 was proven IMPOSSIBLE rather than built.
   - **Istella-S DBSCAN median** for the linear-cluster lane. Its taxi row
     stands (ours 1129.8 ms against cuML brute 13631.0, 2900 clusters, ARI
     0.99999999908) and round 0 on Istella-S ran 337 s against cuML's 55 s,
     so expect that shape to favour cuML at 220 features.
2. **`identity_break` for DEVIATION 2663: the Apple M4 is DONE, AMD gfx942 is
   still owed.** 2663 changes a SHIPPED DEFAULT ON EVERY VENDOR (ExtraTrees
   device batch width 4,096 -> 16,384), so it needs every vendor.
   - **Apple M4, against shipped main 5b7e1e41: cells=45 stable=45 moved=0
     refused=0, and diffed against the H100's own `dflt` set (the same 16,384
     default) `summary: IDENTICAL=45`** -- every rf-clf, rf-reg, et-clf, et-reg
     and iforest fixture equal bit for bit across the two vendors, the
     denormal and denormal_ftz pairs included. Evidence
     `bench/results/forest_finish_2026-09-11/logs/ib_2663_apple_m4.json`;
     `if_check` 8/8 and `device_batched_check` (45 cells) also pass at the new
     default.
   - **AMD gfx942 is what remains.** Run it the same way:
   ```
   for f in bindings/build_*.sh; do MOJOLEARN_NUMERIC_MODE=identical bash "$f"; done
   MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python3 tools/identity_break.py \
     --lanes rf-clf,rf-reg,et-clf,et-reg,iforest --vendor amd-mi300x \
     --json /tmp/ib_2663_amd.json
   PYTHONPATH=python python3 tools/identity_break.py --diff \
     bench/results/forest_finish_2026-09-11/logs/ib_2663_apple_m4.json /tmp/ib_2663_amd.json
   ```
   TWO TRAPS, both of which cost me repeated runs: `--diff` takes JSON files,
   NOT `identity_summary.txt` (that is a text summary); and build EVERY
   `bindings/build_*.sh`, because `identity_break` REFUSES a lane whose
   extension is missing and a partial build looks like a result. The H100
   per-set JSONs live inside
   `~/mojolearn-evidence/forest-finish-2026-09-11/trees_out_FINAL.tgz` at
   `trees_out/ib/*.json` (extract with the path as a plain operand; macOS
   bsdtar has no `--wildcards`).

3. **AMD confirmations for everything merged tonight** (gfx942, Hot Aisle
   first, then DigitalOcean): the three kNN checks; `kde_check` and
   `kde_stage_profile`; `svm/svc_main.mojo` 44/44 plus the hash probe
   (457e29b82bca9df9, 733a383c5699f427, 2b66bc991a9c9ed0); `check-if`,
   `check-forest-resident-layouts` and `device_batched_check`.
4. **The per-pod dataset tax is SOLVED as of 2026-09-12: Cloudflare R2.**
   Every new box used to rebuild Istella-S from library.istella.it (472,129,615
   bytes at 86 KB/s to 1.2 MB/s) and then spend about 18 minutes decoding the
   LETOR text into the 2.25 GB npz. On 2026-09-12 a GBDT leg spent 21 minutes
   of setup to produce 4.5 minutes of measurement, and it had already blown a
   2400 s download timeout the night before.
   - Bucket `mojolearn-data` on Andrew's Cloudflare account, credentials in
     `~/.mojolearn_r2` (mode 600, outside the repo; verified absent from the
     working tree and from history).
   - `tools/dataset_store.sh` holds it: `manifest` pins every file by size and
     sha256 into `bench/results/dataset_store/manifest.tsv`, `push` uploads,
     and **`stage "<ssh flags+target>" <key>...`** is what a leg calls after
     renting. `stage` mints a short-lived presigned URL locally and pipes a
     self-verifying fetch to the box over stdin, so the R2 secret never leaves
     the Mac and the URL never appears in the box's process list. The box
     refuses the file unless size AND sha256 equal the pins.
   - Five keys, 2.9 GB total: taxi npz, Istella-S npz (decoded, so the 18
     minute parse never happens again), the Istella-S source tarball (so a
     decode stays reproducible if the origin server dies), and both neural
     corpora (enwik8, pile_github -- their bytes are gitignored, so they were
     being refetched from mattmahoney.net and HuggingFace the same way).
   - MEASURED, not estimated: the whole 2.9 GB uploaded in 5 min 26 s (about
     8.9 MB/s) and a full round trip -- pull all five back and re-hash -- took
     2 min 17 s with 5 of 5 sha256 matching. I had estimated 40 minutes for the
     upload from a bad 1.1 MB/s sample taken while two scp transfers competed;
     that was wrong by 8x and the old number should not be re-derived.
   - A RunPod network volume is now only an optional hot cache, not the fix.
     It is region-locked and RunPod-only, whereas R2 also serves the
     DigitalOcean and Hot Aisle boxes the AMD legs need.
   - STILL OWED and worth doing on its own: a dataset that failed to download
     must REFUSE, not substitute. On the pointwise pod the fetch was killed at
     464 of 472 MB, the setup script wrote its ready sentinel anyway, and the
     arm silently fell back to synthclf. `body_setup.sh` was since fixed to
     write `data.done` only on a real decode; the arm's own fallback path has
     not been.

5. **Before the next Linux release**: confirm the installed FAST and
   DETERMINISTIC smoke passes on main (`run_installed.py` asks
   `_backend.binding('_mojolearn')` in tiers DEVIATION 2490 removed; the fix
   cc117fdf is on main but was not on the 0.8.3 release branch), and qualify
   sm_89 installed (no L40S, RTX 4090 or L4 stock on RunPod that night).
6. **Speed work still open**, in descending value:
   - ExtraTrees: the 2663 win is DEVIATION 205's rescue paying two extra
     staged sub-batches per cycle, so attack the rescue directly (batch the
     surveys, or skip the re-stage when the retry set is a subset of the batch
     already on the device). That should pay more than width did and would
     help Istella-S, where width bought least. The score and range passes are
     still 77 percent of the Istella-S loop and untouched.
   - kNN: a fused kernel only pays if a block owns SEVERAL query rows, which
     needs cuVS's shape of staging a rows-by-columns distance chunk in shared
     memory. 2667 failed because one block owning one row reads 1.125 operands
     per cell per feature against the register tile's 0.375.
   - UMAP: the remaining quality gap is the order ACROSS vertices, so damping
     (a rate schedule matched to how many moves a vertex applies) is the
     candidate, not a closer imitation of a serial sweep.
   - SVC: cuML runs about the same inner-iteration count we do, so the gap is
     per-iteration cost inside the fused tree (about 3.5 us against 2.5), plus
     the Python side still allocating n_rows x n_cols float32 for support
     vectors on every fit.
   - KDE: Istella-S is about 68 ms against cuML's 7, roughly 41 ms device and
     20 ms host staging. cuML's fused approach never writes the
     n_query x n_train matrix, but under IDENTICAL that needs a summation
     order equal to the staged one, so it is a design question.
   - The kNN query tile row is NVIDIA only; Apple and AMD have never been
     timed at a wider tile and that is the cheapest kNN win left.
7. **Also owed, not speed** (two of these turned out not to be real work):
   - `test_native_helpers.py` on the M4: **DONE, 18 of 18 passed** against
     shipped main with the identical set built. Trap for whoever runs it next:
     **pytest is declared NOWHERE in this project** (no pixi task, no
     dependency) and is absent from both the pixi env and system python, so it
     has to be driven from a throwaway venv (`python3 -m venv`, `pip install
     pytest numpy`; two of its cases import numpy inside the test body even
     though the fixtures themselves are `array.array` by design).
   - `helpers_ident.py` on the M4: **NOT RUNNABLE AS WRITTEN, and it is not
     really owed.** It hardcodes `/root/ctd-data/big-taxi.npz`, a pod path, so
     it is a pod-only script. Its identity claim is ALREADY PROVEN from the
     pod: before (`python_base`) and after (`python`) give identical mu64,
     mu32, center and scale hashes on all four fixtures (taxi, planted_order,
     wide, tiny) with `ymean` equal, recorded in
     `bench/results/linear_cluster_speed_2026-09-11/helpers_ident_after2.log`.
     Getting a VENDOR dimension out of it would need the taxi npz staged
     locally and that path parameterized; until someone wants that, treat this
     as closed rather than outstanding.
   - Apple M4 pointwise model hashes were never compared. STILL OPEN.
   - scikit-learn KDE Istella-S row on AMD. STILL OPEN.

## 7. DEVIATION numbers

Trees and classical have used 2620 to 2629, 2631 to 2638, and 2660 to 2672.
2630 and 2640 to 2659 are neural's. 2639 was held for the orchestrator and is
still free.

Take the next trees or classical number from 2673 up, and BEFORE taking it run
`git grep -n "DEVIATION 26[0-9][0-9]"` on main and on every open branch --
two collisions (2624, 2660) have already come from byte-LM lanes in other
sessions picking from the same range.
## 8. Evidence and worktrees

- Tonight's finish lanes: `~/mojolearn-evidence/{kde,svm,knn,forest,gbdt}-finish-2026-09-11/`
  and `linear-cluster-speed-2026-09-11/`; small copies committed under
  `bench/results/*_2026-09-11/`.
- Earlier: `~/mojolearn-evidence/{kde,svm,knn,linear-cluster,gbdt,forest}-speed-2026-09-11/`,
  `release-0.8.3-2026-09-11/`, `release-0.8.2-2026-09-11/`,
  `svc-cuda-1024-2026-09-11`, `pointwise-hash-drift-2026-09-11`,
  `ptw2624-amd-mi300x-hotaisle`, `verify083-amd-mi300x-hotaisle`,
  `trees-h100-pointwise-ab-2026-09-11`,
  `classical-h100-kde-svc-istella-2026-09-11`.
- The lane worktrees live in the orchestrator's scratchpad under /private/tmp
  and will not survive a reboot. Every branch is pushed and all evidence is
  copied out, so they can be removed.
- TRAP that cost 40 minutes on 2026-09-12: `body_phase2.sh` waits on
  `$OUT/istella_real.done`, which ONLY the resumed-fetch path writes, while the
  normal setup path writes `$OUT/data.done`. A recovery chain that writes
  `data.done` therefore deadlocks phase 2 forever. Write BOTH markers, or key
  phase 2 off the dataset itself.
- Useful traps learned tonight: macOS has no `timeout` (the wrapped command
  never runs); `pkill -f PAT` inside an ssh command matches that command's own
  line and kills the remote shell, so break the pattern (`"istella_re[s]ume"`);
  `git merge -F -` cannot read a message from stdin the way `git commit -F -`
  can.
