# Handoff, September 15 2026, evening

Written at 19:45 ET for a session that holds none of tonight's context. Everything here was verified against the repository at the time of writing, not recalled. Where a fact is owed or unproven, it says so.

Base of this handoff: `origin/main` at `fd74380c9`.

## Read these first, in this order

1. This file.
2. `docs/lanes/RELEASE_086_STATE.md` on branch `release/0.8.6` (`origin/release/0.8.6` at `9f2ccff71`). It is the ground truth for the release, and it is ahead of any local `release/0.8.6` checkout, which sits at the frozen build commit `db9047b9f`.
3. The `docs/lanes/LANE_STATUS_<branch>.md` file on each lane branch below. Each was written tonight to be resumable with no memory of this conversation and carries the exact commands. Do not reconstruct a lane from this file when its own status file exists.

## The rules that bind all of this

- Bitwise identity work uses small fixtures. Only timing work uses large runs.
- Local testing is one core per agent, one Metal job at a time through the scratchpad slot helper. A Metal cell taken under contention is not evidence.
- No GPU boxes are rented between releases. The 0.8.6 record is the authorized exception and is 1 AMD, 1 NVIDIA, 1 Apple.
- A lane fully proves its new or changed cells. Untouched existing lanes get a base fixture spot check only. The full all lane proof happens once, at the release record.
- Never cancel an owed run, local or rented.
- Sabotage must be seen to fail before it counts as a control.
- STOP BEFORE PUBLISHING 0.8.6 TO PyPI. That needs Andrew's explicit word.

## Correction carried into tonight

Earlier tonight I reported that the Mac held 6,754 Metal command queues, that most outlived the processes that created them, and that only a reboot could clear it. **That was wrong and no restart is needed.** `ioreg -l -c AGXCommandQueue` prints no nodes of that class, so the attribution behind that claim compared two different populations. Measured truth, from `lane/metal-queue-leak` at `839611c76` with evidence in `~/mojolearn-evidence/metal-queue-leak-2026-09-15/queue-samples.txt`:

- One live `mojo` process held 1,211 of the machine's 1,243 queues, still climbing at 2,491.
- About 15 seconds after that process exited, the count fell to 34.
- With no mojolearn GPU process alive the machine idles at 40 to 41 queues.

So queues are released on process exit, and the slowdown came from a single process crossing the 512 queue limit inside its own lifetime.

**Two further corrections, from the health check at `be12003b8`, evidence in `~/mojolearn-evidence/metal-queue-leak-2026-09-15/gbdt-metal-health.log`.**

First, **the degraded state is gone**. Like for like against this afternoon's degraded evidence, the same `gbdt_direct.py` at 20,000 rows: SymmetricTree 8.289, 7.735 and 8.288 seconds now against 21.5 to 23.4 seconds then, and Depthwise 11.848, 14.082 and 13.373 seconds now against 27.0 to 30.4 seconds then. That is about 2.7x and 2.2x, with nothing resembling the 20x fits.

Second, **my "about 1 second per fit is healthy" figure is unverified and should not be quoted.** No GBDT Metal timing from before the slowdown is committed anywhere in the repository, so 7 to 8 seconds is neither proven healthy nor proven slow. What is proven is only that the degraded state is gone. Committing a real baseline for this shape is worthwhile separate work.

Third, **do not write the "reuse one `DeviceContext` per process" fix on the strength of this file.** Queue counts held flat at 34 before each run, after every fit, at each process exit and 20 to 30 seconds later, across eleven fits in two processes. The per-call `DeviceContext` in the GBDT path does not accumulate queues in this build, so that change would fix nothing measurable. Phase 2 must first find which workload shape actually accumulates, most likely long-lived contexts held concurrently rather than created and dropped per call. The reproduction at `checks/device_context_queue_repro.mojo` is built to answer exactly that.

The safe mitigation in the meantime is unchanged and cheap: run long recordings as several short processes rather than one long-lived process.

## Release 0.8.6, in progress, do not publish

Frozen at `db9047b9f`. State file on `release/0.8.6` at `9f2ccff71`.

- Three Linux build sets are home in `~/mojolearn-evidence/release-0.8.6/linux-builds/`, each matching its proof, with the 15 host bindings byte identical across AMD, H100 and L40S.
- Linux wheel packed, repaired, stripped and audited, sha256 prefix `7cab1aa3`, 70,862,796 bytes, manylinux_2_35, uploaded to R2 under `releases/0.8.6/` with a verified round trip.
- macOS wheel sha256 prefix `eba69f83`, passing every tier on Python 3.10 through 3.14, kept rather than rebuilt.
- The record is arriving in legs, not one pass. **AMD leg 1 is done: 62 of 192 lanes, 558 cells, all STABLE, no divergent, moved or refused cells, VM verified gone, $2.10.** Its evidence and an explicit 130 lane remainder list are under `~/mojolearn-evidence/release-0.8.6/records/`. AMD leg 2 (skipping the recorded 62) and NVIDIA leg 1 (all 192) were in flight at 20:00, both reporting about 20:45. Expect a third AMD leg and a second NVIDIA leg for the remainder, one box per vendor at a time. Budget: roughly $2 per AMD leg, so the whole record lands nearer $10 to $15 than the $7 spent before it started.
- Both runners refuse a dirty tree, so state file edits must be committed before renting. That cost two blocked rentals and no money.
- Release branch head is `7e23bc670`, which carries the AMD leg 1 record, the Linux `sys.executable` finding, the corrected Metal explanation with the health numbers, and "no more new lanes".
- Remaining after AMD: the NVIDIA record leg (one box per vendor, never two at once), the Apple chunks (chunk 00 is saved; chunks 01 to 06 remain, and the 7 lanes recorded while the GPU was degraded must be rerun and byte compared), the four column diff, the record commit with a regenerated `verify` reference table, the final pack under packing allowlist plan C, the macOS repack, and the final content audit.
- Packing allowlist plan C means only `host_surface.py` record lists and `verify_reference/table.json` may differ from the build proofs.
- Costs are approved and are about $7 so far. Publishing is not approved.

## Lanes

### Merged to main tonight

| Lane | Commit | What landed | Owed |
| --- | --- | --- | --- |
| `lane/gp-optimizer` | `fd74380c9` | GP hyperparameter optimization. `GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", n_restarts_optimizer, random_state)`, scikit-learn `*_bounds` arguments in scikit-learn's order, `log_marginal_likelihood(theta, eval_gradient)` at any theta. DEVIATIONs 2880 and 2881. `optimizer=None` remains the default, so no recorded gp hash moved. CPU 63 of 63 stable, Metal matches CPU on 54 train, 108 infer and model, 54 batch, 36 existing gp lanes unmoved, gradient sabotage moves 9 of 9 new fixtures and 0 of 9 old ones. | Metal column at the merge commit, so `gp-normalize-y` records rather than refuses. NVIDIA and AMD cells at the next release record. |
| `lane/istella-ranking-bench` | `f624c41de` | Istella-S ranking timing on one H100, full size, results in `bench/results/istella_ranking_2026-09-15/`. Our IDENTICAL arm against each opponent's fastest cell is 1.50x CatBoost, 0.90x XGBoost, 0.65x LightGBM CUDA. QueryRMSE and PairLogit gaps are intercept, and 1,278 ms of ours is a Python `_group_sizes` loop. YetiRank is slope, 6.57x CatBoost per tree, so it grows with tree count. | Nothing. Pod deleted and verified. |
| `lane/cpu-training-par-wave3` | `df9234099` | Triage only, 267 lines, of the 28 remaining two device `par-*` lanes for the CPU verifier. 2 coverable now (`par-samba`, `par-samba-clip`, needing two names in `_parallel_pool.py`, two lanes in `host_surface.py`, the par test maps, no Mojo). 24 coverable after named work. 2 not coverable at all (`par-byte-lm-model-pool`, `par-byte-lm-offload`), because with one CPU worker both sides of their claim are the same host arithmetic in one process, so the equality is true by construction. | The 2 easy lanes are the next cheap win. |
| `lane/cpu-verifier-gaps-7` | `05ac56e7d` | Closed the 7 lane CPU verifier gap, with RunPod CPU evidence. | Nothing. |

### Pushed, not merged, each with Metal or CPU columns owed

| Lane | Branch head | State |
| --- | --- | --- |
| `lane/gbdt-rest` | `4d10f60d7` | QuerySoftMax, the softmax learning to rank loss, with four reference kernels plus launcher and a matching host order, wired through the querywise arm, the oracle, `train`, both bindings and the losses oracle. Four loss parameters ride as a fifth string entry so no numeric ABI tail moves. **Proven on CPU only, no identity claim made.** CPU column 3 of 3 stable across infer, model and batch, 21 CPU route tests green, both bindings build. Owed: the Metal column, the Metal against CPU diff, batch sabotage, the host sabotage build and column, a base fixture spot check of ten existing GBDT lanes, the Metal test route, and the CatBoost agreement check whose script exists at `~/mojolearn-evidence/gbdt-rest-2026-09-15/query_softmax_reference.py` but has never been run. Triage at `2355b6a27`, 28 rows as 10 user facing, 13 internal plumbing, 5 excluded. Measured finding pinned by a test: `lambda` enters only the second derivative, so it is inert under QuerySoftMax's own Gradient and Cosine defaults and live under Newton leaves. |
| `lane/neighbors-rest` | `d7967e7e0` | Seven brute force metrics (canberra, correlation, jensenshannon, inner product, braycurtis, hamming, russellrao) on `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`, on GPU, in the host oracle and in public CPU inference. DEVIATIONs 2898 to 2901. Apple column 63 cells stable across train, infer, model and batch, taken twice and byte identical. Metric check 15 distance types over 1,961 cells, 29,415 bit equal, 0 differ. 14 Python tests, saved model 63 of 63 recorded with reload equal, existing neighbor and KDE lanes unmoved. Owed: the x86 CPU column, the sabotage column, the host gate check, and measurement of `kneighbors(X=None)`, which is written and tested but unmeasured. Triage of 31 rows as 2 user facing, 20 internal plumbing, 9 excluded. Fixed in passing: `classical_host_gate.py record` had been dying on `--lane-rule-only` for every caller since that flag landed. |
| `lane/arima-exog` | **MERGED to main at `71dd3dd03`** | ARIMA exogenous regressors, complete. `ARIMA.fit(y, exog)`, `forecast(steps, exog)`, `predict(start, end, exog)`, with `exog` shaped `(batch_size, n_obs, n_exog)`, new `beta_` and `n_exog_`, and `params_` packing `beta` after `mu`. Future regressor values are required exactly when the model has a regression and the horizon passes `n_obs`, refused otherwise. DEVIATIONs 994 to 998, including `EXOG_MAX=17`, the two closed gemms spelled as serial ascending fma from zero, and saved format `mojolearn-arima-2` with `mojolearn-arima-1` byte unchanged. Counts: `OWED x1` on every cell across nine fixtures once the CPU column existed, `owed verdict OK (72 of 72 owed parts moved)`, sabotage `DIVERGENT=18/36/18`, saved models `IDENTICAL (21 fixtures, 3 GPU columns)`, existing ARIMA lanes `IDENTICAL x4`. The narrow exog-only control diverges on both lanes while `ar` still agrees on `arima-exog`, which pins the divergence to the exog arithmetic rather than to the fit in general. statsmodels SARIMAX agrees on `beta` within 4.4e-4 to 6.0e-4 and forecasts within 1.4e-3 to 2.0e-3. Two defects found by running it: `_FORMATS` missing `mojolearn-arima-2` and `_HOST_ARRAYS` missing `_exog`. **Still owed:** the NVIDIA and AMD columns and their `mojolearn-arima-2` model cells at the next release record, the installed-wheel gate (the pod env had no pip, so its pytest, statsmodels and wheel steps did not run and were done on the Mac instead), and a host-infer diff rerun with `--repeats 2`. Superseded detail: the Metal recording finished for both lanes, nine fixtures each, at `352956a54`. `arima-exog` is stable with regression coefficients, AR terms, mean, variance and forecast hashed, and a saved and reloaded model predicting the same bytes, and `arima-exog-seasonal` now carries its full nine fixture recording rather than the partial one described at `51ac98283`. DEVIATIONs 994 to 998. 156 source tests, Metal runtime 8 of 8, CPU only runtime test all pass. Two real bugs found by running it: a missing saved model format registration and a gap in the identity harness array handling. The whole recording took 9 minutes 7 seconds for 18 fixture directories. The earlier apparent crawl was another lane holding the Metal lock, not a slow seasonal fit and not GPU degradation. Owed: the CPU column, the sabotage column and the host gate check, then merge. |
| `lane/metal-queue-leak` | `ffab15fea` | Closed out, no fix written and none currently justified. Carries the measurement in the Correction section above, the GBDT Metal timing baseline the repository never had (committed at `bench/results/classical_host/2026-09-15-apple-m4-gbdt-metal-baseline/gbdt_metal_fit_times.txt`, labeled in the file and the commit message as a first baseline and not a proven healthy figure), and a phase 2 plan written for a context free session: concurrently held contexts in the pools and per-shard sites first, then native `mojo` binaries rather than the Python bindings, then long harness processes. The reproduction is at `checks/device_context_queue_repro.mojo`. The lane also went back and rewrote three of its own earlier sentences that still recommended per-process context reuse, because its flat-at-34 measurement had already retired that advice. |
| `lane/ties-sabotage` | `78311d743` | Neighbor and IVF cells that read inert on the `ties` fixture. Not worked tonight. Its head carries a changelog entry aimed at 0.8.7 and its lane status. |

### Not started, and NOT to be started

**Andrew, September 15 2026 at 19:50 ET: "no more new lanes."** `lane/arima-rest`, `lane/ivf-rest`, `lane/svm-spectral-holtwinters-rest` and `lane/extratrees-rest` were queued earlier in the evening and are now off the board. Do not open them, and do not open any other new lane. Finish, prove and merge what already exists.

Kept only as a record of the pattern that worked tonight, for whenever new work is authorized again: triage every NOT_IMPLEMENTED row into user facing, internal plumbing, or intentionally excluded, each with a precise reason and a citation, then build only the user facing rows, proving new cells fully and spot checking existing ones.

## Coverage, measured this evening

- 195 identity lanes, 156 one device and 39 two device.
- Batch invariance: 178 real parts, 17 named not applicable, 0 undeclared.
- CPU verifier: 167 of 195. The 28 missing are the two device lanes triaged by wave 3 above.
- Public CPU inference: 77 manifest lanes plus the forest and GBDT kinds and the neural and byte LM classes.

## What I would do next, in order

1. Run the GBDT Metal health check once the Metal lock is free. If it reads about 1 second per fit, the GPU is healthy and the release proceeds tonight.
2. Finish 0.8.6. AMD record, then NVIDIA record, then the Apple chunks run as several short processes rather than one long one, then the four column diff, record commit, final pack and audit. Then stop and wait for Andrew.
3. Take the owed Metal columns for `gbdt-rest`, `neighbors-rest` and `gp-optimizer`, and merge those lanes.
4. Do not write the `DeviceContext` fix yet. Run the phase 2 reproduction first to find which workload shape actually accumulates queues, since the per-call path provably does not. Commit a GBDT Metal timing baseline while the machine is known good, so the next slowdown has something to be measured against.
5. Cover `par-samba` and `par-samba-clip`, the two cheap CPU verifier wins.
6. The Istella intercept, that 1,278 ms Python `_group_sizes` loop, is the clearest ranking speed win on the board.

No new lanes. Steps 1 through 6 are all work that already exists and is owed. When they are done, ask Andrew rather than opening anything.

## Cautions earned tonight

- Before attributing a count, check that the tool counts the thing you named. My queue attribution failed exactly there.
- A verification that cannot fail is not a verification. Run the sabotage and watch it fail before trusting the control.
- A truthful "this cannot be covered on CPU" is a result, not a gap. Two byte LM lanes are genuinely uncoverable and saying so is worth more than a green cell that passes by construction.
- Never `git stash` in the shared checkout. Stashes are shared across every worktree.
- Check the current branch immediately before every commit, because other lanes move HEAD.
