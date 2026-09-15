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

So queues are released on process exit. The roughly 20x slowdown comes from a single process crossing the 512 queue limit inside its own lifetime, because `tools/identity_break.py` runs every lane, fixture and repeat in one process while every binding call builds a `DeviceContext`. Reusing one context per process is a real fix. Until it lands, run long recordings as several short processes. Health check is one GBDT Metal fit on the base fixture, about 1 second healthy against about 20 seconds degraded.

## Release 0.8.6, in progress, do not publish

Frozen at `db9047b9f`. State file on `release/0.8.6` at `9f2ccff71`.

- Three Linux build sets are home in `~/mojolearn-evidence/release-0.8.6/linux-builds/`, each matching its proof, with the 15 host bindings byte identical across AMD, H100 and L40S.
- Linux wheel packed, repaired, stripped and audited, sha256 prefix `7cab1aa3`, 70,862,796 bytes, manylinux_2_35, uploaded to R2 under `releases/0.8.6/` with a verified round trip.
- macOS wheel sha256 prefix `eba69f83`, passing every tier on Python 3.10 through 3.14, kept rather than rebuilt.
- AMD record leg started 19:16:15 on a Hot Aisle 8 core MI300X, bounded at 3,217 seconds, reporting about 20:10.
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
| `lane/arima-exog` | `352956a54` | ARIMA exogenous regressors. **The Metal recording finished for both lanes, nine fixtures each**, at `352956a54` (19:40). `arima-exog` is stable with regression coefficients, AR terms, mean, variance and forecast hashed, and a saved and reloaded model predicting the same bytes, and `arima-exog-seasonal` now carries its full nine fixture recording rather than the partial one described at `51ac98283`. DEVIATIONs 994 to 998. 156 source tests, Metal runtime 8 of 8, CPU only runtime test all pass. Two real bugs found by running it: a missing saved model format registration and a gap in the identity harness array handling. The whole recording took 9 minutes 7 seconds for 18 fixture directories. The earlier apparent crawl was another lane holding the Metal lock, not a slow seasonal fit and not GPU degradation. Owed: the CPU column, the sabotage column and the host gate check, then merge. |
| `lane/metal-queue-leak` | `839611c76` | The measurement in the Correction section above, plus a phase 2 protocol written for a context free session and the GBDT health check. No fix written yet. The fix is context reuse per process in the binding layer. |
| `lane/ties-sabotage` | `78311d743` | Neighbor and IVF cells that read inert on the `ties` fixture. Not worked tonight. Its head carries a changelog entry aimed at 0.8.7 and its lane status. |

### Queued, never started

`lane/arima-rest` (after `arima-exog` merges), `lane/ivf-rest`, `lane/svm-spectral-holtwinters-rest`, `lane/extratrees-rest`. Each is a NOT_IMPLEMENTED triage lane. The pattern that worked tonight: triage every row into user facing, internal plumbing, or intentionally excluded, each with a precise reason and a citation, then build only the user facing rows, proving new cells fully and spot checking existing ones.

## Coverage, measured this evening

- 195 identity lanes, 156 one device and 39 two device.
- Batch invariance: 178 real parts, 17 named not applicable, 0 undeclared.
- CPU verifier: 167 of 195. The 28 missing are the two device lanes triaged by wave 3 above.
- Public CPU inference: 77 manifest lanes plus the forest and GBDT kinds and the neural and byte LM classes.

## What I would do next, in order

1. Run the GBDT Metal health check once the Metal lock is free. If it reads about 1 second per fit, the GPU is healthy and the release proceeds tonight.
2. Finish 0.8.6. AMD record, then NVIDIA record, then the Apple chunks run as several short processes rather than one long one, then the four column diff, record commit, final pack and audit. Then stop and wait for Andrew.
3. Take the owed Metal columns for `gbdt-rest`, `neighbors-rest` and `gp-optimizer`, and merge those lanes.
4. Fix the `DeviceContext` per call issue, since it is the root of the slowdown and makes every future long recording cheaper.
5. Cover `par-samba` and `par-samba-clip`, the two cheap CPU verifier wins.
6. Start the queued triage lanes in Andrew's order.
7. The Istella intercept, that 1,278 ms Python `_group_sizes` loop, is the clearest ranking speed win on the board.

## Cautions earned tonight

- Before attributing a count, check that the tool counts the thing you named. My queue attribution failed exactly there.
- A verification that cannot fail is not a verification. Run the sabotage and watch it fail before trusting the control.
- A truthful "this cannot be covered on CPU" is a result, not a gap. Two byte LM lanes are genuinely uncoverable and saying so is worth more than a green cell that passes by construction.
- Never `git stash` in the shared checkout. Stashes are shared across every worktree.
- Check the current branch immediately before every commit, because other lanes move HEAD.
