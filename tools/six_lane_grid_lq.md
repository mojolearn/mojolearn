# Running the IDENTICAL switch grid over lq

`tools/six_lane_grid_lq.py` turns a grid plan into `lq add` lines, one file per box. After the runs it turns the
lq results into the three inputs that `tools/six_lane_grid_decide.py` reads. It never builds, races or connects
to a box. Lanes queue nothing. The orchestrator feeds the lines and copies the results back.

## Commands

```bash
# 0. Freeze one commit for the round and push it. Both arms build from this branch.
#    A moving main mixes heads; collect then reports INCOMPLETE identity.
git push origin <sha>:refs/heads/grid/freeze-<date>
python3 tools/six_lane_grid.py --crosses auto --cap 0 --out ~/mojolearn-evidence/grid-lq/plan

# 1. Render one lines file per box (laptop, metadata only)
for v in nvidia amd; do
  python3 tools/six_lane_grid_lq.py render --plan-dir ~/mojolearn-evidence/grid-lq/plan --vendor $v \
    --branch grid/freeze-<date> --b-repeats 3 --out ~/mojolearn-evidence/grid-lq/$v.lines
done

# 2. Feed each box (orchestrator; one loop per box, resumable).
#    Waits while the box has >= 4 unfinished jobs; checks every 300 s.
bash tools/six_lane_grid_lq_feed.sh nv  ~/mojolearn-evidence/grid-lq/nvidia.lines 4 300 > ~/mojolearn-evidence/grid-lq/feed-nv.log 2>&1 &
bash tools/six_lane_grid_lq_feed.sh amd ~/mojolearn-evidence/grid-lq/amd.lines    4 300 > ~/mojolearn-evidence/grid-lq/feed-amd.log 2>&1 &

# 3. Progress (grep-sized): the run id is in <lines>.json totals.run_id
lq results nv  'MOJOLEARN_GRID_TAG=<run_id>' | tail -5
lq results amd 'NO-RESULT' | tail -5

# 4. Collect (orchestrator copies, per box: /root/lq/results.txt, plus the race-log dump made ON the box with
#    grep -H -E '^ALGOS|digest=' /root/lq/out/*/race-*.log > /root/lq/grid-logs.txt)
python3 tools/six_lane_grid_lq.py collect --plan-dir ~/mojolearn-evidence/grid-lq/plan \
  --results ~/mojolearn-evidence/grid-lq/nv-results.txt ~/mojolearn-evidence/grid-lq/amd-results.txt \
  --logs ~/mojolearn-evidence/grid-lq/nv-grid-logs.txt ~/mojolearn-evidence/grid-lq/amd-grid-logs.txt \
  --out ~/mojolearn-evidence/grid-lq/collected

# 5. Decide (unchanged tool; pass the plan's matrix, not the committed default)
python3 tools/six_lane_grid_decide.py --matrix ~/mojolearn-evidence/grid-lq/plan/grid-matrix.json.gz \
  --verdicts ~/mojolearn-evidence/grid-lq/collected/grid-verdicts.json \
  --identity ~/mojolearn-evidence/grid-lq/collected/summary.json \
  --quality  ~/mojolearn-evidence/grid-lq/collected/quality.json \
  --out ~/mojolearn-evidence/grid-lq/decide
```

`render` options: `--only-lanes ridge-cv,qda` (bench_board_algos lane names), `--phase factorial|pairwise`,
`--budget-hours H` and `--b-group-size N` (default 8 workloads per incumbent line).

`--budget-hours H` keeps A lines in order while the A races plus their new workloads' B repeats fit in H hours.
Each race is costed at half the plan's pair time: nvidia 38 s, amd 24 s per pair. Build time is not included.

`collect` options: `--min-samples` (default 2 incumbent repeats for a floor), `--run-id`.

## What a line is

```
lq add nv RACE <branch> <lane[,lane]> <ds[,ds]> MOJOLEARN_GRID_TAG=<run>.<pack> MOJOLEARN_BUILD_DEFINES=<D1=1,D2=3> BUILDS=build_x,...
lq add nv RACE <branch> <lane@ds,lane@ds,...> PAIRS MOJOLEARN_GRID_TAG=<run>.B<grp>r<k> BUILDS=build_x
```

- **Arm A, one line per build pack.** A pack is the planner's set of configurations that share one define set
  (`grid-plan.json` `builds.packs`), so one build covers all of them. Packs never race the same workload twice.
  The lanes and datasets come from the members' workload ids.
  - The comma form is used when the cells are a full lanes x datasets product.
  - Otherwise the line uses `lane@ds,... PAIRS`, which box_job.sh accepts.
  - Lines are ordered: factorial regime first (a pack is factorial if any member is), then pairwise, then
    config priority, then pack number.
- **Arm B, the incumbent.** The same branch with no defines. Workloads are grouped by binding set: at most
  8 per line, one build each. Every workload's B is repeated `--b-repeats` times, spread evenly through the
  file. The repeats are the noise floor, so no separate A/A pass is needed (as in
  `six_lane_timing.py floors --from-pairs`).
- **`MOJOLEARN_GRID_TAG`.** The run id plus the pack, or the B group and repeat. The run id is a hash of the
  plan files, so collect ignores lines from another campaign. It is the first ENV, so lq's 600-character cut
  never removes it. No build or race reads it.
- **How the defines reach the build.** box_job.sh (2026-10-07) exports every `MOJOLEARN_*=` token before the
  builds and the races. Every binding build script sources `bindings/build_defines.sh`, which expands
  `MOJOLEARN_BUILD_DEFINES` into `-D NAME=VALUE` on its `mojo build` line.
  - When the variable is empty, the mojo argv is unchanged.
  - The older `MOJOLEARN_EXTRA_DEFINES` and `MOJOLEARN_BUILD_EXTRA_DEFINES` still work.
- **`BUILDS=`.** The device build scripts for the pack's bindings and the members' workload bindings.
  If a race needs another binding, overlay_race_job2.sh builds it with the same exported defines and retries.

## Where results come from

- **`/root/lq/results.txt` on each box.** box_job.sh writes one line per race:

  ```
  <id> <nvidia|amd> <branch>@<head> [ MOJOLEARN_GRID_TAG=... MOJOLEARN_BUILD_DEFINES=...] ALGOS lane= dataset= arm=ours status= median_ms= quality={json} digest=<16 hex>
  ```

  The line is cut at 600 characters. Long define sets push the quality JSON, the digest, or the whole ALGOS
  text past the cut.
- **`/root/lq/out/<id>/race-*.log`.** These hold the full `ALGOS` line and the `ALGOS-ROUND ... digest=` lines.
  Pass `--logs` with either form:
  - the lq out directories (`<id>/race-*.log`), or
  - a `grep -H -E '^ALGOS|digest=' /root/lq/out/*/race-*.log` dump.

  Collect fills cut fields from the logs. `collect-report.json` lists `truncated_without_log` and
  `jobs_without_algos` (a tagged job with no ALGOS line and no logs). Both should be 0.
- **`lq results <box>` and `lq log` are for progress only.** They return the last 40 lines, cut at 400 characters.

## What collect writes (`--out`)

| file | schema | read by |
| --- | --- | --- |
| `grid-verdicts.json` | `mojolearn.six-lane-timing-verdicts/1` | decide `--verdicts` |
| `summary.json` | `mojolearn.six-lane-comparison/1` | decide `--identity` |
| `quality.json` | rows | decide `--quality` |
| `floors.json` | `mojolearn.six-lane-aa-floors/1` | the floors from the B repeats |
| `collect-report.json` | | coverage per vendor (MEASURED/FAILED/MISSING/REFUSED), counts, ignored lines, NO-RESULT jobs |

**`grid-verdicts.json`.** Cases list a configuration, a workload_id, and per vendor
`candidate_over_baseline.scored` = A_ms / median(B_ms), `log_ratio` and `floor`.
- B is taken from the same head when it exists.
- floor = log(p90/p10) of the B repeats with 4 or more samples, else log(max/min) (`six_lane_timing._spread`).
- The verdict comes from `six_lane_timing.judge`: FASTER or SLOWER only when the change is beyond the floor on
  both NVIDIA and AMD and both agree in direction. Otherwise NO_VERDICT.
- lq reports one `median_ms` per race, so only the `scored` phase is judged.

**`summary.json`.** Per (configuration_id, workload_id), `arms.A` and `arms.B` compare the NVIDIA digest with the
AMD digest. Each arm is one of:
- MATCH: the digests are equal.
- MISMATCH: same head, digests differ. Also used when the incumbent's digest changes from run to run on one vendor.
- INCOMPLETE: a digest is missing or the race failed. Also used when the digests differ and the vendors raced
  different heads.

`output_sha256` holds the digest per vendor: the bench_board_algos race digest, first 16 hex characters, as
box_job.sh records it.

**`quality.json`.** One row per configuration, lane, dataset and vendor. `candidate_vs_baseline.verdict` is set
from the board quality JSON:
- WORSE or BETTER: a metric moves beyond 1e-6 relative.
  - Lower is better: rmse, logloss, inertia, error, residual, diff, ...
  - Higher is better: accuracy, auc, r2, trustworthiness, recall, silhouette, ...
- SAME: every judged metric is within that tolerance.
- FAIL: the A race failed.
- PENDING: there is no incumbent.

Keys with no known direction (`n_clusters`, `fraction_flagged`, ...) are listed in `unjudged`, not judged.

## Coverage at 8d8771a8e (`--crosses auto --cap 0`, both vendors identical)

| | nvidia | amd |
| --- | --- | --- |
| lines (= builds) | 164 | 164 |
| A lines (packs) | 137 (factorial 109, pairwise 28) | 137 |
| B lines (9 groups x 3 repeats) | 27 | 27 |
| A races (cells) | 509 of 3,358 | 509 of 3,358 |
| B races | 144 | 144 |
| configurations / workloads | 269 / 48 | 269 / 48 |
| projected race hours (builds excluded) | 3.45 | 2.18 |

`lq RACE` runs `tools/bench_board_algos.py race` only. The other 2,849 cells per vendor are refused by name:

| refused | cells per vendor | reason |
| --- | --- | --- |
| trees (`rf:`, `et:`, `gbdt-*:`, `iforest:`) | 1,659 | Tree board workloads (`<lane>:<dataset>`). Not bench_board_algos lanes. |
| neural (`neural:*`) | 932 | bench_board_neural workloads. |
| classical (dbscan, hdbscan, kde, kmeans, knn, ols, pca, svc) | 174 | Classical harness lanes. Not in bench_board_algos `LANES`. |
| more (arima, ets, gmm, ivf, linearsvc/svr, logreg, nystroem, rbf-sampler, ridge, svr, tsvd) | 52 | bench_board_more lanes. Not in `LANES`. |
| expanded time-series (auto-theta, autoarima, damped-ets, dynamic(-optimized)-theta, optimized-theta, theta) | 26 | The grid names `dataset=taxi/istella`, but these lanes race `taxi-hourly` and `synthetic`. |
| expanded `lu-solve@dataset=taxi/istella` | 2 | The lane races `synthetic` only (its `synthetic` cells are kept). |
| expanded `sgd-reg@regression_report` | 4 | A lane variant with no lq RACE form. |

Accepted with a note: 16 `@input=classification-full-v1` and `@input=tsvd-full-v1` workloads (gaussian-nb, lda-clf,
minibatch-kmeans, qda, randomized-svd, ridge-clf, standard-scaler, target-encoder). lq races the board's rows-full
block for these lanes, not the registered full-input variant. A and B share that input, so the A/B comparison holds,
but the input is not the saved recipe's.

Reaching the refused families needs a harness entry that lq can run with exported defines. Today box_job.sh `CMD`
does not export ENV tokens before its builds. Running time-series cells needs a planner mapping to `taxi-hourly` or
`synthetic`. Neither is substituted here.
