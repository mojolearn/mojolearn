# Running the IDENTICAL switch grid on a box

`tools/six_lane_grid_run.sh` runs the whole grid on one Linux GPU box. `tools/six_lane_grid_decide.sh` then
turns both boxes' receipts into one decision per switch arm. Together they replace the five manual steps in
`experiments/six_lane_integration/grid/GRID.md` ("How to queue"). The scripts call the existing tools; the
decision rules are unchanged. Only the orchestrator types these commands. Lanes never compile or measure.

## The commands

On the NVIDIA box (RunPod L40S) and the AMD box (DO MI325X), in parallel:

```bash
cd /root/mojolearn && git fetch -q origin && git checkout -q main && git reset -q --hard <freeze-sha>   # clean, on main
bash tools/six_lane_grid_run.sh nvidia /root/grid-run-<freeze> --kit /root/grid-kit     # on the nv box
bash tools/six_lane_grid_run.sh amd    /root/grid-run-<freeze> --kit /root/grid-kit     # on the amd box
```

Then, with both run directories copied into one place:

```bash
bash tools/six_lane_grid_decide.sh <run-nvidia> <run-amd> [--out <dir>]
```

Options for `six_lane_grid_run.sh`:

- `--phase factorial|pairwise|all`: the phase to run. The default, `all`, runs the factorial regime first, then pairwise.
- `--dry-run`: prints every command and checks the inputs (freeze, grid, kit, data, builds). It runs nothing; the only exception is planning the grid into a temporary directory. It works on the laptop.
- `--data-dir DIR` (repeatable): the directory searched by basename when a full input is not at its original path. The default is `/root/six-lane-full-ab-20261006/data`.
- `--builds DIR`: where the compile output goes. The default is `<run>/builds`. If the builds were compiled elsewhere, copy them in and `touch <builds>/.external` to skip compiling.
- `--compile-jobs N`: the `mojo build -j` value. The default is 4.
- `--compile-shards N`: how many per-binding compile processes run at once. The default is 3.
- `--retry-failed`: runs the failed cells again in fresh attempts. The earlier attempts are kept.
- `--limit-cells N`: runs at most N cells in this invocation. Use it for a smoke run.

Environment overrides:

- `GRID_AUTHORIZATION`: the text recorded in every queue and worker. By default it records the host, the time and that the orchestrator ran the script.
- `GRID_CELL_TIMEOUT`: seconds per cell. The default is 3600.

## What each step does, and what it needs on the box

| step | tool | needs | where it is on the box |
| --- | --- | --- | --- |
| 0 preflight | the script | a clean checkout on `main` or `integration/*` at the freeze: `six_lane_ab compile`, `materialize` and the worker all refuse a dirty tree or another branch, so `lq`'s detached CMD worktrees do not qualify. The run dir must be outside the checkout. | nv: `/root/mojolearn`; amd: `/root/mojolearn-ab2` (from `~/mojolearn-evidence/lq/box_job.sh`); pixi envs `default` (mojo) and `bench` (worker python) |
| 1 grid | `six_lane_grid.py --crosses auto --cap 0 --out <run>/grid` | the committed `grid_controls/*.json` and `core/six_lane_experiment_guards.mojo` | generated into `<run>/grid`; `grid/.source` holds the commit. The committed `experiments/six_lane_integration/grid/` is never written. |
| 2 compile | `six_lane_ab.py compile --plan <run>/grid/grid-build-plan.json --vendor V` with `--nvidia-target native --nvidia-arch sm_89` or `--accelerator gfx942`, plus `--keep-going --binding B --key K...` | the compiler `<repo>/.pixi/envs/default/bin/mojo` | `<run>/builds/<binding>/<key>/receipt.json`. There is one shard per binding. `--phase factorial` compiles only the factorial configurations' A builds plus every B build. |
| 2b math | `packaging/portable_math/stage.build` | the `default` pixi env | `<run>/math/libMojolearnMath.so`, built outside the tree; it is copied into every package's `mojolearn/.libs/` |
| 3 kit | `six_lane_grid_run.py install-kit` | the kit (see below) and the full inputs | copies the kit's variant evidence back to its original `/root/six-lane-full-ab-20261006/...` paths, after checking each sha256 |
| 4 stage | `six_lane_grid_run.py stage` → `six_lane_materialize.materialize` → `six_lane_ab.queue` | saved facts, loaded bindings, compile receipts and full inputs | `<run>/stage/configs/<cfg>/{facts,deployments,matrix,queue}.json`, plus `cells/<key>.json` (one authorized queue per cell) and `<run>/stage/cells-<phase>.json` (the ordered index). Packages go to `<run>/packages/<vendor>/<hash>/mojolearn/identical/*.so`: every B build is hard-linked in, and the cell's A builds replace their bindings. |
| 5 run | `six_lane_grid_run.py run` → `performance_full_ab_queue.py --config <cell queue> --output <run>/results/<key>` | the staged cells | receipts at `<run>/results/<key>/<key>/attempts/attempt-N/receipt.json`, logs at `<run>/logs/cells/<key>.log` |
| 6 evidence | `six_lane_timing.py floors --from-pairs <run>/results`; `six_lane_grid_run.py quality` | the receipts | `<run>/evidence/floors-<vendor>.json`, `<run>/evidence/quality-<vendor>.json` |

`<run>/status.txt` holds a one-line status, for example
`grid-run vendor=nvidia phase=all step=run state=MEASURING cells=434 measured=120 failed=2 current=<key> ...`.
`<run>/state/*.done` marks finished steps. Run the same command again to resume:

- Finished steps and finished compile shards are skipped.
- An interrupted shard's job directories without a COMPILED receipt are moved to `builds/.interrupted/` and compiled again.
- Configurations that are already staged are skipped.
- Measured cells are skipped.
- A failed cell stays failed until `--retry-failed`.
- An interrupted cell gets a fresh attempt, and its old attempt directory is kept.

The A/A step is gone. Arm B, the shipped default build, is the same in every cell of a workload on a box, so
`floors --from-pairs` measures the noise floor from those repeats. A workload with fewer than 2 B repeats gets
no floor and is listed as `rejected`.

## The kit (built on the laptop, copied to each box)

```bash
python3 tools/six_lane_grid_run.py kit --out ~/mojolearn-evidence/grid-kit-<date> \
  --search ~/CascadeProjects/mojolearn-six-lane-full-tsvd-20261006 --search ~/mojolearn-evidence/targeted-ab-20261007/planning
rsync -a ~/mojolearn-evidence/grid-kit-<date>/ <box>:/root/grid-kit/
```

The kit is 1.8 MB:

- `vendor-workload-facts.json`: the saved full-workload facts, per vendor and workload id. They were retained from the Oct 6 campaign under `~/mojolearn-evidence/six-lane-full-ab-20261006/{nvidia-native/capture-attempt-02,amd/capture-attempt-01}/artifacts/*/{workload-facts,registered/facts}.json`. This is the same set `targeted-ab-20261007/planning/prepare-vendor-facts.py` used: 100 workloads for NVIDIA and 84 for AMD.
- `required-bindings.json`: the bindings each workload's scored worker attested as loaded. It comes from the retained `deployments.json` files and the committed receipts in `measurements/20261006/receipts`.
- `files/` and `files.json`: the small sha-bound evidence that the registered input variants read: tsvd proposals, projection receipts and plans, classification preparation plans, and the contract.
- `data-manifest.json`: every full input file, with its sha256, its original path on the box and its candidate copies on the laptop. These files are not copied into the kit.

### Full inputs

Put the full inputs on each box at their original paths, under `/root/six-lane-full-ab-20261006/data/` (24 files per vendor):

| files | original box path | laptop copy |
| --- | --- | --- |
| `big-*.{npz,json}` and `reg-*.{npz,json}` (istella and taxi) | `data/` | `~/mojolearn-evidence/six-lane-full-ab-20261006/preserved-inputs/big-reg-full/` |
| `cls-*`, `cat-*`, `raw-*` | `data/classification-full-v1-1bda35ed/` | `.../preserved-inputs/classification-full-v1-1bda35ed/` |
| `tsvd-*.{npz,json}` | NVIDIA `data/tsvd-full-v1/<ds>/`; AMD `data/tsvd-full-v1-repair01/<ds>/` | **not on the laptop.** The tsvd validator refuses relocated inputs, so these must sit at exactly these paths. Recreate them with `tools/six_lane_prepare_full_tsvd.py project`, or the 4 tsvd cells per configuration stay blocked. |

## Decide (both vendors)

`six_lane_grid_decide.sh <run-nv> <run-amd>` writes into `--out`. The default is `<parent>/grid-decide`. It runs these steps:

1. `floors --from-pairs` over both vendors' results: `floors.json`.
2. `verdicts`: `verdicts.json`. A verdict is FASTER or SLOWER only when the change is beyond both vendors' floors and the two vendors agree on the direction.
3. `six_lane_grid_run.py manifest`: `compare-input.json`. It holds one case per configuration and workload, with the columns `nvidia-native` and `amd`. The pinned scope is taken from the NVIDIA receipt. Unpaired cells go to `compare-input-unpaired.json`.
4. `six_lane_compare_results.py`: `compare-<UTC>/report.json`. This is the NVIDIA == AMD check per arm. Exit 1 means a mismatch and exit 2 means incomplete cases; both are results, not tool errors.
5. `six_lane_grid_run.py quality`: `quality.json`. It compares candidate A against incumbent B on the saved task metrics, using `tools/af_quality.py` (rel 1e-3, abs 1e-6).
6. `six_lane_grid_decide.py --matrix <run-nv>/grid/grid-matrix.json.gz --verdicts --identity compare-*/report.json --quality --out decisions`: `GRID_DECISIONS.md` and `grid-decisions.json`. `GRID.md` names `summary.json`, but the comparator writes `report.json`, which carries the per-case `arms` that decide reads.

## Coverage today (dry run at f0ad44ab5, `--crosses auto --cap 0`)

The grid has 3,358 cells per vendor: 1,044 in the factorial regime and 2,314 in the pairwise regime.

| vendor | factorial cells admissible | pairwise cells admissible | workloads with saved facts |
| --- | --- | --- | --- |
| NVIDIA | 434 of 1,044 | 0 of 2,314 | 46 of 161 |
| AMD | 432 of 1,044 | 0 of 2,314 | 44 of 161 |

Every blocked cell is blocked for the same reason: no saved full-workload facts exist for its workload. The saved
facts cover the regression family, kmeans/ols/pca, pls, gmm, isotonic/cross-val and the registered variants
classification-full-v1 and tsvd-full-v1. None of the 20 pairwise-regime algorithms has saved facts: trees, neural
and the large expanded/more lanes have no saved full recipe yet. Those cells stay BLOCKED in `stage/cells-*.json`,
with that reason, until such facts are captured. Rerun `stage` after adding them to the kit; configurations that
are already staged are not revisited, so use a new run directory.
