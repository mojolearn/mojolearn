# LANE STATUS: the `parallel_groves` forest engine on the CPU, and its GPU kernel speed (2026-09-17)

Branch `lane/forest-groves-cpu-and-speed`, cut from `main` at `c85657041`
and fast-forwarded to `f3bff5be4` (the R2 forest datasets) before any commit.
Written for a session with no memory. Small summaries are committed under
`bench/results/forest_groves_2026-09-17/`; every log, JSON and `.so` digest
is under `~/mojolearn-evidence/forest-groves/leg_out/` (outside the repo).

## What Andrew asked for, and what this lane did

Random Forest and Extra Trees have two inference engines, `sequential` (the
host walk, tree-order accumulation, the default) and `parallel_groves` (the
GPU kernel, 32 fixed tree groups, a fixed 16/8/4/2/1 fold). Both are
features of a saved model. This lane:

1. gave `parallel_groves` its public CPU door: `mojolearn.host_model` on a
   `-parallel-groves-1` archive now predicts the GPU groves engine's bits
   through the shipped forest host binding (it refused such archives by name
   before), threaded over rows (DEVIATION 2960), with two negative controls
   (DEVIATION 2961 and the existing forest host sabotage);
2. made the groves GPU kernel faster without touching its topology, fold or
   FTZ rules: the packed node layout with one 16-byte node load is now the
   resident default (DEVIATION 2963), and the per-call host finiteness scans
   run 16 lanes wide across the host pool (DEVIATION 2962);
3. timed cuML FIL on the same box, on our own trees and on cuML's own forest.

`sequential` stays the default and the standard identity record moved no
bit.

## DEVIATIONs

| number | where | what | why no bit moves |
|---|---|---|---|
| 2960 | `core/forest_host_groves.mojo` | the host grove walk fans rows out to host threads (`MOJOLEARN_CPU_THREADS`, DEVIATION 2900's reading) | a thread owns whole rows and its own 32-lane scratch; the per-row body is the pre-2960 loop moved into `_predict_row` unchanged |
| 2961 | `core/forest_host_groves.mojo` | `-D MOJOLEARN_FOREST_GROVES_SABOTAGE=1`: the 16/8/4/2/1 fold becomes a lane-order left fold | a negative control, default off, refused outside the gate by `_forest_host.py`; the existing `MOJOLEARN_FOREST_HOST_SABOTAGE` now also divides the grove fold by `trees + 1` |
| 2962 | `core/forest_inference_model.mojo` | the resident path's input and output finiteness scans are SIMD-16 across the host pool (`scan_finite_f32`); `-D MOJOLEARN_FOREST_PINNED_STAGE=1` fuses the input scan with a copy into a pinned stage (default off); `-D MOJOLEARN_FOREST_PROFILE=1` prints per-stage nanoseconds (diagnostic) | the same predicate on the same bits; host glue only |
| 2963 | `core/forest_inference.mojo` | the packed node layout reads a node's four words with one 16-byte load and is the resident default (`FOREST_PACKED_NODES`; `-D MOJOLEARN_FOREST_SEPARATE_NODES=1` is the separate-arrays arm); `-D MOJOLEARN_FOREST_SHARED_ROWS=1` tiles a block's four rows in shared memory (default off, slower) | the same words compared in the same order; the archive arrays, the comparison and the fold are unchanged |

Files: `core/forest_host_groves.mojo`, `core/forest_inference.mojo`,
`core/forest_inference_model.mojo`, `core/forest_inference_pool.mojo`,
`bindings/_mojolearn_forest_host.mojo` (four new exports:
`forest_host_groves_prepare`, `forest_host_groves_predict`,
`forest_host_groves_release`, `forest_host_groves_sabotage`),
`bindings/forest_inference_binding.mojo` and `checks/forest_inference_model.mojo`
(read the layout switch from the kernel module), `python/mojolearn/_forest_host.py`,
`tools/forest_groves_identity.py`, `tools/forest_groves_speed.py`,
`tools/forest_groves_fil.py`, `tools/forest_groves_body.sh`,
`tools/check_forest_resident_layouts.sh`, `docs/FOREST_INFERENCE_ENGINES.md`.

OWED IN A SHARED FILE (not edited by this lane, exact diff in
`~/mojolearn-evidence/forest-groves/report/host_surface.diff`):
`python/mojolearn/host_surface.py`, the `forest` family's `exports` tuple
gains the four names above and its `host_modules` gains
`core/forest_host_groves.mojo`; until then
`python/mojolearn/tests/test_host_surface.py::test_binding_exports_exactly_the_manifest[forest]`
fails on this branch.

## Box

RunPod `oeb71n6q3y70sy`, NVIDIA L40S (driver 580.159.03; the RTX 4090 create
returned a 500 from RunPod, so the list walked to the L40S), 16 vCPU,
$1.09 per hour, created 17:35Z. Mojo 1.0.0, cuML 26.08.00, treelite 4.7.2.
BEFORE is `main` at `f3bff5be4`; AFTER is the branch at `a9ba2024f` for the
A/B (separate arrays, the SIMD scan) with the four candidates as one-define
rebuilds of the same tree, and `317b21743` (packed default) for the final
column, labeled `after2`.

## Groves CPU identity (tools/forest_groves_identity.py)

Ten lanes (`rf-clf`, `rf-reg`, `et-clf`, `et-reg`, `rf-clf-entropy-log2-noboot`,
`rf-clf-balanced-parallel`, `rf-reg-poisson`, `rf-reg-gamma-ig`,
`et-clf-entropy-bestfirst`, `et-reg-bootstrap-parallel`; `rf-score-weighted`
is a scoring lane with no estimator, n/a), five fixtures (`base`, `ties`,
`odd`, `dupes`, `wide`), every fit switched to `parallel_groves`, saved,
reloaded through `host_model` on the held-out rows, `predict` and
`predict_proba`:

| host set | IDENTICAL | DIVERGENT | REFUSED |
|---|---|---|---|
| production (`_mojolearn_forest_host.so` 16754e6b) | 75 | 0 | 0 |
| `MOJOLEARN_FOREST_HOST_SABOTAGE` (35cce62b) | 25 | 50 | 0 |
| `MOJOLEARN_FOREST_GROVES_SABOTAGE` (7d2a48ab) | 25 | 50 | 0 |

The 25 cells neither sabotage moves are the five classifier lanes' `predict`
(class labels, the argmax of a vote both controls scale or reassociate
without changing the winner); every `predict_proba` and every regression
`predict` moves under both. The fixtures discriminate the engines: the
groves and sequential answers differ on most rows of every numeric cell
(the per-cell counts are in `groves/production.json`).

Large models (500,000 HIGGS rows, all 581,012 Covtype rows, all 515,345
Year rows; production host set, 16 threads): 9 of 9 cells IDENTICAL, GPU
groves against host groves, for `rf-higgs-100x16` (predict, proba),
`et-higgs-100x16`, `rf-covtype-100x16` (seven outputs), `et-year-100x16`
and `rf-higgs-500x16`. Host time per call at 16 threads: RF/HIGGS 100 trees
7.9 s, 500 trees about 40 s, Covtype 2.9 s.

Mac, one core, before the pod: the host grove engine matched a numpy
restatement of the GPU kernels bit for bit on the 24 recorded forest host
models at 1 and 4 threads, and `tools/forest_host_gate.py check` read
IDENTICAL on all 24 sequential fixtures with the new binary.

## Standard identity (tools/identity_break.py, sequential default)

Eleven lanes, five fixtures, two repeats; the CUDA columns `--skip` the two
parallel lanes (their batch part hangs a one-GPU box, lane/infer-speed-trees).

| diff | infer/model | batch |
|---|---|---|
| before-cuda vs after-cuda | IDENTICAL=80, N/A=10 | IDENTICAL=40, N/A=5 |
| before-cpu vs after-cpu | IDENTICAL=80, ONE-COLUMN=20, N/A=10 | IDENTICAL=50, N/A=5 |
| after-cuda vs after-cpu | IDENTICAL=80, ONE-COLUMN=20, N/A=10 | IDENTICAL=40, ONE-COLUMN=10, N/A=5 |
| before-cuda vs before-cpu | IDENTICAL=80, REFUSED=20, N/A=10 | IDENTICAL=40, ONE-COLUMN=10, N/A=5 |
| after-cpu vs sabotage-cpu | DIVERGENT=50, RELOAD-MOVED=50 | IDENTICAL=50 |
| after-cpu vs groves-sabotage-cpu | DIVERGENT=10, RELOAD-MOVED=10, IDENTICAL=80 | IDENTICAL=50 |

Nothing moved. The 20 ONE-COLUMN cells are `rf-clf-balanced-parallel` and
`et-reg-bootstrap-parallel` infer and model on the CPU column, the cells the
BEFORE column REFUSED (its `host_model` refused groves archives) and the
AFTER column answers; the groves script above is their GPU comparison. The
batch cells stay IDENTICAL under both sabotage sets because those sets
rebuild the forest family only and copy the rf and trees host families as
built; the association sabotage moves exactly the ten cells of the two
groves-engine lanes and nothing else, as it should.

## Speed (tools/forest_groves_speed.py, L40S IDENTICAL, ms per public call)

Per model, two processes per arm, each one warmup, seven single calls and
seven blocks of eight calls; the gate is five or more calls, spread at most
1.10, hashes equal. Every arm of every model hashed the same output.
`after` is the SIMD scan alone; `v-packed`, `v-pinned`, `v-shared` are the
candidates on top of it; `after2` is the packed default. Single call
medians (both processes) and eight-call block medians:

| model | rows | before | after | v-packed | v-pinned | v-shared | after2 (packed default) |
|---|---|---|---|---|---|---|---|
| RF HIGGS 100x16, proba | 500,000 | 45.9u, 45.4 / 45.9, 46.1 | 37.9, 38.5 / 38.1, 38.3 | 26.5, 24.0 / 27.3u, 24.0 | 38.6, 38.4 / 39.3, 38.3 | 41.7, 41.9 / 41.5, 44.3 | 24.4, 24.5 / 24.8, 24.8 (block ratio 1.85) |
| ET HIGGS 100x16, proba | 500,000 | 92.9u, 99.1u / 94.9, 98.1 | 91.7, 91.8u / 94.1, 95.1 | 73.8u, 76.0u / 75.3, 75.2 | 92.2, 91.5u / 92.5, 91.6 | 96.3u, 93.7 / 95.9, 95.0 | 70.7u, 75.5u / 77.2, 78.1 (block ratio 1.24) |
| RF Covtype 100x16, proba (7 outputs) | 581,012 | 71.3, 71.6u / 75.4, 77.7u | 58.1u, 54.7u / 62.1, 58.7 | 27.0u, 24.4 / 30.8, 28.3 | 53.6, 54.7 / 57.5, 58.0 | 57.1u, 57.7 / 60.9, 61.6 | 24.0u, 27.0 / 28.1, 31.1 (block 2.6, before block unstable) |
| ET Year 100x16, predict | 515,345 | 90.1, 88.4u / 90.7, 90.7 | 74.9u, 74.0 / 74.1, 73.7 | 64.5, 64.4 / 64.6, 64.2 | 72.3u, 72.1 / 71.1, 71.3 | 77.5u, 79.0 / 82.8, 79.5 | 62.9u, 67.9 / 63.3, 64.3u (block about 1.42, one process unstable) |
| RF HIGGS 500x16, proba | 500,000 | 223.2, 222.3 / 223.6, 224.8 | 216.0, 215.5 / 215.5, 216.7 | 127.2, 125.7 / 126.3, 127.0 | 215.9, 214.0 / 217.2, 215.2 | 233.3, 233.2 / 234.7, 234.6 | 126.9, 129.0 / 126.9, 129.5 (single 1.74, block 1.75) |

`u` marks a process outside the 1.10 spread gate (single calls at 25 to 90 ms
jitter on this box; the eight-call blocks are the steadier scope). The two
HIGGS 100-tree BEFORE processes ran in the final phase (their first run
failed because the BEFORE tree lacked the tool). Ratios are quoted only
where both arms passed the gate in both processes; `after2` against
`before`, eight-call blocks: RF/HIGGS 100x16 1.85, RF/HIGGS 500x16 1.75,
ET/HIGGS 1.24; Covtype and Year have one unstable process on one side and
are quoted as observed, not qualified. Stage profile of the
separate-arrays build, RF/HIGGS 100x16, mean of six calls: input scan 0.7 ms
(one 5 ms outlier), upload 6.2 ms, kernel 30.2 ms, readback 0.5 ms, output
scan 0.25 ms. The kernel is the cost; the pinned stage cannot pay on this
box and did not; the shared-row tile costs more than it saves. The SIMD scan
alone is what moved Covtype (71 to 55 ms: 31 million elements scanned one
compare at a time before) and Year (90 to 74 ms).

Host context, AFTER, 16 threads, single calls: host groves RF/HIGGS 100x16
4.0 s per call, ET/HIGGS 1.8 s, Covtype 1.3 s, Year 3.0 s; the GPU class's
`sequential` engine (the threaded host walk) 5.7, 3.9, 2.1, 4.5 s.

### cuML FIL on the same box (tools/forest_groves_fil.py, /usr/bin/python3)

`fil-ours` is OUR saved forest rebuilt as a treelite model (every node,
`<=` left, vector leaves, `average_tree_output`) and loaded into
`cuml.fil.ForestInference`; its outputs differ from our groves engine by at
most 1.8e-7 (HIGGS, Covtype probabilities) and 3.7e-4 (Year, values near
2000), the association difference, never claimed equal. `cuml-rf` is cuML's
own RandomForest fit on the same training rows with the same trees, depth,
n_bins and max_features, timed through its cached nvForest model
(independent trees). Seven rounds; single call / eight-call block, ms per call:

| model | fil-ours | cuml-rf | ours after2 (packed) |
|---|---|---|---|
| RF HIGGS 100x16 | 18.4u / 14.6u | 17.9u / 16.4u | AFTER2 |
| ET HIGGS 100x16 | 13.0 / 14.1u | (no cuML ET) | AFTER2 |
| RF Covtype 100x16 | 24.1u / 19.5u | 19.6 / 22.0u | AFTER2 |
| ET Year 100x16 | 26.9 / 27.3u | (no cuML ET) | AFTER2 |
| RF HIGGS 500x16 | 40.9 / 41.9 | 48.3u / 43.5 | AFTER2 |

FILRERUN

## Commands

```sh
export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key
export TREES_LEG_STATE=$HOME/mojolearn-evidence/forest-groves/pod
TREES_LEG_NAME=mojolearn-forest-groves TREES_LEG_CUDA_VERSIONS=13.0 \
  MOJOLEARN_STAGE_KEYS="gbm-bench/higgs/higgs_speed.npz gbm-bench/covtype/covtype_speed.npz gbm-bench/year/year_speed.npz" \
  sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 180
# ship main at the base commit to /root/mojolearn-before with a COMMIT file (git archive over ssh), then
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/forest_groves_body.sh setup'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/forest_groves_body.sh variants'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/forest_groves_body.sh identity'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/forest_groves_body.sh groves'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/forest_groves_body.sh speed'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/forest_groves_body.sh fil'
sh tools/trees_leg.sh ssh 'cd /root/mojolearn && bash tools/forest_groves_body.sh final'
sh tools/trees_leg.sh pull /root/leg_out/ $HOME/mojolearn-evidence/forest-groves/leg_out/
sh tools/trees_leg.sh reap
```

Mac, one core (the host door only, no Metal):

```sh
nice -n 19 env OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1 \
  sh bindings/build_forest_host.sh
nice -n 19 env OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical \
  python3 tools/forest_host_gate.py check bench/results/forest_host/2026-09-13-*
```

## Not done, and what was rejected

- Rejected on the box: the pinned host stage (no gain; the upload is 6 of 39
  ms) and the shared-row tile (slower on every model). Both stay in the
  source behind their defines as measured arms.
- Not attempted because it would change the reduction graph, so it is a new
  engine and not this lane: a different thread mapping (rows across lanes, a
  group per thread loop), tree-group chunking by occupancy (nvForest's
  device-derived grove count), or a depth-first node reordering that changes
  which trees a lane holds.
- Not built: a depth-first node order INSIDE the packed layout that keeps the
  lane-to-tree assignment (a cache-locality candidate that keeps the graph);
  a two-stream overlap of the upload with the kernel on chunks of rows.
- The Apple and AMD columns of the next release record; this lane rented
  NVIDIA only. The packed layout passed its Metal correctness matrix on
  2026-09-10 (bench/results/forest_packed_nodes_2026-09-10) but Metal speed
  under the new default was not measured here.
- `tools/forest_host_gate.py` still refuses a groves archive by name (its
  recordings are sequential fixtures); a groves fixture kind for the gate is
  a follow-up.
- No pytest was added for the host groves door (the tests directory is not
  this lane's); `tools/forest_groves_identity.py lanes` is the check.

## False claims found in docs this lane read

- `docs/FOREST_INFERENCE_ENGINES.md` said a `parallel_groves` archive "is
  refused by name" by the host door: true until this lane; rewritten.
- `docs/lanes/BRIEF_forest_host_inference_2026-09-13.md` line 155 still says
  `-parallel-groves-1` archives are refused by name (not this lane's file;
  true text: they predict through the host grove engine since 2026-09-17).
- `python/mojolearn/_backend.py` `_HOST_MODULES` comment and
  `python/mojolearn/host_surface.py` line 709 describe the rf and trees host
  families' resident groves entries as the only CPU groves path; the forest
  family now carries one too (the manifest diff above).
