# LANE STATUS: the `parallel_groves` kernel's schedule, one thread per row (2026-09-17)

Branch `lane/forest-groves-row-schedule`, cut from `main` at `32acd33d8`.
Written for a session with no memory. One deviation, DEVIATION 2964, in
`core/forest_inference.mojo`. Evidence outside the repo under
`~/mojolearn-evidence/forest-groves-row/`.

## What changed, and why no bit can move

`forest_grove32_kernel` and `forest_vector_grove32_kernel` give one row 32
threads. Thread `lane` sums trees `lane, lane + 32, ...` ascending from +0.0;
the 32 sums meet in shared memory and fold 16/8/4/2/1 across seven barriers.
So the 32 threads of a thread group walk 32 DIFFERENT trees at the same
moment, and no two neighbors ever read the same node.

With `-D MOJOLEARN_FOREST_ROW_THREADS=1` one thread owns an item (a row and
output, or a row under vector leaves). It walks the trees in ascending order,
adds each leaf into private sum `tree % 32`, and folds its 32 private sums
16/8/4/2/1. Lane `l` therefore still receives trees `l, l + 32, ...` in that
order from +0.0, and every fold step adds the same pair with the same
`forest_add`. Each floating point addition has the operands and the order it
had; what changed is which thread performs it. There is no shared memory and
no barrier, and the threads of a group now walk the SAME tree on adjacent
rows. `core/forest_host_groves.mojo` has run this schedule on the CPU to the
GPU's bits since lane/forest-groves-cpu-and-speed.

lane/forest-groves-cpu-and-speed's status document listed "a different thread
mapping" under changes that would alter the reduction graph. That is true of
nvForest's device-derived grove count and false of this mapping, which keeps
the 32 lanes, their tree order and the fold.

`-D MOJOLEARN_FOREST_ROW_THREADS_SABOTAGE=1` is the negative control: the 32
sums fold in lane order. Default off, never shipped.

## Evidence

### M4 Metal, one core, `checks/forest_inference_gpu.mojo`, IDENTICAL

The check restates the grove graph independently (`reference_mean`) and
asserts bit equality, including a cancellation case where ordered and grove
sums must differ.

| arm | result |
|---|---|
| default | PASS |
| `MOJOLEARN_FOREST_ROW_THREADS=1` | PASS; the 1,440 `BITS` lines hash equal to the default arm's (`2a1e2a95b976b056`) |
| `..._ROW_THREADS=1` and `..._SABOTAGE=1` | FAILS (exit 1) at the cancellation case |

`checks/forest_inference_model.mojo` (the resident model check) also passes
under the define. Untimed on Metal.

### NVIDIA

Pod `z8d2a8y7d5kzii`, RTX 4090 (driver 580.159.04), 26 vCPU, $0.74 per hour,
2026-09-18 01:30Z to about 03:00Z. Every model prepared once at commit
`86fc316b6` (`tools/forest_groves_speed.py prepare`, the whole dataset as the
prediction rows for taxi, taxireg, istella, istellareg, covtype and year;
500,000 HIGGS rows); three arms built from one tree: `after` (the default
build, its row kernels off), `v-rowthreads2` (`-D MOJOLEARN_FOREST_ROW_THREADS=1`)
and `v-rowsab2` (plus the sabotage). Seven rounds, one warmup, eight-call
blocks, two processes per arm; `u` marks a spread above 1.10. Medians per
process, ms per call. Summaries in `bench/results/forest_groves_row_2026-09-18/`;
the JSONs and logs in `~/mojolearn-evidence/forest-groves-row/pod2/`.

| model | rows | columns | outputs | 32-thread kernels (default) ms | row schedule ms | ratio | hashes | sabotage hash |
|---|---|---|---|---|---|---|---|---|
| rf-taxi-100x16 | 4,610,786 | 16 | 2 | 189.4 / 190.7 | 125.7 / 127.5 | 1.49 | equal | differs |
| rf-taxireg-100x16 | 5,750,086 | 16 | 1 | 237.9 / 239.0 | 157.4 / 158.1 | 1.51 | equal | differs |
| rf-higgs-100x16 | 500,000 | 28 | 2 | 24.0 / 24.3 | 20.4 / 19.9 | 1.19 | equal | differs |
| et-higgs-100x16 | 500,000 | 28 | 2 | 66.5 / 67.4 | 68.0u / 63.5 | 0.99 | equal | differs |
| rf-higgs-500x16 | 500,000 | 28 | 2 | 137.6 / 138.6 | 104.0 / 105.1 | 1.32 | equal | differs |
| rf-covtype-100x16 | 581,012 | 54 | 7 | 27.0u / 27.4 | 32.0 / 32.5u | 0.84 | equal | differs |
| et-year-100x16 | 515,345 | 90 | 1 | 56.5 / 54.1 | 63.8 / 64.9 | 0.87 | equal | differs |
| rf-istella-100x16 | 2,543,304 | 220 | 2 | 275.2 / 273.8 | 619.4 / 616.4 | 0.44 | equal | differs |
| rf-istellareg-100x16 | 2,543,304 | 220 | 1 | 263.1 / 257.5 | 545.0 / 546.3 | 0.48 | equal | differs |

geomean all nine 0.94; the dispatch takes the row schedule only for columns <= 32

Every cell hashed equal between the default and the row schedule and every
sabotage cell hashed differently, on 500,000 to 5,750,086 rows. The schedule
wins on 16 and 28 column rows (RF 1.19x to 1.51x, ET even) and LOSES above
that, down to 0.44x at 220 columns: with one row per thread, adjacent
threads read 32 different rows, and past two cache lines a row the feature
reads stop coalescing. The first version of the row kernels (commit
`13493dc27`, a per-tree read-modify-write of the private sums) was 0.84x on
Covtype too; the lane-by-lane rewrite did not change that verdict, so the
loss is the memory pattern, not the register pressure.

### The dispatch, and the final build

Commit `618e7b5f9`: `checks/kernel_matrix.mojo::forest_row_threads_for`
turns the schedule on for NVIDIA only; `core/forest_inference.mojo` takes it
when `n_features <= FOREST_ROW_THREADS_MAX_FEATURES` (32) and the 32-thread
kernels otherwise. Rebuilt on the same pod with NO defines as
`/root/mojolearn-final` and checked against the default build:

- `tools/identity_break.py`, nine forest lanes (rf-clf, rf-reg, et-clf,
  et-reg, rf-clf-entropy-log2-noboot, rf-reg-poisson, rf-reg-gamma-ig,
  et-clf-entropy-bestfirst, rf-score-weighted; the two `-parallel` lanes
  skipped at the time, because they hung a one-GPU box; DEVIATION 3010
  unblocked them on 2026-09-18), five fixtures, two repeats, cuda
  column: default vs final IDENTICAL=45 train, IDENTICAL=80 infer/model
  (N/A=10), IDENTICAL=40 batch (N/A=5). Nothing moved.
- Confirmation timing, two processes each (`final_confirmation.txt`): taxi RF
  198.8 / 201.9 default against 135.0 / 131.7 final (the row schedule,
  1.49x); HIGGS RF 25.0u / 25.2u against 21.7u / 21.3 (1.16x); Istella-S RF
  304.1u / 285.4 against 290.5 / 289.4 (the 32-thread kernel kept, parity).
  Hashes equal in every pair.

## Owed

- Apple and AMD columns at the next release record: the row kernels compile
  on every column and are bit-identical on Metal (the oracle above) but are
  untimed there, so the default stays NVIDIA-only.
- The 32-column cut sits between the measured 28 (wins) and 54 (loses); a
  model with 33 to 53 columns was not measured and takes the old kernel.
- ET inference is about 5x behind cuML FIL on the same trees regardless of
  schedule (66 ms against about 14 on the L40S for ET/HIGGS); ET/HIGGS has
  FEWER nodes (1.88M) than RF/HIGGS (3.89M) yet runs 3x slower, so it is path
  length or layout, not size. Not this lane's; measure average path length
  first.
- `tools/forest_groves_body.sh` gained overridable roots, `setup-lite` and
  `SKIP_HOST_WALKS`; `tools/forest_groves_speed.py` gained the taxi and
  Istella-S models and a prepare that adds to the manifest. An ET REGRESSOR
  fit took 377.8 s on taxireg (5.25M x 16) and 179.0 s on Year where the RF
  classifiers take 1 to 5 s and RF regressors 8 to 37 s: reported to
  lane/forest-train-speed, not touched here.
- The deadlock in `tools/forest_groves_identity.py large` (below) was
  reported here and FIXED on 2026-09-18 by DEVIATION 3010.

## A deadlock on main, found on the way

`tools/forest_groves_identity.py large` hung on the DEFAULT build (main's
kernels) on 2026-09-17 19:47Z on a 96-vCPU RTX 4090 pod: `et-higgs` finished,
then 195 threads in `futex_wait`, 0 percent CPU, GPU idle, for 30 minutes on
the next model. Reproduced on the 26-vCPU pod with a 150 s watchdog and
`faulthandler`: every SINGLE model passes (Covtype at default, 16 and 1 host
threads; RF/HIGGS), and `et-higgs` followed by `rf-covtype` in ONE process
hangs in `python/mojolearn/_forest_protocol.py:28`, `_ResidentForest.__init__`,
inside `native.forest_prepare_gpu(...)` for the SECOND model. Each
`ResidentForest` (`core/forest_inference_model.mojo`) creates its OWN
`DeviceContext()`, so a second model is a second device context created
while the first is still alive (its Python finalizer runs at collection).
The speed tools never see it because they load one model per process. This
is the same signature as the known hang of the two `-parallel` lanes in
`tools/identity_break.py` on a one-GPU box, which every lane has been
skipping. Probe logs: `~/mojolearn-evidence/forest-groves-row/pod2/leg_out/hang/`.

Pinned further with `bench/results/forest_groves_row_2026-09-18/ctx_probe.py`
(two saved `parallel_groves` models through the public `predict` on 2,000
rows in one process, 120 s watchdog; `ctx_probes.txt`):

| sequence | result |
|---|---|
| RF/HIGGS then RF/Covtype, first model KEPT alive | both predict (0.78 s, 0.28 s) |
| ET/HIGGS then ET/Year, kept | both predict |
| RF/HIGGS then RF/HIGGS 500 trees, kept | both predict |
| RF/HIGGS, then `del` + `gc.collect()` (the finalizer calls `forest_release_gpu`), then RF/Covtype | HANGS in the second model's `forest_prepare_gpu` (watchdog, exit 124) |

So the deadlock is RELEASE THEN PREPARE: after a resident forest's device
snapshot is released (`resident_release`, `ResidentForest.close`, its
`DeviceContext` dropped), the next `resident_prepare` in the same process
never returns. Two live snapshots are fine. Any program that fits or loads a
second groves forest after the first was collected hits this, which is what
`tools/forest_groves_identity.py large` does between models. **FIXED 2026-09-18 by DEVIATION 3010
(lane/forest-deadlock).** The suspect named here was right, and the
mechanism was already on record as DEVIATION 2520: `ResidentForest.close`
synchronized BEFORE its ten releases and destroyed the context immediately
after them, so the context died with their buffer frees in flight and the
MAX runtime allocator's lock stayed held for the whole PROCESS (which is
also why a live snapshot's next allocation can block). Watched failing and
then passing on an RTX 4090 with a native backtrace, 70 forest/GBDT/KDE
cells bitwise identical across the change, and the two `-parallel`
identity_break lanes now run the full protocol on one GPU. See
`docs/lanes/LANE_STATUS_lane-forest-deadlock.md`; the reproduction is
`tools/forest_release_prepare_repro.py`, which needs no dataset.
