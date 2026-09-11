# Lane forest-finish, H100 leg, 2026-09-11 night

The leg that BUILT and RAN what lane/forest-speed left as source: DEVIATION
2637 (RF and ET row-major staging into the pinned buffer across the host
pool), DEVIATION 2638 (the isolation forest lends X by address), and a new
DEVIATION 2663 trial (the ExtraTrees frontier batch width).

## The box

| fact | value |
|---|---|
| pod | `8gsem9f3thnhvu` (RunPod SECURE, reaped at the end of the leg) |
| GPU | NVIDIA H100 80GB HBM3, driver 580.126.09, `GPU-b645d4a6-3c99-a75a-6492-0b26a9929031` |
| CPU | Intel Xeon Platinum 8470, 208 logical CPUs visible, cgroup quota 22.1 CPUs, joblib `cpu_count` 23 |
| container | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| opponents | cuML 26.08.00, scikit-learn 1.9.1, numpy 2.4.6, CatBoost 1.2.10, LightGBM 4.7.0 |
| tier | IDENTICAL (`MOJOLEARN_NUMERIC_MODE=identical`) everywhere |
| rows | 1,000,000 training rows on both datasets, 100 trees, depth 16, sqrt features, 128 bins for RF, bootstrap for RF only, seed 7 |

## The binary sets (`/root/bins/<set>`, swapped into `python/mojolearn/identical/`)

| set | what it is |
|---|---|
| `baseline` | main `4dc4346a`, the commit this lane's merge took: `_mojolearn_rf.so`, `_mojolearn_trees.so`, `_mojolearn_svm.so` built in a second checkout (`/root/mojolearn_main`) on this pod. `_mojolearn.so` and `_mojolearn_gbdt.so` are byte-identical source on both sides and come from the setup build. |
| `rowmajor` | lane/forest-finish `600dcec0` (DEVIATIONS 2637 and 2638), all five extensions |
| `ctl` | `rowmajor` with `_mojolearn_trees.so` rebuilt from the DEVIATION 2663 source at its default width (4096), the A/B control |
| `stats` | `ctl` plus `-D MOJOLEARN_ET_CYCLE_STATS=1` (level cycles, searched nodes, DEVIATION 205 surveys) |
| `bw16k` / `bw32k` | `-D MOJOLEARN_ET_DEVICE_BATCH_16384=1` / `_32768=1` |

## The batches (`logs/batch{A,B,C,D}.sh`, run on the pod in that order)

- **A** builds both sets, runs `identity_break` on each and the diff, `check-if`
  under IDENTICAL, a non-finite refusal probe through the Python surface on
  both sets, then the same-pod speed cells: pair 1 full (opponent interleaved,
  main then lane), pair 2 ours-only in the reverse order (ABBA), taxi first and
  Istella-S after the download, then one untimed stage replicate per cell.
- **B** proves REACH (the `*_rowmajor` entries are called once for a C-order
  float32 fit and never for an F-order one) and splits the Python-side host
  time on Istella-S for both sets.
- **C** builds the DEVIATION 2663 sets, gates identity against `rowmajor`, runs
  `device_batched_check`, prints the cycle stats, and runs the rotated
  ours-only A/B on both datasets.
- **D** runs `tools/flip_verdict.py` for every switch.

## Results

Filled in when the leg finishes; the tables live in
`bench/OPPONENT_REFERENCE.md` and the lane's commit messages. Big logs are
outside the repo in `~/mojolearn-evidence/forest-finish-2026-09-11/`.
