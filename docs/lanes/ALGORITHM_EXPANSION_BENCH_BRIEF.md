# Lane 10: bench. Get the board ready for every new algorithm; measure NOTHING

Plan: [ALGORITHM_EXPANSION_PLAN.md](ALGORITHM_EXPANSION_PLAN.md). The board is
`tools/bench_board.py`; also read `tools/bench_board_more.py` and
`docs/BENCH_BOARD.md`.

**Goal:** when Andrew says "measure", one board run on each box times every
new algorithm against its opponents with no further code. Until then, run
nothing at measurement size. Speed work across all lanes comes first.

## Setup

- Worktree `~/mojolearn-wt/algos-bench`, branch `lane/algos-bench`, from
  `origin/main`.
- You own `tools/bench_board.py`, `tools/bench_board_more.py`,
  `docs/BENCH_BOARD.md` and their tests. No other lane touches them.
- Pod: `tools/dev_pod.sh up bench` (it retries while RunPod has no stock).
  Use it for opponent installs and plumbing smokes. You never need a Mac.
- At most 1 laptop core, only through `tools/mac_slot.sh run`; avoid even
  that.

## What to add, per new algorithm (the 50 in the plan's 9 lanes)

1. **A race entry** in the right family (trees / classical / classical2 /
   neural, or a new family if a group doesn't fit):
   - dataset(s), following the two-datasets-of-different-kind rule: taxi +
     Istella, never HIGGS alone
   - shape: 1M+ rows or the family's realistic large shape
   - task (clf / reg / transform / cluster / forecast / search)
   - the quality metric printed beside the time
   - training AND inference timing where both exist
2. **Opponents, fastest real implementation per box, matched settings:**
   - scikit-learn (CPU) always
   - cuML on NVIDIA where it has the algorithm: MBSGD, LinearSVC/SVR,
     Lars, RandomProjection, NaiveBayes, TSNE, KernelDensity-style, etc.
   - cuVS / FAISS for IVF-PQ and CAGRA / HNSW
   - LightGBM and XGBoost for DART
   - PyTorch for LSTM / GRU / Conv (eager/compile x fp32/bf16/TF32, same as
     the existing neural races)
   - statsmodels for STL / VAR
   - statsforecast or pmdarima for AutoARIMA
   - Pin every opponent version the way the board already does.
3. **Our arms:** FAST and IDENTICAL (neural too, per Andrew's Sep 27
   decision) and `ours-cpu`.
   - Our side calls the public class by name. If `mojolearn` doesn't export
     it yet, the race reads **SKIPPED: not built yet**, never an error. The
     board must stay usable while lanes land.

## Proof (no measurement)

- `tools/bench_board.py --dry-run` on each vendor (`apple`, `nvidia`, `amd`)
  lists every new race with its arms, opponents, datasets and rows. Commit
  the three plan summaries under `~/mojolearn-evidence/algos-bench/`.
- On your pod: one TINY smoke (`--rows` at the smallest allowed) of the
  OPPONENT side of each new race, to prove installs, imports and the quality
  metric work. Label it a plumbing smoke; never quote its times.
- Existing races unchanged: the dry-run of the existing 93 races is
  identical before and after your change (diff the plan output).

## Merging

Merge `origin/main` in, rerun the dry-runs, then
`git push origin HEAD:main` (fast-forward only; retry on reject). Merge as
soon as the plumbing is green; don't wait for the other lanes. Add new
algorithms' races as the classes land, or up front with SKIPPED handling.

Report to the orchestrator when your pod is up, at each merge, and at the
end: the race count before and after, and the opponent per algorithm (a
table).
