# bench: progress

Lane `bench` of the algorithm expansion: make `tools/bench_board.py` race every
new algorithm, measure nothing. Worktree `~/mojolearn-wt/algos-bench`, branch
`lane/algos-bench`. Evidence: `~/mojolearn-evidence/algos-bench/`.

## Where it stands

- NEW family `algos` (`tools/bench_board_algos.py`, wired into
  `tools/bench_board.py`, in the default families): 151 algorithm lanes, 282
  races (taxi + Istella-S, or the lane's own data: text, taxi-hourly,
  taxi-zones, enwik8, synthetic). Before: 93 races on every vendor. After:
  375 races (Apple 1,623 fit cells, NVIDIA 1,408, AMD 1,275). The existing
  93 races' plan lines are unchanged (dry-run diff before/after, evidence
  `before_*.txt` / `after_*.txt`).
- Our arms on every algos race: `ours` (IDENTICAL), `ours-fast` (Apple),
  `ours-cpu`. A class the installed wheel does not export reads
  `SKIPPED: not built yet` (worker event `skipped`, cell status
  `bench_board.ALGOS_SKIPPED`); opponents still race.
- Tests: `tools/test_bench_board_algos.py` (new) plus the family counts in
  `test_bench_board.py` / `test_bench_board_cpu_mem.py` (the cpu_mem counts
  were stale at 88 races on main; fixed to 93 with the pre-expansion
  families named). 86 pass on the pod.
- Plumbing smoke (opponent side, `--max-rows 2000`, one round, labelled
  plumbing, no times quoted): `~/mojolearn-evidence/algos-bench/plumbing_smoke.py`,
  log `smoke_nvidia.log`.

## Adding a class name or contract

A lane whose class is exported under another name: add it to the lane's
`ours` candidates in `LANES` (`tools/bench_board_algos.py`). The class
contract the board calls is in that file's docstring.

## Not in R2 (reported, not fetched)

An image set (CIFAR/ImageNet) and an implicit-feedback set
(MovieLens/Last.fm). CNN layers run on seeded tensors; ALS on taxi-zones and
text counts.
