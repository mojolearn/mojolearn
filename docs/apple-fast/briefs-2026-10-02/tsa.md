# tsa
Worktree ~/mojolearn-wt/tsa, branch lane/apple-fast-tsa (1 WIP commit 6c72cf77a; merges main cleanly).
1. `git merge origin/main` (clean), push.
2. Finish the WIP: arima/impl/batched_kalman.mojo (+234), arima/impl/batched_arima.mojo, tsa/arima_common.mojo, new arima/impl/fast_eval_ws.mojo (+226). Read the WIP diff with `git show 6c72cf77a --stat` and grep for its switch names / TODOs. Complete the batched Kalman FAST evaluation workspace behind one `-D MOJOLEARN_<NAME>` define, FAST + Apple only, GPU only, no device-to-host round trips inside the L-BFGS iteration. Main already has FIT_COMPACT (arima/impl/batched_fit.mojo) and ARIMA_FAST_BATCH_GRAD: build on them, do not duplicate them.
3. Add docs/apple-fast/ab/tsa.txt: arima lanes on taxi-hourly (AFC_FAMILY=algos; grep tools/bench_board_algos.py for the arima lane names), one tag per lane, and docs/apple-fast/ab/tsa.md.
