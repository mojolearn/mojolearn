# ets: damped-ets (ETS(A,Ad,N)) one series per block

Gap: damped-ets taxi-hourly FAST 612 ms vs statsforecast-cpu 185 ms (3.3x); synthetic 3.2x.
Cause: `op_ets` (sequence/ets.mojo) runs each series' whole Nelder-Mead (up to 1000
iterations, 1-2 serial walks of the 1,392 points each) inside ONE thread; 64 series =
64 threads. The walk also evaluates `identical_log(|f|)` per point for the
multiplicative-error term that additive errors never use.

## Switch

`-D MOJOLEARN_ETS_TEAM=1` (comptime `ETS_TEAM` in sequence/ets_team.mojo:
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined[...]`).
Default OFF. IDENTICAL, NVIDIA, AMD and the host column compile main's `op_ets` unchanged.

## What it changes

- `sequence/ets_team.mojo` (new): `ets_team`, one block of SEQ_TEAM_TPB (64) threads per
  series in the GARCH/Prophet team shape (sequence/fit_team.mojo): the Nelder-Mead runs in
  every thread over its private row, the objective is the block's. ETS(A, A|Ad, N) is an
  affine recurrence in (l, b), so the series is split into 64 chunks: pass 1 composes each
  chunk's 2x2 affine map with the start unknown, the exclusive prefix gives each chunk its
  start state, pass 2 walks the chunk and sums e^2; the likelihood n log(sum e^2) is every
  thread's ascending fold over the 64 partials (uniform control flow). Same model, bounds,
  start values, stall stop and Nelder-Mead as `op_ets`; no per-point log.
- `sequence/ets_team_py.mojo` (new): `ets_team_py`, ets_py's contract; one group launch
  through `_run_team` (Apple slices only when the command-buffer bound needs it: at
  n 1,392 the budget is ~7,000 iterations, so one launch) and one wait, then the forecast
  and info downloads. `ets_team_applies(ip)`: error A, trend A, season N.
- `sequence/exec_device.mojo` `team_kernel`: `elif ETS_TEAM and OP == OP_ETS: ets_team(...)`.
- `bindings/_mojolearn_x_sequence.mojo` `ets_binding`: under `comptime if ETS_TEAM`, routes
  the applying models to `ets_team_py`; everything else `ets_py`.
- `tools/afc_ab_def.sh`: copied from origin/lane/apple-fast-tier.

Bits: FAST only. The error fold order and the trend update (`beta e` in place of
`(beta / alpha) ((l' - l) - phi b)`) differ from op_ets; quality must hold within FAST
run-to-run spread (same optimizer from the same start on the same objective).

## Risky compile sites

- sequence/ets_team.mojo `EtsTeamObj.eval`: `Tuple[Float32 x5]` unpack; `self.team.sync()`
  inside a `mut self` eval (as GarchTeamObj).
- sequence/ets_team.mojo `ets_team`: `nm_steps(obj, s, lo, hi, k, nm_scr, 1000, 1e-4,
  stall_iters=a.i2, stall_rel=a.f4, budget=left)` skips `snap` by keyword (op_ets's form);
  `ets_init_state(y, n, True, SEAS_N, 1, nm_scr, nm_scr, nm_scr)` (season N touches none
  of the scratch pointers).
- sequence/ets_team_py.mojo: imports the underscore helpers `_group, _run_team, _saves,
  _team_budget, _zero` from sequence.fit_team_py (fit_team_py itself imports `_prophet_X`
  from pyapi the same way).
- bindings/_mojolearn_x_sequence.mojo `ets_binding`: `comptime if ETS_TEAM:` with a runtime
  `if ...: return` inside, then the fallthrough return (prophet_fit_team_py's form).

## Request

docs/apple-fast/ab/ets.txt: `ets-team-taxi` (afc_ab_def.sh, binding x_sequence, lane
damped-ets, taxi-hourly, 1 2, arm B `-D MOJOLEARN_ETS_TEAM=1`). synthetic only after
taxi-hourly wins.
