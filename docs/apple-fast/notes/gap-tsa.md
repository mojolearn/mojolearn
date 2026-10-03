# lane/apple-fast-gap-tsa: FAST Apple time-series rows slower than the best opponent

Rows (docs/apple-fast/BOARD_M3_FAST.md at 18dc09a7a; ours FAST ms vs statsmodels ms):
var taxi-hourly 5.1 vs 2.8, synthetic 4.5 vs 2.6; kpss synthetic 3.9 vs 2.6, taxi-hourly 3.8 vs 2.6;
select-d taxi-hourly 7.0 vs 3.4, synthetic 6.2 vs 5.1; theta taxi-hourly 219 vs 171.
Board shapes (tools/bench_board_algos.py:104, :771): 64 series x 1,392 fit points, h = 48;
VAR takes the first 16 series, lag 2 (`m = 33` design columns).

This branch carries lane/apple-fast-tsa2 (b27c8169b, merging to main): TSA2_VAR and TSA2_STL default on,
TSA2_KPSS opt-in. Hypotheses only (no measurement here); the A/Bs below decide.

## Where the time goes (hypothesis)

### kpss (3.8 / 3.9 ms; statsmodels 2.6)
Millisecond row: fixed overhead, the device work is ~90 K floats.
- python/mojolearn/_tsa_impl.py:161 `as_f32_colmajor` copies the board's (n, 64) C array into
  series-major order (both arms pay it; not changed here).
- tsa/estimator.mojo:147 `_upload_f32`: a host stage allocated, filled, copied, then a WAIT (estimator.mojo:73).
- tsa/impl/timeSeries/stationarity.mojo:475 `_refuse_non_finite`: a second download of the whole input,
  a WAIT (:502) and a host scan of 89 K words.
- stationarity.mojo:357-433 `_kpss_test`: eight buffers allocated, eight launches (four of them one block
  per series, `cumsum_by_series_kernel` one THREAD per series serial over 1,392 points), a WAIT (:433).
- estimator.mojo:151 `download_results`: two host buffers, two copies, a WAIT (:522).
Four waits, ~14 allocations, 9 launches, one serial-per-series scan.

### select-d (7.0 / 6.2 ms; statsmodels 3.4 / 5.1)
tsa/impl/auto_arima.mojo:69-73: kpss_test per order d = 0, 1, each round the whole kpss route above
(upload once, then per round: finite download + wait + host scan, 8 launches + wait, download + wait),
the mask on the host. ~7 waits. `MOJOLEARN_SELECT_D` (tsa/impl/select_d_fast.mojo, lane apple-fast-select)
already queues every round with one wait but had never compiled: its launches took two pointers off one
workspace buffer, which the launch's aliasing check refuses (fixed here: each region its own sub-buffer;
main's tsa binding failed on this too, in every mode).

### var (5.1 / 4.5 ms; statsmodels 2.8 / 2.6)
TSA2_VAR already queues the fit on one wait (sequence/pyapi.mojo:509 `_var_fit_queued`, the wait at :611).
What is left per board call: TWO binding calls, each a DeviceExec, pooled allocation, staging copies and a
WAIT: `var_fit` (python/mojolearn/_x_sequence_var.py:93) and `var_forecast` (_x_sequence_var.py:52,
pyapi.mojo:655 a blocking download). Inside the fit, four device-to-host copies (pyapi.mojo:~600-610:
status, params, sigma_u, resid) of words that sit in ONE span of the workspace. The device work (design
1,390 x 33, two GEMMs, a 33 x 33 threadgroup Cholesky, the residual chain) is tens of microseconds.

### theta taxi-hourly (219 ms; statsmodels 171)
sequence/pyapi.mojo:1225 `ex.launch[OP_THETA](a, B)`: ONE THREAD PER SERIES (64 threads on the GPU).
Each thread runs statsforecast's Nelder-Mead (sequence/nm.mojo, up to 1,000 iterations, theta.mojo:590) and
every objective is a whole-series serial pass `theta_run_reg` (theta.mojo:170):
- theta.mojo:190 and :230: two more whole-series folds per evaluation that depend on the series alone
  (sum y, sum (i+1) y for the static A, B; mean|y|), recomputed every evaluation;
- the running mean `my` (a divide per step) is computed for the static models that never read it, and the
  last row is stored every evaluation (only the final run needs it);
- the evaluations of one iteration run one after another: reflection then expansion or contraction is two
  serial passes (nm.mojo:218, :229, :245, :255), a shrink 2 + k (nm.mojo:269), the start k + 1 (:96).
Latency-bound serial chains on 64 threads; the GPU is nearly empty.

## Candidates (all FAST + Apple, default OFF, quality the same metric as the board's)

| define | rows | what | bits |
|---|---|---|---|
| `MOJOLEARN_TSA2_KPSS` (lane tsa2, compile fixed here) | kpss | one launch, one block per series (kpss_fused.mojo), queued upload, finite check in the kernel, three downloads on one wait | stat: FAST block scan for eta (stages 1-3 the 8-launch fold); flag agreement is the gate |
| `MOJOLEARN_TSA_FAST_KPSS_PACK` | kpss | as TSA2_KPSS with one device buffer + one host stage for input and packed outputs (2 allocations, not 8), one download, one wait (kpss_fused.mojo `kpss_rounds`, estimator.mojo `kpss_test_host`) | as TSA2_KPSS |
| `MOJOLEARN_SELECT_D` (lane apple-fast-select, compile fixed here) | select-d | every round queued on the device, one download, one wait | same words as main |
| `MOJOLEARN_TSA_FAST_SELD_FUSED` | select-d | every round of a series inside its block (round 1 differenced from device memory), the first stationary order chosen in the kernel, one upload, one launch, one download, one wait (`kpss_rounds`, estimator.mojo `select_d_host`) | eta's FAST block scan; d agreement is the gate |
| `MOJOLEARN_SEQ_FAST_VAR_SPEC` | var | the fit also queues the forecast recursion from the bound endog's last p rows (64 rows) and downloads it on the fit's wait; `VARResults.forecast` returns those rows when asked for <= 64 steps from the same last rows (byte compare), else its own call: one binding call and one wait fewer (pyapi.mojo `_var_fit_queued`, `var_spec_steps_py`; _x_sequence_var.py) | same words (same kernel, same inputs) |
| `MOJOLEARN_SEQ_FAST_VAR_ONECOPY` | var | params, resid, sigma_u, status (and the speculative rows) as one device-to-host copy, split on the host | same words |
| `MOJOLEARN_SEQ_FAST_THETA_HOIST` | theta | the series-only folds once per series (`theta_invariants`), the objective without the store and without the static models' running mean (`theta_sse_hoisted`) | same words |
| `MOJOLEARN_SEQ_FAST_THETA_SPEC` | theta | one simdgroup per series (coop kernel); lanes 0..3 evaluate reflection, expansion, outside and inside contraction AT ONCE, lanes 0..k the start and shrink vertices; values broadcast, `nm_steps`' decision on every lane, the simplex in registers (sequence/theta_spec.mojo) | same words (each value is the serial value; the serial decision) |

## A/B lines

docs/apple-fast/ab/gap-tsa.txt: one afc_ab_def line per candidate x row (arm A = the branch default, arm B
= the define). The tag is `gaptsa-<cand>-<row>`.
