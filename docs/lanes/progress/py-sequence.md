# lane py-sequence (Python-work fixes, sequence family)

Brief: ~/mojolearn-evidence/py_work_brief.md; findings: ~/mojolearn-evidence/python_work_audit.md
(Family: sequence). Base: lane/apple2-merged 0a11b50c7. Branch lane/py-sequence.

## What changed

| # | audit row | change | bits |
|---|---|---|---|
| 1 | sequence 1, DEVIATION 5540 | `_x_sequence_sched.py`: StepLR / ExponentialLR / OneCycleLR enclose the exact value in a fixed-width integer interval (128-bit ends, directed rounding at every product; OneCycle's cosine a 160-bit fixed-point Taylor series with its Lagrange remainder and pi from `_training_impl`'s rational bracket) and return a float32 only when BOTH ends round to it. Rounding with flush is monotone, so that float32 is the exact value's. Otherwise the old all-Fraction evaluation decides (kept as `_exact_lr_at`, also the reference). gamma^e is carried across calls (one product per step); StepLR caches per exponent. | same contract (exact rational, one rounding); proven by construction, `sched_check.py` (independent 70-digit oracle) and `ab.py sched_fast_vs_exact_dense` |
| 2 | sequence 2 | `opt_step_py` takes an optional 6th address (float32[3]: beta1^t0, beta2^t0, NAdam mu product) and t0; it replays only t0+1..t-1 (none on a normal step) and writes the state back. `_SeqOptimizer` carries it (state_dict gains `scalars`; an old state dict replays once). LAMB likewise (5th address, fp[6] = t0). | same products in the same order |
| 3 | sequence 8 (part) | `opt_step_py` binds, uploads and downloads only the state slots the kind touches (`opt_slots`: SGD s1 iff momentum; RMSprop s2, s1 iff momentum, s3 iff centered; Adagrad s2; Lion s1; Adam/AdamW/Adamax/NAdam s1, s2). An unused slot is handed the params buffer, which the element body never dereferences for it. | element body unchanged |
| 4 | sequence 9 | RNN `_schedule` builds the step table with numpy (the permutations are drawn in the same order); the binding's order-LENGTH cap 2^24 is lifted to the int32 offset range (indices stay < N < 2^24, exact in float32), so 1M rows x more than 16 epochs is no longer refused. | same order and steps (checked byte for byte on 5 shapes) |
| 5 | sequence 10 | LayerNorm backward passes y = 0; the binding skips the (M, D) y download (and Python no longer allocates it). | dx, dw, db unchanged |
| 6 | sequence 7 | Theta `model_` by one `np.take` (was `list(_MODELS)` rebuilt per series); Theta and ETS keep the last forecast call (key: every argument that reaches the binding) so a repeat predict does not re-run the whole fit; fit snapshots y (a private copy), so the stored answer cannot go stale. | same binding call on a miss |
| 7 | sequence 13 | Prophet: sortedness by one O(n) comparison, not a stable argsort; NaN in t refused by name (it used to pass the argsort test at the end). | unchanged for valid input |
| 8 | sequence 5 (part) | AutoARIMA BIC: `_portable_math.log`, not the host libm's `math.log` (an identity defect: an argmin near a tie could flip between hosts). Candidate batching NOT done. | equal wherever libm's log is correctly rounded |
| 9 | sequence 6, DEVIATIONs 2421, 2422 | Holt-Winters: `level_` / `trend_` / `season_` are built on first use, not at every fit and load; `get_level(index)` (and trend, season) and `forecast(index=...)` / `predict(index=...)` take the ONE series' strided slice instead of un-interleaving all series. | same bytes |
| 10 | sequence 11 (part) | KPSS flags: a C-level `astype('<u1')` of the binding's 0/1 int32 (was a Python loop). CPU `select_d` loop NOT changed. | same bytes |
| 11 | sequence 3 | `parallel_forecasting` Holt-Winters shard copies: one strided slice per series column (`_gather_cols` / `_scatter_cols`) instead of one `memcopy` per time row per component. | same bytes (checked against the memcopy loop locally) |

## Before / after

Job: `bench/py_sequence/job.sh` on the shared NVIDIA pod (both trees in one job: BASE =
0a11b50c7 worktree, NEW = this branch; GPU column and x86 CPU column).
PENDING: the shared pod was down when the lane was ready (19:3xZ); no pod, no numbers yet.

## DEVIATION changes

- 5540: contract kept (exact rational, one rounding to float32); the Python Fraction
  implementation retired from the hot path (it remains only as the undecided-case fallback
  and the reference). IDENTITY_PATHS.md row 154 text: unchanged contract.
- 2421: the un-interleave is now done on first use, not at every fit/load (docstring updated).
- 2422: `index=` returns one strided slice.
- 991: NOT retired (aic_/bic_ still a Python O(B) comprehension).

## Not done / unproven

- Optimizer state RESIDENT on the device (brief item 2, second half): waits for py-shared's
  "SHARED API READY" (not present on origin/lane/py-shared at 19:30Z). Today each step still
  uploads params + grads + the USED state slots and downloads params + used slots.
- Adafactor multi-tensor call; persistent packed buffer for multi-tensor param lists.
- AutoARIMA candidate batching (IC and argmin in Mojo).
- ARIMA aic_/bic_ into the binding (DEVIATION 991); CPU `select_d` host export.
- SmallMLPTrainer fused step (audit row 4), MoE resident weights (row 14).
