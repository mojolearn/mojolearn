# sequence: progress

Pass 1 (code first). Pod: dev_pod `sequence`, NVIDIA A40. Gate per algorithm:
builds, sanity vs reference, `tools/algos_lane_check.sh <lane>` AGREE (CPU == NVIDIA).

Shared machinery (first commit): `sequence/` — `ops.mojo` (one scalar body per
output element, IDENTICAL seams), `exec.mojo` (`Exec` trait, `HostExec`),
`exec_device.mojo` (`DeviceExec`, one GPU thread per element), `recurrent.mojo`
(RNN/LSTM/GRU BPTT + optimizers), `pyapi.mojo` (the address contract both
bindings export). Python: `python/mojolearn/_x_sequence_rnn.py`.

| # | algorithm | lane | commit | AGREE line | sanity |
|---|---|---|---|---|---|
| 1 | LSTM (LSTMRegressor, LSTMClassifier) | sequence-lstm | bb6a93893 | CLEAN: sequence-lstm: AGREE: compared infer 9, train 9 (cuda column vs CPU column) | torch nn.LSTM, 5 full-batch steps, float64 torch: max param diff SGD 4e-8, Adam(2 layers) 1e-6, CE Adam 8e-7, RMSprop(centered, momentum) 2e-5, Adagrad 2e-7, AdamW 1e-6 |
| 2 | GRU (GRURegressor, GRUClassifier) | sequence-gru | bbd4a08e0 | CLEAN: sequence-gru: AGREE: compared infer 9, train 9 | torch nn.GRU: SGD 5e-8, Adam 2 layers 1e-6, CE Adam 8e-7, RMSprop centered 2e-5 |
| 3 | RMSprop (torch.optim-shaped, in-place float32 params) | sequence-rmsprop | 3b3648bc1 | CLEAN: sequence-rmsprop: AGREE: compared train 9 | torch.optim.RMSprop float64, 8 steps: default 2.5e-7, centered+momentum+wd 5.1e-7, alpha 0.9 3.7e-7 |
| 4 | Adagrad (torch.optim-shaped) | sequence-adagrad | c7ff7aee1 | CLEAN: sequence-adagrad: AGREE: compared train 9 | torch.optim.Adagrad float64, 8 steps: default 1.7e-7, lr_decay+wd+init acc 1.3e-7 |
| 5 | AutoARIMA (cuML auto_arima.pyx search over mojolearn.ARIMA + select_d) | sequence-autoarima | e47ca54e3 | CLEAN: sequence-autoarima: AGREE: compared train 9 (runs _mojolearn_arima, _mojolearn_tsa and their hosts) | 4 simulated series (AR1, MA1, random walk, ARMA(2,1)), p,q in 0..2, aic: same orders and AICs as statsmodels ARIMA (to 3e-3), forecasts to 1e-3 |
| 6 | STL (statsmodels STL, batched, one series per GPU thread) | sequence-stl | d5cbc28c3 | CLEAN: sequence-stl: AGREE: compared train 9 | statsmodels STL float64 on 150 obs with outliers: default 3e-5, robust 4e-5, deg-0 + jumps 8e-6, period 7 robust deg 0 1e-6 (season, trend, weights) |
| 7 | VAR (statsmodels VAR, OLS per equation, fixed lag) | sequence-var | 812d79dff | CLEAN: sequence-var: AGREE: compared train 9 | statsmodels VAR on a simulated 3-variable VAR(2): params 9e-6, sigma_u 7e-7, 5-step forecast 2e-5 (p=1,2 with constant, p=3 without) |
| 8 | MLPClassifier / MLPRegressor (sklearn-shaped, own trainer) | sequence-mlp | 70349dafc | CLEAN: sequence-mlp: AGREE: compared infer 9, train 9 | sklearn MLP (shuffle=False, same random_state): loss curves and predictions within 3e-6 for Adam+tanh+L2 regressor, relu 3-class, logistic binary, SGD invscaling, SGD adaptive (both stop at epoch 57) |
| 9 | RNN (RNNRegressor, RNNClassifier; tanh / relu) | sequence-rnn | 8035fcbbf | CLEAN: sequence-rnn: AGREE: compared infer 9, train 9 | torch nn.RNN float64, 5 steps: tanh SGD 5e-8, relu Adam 2 layers 2e-6, tanh CE Adam 8e-7, relu CE AdamW 2 layers 1e-6 |
| 10 | Lion (torch.optim-shaped; also `optimizer="lion"` in the recurrent estimators) | sequence-lion | d1f97820a | CLEAN: sequence-lion: AGREE: compared train 9 | float64 restatement of lion-pytorch's step, 6 steps, defaults and betas (0.95, 0.98) + wd 0.1: within rtol 2e-5 (tests/test_x_sequence_optim.py, GPU and CPU) |
| 11 | Adafactor (torch 2.5 rule; 1-D and 2-D tensors) | sequence-adafactor | cc6ca19b8 | CLEAN: sequence-adafactor: AGREE: compared train 9 | torch.optim.Adafactor (2.5.1, float32), 10 steps on a 6x5 matrix + a vector: defaults 7e-9, lr/beta2_decay/d/wd moved 1.2e-7 |
| 12 | LAMB (timm Lamb statement: global clip, trust ratio, trust_clip, always_adapt) | sequence-lamb | 7f722502c | CLEAN: sequence-lamb: AGREE: compared train 9 | timm.optim.Lamb (float32), 8 steps: defaults 1.2e-7, no-decay always_adapt trust_clip no-clip 3.6e-7, wd 0.1 clip 0.5 1.2e-7 |
| 13 | Adamax (torch.optim-shaped; also `optimizer="adamax"` in the recurrent estimators) | sequence-adamax | 315b89a64 | CLEAN: sequence-adamax: AGREE: compared train 9 (and the six optimizer-using lanes re-AGREE after the host-scalar refactor) | torch.optim.Adamax float64, 8 steps: defaults 2.4e-7, betas/wd moved 2.9e-7 |
| 14 | NAdam (torch.optim-shaped; also `optimizer="nadam"`) | sequence-nadam | d47e0a82c | CLEAN: sequence-nadam: AGREE: compared train 9 | torch.optim.NAdam float64, 8 steps: defaults 5.2e-7, decoupled wd + momentum_decay 0.01 6.8e-7 |
| 15 | LR schedulers StepLR, ExponentialLR, OneCycleLR (exact rational, float32 once; `lr_schedule=` on the lane's optimizers and recurrent estimators) | sequence-lr-schedulers | 0e01f3930 | CLEAN: sequence-lr-schedulers: AGREE: compared train 9 | torch.optim.lr_scheduler (float64), 40 steps: StepLR 1.5e-8, ExponentialLR 5.3e-8, OneCycle cos 4.8e-8, linear three-phase 5.4e-8 relative |
| 16 | LayerNorm (module + layer_norm_forward / layer_norm_backward) | sequence-layernorm | 7b732166d | CLEAN: sequence-layernorm: AGREE: compared infer 9, train 9 | torch F.layer_norm float64 with weight and bias: y 3.5e-7, dx 2.1e-7, dw 8.7e-6, db 3.0e-6; (2, 5) normalized shape 2.8e-7 |
| 17 | Theta forecasters (Theta, OptimizedTheta, DynamicTheta, DynamicOptimizedTheta, AutoTheta; statsforecast) with statsforecast's Nelder-Mead (`sequence/nm.mojo`, reusable) | sequence-theta | 462d18391 | CLEAN: sequence-theta: AGREE: compared train 9 | statsforecast 2.1.1 on 4 series (seasonal, random walk, trend, additive seasonal), 12-step forecasts: max relative diff 7.5e-7 except OTM/DOTM on the trend series 6.7e-5 / 1.7e-3 (float32 Nelder-Mead path) |
| 18 | Croston (CrostonClassic, CrostonOptimized, CrostonSBA; statsforecast) | sequence-croston | 5612c9178 | CLEAN: sequence-croston: AGREE: compared train 9 | statsforecast 2.1.1 on 6 intermittent series (one all-zero, one with a negative event): classic 4.6e-8, SBA 6.7e-8, optimized 2.1e-4 relative (float32 golden section) |
| 19 | Damped-trend ETS (ETS for ANN/AAN/MNN/MAN and damped, DampedETS; statsforecast) | sequence-ets | 6b0a36994 | CLEAN: sequence-ets: AGREE: compared train 9 | statsforecast AutoETS on 4 series, 10-step forecasts: AAdN <= 8.6e-4, MAdN <= 1.2e-3, AAN <= 6.1e-4, ANN <= 1.8e-6, MNN <= 3.7e-5 relative |
| 20 | GARCH(p, o, q), constant or zero mean, normal (arch package statement; Nelder-Mead instead of SLSQP) | sequence-garch | 5baf7a3f3 | CLEAN: sequence-garch: AGREE: compared train 9 | arch 8.0 on 3 simulated series (n=1000): GARCH(1,1) params within 1e-3, loglik within 0.004, 5-step variance forecast <= 1.3e-3 relative; GJR zero-mean loglik within 0.03 |
| 21 | Prophet-style forecaster (ProphetForecaster: linear trend with changepoints, Fourier seasonality, holiday regressors, additive / multiplicative, MAP by our L-BFGS) | sequence-prophet | e63e27fa8 | CLEAN: sequence-prophet: AGREE: compared train 9 | prophet 1.4.0 (Stan) on 800 daily points with a trend break, weekly + yearly seasonality and a holiday regressor: max diff / max y 2.2e-4 history, 3.2e-4 30-day future (additive); 9.3e-4 / 4.2e-4 (multiplicative) |
| 22 | Mixture-of-experts block (MoEBlock, Mixtral sparse block forward; top-k with the lower-index tie rule) | sequence-moe | (the commit adding this row) | CLEAN: sequence-moe: AGREE: compared infer 9, train 9 | float64 torch restatement of MixtralSparseMoeBlock (HF layout) on 50 tokens, 8 experts, top 2: max diff 9.6e-10 (max y 3.3e-3), same experts, logits 1.2e-6 |

Pod setup note: the ARIMA path needs `python/mojolearn/.libs/libMojolearnMath.so`; build it on a fresh pod with `packaging/portable_math/stage.py`'s `build()`.

PASS 1 COMPLETE (main table and every Addition merged).

Merge-gate notes: test_host_surface passes; tools/test_lane_select.py has one
failure that is not this lane's (core/forest_host_predict.mojo answers 81
lanes, not 80: all trees/gbdt lanes, none sequence-*), as of the Prophet merge.

## PASS 2 (proof)

Seams: 17, DEVIATIONS 5500-5516 (`sequence/README.md`), IDENTITY_PATHS rows
150-159. Host oracles `sequence/checks/oracle.mojo` (from the reference
semantics, each with its `alt` spelling); driver `sequence/checks/seams_check.mojo`
(fixture must separate pinned vs alt, then device == oracle and host ==
oracle, stage `sequence.<seam>` on the card). Arms: `sequence/checks/sabotage/seam_55xx_*.patch`,
listed in `tools/identity_lanes/sequence.checks`. `DeviceExec` now takes ONE
process-lifetime DeviceContext (`exec_device.mojo::sequence_ctx`, `_Global`
per numeric tier); torch.lerp is one body (`ops.mojo::lerp`) for Adamax,
NAdam and Adafactor.

- NVIDIA A40 (fixed lane check, after 02b63f107), `--pass 2` over all 22
  lanes: every one of the 17 arms PASS / FAIL (exit 1) / PASS after reversal;
  all 22 lanes CLEAN AGREE (CPU == cuda). DONE, never re-run.
- End-to-end CPU-column sabotage for the stewards:
  `sequence/checks/sabotage/e2e_host_download_bit.patch` (21 lanes: HostExec
  download flips the low bit) and `e2e_arima_host_param_bit.patch`
  (sequence-autoarima: the ARIMA host binding's params). Both proven on the
  A40: AGREE, DISAGREE under the patch on every lane, AGREE after reversal.
- Merge gate on the A40: test_host_surface 196 passed; tools/test_lane_select.py OK, 0 failures.
- Steward requests at d5a849bb5 (m2pro + do-amd; m3ultra spooled):
  1790537359123-sequence-d5a849bb57 (21 lanes), 1790537368020-sequence-d5a849bb57
  (autoarima). do-amd PASS on both.

### The family's earlier algorithms (LANE CHARTER: ARIMA, ExponentialSmoothing, KPSS)

Their own drivers and oracles (`arima/checks/arima_check.mojo`,
`fit_check.mojo`, `holtwinters/checks/hw_check.mojo`,
`tsa/checks/stationarity_check.mojo`) are now listed in
`tools/identity_lanes/sequence.checks` with 15 source arms 5520-5535
(`sequence/README.md` maps each to its DEVIATION / SEAMS.tsv row).
Unarmable seams, written down there: `F = Z P Z'` and the Jones inverse's
`fma(sign, prod, x)` (no separating spelling).

- NVIDIA A40, `--pass 2` over arima, arima-011, arima-seasonal-c, arima-exog,
  arima-exog-seasonal, holtwinters, holtwinters-multiplicative, kpss,
  sequence-autoarima: all 32 arms of sequence.checks PASS / FAIL / PASS;
  all 9 lanes CLEAN AGREE. DONE, never re-run.
- E2E steward sabotages: `e2e_arima_host_param_bit.patch` (the five arima
  lanes), `e2e_tsa_host_bits.patch` (holtwinters x2, kpss).

### Apple build fix (m2pro FAIL on d5a849bb5, both requests: the Metal build)

`seq_kernel` had 33 arguments (Metal binds each to a buffer slot, 31 max):
the 12 ints and 8 floats now travel packed two per Int64 (bit-exact, ints
checked to fit Int32). Theta, ETS and GARCH then segfaulted Apple's air-lld
(`LazyLinker::LinkDefinition`): their helpers (and nm.mojo's) are now
`@always_inline`. Bisected on the M2 Pro (`~/seqdbg`, compile only): all 55
operations and seams_check build for Metal. A40 re-check after the fix
(e71c8f66a): all 32 arms bite, all 22 sequence lanes AGREE. DONE.
One batched steward request per hour covers the family; its end-to-end
sabotage is `sequence/checks/sabotage/e2e_family_host_bits.patch` (the three
e2e patches in one).

Batched steward request (all 30 family lanes, e2e_family_host_bits.patch) at
3607aa10e: 1790537359124-sequence-3607aa10ee on m2pro, m3ultra, m4pro-a, do-amd.

### Steward verdict on 3607aa10e (read 2026-09-27 evening)

1790537359124-sequence-3607aa10ee: do-amd PASS; m2pro, m3ultra, m4pro-a FAIL
on ONE lane, `clean: sequence-adafactor DISAGREE` (every other family lane
AGREE on Metal, every arm bites). The Metal cells: `plain_state` and
`moved_state` (row_var / col_var / variance) EQUAL the CPU column; `plain`
and `moved` (the params) differ in all nine fixtures, and Metal's params are
the SAME for `base` and `dupes`, which share rows 0..40 (the initial params)
and differ only in g1 (columns 14, 15). So on Metal the matrix update does
not reach the params (U = 0, or the one-thread scalars sc[1..3] are lost);
the vector arm's grad is column 1, equal in both fixtures, so it says nothing.
Logs pulled to ~/mojolearn-evidence/sequence/adafactor_metal/{cpu,gpu}.json.

Probe: `sequence/checks/af_probe.mojo` (99a1467ec) runs one step, matrix
32x8 and vector 8, DeviceExec vs HostExec, the scalars / row_var / U / params
after every launch (synced) and once more with a single sync at the end
(an ordering bug would hide behind the syncs). Host vs host is clean on the
laptop CPU (mac_slot, 1 core). NOTE for probes: `FP(unsafe_from_address=...)`
carries no origin, so a List whose last use is `_fp(l)` is freed BEFORE the
copy that reads it (ASAP destruction); the probe keeps `g` alive with `_ = g^`.
Queued on m2pro as speed request 1790553926253-speed-sequence-99a1467ec8
(`python3 tools/apple_steward.py status | grep speed-sequence`; stdout in the
verdict's speed.stdout on the Mac).

### Session 2026-09-28 (owed gates)

Pod: dev_pod `sequence` back up (RunPod H100 80GB HBM3, f0ouvpsfix31kz).

**Adafactor on Metal: FIXED at the root** (commit "alpha's max(eps2, rms) spelled max()").
The stage probe (1790553926253 m2pro, 1790557727991 m4pro-a) named OP_AF_ALPHA: sc[1] = 0 on
Metal, every other stage equal. Then on m3ultra-b (all logs in
~/mojolearn-evidence/sequence/adafactor_metal/): `af_alpha_probe.mojo` showed the store NEVER
lands (a 7-filled slot stays 7); `af_order_probe.mojo` showed it is not launch order (RMEAN
always lands, ALPHA never, in any order); throwaway prefix ops of the body showed the kernel
runs up to `rms` and dies (no store lands, not even an entry marker) as soon as a float
compare-and-select takes `rms` (portable_sqrtf over the _sumsq loop): `rms if rms > f0 else f0`,
`f0 >= rms ? f0 : rms` and `sqrt(sumsq) > f0 ? ..` all die; `max(rms, f0)`, a negated compare
and an integer compare run. An Apple Metal compiler fault on that pattern. Fix: `max(rms, eps2)`
(exact; torch's max(eps2, rms) incl. NaN -> eps2). Verified on the M3 Ultra: every stage of
af_probe.mojo (matrix and vector, staged and unsynced) == CPU, every launch order == CPU.
(The custom-kernel harness `af_kb/` lost even a constant store on Metal for a separate reason,
an out-of-line Args-returning helper, and crashed Apple's compiler on DENOM; not used for the
conclusion.)

**Seasonal ETS: DONE.** statsforecast 2.1.1 sanity on the H100 over 40 cases (m 4, 7, 12, 24;
AAA, AAAd, ANA, MAM, MNM, MAMd, MNA, MAA; scripts ~/mojolearn-evidence/sequence/ets_init/):
initial states == `initstate` to 2.2e-5; likelihood at our parameters == Calc to 1e-4;
Nelder-Mead identical in steps and constants. Where the reference converges (m 4, most m 7)
forecasts agree to <= 1.6e-3. At m 12 and 24 the REFERENCE ALSO stops at its 1000-iteration
cap (captured from `_ets.optimize`), so both return unconverged iterates; the float64 path is
stable under 6e-8 input perturbation, the float32 path ends elsewhere (max forecast diff
4.4e-2, MNM n=200 m=24), and our -2loglik is lower than the reference's in about half the cases
(net sum of differences -4 over 40). Same nature as Theta's float32 NM rows.
Seams 5517 (seasonal update) and 5518 (decomposition moving average): host oracles
`o_ets_calc` / `o_ets_init` (from the reference, explicit shifted state vector), seams_check
blocks through OP_ETS_LIK / OP_ETS_INIT (separate 53 / 44 cells), arms
`seam_5517_ets_seasonal_update.patch`, `seam_5518_ets_decompose_ma.patch`, README rows,
IDENTITY_PATHS row 159 extended. sequence-ets gained aaa (+info, states), mamd, ana30 (Fourier
init), mnm5 cells.
Gate on the H100 (`--pass 2`, all 22 sequence lanes): every arm PASS / FAIL / PASS incl. 5517
and 5518; all 22 CLEAN AGREE; existing bits vs the merge base (same lane check at f237f1996):
21 lanes' hashes identical, sequence-ets's 72 existing part hashes identical (only new parts).

Merge gate for both (H100, after merging origin/main): sequence-adafactor + sequence-ets
`--pass 2` AGREE, every arm bites (5507 incl.); sequence-adafactor hashes == the merge base on
both columns (max is exact); test_host_surface 200 passed; tools/test_lane_select.py OK,
0 failures (sequence.py / sequence.checks are its inputs). MERGED to main.
Post-merge: ONE batched steward identity request of the 30 family lanes
(e2e_family_host_bits.patch): 1790564279351-sequence-ee26312f08 on m2pro, m3ultra-b, m4-a,
do-amd, at main ee26312f0 (queued 2026-09-28 ~02:58Z).

NEXT (a fresh session starts here): read the verdict of 1790564279351-sequence-ee26312f08;
a FAIL is a fix commit at the root. Then per the brief, GPU speed (FAST and IDENTICAL) for
every sequence algorithm on NVIDIA, AMD and Apple, largest real-world cost first, profile
(sum the stage timings) before changing code, judge at 1M+ rows (R2 data). Remaining option
parity (MoE backward, prediction intervals, NOT_IMPLEMENTED rows) waits behind speed per
the 2026-09-28 brief. Apple: m2pro / m3ultra released ~12:35Z Sep 28, m4pro-a/b, m4-a,
m3ultra-b ~21:15Z Sep 28; submit Apple speed requests early and batched.

### Session 2026-09-28 ~05Z (RunPod out of money: no NVIDIA)

RunPod's balance ran out: pod f0ouvpsfix31kz is gone (404), every RunPod pod is gone, and
`dev_pod.sh up sequence` is refused ("account balance is too low"). Per the coordinator, no
retry. This session worked on the central Hot Aisle box (tools/amd_central.sh, MI300X, its
Xeon for the CPU column) and the Apple steward.

**Step 0 coverage audit (all 30 family lanes, `~/mojolearn-evidence/sequence/family_lanes.txt`):**
every algorithm has a verifier lane with a CPU and a GPU arm (22 sequence-* lanes in
tools/identity_lanes/sequence.py; arima x5, holtwinters x2 and kpss in sequence.core), and a
source sabotage per seam in tools/identity_lanes/sequence.checks (42 arms: 5500-5518,
5520-5535, 5536-5543 new in step 0, 5507 through NAdam and Adafactor), plus the family's
e2e patch `e2e_family_host_bits.patch` for the stewards. Seam 5542's arm now unfuses both the
ARCH and GARCH terms (the ARCH term alone did not separate; Metal showed it on m3ultra-b).

**Verdict 1790564279351-sequence-ee26312f08:** m2pro, m3ultra-b and m4-a FAILED on ONE lane,
`clean: sequence-adafactor DISAGREE`, after the alpha fix (do-amd still queued then). The same
Metal compare-select fault in three more Adafactor clamps (update denominator, row mean,
update clamp), now spelled max(): commit 266d5cedc. Metal == CPU on m3ultra-b and m4-a
(speed probes 1790565936440, 1790565937835).

**IDENTICAL speed 1: `ops.mojo::sumsq_fold`.** LAMB's and Adafactor's norms and MLP's L2 term
are one GPU thread folding a sum of squares over up to 4M values, each load waited out.
The fold now LOADS 64 values (16-byte vector loads on contiguous runs) before it folds them:
the same fma chain in the same order, so the bits are the plain loop's. Stage sizes tried on AMD
(fit s, same digests): f2 LAMB 4.23; s16 LAMB 3.34 / Adafactor 3.07; s64 3.14 / 2.64; v4 3.31 / 2.66.
Kept 64.

| algo (4M params x 10 steps; MLP 1M x 28, (256,), 1 epoch) | AMD MI300X before -> after | Apple M3 Ultra before -> after | digest (unchanged) |
|---|---|---|---|
| LAMB | 11.75 -> 3.14 s | 13.58 -> 2.66 s | 16c00643bf296195 |
| Adafactor | 6.74 -> 2.64 s | 7.85 -> 1.40 s | 1c495bd78d7e2734 |
| MLPClassifier fit | 2.15 -> 1.43 s | 3.15 -> 2.03 s | 983c4e5fbd05feaf |

Full IDENTICAL "before" table, every algorithm (tools/sequence_speed.py, HIGGS 1M from R2), is in
~/mojolearn-evidence/sequence/amd/ev-sequence/before_amd_identical.json (AMD) and
~/mojolearn-evidence/sequence/speed/apple_m3ultra-b_before_308d10460.jsonl (Apple, steward
1790571530078). All 19 digests are EQUAL on AMD and Metal. The largest remaining costs:
GARCH (AMD 4.87 s, Apple 4.17 s at 10000 series x 100), LSTM / GRU / RNN (1.1-2.1 s),
VAR 0.93 / 1.18 s. AutoARIMA on AMD needed libMojolearnMath.so (built on the box).

**Gate on AMD + CPU (the NVIDIA gate is OWED):** at c6d6c220e (step 0, the Adafactor clamps, the
fold, origin/main 3fa29cd1f merged): `algos_lane_check.sh <30 lanes> --pass 2` on the central
MI300X: all 42 arms PASS / FAIL / PASS after reversal; all 30 lanes CLEAN AGREE (hip vs CPU
Xeon 8470); RESULT: PASS. Log: ~/mojolearn-evidence/sequence/amd/ev-sequence/gate1.log.

**OWED when an NVIDIA pod is back (then merge):** `dev_pod.sh up sequence`, sync, the same
`--pass 2` over the 30 lanes on NVIDIA; existing bits vs the merge base (lane check hashes);
test_host_surface; tools/test_lane_select.py (sequence.checks / sequence.core changed); merge
to main and push in one command. NOT merged until then.

Batched family identity request (30 lanes, e2e_family_host_bits.patch) at 90a7e25ad: 1790574625975-sequence-90a7e25ad6 on m2pro, m3ultra-b, m4pro-a, do-amd. The next session reads its verdict first.

## PHASE 5: CPU speed (lane sequence-cpu, branch lane/sequence-cpu, 2026-09-28)

Pod `sequence-cpu` (RunPod H100 box, Intel Xeon Platinum 8470, cgroup quota
22.1 CPUs). What changed, all bit-identical by construction and proven so:

1. **`sequence/exec.mojo::HostExec.launch` splits a launch's elements across
   `core/host_parallel.mojo::host_parallelize`** (the caller's FP environment,
   DEVIATION 5900). An element is a GPU thread (no element reads another's
   write), so which thread runs it moves no bit. Task count:
   `core/host_predict_threads.mojo` (MOJOLEARN_CPU_THREADS, else one per
   physical core), cut so a task keeps `HOST_LAUNCH_GRAIN` (2^15) units of
   `_element_weight` work; small recurrent launches stay on the caller.
   Covers all 22 x_sequence lanes' host paths.
2. **`sequence/host_gemm.mojo`**: OP_GEMM on the host computes op_gemm's cells
   a native vector of columns x 4 rows at a time (one IEEE fma per lane, the
   bitwise flush `_ftz_v`, k ascending; B with n stride != 1 is copied into a
   flushed [K, N] panel first), rows split across threads. Seam **5544** (5540 on lane/sequence-cpu; renumbered on lane/merged, 5540 is the LR schedulers' seam)
   (`seams_check.mojo`, oracle alternative `o_gemm_split`, arm
   `seam_5544_host_gemm_unfused.patch`, IDENTITY_PATHS row 150 amended,
   sequence/README.md table). The host sabotage (k descending) is honored.
3. **`arima/host/arima_oracle.mojo::_kalman`**: the matrices/initial-state
   loop and the filter loop run series ranges on host_parallelize (pointers
   only; the Lists they reach are kept alive to the join).
4. **`holtwinters/host/hw_oracle.mojo::_oracle_estimate`** (the public
   default, `initialization_method="estimated"`): series ranges on
   host_parallelize, one scratch per task.
5. **Optimizer step in place on the host** (`Exec.bind`: HostExec returns the
   caller's pointer, DeviceExec allocates and uploads; `download` skips a
   self-copy) and `_SeqOptimizer.step` skips the Python pack/unpack for one
   C-contiguous float32 param.
6. e2e host sabotage patches regenerated against the new `exec.mojo`.

`core/host_parallel.mojo` is carried on this branch at lane/cpu's bytes
(41f60919d) until lane/cpu lands it on main. **This branch merges only after
that** (brief: never parallelize a host loop without it on main).

### Timings (seconds, fit + forecast/predict, one run each; bench board shapes)

Data: taxi-hourly from R2 (`gbm-bench/taxi/taxi_speed.npz`, 64 busiest zones x
1392 fit hours; `tsi` intermittent zones, `tsr` log-return diffs), windows of
24 for the recurrent models (87,552 windows, 1 epoch, hidden 64, batch 256),
MLP 64,000 x 32 (256, 256) 1 epoch, LayerNorm (4096, 1024) fwd+bwd, MoE
(512, 1024) E 8 F 2816 top 2 forward, optimizers 16.7M params x 10 steps.
Script: ~/mojolearn-evidence/sequence-cpu/seq_time.py. "before" = origin/main
host bindings (serial); "after" at the default thread count and at 3.

| algorithm | before | after (default) | after (T=3) |
|---|---|---|---|
| LSTM (reg) | 258.0 | 8.68 | 21.4 |
| GRU (reg) | 158.7 | 6.52 | 13.2 |
| RNN (reg) | 55.5 | 3.00 | 7.36 |
| MLPClassifier | 82.3 | 0.94 | 2.88 |
| MoE forward | 16.2 | 1.00 | 5.21 |
| GARCH(1,1) | 17.6 | 1.87 | 6.03 |
| ARIMA(1,1,1) | 13.1 | 1.23 | 4.68 |
| AutoARIMA (p,q 0..1, d 0..1) | 6.16 | 0.79 | 5.63 (before the task-count fix) |
| Prophet | 3.79 | 0.24 | 1.29 |
| STL robust | 4.69 | 0.22 | 1.53 |
| STL | 0.66 | 0.08 | 0.26 |
| OptimizedTheta | 3.35 | 0.63 | 1.56 |
| Theta | 2.34 | 0.35 | 1.11 |
| ETS (AAdN) | 1.46 | 0.26 | 0.68 |
| ExponentialSmoothing (HW, estimated) | 1.79 | 0.17 | 0.64 |
| LayerNorm fwd+bwd | 0.25 | 0.08 | 0.13 |
| RMSprop / Adagrad / Adamax / NAdam / Lion (10 steps) | 1.58 / 1.50 / 1.18 / 2.26 / 0.80 | 0.56 / 0.57 / 0.56 / 0.64 / 0.41 (before item 5) | |
| Adafactor / LAMB | 2.02 / 2.30 | 0.76 / 0.90 | |
| VAR, KPSS, Croston | 0.017 / 0.015 / 0.012 | 0.007 / 0.013 / 0.002 | |

**Bits:** every one of the 27 cases' output sha256 at MOJOLEARN_CPU_THREADS=1,
3 and the default EQUALS the serial origin/main digest.

### Session 2026-09-28 ~05Z (lane sequence-cpu, resumed)

**Step 0 coverage audit (the 30 family lanes in
~/mojolearn-evidence/sequence-cpu/lanes.txt):** every algorithm has a verifier
lane with a CPU arm (host binding) and a GPU arm (tools/identity_lanes/sequence.py
+ the arima/holtwinters/kpss lanes in tools/identity_break.py), and
`sequence/checks/sabotage/e2e_family_host_bits.patch` (a source edit of the
host downloads / ARIMA / TSA host params) makes every one of the 30 DISAGREE
(A40, pass 2 above); 35 seam arms (5500-5518, 5520-5535, 5544) in
tools/identity_lanes/sequence.checks. No gap. The cpu lane's audit table lists
`parallel_forecasting.*` as NO LANE: those are the multi-device drivers, whose
lanes are par-arima / par-holtwinters / par-forecast-* (its name map misses
them), not a sequence-family gap.

**CPU FAST (phase 3): there is no FAST tier on a CPU-only install.**
`bindings/build_host_family.sh` builds IDENTICAL only and
`_backend._cpu_only_binding` refuses `numeric_mode='fast'` by name, for every
family. CPU speed serves the one tier; the phase 5 table above is it.

**Pod:** `sequence-cpu` (2lshqhccqf35wn) was gone (404) and RunPod is out of
funds (the re-rent was refused: balance too low; nothing created). Per the
orchestrator: no pod until funded. Branch merged origin/main (d9b966c63);
`core/host_parallel.mojo` now at lane/cpu's tip bytes (a doc-comment change).

**MERGE BLOCKER:** `core/host_parallel.mojo` is NOT on origin/main (as of
3fa29cd1f), although lane/cpu's progress file says "MERGED to main (session 2)":
lane/cpu's tip (c4716ec93) is not an ancestor of origin/main. This branch
merges only after it lands.

**OWED on an NVIDIA pod (when RunPod is funded), then merge:**
1. `tools/algos_lane_check.sh <the 30 lanes> --pass 2` on the branch: every
   lane CLEAN AGREE (CPU == cuda), every arm of sequence.checks PASS / FAIL /
   PASS (5544, then numbered 5540, included).
2. Existing bits: the CUDA and CPU columns of the 30 lanes == the merge base's
   (same lane check at origin/main).
3. The lane check's CPU column at MOJOLEARN_CPU_THREADS=1, 3 and unset: equal.
   (seq_time.py digests at 1/3/default already equal origin/main's serial
   digests on all 27 cases.)
4. test_host_surface; test_lane_select only if its inputs changed.
5. Re-time AutoARIMA at T=3 (after the task-count fix) and the optimizers
   (after the in-place step) with ~/mojolearn-evidence/sequence-cpu/seq_time.py.

**AMD central box (queued 2026-09-28 ~05:05Z, partial gate while no NVIDIA pod):**
tree /root/mojolearn-sequence-cpu at d9b966c63 + worktree, host bindings
prebuilt (`/root/ev-sequence-cpu/hostbuild.log`). Job
`/root/ev-sequence-cpu/amd_gate.sh` (copy: ~/mojolearn-evidence/sequence-cpu/amd_gate.sh):
the 30 lanes `--pass 2` (CPU == hip, every arm) at default threads, then
pass 1 at MOJOLEARN_CPU_THREADS=1 and 3. Both slots were busy; the Mac-side
waiter (nohup, log ~/mojolearn-evidence/sequence-cpu/amd_gate_launch.log)
starts it when a slot frees (it gives up after 120 min: exit 75, relaunch
the same command). Results: `/root/ev-sequence-cpu/gate.status`, `p2.log`,
`t1.log`, `t3.log`; read them first next session
(`tools/amd_central.sh sh sequence-cpu 'cat /root/ev-sequence-cpu/gate.status'`).
It does not replace the owed NVIDIA steps above.
