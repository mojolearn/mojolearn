# Apple FAST manager state (2026-10-04, written before a credit pause)

Restart: read this file, then run `bash ~/mojolearn-evidence/apple_watch.sh`, then `python3 ~/mq/ab_extract.py '^rab'` on M3.
Recreate the watcher cron (it is session-only): every 20 min run apple_watch.sh and act on the results.

## Done and on main
- Recovery branch landed (7aeccb4d7): all lost candidates ported and fixed behind flags; FAST quality rule; quality fixes (ARD, tree bins,
  LU/SVD/TSVD, knn refine, IVF coarse init, perceptron averaging, float64 predict_proba for NB/QDA/nearest-centroid).
- New defaults (all M3 A/B, full board size, 1 run/arm): VSEARCH bundle, RSVD_FAST_DIRECT_IN, RESAMPLE_FAST_PERM_SELECT, SHAP_PERM_CACHE,
  MCD_SKIP_PINVH, TRAIN_OPT_FAST_PIPE_DOWN, OPT_FAST_STREAM+OPT_RAW_UP, LN_FAST_NOFILL, X_PREP_FAST_TE_GLOBAL/TE_ENC, PREP2_FAST_EIGH_BLOCK,
  HUBER_DEVICE_LBFGS, FA_FAST_QRR.
- FAST board applied (90176bccd): 354/377 faster, geomean 0.222, quality-gated.

## Waiting on M3 results (queue tags), then apply verdicts
- rab3-*: dbscan denseball, cc_fast, mcd ordered/G1/deflate, eigh panel df, legacy-narrow A/Bs (A = new general rule, B = old window).
  Already in: OMP_BLOCK accept (-24%, same digest); QR_FAST_DEV, SVD_FAST_CHOLQR, DECOMP_FAST_GEMM_TILED drop (slower).
- rab4-*: trees (sym umbrellas, ORD_ALL, ET_BINNED legacy).
- rab5-*: quality fixes, A = _QOLD (old), B = fix. Accept a fix when quality improves; note the speed cost.
- rab6-*: FA quality fix (A = FAST main, B = FA_ALL with the double-float Gram). Accept if faster and log-likelihood equals main's 99.49.
- rab7-*: deep-queue singles (sym per-define, optimizer variants).
- rab1d-*: tree-shap SHAP_TREE_TAB and the CV_FAST_* (harness: rf/metrics prebuild).
- Opponent job opp9c-* stays last.

## Decisions for Andrew
- gap-cls2 GRP_NOSCAN (gaussian-rp taxi 2.14 -> 1.14 ms) / GRP_LAZY: faster only because fit skips sklearn's NaN check / defers components_.
- Perceptron averaging (new FAST default) departs from sklearn's last-iterate Perceptron; MOJOLEARN_SGD_PERC_QOLD restores it.
- LLE_FAST_DEV_LU: 42% faster, but trustworthiness 0.866 -> 0.841 vs FAST main (still above the opponent). HOLD under the rule.

## Owed
- Multi-seed checks: bayesian-gmm taxi, als taxi-zones, adaboost-reg taxi (TE_ADA_SESSION_OFF probe), mb-dict-learning seed spread.
- KMEANS FA GRAM_ONCE-alone path still float32 (not covered by the FA fix).

## Update 2026-10-05 (main 9d3fb4278)
- Landed since: verdicts batch 4 (CC_FAST, SHAP_TREE_TAB, CV_FAST_SLICE/TRUST, MCD_DEFLATE, SYM_CTR_ALL, SYM_EST_ALL, PL_GROUP_NARROW,
  FA_ITER_DEVICE + double-float Gram; quality fixes LU, PROBA64, DT bins (adaboost-reg r2 -0.61 -> 0.60), ARD, TSVD; reverts of
  perceptron averaging, IVF coarse init, KNN refine, SVD qfix); MOE_FAST_MMA default (moe -25%); LSTM scan candidates (still broken, off).
- Board: 354/377 faster, geomean 0.220.
- Not on main yet, pushed to GitHub: lane/apple-fast-general-speed (LDA tiers, SVM ws rule, ARD one-pass: these CHANGE DEFAULTS, so merge after
  rab11), lane/apple-fast-s-small (RBF_PIPE, off), lane/apple-fast-s-ts (6 time-series candidates, off), compiling in M2 bq 98-merge.
- Running speed lanes: apple-fast-s-linalg (incl. TSVD speed), apple-fast-s-shap (incl. LLE quality).
- M3 queue: rab10 rest, rab11 (LDA/SVC/ARD fixes), rab12 (rbf pipe), rab13 (time series); opponent job last.

## Update 2026-10-05 late (main 2aa413135+)
- Landed: verdicts-5 (adafactor resident/nofill, rbf pipe, LDA tiers, time-series candidates off), main Apple build fixes (out->dst, ptr casts,
  sf64 import, ivf stray docstring), trees Metal fix (inlined copy constructors for ET records). Board 356/377 faster, geomean 0.220.
- PENDING:
  1. verdicts-6 (subagent writing on lane/apple-fast-verdicts-6; brief briefs-2026-10-04/apple-fast-verdicts-6.md): LU_FAST_RESIDENT, CHOL_FAST_POOLIO,
     MBK_FAST_DEVSCAN, RSVD_FAST_DEVSCAN+ORTH_WS, TSVD_FAST_POOL/COLVAR, ARIMA_FAST_SEARCH_REUSE, SCHED_FAST_TABLE, SEQ_FAST_VAR_COOP, HUBER_FAST_BLOCK512,
     SVGP_FAST_BSPLIT, PCA_FAST_COLMEAN -> M2 compile -> main -> board rows.
  2. lane/apple-fast-no-narrow-2 @ 1af808461: M2 compile (bq 004-nn2) -> land -> requeue rab17 A/Bs (general rule vs LEGACY_NARROW_*).
  3. M3 requeues: rab21 (CTR re-measure on current main, bgmm, nb-cat atomic, FA livebuf, dbscan CC batch) and ripple checks (rab20).
  4. Shape sweeps: SGD_PS_SIMD and QN_ALL general rules confirmed; KDE/SVC sweeps blocked (classical_two_datasets.py rejects s-* datasets: fix the driver).
  5. Decisions for Andrew: gap-cls2 NOSCAN/LAZY; optimizer/layernorm lanes need device-resident API or protocol change; LLE DEV_LU (istella -2% trust).
  6. Open quality gaps vs opponent: gbdt-categorical AUC, pairlogit MAP (quality lane), svgp r2 negative on both arms.
- Tools: ab_extract.py now matches rab* and ssl-* tags. Sweep: tools/afc_shape_sweep.py (strip 'lq add apple' prefix for the M3 queue).
