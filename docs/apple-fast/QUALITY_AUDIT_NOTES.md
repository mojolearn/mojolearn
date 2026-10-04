# FAST quality audit notes

Rows of `tools/af_board_quality_audit.py` (board-quality-audit-2026-10-04) that are metric artifacts or stale measurements, with the reason, so the audit can whitelist them. Each lane appends its own section.

## Classifiers (lane/apple-fast-q-clf, 2026-10-04)

### pa-clf taxi: stale row (no code change)

- Audit: FAST accuracy 0.58383 vs sklearn 0.74474 (and IDENTICAL 0.76036), from the 0.8.34 board JSON.
- The 0.8.34 wheel (release 42b06235e) fit PassiveAggressiveClassifier per sample (no `batch_size`). Main fits it in batches of 256, the batch moving by the mean of its rows' PA-I steps (lane/neural-pass132, merged after 0.8.34; python/mojolearn/_expansion_linear.py `PassiveAggressiveClassifier`, x_linear/sgd.mojo `mb_row_dot` / `mb_step`). The row measures code that is no longer on main.
- A float32 numpy model of main's step on the board's taxi block (1M fit rows, standardized, 20 epochs, C=1) gives 0.7717 / 0.7702 / 0.7665 / 0.7667 / 0.7703 over five shuffle seeds, all above sklearn's 0.74474 (`~/mojolearn-evidence/apple-fast-q-clf/sim_mb_perceptron.py`).
- Needed: one FAST re-measure of pa-clf taxi on main. Whitelist the 0.8.34 row.

### decision-tree-clf taxi: metric artifact (no code change)

- Audit: accuracy SAME (0.7563 vs 0.75656), log loss 1.20224 vs sklearn 1.14955. FAST and IDENTICAL are the same words.
- Cause: a depth-16 unpruned tree's leaves are mostly pure, so `predict_proba` is exactly 0 or 1 in every library (leaf frequencies, not a float32 effect). The board clips at 1e-15, so each test row that falls in a pure leaf of the wrong class costs 34.5 nats. The 0.053-nat gap is about 0.15% of the 100,000 test rows landing in wrong pure leaves.
- Which rows those are depends on the split candidates: ours splits on 128 quantile bins per feature (the documented mismatch in tools/bench_board_algos.py `decision-tree-clf`), sklearn on exact thresholds. The sign flips by dataset: on Istella ours is better (0.7583 vs sklearn 0.7917).
- Not a quality loss: accuracy is equal, and the log loss of 0/1 leaves measures how many errors fall in pure leaves. Changing n_bins to match this board would be tuning to the dataset.

### Fixed in code (see EXPERIMENTS.md, "Quality fixes, classifiers")

- perceptron taxi: the minibatch last iterate is a lottery. The fix averages the epoch-end iterates (`SGD_PERC_AVG`, QOLD `MOJOLEARN_SGD_PERC_QOLD`). The audit row itself is also stale: 0.8.34 fit per sample.
- gaussian-nb, bernoulli-nb, multinomial-nb, complement-nb, qda, nearest-centroid on istella: float32 probabilities saturated at 1.0. Fixed by float64 `predict_proba` (QOLD `MOJOLEARN_PROBA64_QOLD`). The two text rows (multinomial-nb 0.559529 vs 0.557319, complement-nb 0.559491 vs 0.557285) have the same cause and are covered by the same fix.
