# E2 verdicts, reference = APPLE @ c7a349324aaa178e3d1beab395b55b630c4ac8ab

- NVIDIA: commit c7a349324aaa178e3d1beab395b55b630c4ac8ab, inputs IDENTICAL (91 cells)
- AMD: commit c7a349324aaa178e3d1beab395b55b630c4ac8ab, inputs IDENTICAL (91 cells)

| cell | NVIDIA | AMD |
|---|---|---|
| et_clf | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_062356-MacBook-Air-1-terrabyte/et_clf.card:104: seq out of order: expected 99, got 0 (seq must be 0-based and increase by exactly 1))) | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_062356-MacBook-Air-1-terrabyte/et_clf.card:104: seq out of order: expected 99, got 0 (seq must be 0-based and increase by exactly 1))) |
| et_clf_3class | IDENTICAL (99 stages) | IDENTICAL (99 stages) |
| et_clf_allfeat | IDENTICAL (99 stages) | IDENTICAL (99 stages) |
| et_clf_bootstrap | REFUSED= | REFUSED= |
| et_clf_cpu | **IDENTICAL(no-card)** | **IDENTICAL(no-card)** |
| et_clf_deep | IDENTICAL (567 stages) | IDENTICAL (567 stages) |
| et_clf_entropy | REFUSED= | REFUSED= |
| et_clf_maxleaf | REFUSED= | REFUSED= |
| et_clf_minleaf5 | IDENTICAL (99 stages) | IDENTICAL (99 stages) |
| et_reg | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| et_reg_cpu | **IDENTICAL(no-card)** | **IDENTICAL(no-card)** |
| et_reg_halffeat | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| et_reg_mid | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| et_reg_minsplit20 | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| gbdt_crossentropy | **DIVERGENT@tree000.winners.scores** | **DIVERGENT@proba** (cards agree on 302 stages) |
| gbdt_expectile | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_huber | **DIVERGENT@tree000.depth01.hist** | IDENTICAL (302 stages) |
| gbdt_loglinquantile | **DIVERGENT@tree000.depth01.hist** | IDENTICAL (302 stages) |
| gbdt_logloss | **DIVERGENT@proba** (cards agree on 0 stages) | **DIVERGENT@proba** (cards agree on 0 stages) |
| gbdt_logloss_bayesian | **OUTPUT-ONLY@tree000.depth00.hist** | IDENTICAL (302 stages) |
| gbdt_logloss_border | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_logloss_cat | **DIVERGENT@tree000.winners.scores** | **DIVERGENT@proba** (cards agree on 362 stages) |
| gbdt_logloss_class_weights | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_logloss_exact | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_logloss_gradient | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_logloss_newton3 | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_logloss_newtoncosine | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_logloss_newtonl2 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_pointwise | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (182 stages) |
| gbdt_lq | **DIVERGENT@tree000.depth00.hist** | **DIVERGENT@tree000.depth00.hist** |
| gbdt_mae | **DIVERGENT@tree000.depth01.hist** | IDENTICAL (302 stages) |
| gbdt_mae_gradient | **DIVERGENT@tree000.depth01.hist** | IDENTICAL (302 stages) |
| gbdt_mape | **OUTPUT-ONLY@tree004.winners.scores** | IDENTICAL (302 stages) |
| gbdt_multiclass | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_multiclass_cat | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (362 stages) |
| gbdt_multiclass_pointwise | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (182 stages) |
| gbdt_poisson | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_quantile | **DIVERGENT@tree000.depth01.hist** | IDENTICAL (302 stages) |
| gbdt_quantile_a03 | **DIVERGENT@tree000.depth01.hist** | IDENTICAL (302 stages) |
| gbdt_quantile_newton | **DIVERGENT@tree000.depth01.hist** | IDENTICAL (302 stages) |
| gbdt_rmse | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_062356-MacBook-Air-1-terrabyte/gbdt_rmse.card:307: seq out of order: expected 302, got 0 (seq must be 0-based and increase by exactly 1))) | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_062356-MacBook-Air-1-terrabyte/gbdt_rmse.card:307: seq out of order: expected 302, got 0 (seq must be 0-based and increase by exactly 1))) |
| gbdt_rmse_bayesian | **DIVERGENT@tree000.depth00.hist** | IDENTICAL (302 stages) |
| gbdt_rmse_bernoulli | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_bins1 | **OUTPUT-ONLY@tree002.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_bins15 | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_bins254 | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_bins64 | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_borders_sub5k | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_cat | **DIVERGENT@tree000.winners.scores** | IDENTICAL (362 stages) |
| gbdt_rmse_cat_estperm0 | **DIVERGENT@tree000.winners.scores** | IDENTICAL (322 stages) |
| gbdt_rmse_cat_numeric_control | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_cat_onehot | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (362 stages) |
| gbdt_rmse_cat_perm1 | **DIVERGENT@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_cat_perm2 | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (322 stages) |
| gbdt_rmse_cat_pointwise | **OUTPUT-ONLY@tree002.winners.scores** | IDENTICAL (362 stages) |
| gbdt_rmse_depth10 | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (462 stages) |
| gbdt_rmse_depth2 | **OUTPUT-ONLY@tree002.winners.scores** | IDENTICAL (142 stages) |
| gbdt_rmse_gradient | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_l2 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_l2reg0 | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_nan_max | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_nan_min | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_newtoncosine | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_newtonl2 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_no_boost_from_avg | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_noboot | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_od_inctodec | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_od_iter | **OUTPUT-ONLY@tree001.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_pointwise | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (182 stages) |
| gbdt_rmse_pointwise_bayesian | **OUTPUT-ONLY@tree000.depth00.hist.OneByteFeatures** | IDENTICAL (182 stages) |
| gbdt_rmse_pointwise_rs1 | **OUTPUT-ONLY@tree000.winners.scores** | **OUTPUT-ONLY@tree002.winners.scores** |
| gbdt_rmse_poissonboot | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_rmse_rs1 | **OUTPUT-ONLY@tree001.winners.scores** | **OUTPUT-ONLY@tree004.winners.scores** |
| gbdt_rmse_sample_weight | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| gbdt_tweedie | **OUTPUT-ONLY@tree000.winners.scores** | IDENTICAL (302 stages) |
| rf_clf | IDENTICAL (578 stages) | IDENTICAL (578 stages) |
| rf_clf_3class | IDENTICAL (503 stages) | IDENTICAL (503 stages) |
| rf_clf_allfeat | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_clf_deep | IDENTICAL (2098 stages) | IDENTICAL (2098 stages) |
| rf_clf_entropy | IDENTICAL (638 stages) | IDENTICAL (638 stages) |
| rf_clf_maxleaf | IDENTICAL (388 stages) | IDENTICAL (388 stages) |
| rf_clf_nobootstrap | IDENTICAL (443 stages) | IDENTICAL (443 stages) |
| rf_clf_streams1 | IDENTICAL (578 stages) | IDENTICAL (578 stages) |
| rf_reg | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_062356-MacBook-Air-1-terrabyte/rf_reg.card:518: seq out of order: expected 513, got 0 (seq must be 0-based and increase by exactly 1))) | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_062356-MacBook-Air-1-terrabyte/rf_reg.card:518: seq out of order: expected 513, got 0 (seq must be 0-based and increase by exactly 1))) |
| rf_reg_bins32 | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_gamma | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_invgauss | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_maxsamples | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_mid | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_minleaf5 | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_poisson | IDENTICAL (513 stages) | IDENTICAL (513 stages) |

- NVIDIA: DIVERGENT 15, IDENTICAL 29, IDENTICAL(no-card) 2, OUTPUT-ONLY 42, REFUSED= 3
- AMD: DIVERGENT 4, IDENTICAL 80, IDENTICAL(no-card) 2, OUTPUT-ONLY 2, REFUSED= 3
