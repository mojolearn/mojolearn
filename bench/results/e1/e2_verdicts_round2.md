# E2 verdicts, reference = APPLE @ 53d56ef1bf2f96c2cdf58c99b837e57f84284583

- NVIDIA: commit 53d56ef1bf2f96c2cdf58c99b837e57f84284583, inputs IDENTICAL (99 cells)
- AMD: commit 53d56ef1bf2f96c2cdf58c99b837e57f84284583, inputs IDENTICAL (99 cells)

| cell | NVIDIA | AMD |
|---|---|---|
| et_clf | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/et_clf.card:104: seq out of order: expected 99, got 0 (seq must be 0-based and increase by exactly 1))) | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/et_clf.card:104: seq out of order: expected 99, got 0 (seq must be 0-based and increase by exactly 1))) |
| et_clf_200k | IDENTICAL (99 stages) | IDENTICAL (99 stages) |
| et_clf_3class | IDENTICAL (99 stages) | IDENTICAL (99 stages) |
| et_clf_allfeat | IDENTICAL (99 stages) | IDENTICAL (99 stages) |
| et_clf_bootstrap | REFUSED= | REFUSED= |
| et_clf_cpu | **IDENTICAL(no-card)** | **IDENTICAL(no-card)** |
| et_clf_deep | IDENTICAL (567 stages) | IDENTICAL (567 stages) |
| et_clf_entropy | REFUSED= | REFUSED= |
| et_clf_maxleaf | REFUSED= | REFUSED= |
| et_clf_minleaf5 | IDENTICAL (99 stages) | IDENTICAL (99 stages) |
| et_reg | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| et_reg_200k | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| et_reg_cpu | **IDENTICAL(no-card)** | **IDENTICAL(no-card)** |
| et_reg_halffeat | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| et_reg_mid | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| et_reg_minsplit20 | IDENTICAL (92 stages) | IDENTICAL (92 stages) |
| gbdt_crossentropy | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_expectile | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_huber | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_huber_d1_saturated | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_loglinquantile | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/gbdt_logloss.card:307: seq out of order: expected 302, got 0 (seq must be 0-based and increase by exactly 1))) | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/gbdt_logloss.card:307: seq out of order: expected 302, got 0 (seq must be 0-based and increase by exactly 1))) |
| gbdt_logloss_200k | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_bayesian | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_border | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_cat | IDENTICAL (362 stages) | IDENTICAL (362 stages) |
| gbdt_logloss_class_weights | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_exact | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_gradient | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_newton3 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_newtoncosine | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_newtonl2 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_logloss_pointwise | IDENTICAL (182 stages) | IDENTICAL (182 stages) |
| gbdt_lq | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_mae | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_mae_gradient | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_mape | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_multiclass | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_multiclass_cat | IDENTICAL (362 stages) | IDENTICAL (362 stages) |
| gbdt_multiclass_pointwise | IDENTICAL (182 stages) | IDENTICAL (182 stages) |
| gbdt_poisson | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_quantile | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_quantile_a03 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_quantile_newton | REFUSED= | REFUSED= |
| gbdt_rmse | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/gbdt_rmse.card:307: seq out of order: expected 302, got 0 (seq must be 0-based and increase by exactly 1))) | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/gbdt_rmse.card:307: seq out of order: expected 302, got 0 (seq must be 0-based and increase by exactly 1))) |
| gbdt_rmse_200k | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_bayesian | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_bernoulli | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_bins1 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_bins15 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_bins254 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_bins64 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_borders_sub5k | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_cat | IDENTICAL (362 stages) | IDENTICAL (362 stages) |
| gbdt_rmse_cat_estperm0 | IDENTICAL (322 stages) | IDENTICAL (322 stages) |
| gbdt_rmse_cat_numeric_control | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_cat_onehot | IDENTICAL (362 stages) | IDENTICAL (362 stages) |
| gbdt_rmse_cat_perm1 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_cat_perm2 | IDENTICAL (322 stages) | IDENTICAL (322 stages) |
| gbdt_rmse_cat_pointwise | IDENTICAL (362 stages) | IDENTICAL (362 stages) |
| gbdt_rmse_depth10 | IDENTICAL (462 stages) | IDENTICAL (462 stages) |
| gbdt_rmse_depth2 | IDENTICAL (142 stages) | IDENTICAL (142 stages) |
| gbdt_rmse_gradient | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_l2 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_l2reg0 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_nan_max | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_nan_min | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_newtoncosine | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_newtonl2 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_no_boost_from_avg | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_noboot | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_od_inctodec | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_od_iter | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_pointwise | IDENTICAL (182 stages) | IDENTICAL (182 stages) |
| gbdt_rmse_pointwise_200k | IDENTICAL (182 stages) | IDENTICAL (182 stages) |
| gbdt_rmse_pointwise_bayesian | IDENTICAL (182 stages) | IDENTICAL (182 stages) |
| gbdt_rmse_pointwise_rs1 | IDENTICAL (182 stages) | IDENTICAL (182 stages) |
| gbdt_rmse_poissonboot | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_rs1 | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_rmse_sample_weight | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| gbdt_tweedie | IDENTICAL (302 stages) | IDENTICAL (302 stages) |
| rf_clf | IDENTICAL (578 stages) | IDENTICAL (578 stages) |
| rf_clf_200k | IDENTICAL (353 stages) | IDENTICAL (353 stages) |
| rf_clf_3class | IDENTICAL (503 stages) | IDENTICAL (503 stages) |
| rf_clf_allfeat | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_clf_deep | IDENTICAL (2098 stages) | IDENTICAL (2098 stages) |
| rf_clf_entropy | IDENTICAL (638 stages) | IDENTICAL (638 stages) |
| rf_clf_maxleaf | IDENTICAL (388 stages) | IDENTICAL (388 stages) |
| rf_clf_nobootstrap | IDENTICAL (443 stages) | IDENTICAL (443 stages) |
| rf_clf_streams1 | IDENTICAL (578 stages) | IDENTICAL (578 stages) |
| rf_reg | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/rf_reg.card:518: seq out of order: expected 513, got 0 (seq must be 0-based and increase by exactly 1))) | IDENTICAL (0 stages (card parse error: bench/results/e1/2026-08-23_065934-MacBook-Air-1-terrabyte/rf_reg.card:518: seq out of order: expected 513, got 0 (seq must be 0-based and increase by exactly 1))) |
| rf_reg_200k | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_bins32 | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_gamma | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_invgauss | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_maxsamples | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_mid | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_minleaf5 | IDENTICAL (513 stages) | IDENTICAL (513 stages) |
| rf_reg_poisson | IDENTICAL (513 stages) | IDENTICAL (513 stages) |

- NVIDIA: IDENTICAL 93, IDENTICAL(no-card) 2, REFUSED= 4
- AMD: IDENTICAL 93, IDENTICAL(no-card) 2, REFUSED= 4
