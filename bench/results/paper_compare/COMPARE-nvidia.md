# Paper measurements: 0.8.55 against 0.8.66 on nvidia

Paper GPU: NVIDIA L40S. New GPU: NVIDIA L40S.
Ratio = new / 0.8.55 median ms; REGRESSION above 1.1, FASTER below 0.9; quality judged by mojolearn.quality-parity.v9. Opponents are the paper's stored cells (never rerun); 'vs opp' is ours over the paper's fastest paired opponent on that comparison's clock.

Flags: FASTER 13, FIXED 4, NOT-RUN 1, NOW-LOSES 17, NOW-WINS 2, PARAMS-DIFF 14, REGRESSION 42, SAME 48

| recipe | 0.8.55 ms | new ms | ratio | flags | quality 0.8.55 | quality new | q | best opponent (ms) | vs opp 0.8.55 | vs opp new | note |
|---|---:|---:|---:|---|---|---|---|---|---:|---:|---|
| classical/dbscan/istella/rows=full fit ours | 602959 | - | - | NOT-RUN | - | - | - | - | - | - | race not in the new board |
| classical/dbscan/taxi/rows=full fit ours | 1582293 | 929653 | 0.59 | FASTER | - | - | INSUFFICIENT | cuml-gpu (439533) | 3.60 | 2.12 |  |
| classical/hdbscan/istella/rows=full fit ours | 1193 | 1593 | 1.34 | REGRESSION | silhouette=-0.003856 | silhouette=-0.003856 | PASS | cuml-gpu (905) | 1.32 | 1.76 |  |
| classical/hdbscan/taxi/rows=full fit ours | 423 | 435 | 1.03 | SAME | silhouette=0.001505 | silhouette=0.001505 | PASS | cuml-gpu (365) | 1.16 | 1.19 |  |
| classical/kde/istella/rows=full fit ours | 34.47 | 34.64 | 1.01 | SAME | mean_log_likelihood=-222.3 | mean_log_likelihood=-222.3 | PASS | cuml-gpu (75.45) | 0.46 | 0.46 |  |
| classical/kde/taxi/rows=full fit ours | 5.98 | 6.06 | 1.01 | SAME | mean_log_likelihood=-14.83 | mean_log_likelihood=-14.83 | PASS | cuml-gpu (12.42) | 0.48 | 0.49 |  |
| classical/kmeans/istella/rows=full fit ours | 462 | 7551 | 16.34 | REGRESSION NOW-LOSES | inertia=6.051e+17 | inertia=6.051e+17 | PASS | cuml-gpu (1566) | 0.30 | 4.82 |  |
| classical/kmeans/istella/rows=full infer ours Xq | 38.56 | 30.90 | 0.80 | FASTER | eval_inertia=1.417e+17 | eval_inertia=1.417e+17 | PASS | cuml-gpu (266) | 0.14 | 0.12 |  |
| classical/kmeans/taxi/rows=full fit ours | 186 | 861 | 4.64 | REGRESSION NOW-LOSES | inertia=3.093e+08 | inertia=3.093e+08 | PASS | cuml-gpu (526) | 0.35 | 1.64 |  |
| classical/kmeans/taxi/rows=full infer ours Xq | 3.13 | 2.54 | 0.81 | FASTER | eval_inertia=4.824e+07 | eval_inertia=4.824e+07 | PASS | cuml-gpu (20.18) | 0.16 | 0.13 |  |
| classical/knn/istella/rows=full fit ours | 43.03 | 44.61 | 1.04 | SAME | recall_at_k=0.9762 | recall_at_k=0.9762 | PASS | cuml-gpu (333) | 0.13 | 0.13 |  |
| classical/knn/taxi/rows=full fit ours | 24.56 | 24.93 | 1.02 | SAME | recall_at_k=0.9998 | recall_at_k=0.9998 | PASS | cuml-gpu (51.80) | 0.47 | 0.48 |  |
| classical/ols/istella/rows=full fit ours | 504 | 7704 | 15.29 | REGRESSION NOW-LOSES | r2=0.3325 rmse=0.6817 | r2=0.3325 rmse=0.6817 | PASS | torch-gpu (581) | 0.74 | 13.14 |  |
| classical/ols/istella/rows=full infer ours Xq | 39.42 | 1772 | 44.96 | REGRESSION | predict_max_rel_err_own_fp64=8.451e-06 r2_eval=0.3325 rmse_eval=0.6817 | predict_max_rel_err_own_fp64=8.451e-06 r2_eval=0.3325 rmse_eval=0.6817 | PASS | torch-gpu (19.49) | 2.02 | 90.92 |  |
| classical/ols/taxi/rows=full fit ours | 16.48 | 763 | 46.30 | REGRESSION NOW-LOSES | r2=0.9088 rmse=4.696 | r2=0.9088 rmse=4.696 | PASS | torch-gpu (39.09) | 0.23 | 19.32 |  |
| classical/ols/taxi/rows=full infer ours Xq | 5.45 | 90.32 | 16.57 | REGRESSION | predict_max_rel_err_own_fp64=9.438e-08 r2_eval=0.9088 rmse_eval=4.696 | predict_max_rel_err_own_fp64=9.438e-08 r2_eval=0.9088 rmse_eval=4.696 | PASS | torch-gpu (1.89) | 2.89 | 47.82 |  |
| classical/pca/istella/rows=full fit ours | 176 | 342 | 1.94 | REGRESSION | explained_variance_ratio_sum=1 | explained_variance_ratio_sum=1 | PASS | cuml-gpu (1173) | 0.15 | 0.29 |  |
| classical/pca/istella/rows=full infer ours Xq | 51.94 | 90.92 | 1.75 | REGRESSION | transform_max_rel_err_own_fp64=3.69e-07 | transform_max_rel_err_own_fp64=3.69e-07 | PASS | cuml-gpu (271) | 0.19 | 0.34 |  |
| classical/pca/taxi/rows=full fit ours | 17.07 | 34.61 | 2.03 | REGRESSION | explained_variance_ratio_sum=1 | explained_variance_ratio_sum=1 | PASS | cuml-gpu (167) | 0.10 | 0.21 |  |
| classical/pca/taxi/rows=full infer ours Xq | 18.60 | 14.93 | 0.80 | FASTER | transform_max_rel_err_own_fp64=9.857e-08 | transform_max_rel_err_own_fp64=9.857e-08 | PASS | cuml-gpu (19.88) | 0.94 | 0.75 |  |
| classical/svc/istella/rows=full fit ours | 32.70 | 32.97 | 1.01 | SAME | accuracy=0.9222 | accuracy=0.9222 | PASS | cuml-gpu (46.23) | 0.71 | 0.71 |  |
| classical/svc/istella/rows=full infer ours Xq | 10.96 | 2.23 | 0.20 | FASTER | accuracy_eval=0.9222 | accuracy_eval=0.9222 | PASS | - | - | - |  |
| classical/svc/taxi/rows=full fit ours | 761 | 759 | 1.00 | SAME | accuracy=0.7675 | accuracy=0.7675 | PASS | cuml-gpu (383) | 1.98 | 1.98 |  |
| classical/svc/taxi/rows=full infer ours Xq | 6.07 | 5.23 | 0.86 | FASTER | accuracy_eval=0.7675 | accuracy_eval=0.7675 | PASS | - | - | - |  |
| classical2/kernel-ridge/istella/rows=full fit ours | 533 | 538 | 1.01 | SAME | r2=0.3854 rmse=0.6464 | r2=0.3854 rmse=0.6464 | PASS | cuml-gpu (341) | 1.56 | 1.58 |  |
| classical2/kernel-ridge/taxi/rows=full fit ours | 533 | 533 | 1.00 | SAME | r2=0.7265 rmse=8.33 | r2=0.7265 rmse=8.33 | PASS | cuml-gpu (337) | 1.58 | 1.58 |  |
| classical2/knn-clf/istella/rows=full fit ours | 40.76 | 236 | 5.80 | REGRESSION NOW-LOSES | accuracy=0.9263 | accuracy=0.9263 | PASS | cuml-gpu (201) | 0.20 | 1.17 |  |
| classical2/knn-clf/taxi/rows=full fit ours | 14.18 | 24.87 | 1.75 | REGRESSION | accuracy=0.7418 | accuracy=0.7418 | PASS | cuml-gpu (40.51) | 0.35 | 0.61 |  |
| classical2/knn-reg/istella/rows=full fit ours | 38.20 | 238 | 6.23 | REGRESSION NOW-LOSES | r2=0.4181 rmse=0.6254 | r2=0.4181 rmse=0.6254 | PASS | cuml-gpu (203) | 0.19 | 1.17 |  |
| classical2/knn-reg/taxi/rows=full fit ours | 11.52 | 23.52 | 2.04 | REGRESSION NOW-LOSES | r2=0.9373 rmse=3.842 | r2=0.9373 rmse=3.842 | PASS | cuml-gpu (20.50) | 0.56 | 1.15 |  |
| classical2/linearsvc/istella/rows=full fit ours | 468 | 469 | 1.00 | SAME | accuracy=0.9235 | accuracy=0.9235 | PASS | cuml-gpu (877) | 0.53 | 0.53 |  |
| classical2/linearsvc/taxi/rows=full fit ours | 68.73 | 19.80 | 0.29 | FASTER NOW-WINS | accuracy=0.7633 | accuracy=0.7633 | PASS | cuml-gpu (60.98) | 1.13 | 0.32 |  |
| classical2/linearsvr/istella/rows=full fit ours | 135 | 156 | 1.16 | REGRESSION | r2=-0.1068 rmse=0.8788 | r2=-0.1068 rmse=0.8788 | PASS | cuml-gpu (714) | 0.19 | 0.22 |  |
| classical2/linearsvr/taxi/rows=full fit ours | 35.08 | 37.86 | 1.08 | SAME | r2=0.8998 rmse=5.041 | r2=0.8998 rmse=5.041 | PASS | cuml-gpu (68.80) | 0.51 | 0.55 |  |
| classical2/svr/istella/rows=full fit ours | 63.87 | 67.66 | 1.06 | SAME | r2=0.3183 rmse=0.6808 | r2=0.3183 rmse=0.6808 | PASS | cuml-gpu (58.83) | 1.09 | 1.15 |  |
| classical2/svr/taxi/rows=full fit ours | 65.33 | 64.17 | 0.98 | SAME | r2=0.7676 rmse=7.68 | r2=0.7676 rmse=7.68 | PASS | cuml-gpu (49.14) | 1.33 | 1.31 |  |
| classical2/tsvd/istella/rows=full fit ours | 91.55 | 3561 | 38.90 | REGRESSION NOW-LOSES | explained_variance_ratio_sum=1 relative_reconstruction_error=0.0001314 | explained_variance_ratio_sum=1 relative_reconstruction_error=0.0001314 | PASS | cuml-gpu (603) | 0.15 | 5.90 |  |
| classical2/tsvd/taxi/rows=full fit ours | 5.51 | 174 | 31.58 | REGRESSION NOW-LOSES | explained_variance_ratio_sum=1 relative_reconstruction_error=0.003257 | explained_variance_ratio_sum=1 relative_reconstruction_error=0.003257 | PASS | cuml-gpu (50.31) | 0.11 | 3.46 |  |
| neural/gemm-bf16/gaussian/shape=full fit ours | 65.95 | 59.11 | 0.90 | FASTER | max_rel_err_vs_fp64=1.155e-07 | max_rel_err_vs_fp64=1.155e-07 | PASS | torch-eager-bf16 (39.71) | 1.66 | 1.49 |  |
| neural/gemm-int8/gaussian/shape=full fit ours | 43.75 | 54.17 | 1.24 | REGRESSION | max_rel_err_vs_fp64=0 | max_rel_err_vs_fp64=0 | PASS | torch-eager-int8 (37.75) | 1.16 | 1.43 |  |
| neural/gemm/gaussian/shape=full fit ours | 45.98 | 70.73 | 1.54 | REGRESSION | max_rel_err_vs_fp64=2.399e-07 | max_rel_err_vs_fp64=2.399e-07 | PASS | torch-compile-bf16 (34.03) | 1.35 | 2.08 |  |
| neural/lm-forward/bytes/shape=full fit ours | 44.89 | 60.26 | 1.34 | REGRESSION | mean_nll=9.016 | mean_nll=9.014 | PASS | torch-compile-bf16 (31.06) | 1.45 | 1.94 |  |
| neural/lm-train-step/bytes/shape=full fit ours | 35.50 | 35.61 | 1.00 | SAME | loss_last_step=8.406 | loss_last_step=8.403 | PASS | torch-compile-bf16 (10.83) | 3.28 | 3.29 |  |
| neural/mamba1-forward/gaussian/shape=full fit ours | 3.43 | 5.58 | 1.63 | REGRESSION | rel_fro_vs_torch_eager_fp32=6.377e-09 | rel_fro_vs_torch_eager_fp32=6.377e-09 | PASS | mamba-ssm-tf32 (3.20) | 1.07 | 1.75 |  |
| neural/mamba2-forward/gaussian/shape=full fit ours | 5.57 | 8.09 | 1.45 | REGRESSION | rel_fro_vs_torch_eager_fp32=1.497e-07 | rel_fro_vs_torch_eager_fp32=1.497e-07 | PASS | mamba-ssm-tf32 (2.81) | 1.98 | 2.88 |  |
| neural/mamba3-decode/gaussian/shape=full fit ours | 1815 | 5150 | 2.84 | REGRESSION | - | - | INSUFFICIENT | - | - | - |  |
| neural/mamba3-forward/gaussian/shape=full fit ours | 3.95 | 7.06 | 1.78 | REGRESSION | rel_fro_vs_torch_eager_fp32=5.551e-08 | rel_fro_vs_torch_eager_fp32=5.551e-08 | PASS | mamba-ssm-tf32 (3.07) | 1.29 | 2.30 |  |
| neural/mlp-train-step/gaussian/shape=full fit ours | 1.90 | 0.971 | 0.51 | FASTER NOW-WINS | loss_last_step=1.123 | loss_last_step=1.123 | PASS | torch-eager-tf32 (1.68) | 1.13 | 0.58 |  |
| neural/ops-session-step/gaussian/shape=full fit ours | 2.30 | 2.71 | 1.18 | REGRESSION | loss=2.798 | loss=2.798 | PASS | torch-eager-tf32 (2.10) | 1.09 | 1.29 |  |
| neural/program-run/gaussian/shape=full fit ours | 1.04 | 1.19 | 1.14 | REGRESSION | - | - | INSUFFICIENT | torch-eager-tf32 (0.580) | 1.79 | 2.05 |  |
| neural/samba-decode/bytes/shape=full fit ours | 2517 | 5257 | 2.09 | REGRESSION | mean_nll=5.64 | mean_nll=5.64 | PASS | - | - | - |  |
| neural/samba-forward/bytes/shape=full fit ours | 4.60 | 10.58 | 2.30 | REGRESSION | mean_nll=5.64 | mean_nll=5.64 | PASS | torch-graph-tf32 (1.38) | 3.34 | 7.68 |  |
| neural/samba-train-step/bytes/shape=full fit ours | 45.72 | 47.43 | 1.04 | SAME | loss_last_step=4.822 | loss_last_step=4.805 | PASS | torch-eager-tf32 (11.78) | 3.88 | 4.03 |  |
| neural/sequential-fit/synthetic/shape=full fit ours | 271 | 274 | 1.01 | SAME PARAMS-DIFF | accuracy=0.8991 auc=0.969 | accuracy=0.8991 auc=0.969 | PASS | torch-compile-fp32 (1986) | 0.14 | 0.14 | params: learning_rate |
| neural/sequential-predict/synthetic/shape=full fit ours | 4.27 | 2.80 | 0.66 | FASTER PARAMS-DIFF | accuracy=0.8991 auc=0.969 | accuracy=0.8991 auc=0.969 | PASS | torch-eager-tf32 (1.38) | 3.09 | 2.03 | params: learning_rate |
| neural/transformer-decode/gaussian/shape=full fit ours | 1105 | 1455 | 1.32 | REGRESSION | - | - | INSUFFICIENT | torch-graph-fp32 (326) | 3.38 | 4.46 |  |
| neural/transformer-forward/gaussian/shape=full fit ours | 5.57 | 7.99 | 1.44 | REGRESSION | rel_fro_vs_torch_eager_fp32=6.056e-08 | rel_fro_vs_torch_eager_fp32=6.056e-08 | PASS | torch-graph-fp32 (2.36) | 2.36 | 3.38 |  |
| trees/gbdt-depthwise/istella/rows=full fit ours | 7262 | 7781 | 1.07 | SAME | auc=0.9836 logloss=0.1501 | auc=0.9836 logloss=0.1501 | PASS | catboost-gpu (9601) | 0.76 | 0.81 |  |
| trees/gbdt-depthwise/istella/rows=full fit ours-prebinned | 8128 | 8604 | 1.06 | SAME NOW-LOSES PARAMS-DIFF | auc=0.9837 logloss=0.1494 | auc=0.9837 logloss=0.1494 | PASS | xgboost-gpu (8557) | 0.95 | 1.01 | params: input_binning |
| trees/gbdt-depthwise/istella/rows=full infer ours test | 36.25 | 36.52 | 1.01 | SAME | auc=0.9836 logloss=0.1501 | auc=0.9836 logloss=0.1501 | PASS | catboost-gpu (455) | 0.08 | 0.08 |  |
| trees/gbdt-depthwise/istella/rows=full infer ours-prebinned test | 37.79 | 37.27 | 0.99 | SAME PARAMS-DIFF | auc=0.9837 logloss=0.1494 | auc=0.9837 logloss=0.1494 | PASS | xgboost-gpu (99.24) | 0.38 | 0.38 | params: input_binning |
| trees/gbdt-depthwise/istella/rows=full infer ours large | 69.08 | 68.87 | 1.00 | SAME | - | - | INSUFFICIENT | catboost-gpu (712) | 0.10 | 0.10 |  |
| trees/gbdt-depthwise/istella/rows=full infer ours-prebinned large | 73.25 | 71.04 | 0.97 | SAME PARAMS-DIFF | - | - | INSUFFICIENT | xgboost-gpu (192) | 0.38 | 0.37 | params: input_binning |
| trees/gbdt-depthwise/taxi/rows=full fit ours | 3714 | 3711 | 1.00 | SAME | auc=0.6321 logloss=0.528 | auc=0.6321 logloss=0.528 | PASS | catboost-gpu (6205) | 0.60 | 0.60 |  |
| trees/gbdt-depthwise/taxi/rows=full fit ours-prebinned | 3807 | 3832 | 1.01 | SAME PARAMS-DIFF | auc=0.6318 logloss=0.5281 | auc=0.6318 logloss=0.5281 | PASS | xgboost-gpu (3039) | 1.25 | 1.26 | params: input_binning |
| trees/gbdt-depthwise/taxi/rows=full infer ours test | 9.36 | 9.42 | 1.01 | SAME | auc=0.6321 logloss=0.528 | auc=0.6321 logloss=0.528 | PASS | catboost-gpu (317) | 0.03 | 0.03 |  |
| trees/gbdt-depthwise/taxi/rows=full infer ours-prebinned test | 9.35 | 9.69 | 1.04 | SAME PARAMS-DIFF | auc=0.6318 logloss=0.5281 | auc=0.6318 logloss=0.5281 | PASS | xgboost-gpu (10.83) | 0.86 | 0.89 | params: input_binning |
| trees/gbdt-depthwise/taxi/rows=full infer ours large | 15.21 | 15.51 | 1.02 | SAME | - | - | INSUFFICIENT | catboost-gpu (555) | 0.03 | 0.03 |  |
| trees/gbdt-depthwise/taxi/rows=full infer ours-prebinned large | 15.24 | 15.72 | 1.03 | SAME PARAMS-DIFF | - | - | INSUFFICIENT | xgboost-gpu (18.62) | 0.82 | 0.84 | params: input_binning |
| trees/gbdt-lossguide/istella/rows=full fit ours | 896244 | 897177 | 1.00 | SAME | - | auc=0.9854 logloss=0.1279 | INSUFFICIENT | catboost-gpu (30510) | 29.38 | 29.41 |  |
| trees/gbdt-lossguide/istella/rows=full fit ours-gain0 | 891470 | 892250 | 1.00 | SAME PARAMS-DIFF | - | auc=0.9855 logloss=0.1281 | INSUFFICIENT | - | - | - | params: input_binning |
| trees/gbdt-lossguide/istella/rows=full infer ours test | - | 39.87 | - | FIXED | - | auc=0.9854 logloss=0.1279 | INSUFFICIENT | - | - | - | old: UNKNOWN(no inference lines) |
| trees/gbdt-lossguide/istella/rows=full infer ours-gain0 test | - | 39.27 | - | FIXED PARAMS-DIFF | - | auc=0.9855 logloss=0.1281 | INSUFFICIENT | - | - | - | old: UNKNOWN(no inference lines); params: input_binning |
| trees/gbdt-lossguide/istella/rows=full infer ours large | - | 74.93 | - | FIXED | - | - | INSUFFICIENT | - | - | - | old: UNKNOWN(no inference lines) |
| trees/gbdt-lossguide/istella/rows=full infer ours-gain0 large | - | 77.78 | - | FIXED PARAMS-DIFF | - | - | INSUFFICIENT | - | - | - | old: UNKNOWN(no inference lines); params: input_binning |
| trees/gbdt-lossguide/taxi/rows=full fit ours | 64300 | 64609 | 1.00 | SAME | auc=0.632 logloss=0.5279 | auc=0.632 logloss=0.5279 | PASS | catboost-gpu (20453) | 3.14 | 3.16 |  |
| trees/gbdt-lossguide/taxi/rows=full fit ours-gain0 | 62996 | 63303 | 1.00 | SAME PARAMS-DIFF | auc=0.632 logloss=0.5278 | auc=0.632 logloss=0.5278 | PASS | xgboost-gpu (11422) | 5.52 | 5.54 | params: input_binning |
| trees/gbdt-lossguide/taxi/rows=full infer ours test | 9.83 | 9.99 | 1.02 | SAME | auc=0.632 logloss=0.5279 | auc=0.632 logloss=0.5279 | PASS | catboost-gpu (269) | 0.04 | 0.04 |  |
| trees/gbdt-lossguide/taxi/rows=full infer ours-gain0 test | 9.88 | 10.21 | 1.03 | SAME PARAMS-DIFF | auc=0.632 logloss=0.5278 | auc=0.632 logloss=0.5278 | PASS | xgboost-gpu (7.91) | 1.25 | 1.29 | params: input_binning |
| trees/gbdt-lossguide/taxi/rows=full infer ours large | 15.51 | 15.80 | 1.02 | SAME | - | - | INSUFFICIENT | catboost-gpu (436) | 0.04 | 0.04 |  |
| trees/gbdt-lossguide/taxi/rows=full infer ours-gain0 large | 18.95 | 16.28 | 0.86 | FASTER PARAMS-DIFF | - | - | INSUFFICIENT | xgboost-gpu (14.34) | 1.32 | 1.14 | params: input_binning |
| trees/gbdt-multiclass/istella/rows=full fit ours | 14120 | 13687 | 0.97 | SAME | accuracy=0.9076 mlogloss=0.2584 | accuracy=0.9076 mlogloss=0.2584 | PASS | catboost-gpu (14624) | 0.97 | 0.94 |  |
| trees/gbdt-multiclass/istella/rows=full infer ours test | 42.05 | 42.43 | 1.01 | SAME | accuracy=0.9076 mlogloss=0.2584 | accuracy=0.9076 mlogloss=0.2584 | PASS | catboost-gpu (197) | 0.21 | 0.22 |  |
| trees/gbdt-multiclass/istella/rows=full infer ours large | 83.38 | 80.39 | 0.96 | SAME | - | - | INSUFFICIENT | catboost-gpu (305) | 0.27 | 0.26 |  |
| trees/gbdt-multiclass/taxi/rows=full fit ours | 6934 | 6627 | 0.96 | SAME | accuracy=0.5994 mlogloss=1.013 | accuracy=0.5994 mlogloss=1.013 | PASS | catboost-gpu (9881) | 0.70 | 0.67 |  |
| trees/gbdt-multiclass/taxi/rows=full infer ours test | 27.66 | 11.91 | 0.43 | FASTER | accuracy=0.5994 mlogloss=1.013 | accuracy=0.5994 mlogloss=1.013 | PASS | catboost-gpu (73.43) | 0.38 | 0.16 |  |
| trees/gbdt-multiclass/taxi/rows=full infer ours large | 22.20 | 22.82 | 1.03 | SAME | - | - | INSUFFICIENT | catboost-gpu (130) | 0.17 | 0.18 |  |
| trees/gbdt-rank-pairlogit/istella/rows=full fit ours | 2700 | 2123 | 0.79 | FASTER | map=0.0883 ndcg10=0.01799 ndcg5=0.01816 | map=0.0883 ndcg10=0.01799 ndcg5=0.01816 | PASS | catboost-gpu (3521) | 0.77 | 0.60 |  |
| trees/gbdt-rank-pairlogit/istella/rows=full infer ours test | 38.02 | 40.56 | 1.07 | SAME | map=0.0883 ndcg10=0.01799 ndcg5=0.01816 | map=0.0883 ndcg10=0.01799 ndcg5=0.01816 | PASS | catboost-gpu (123) | 0.31 | 0.33 |  |
| trees/gbdt-rank-pairlogit/istella/rows=full infer ours large | 52.19 | 52.25 | 1.00 | SAME | - | - | INSUFFICIENT | catboost-gpu (119) | 0.44 | 0.44 |  |
| trees/gbdt-symmetric/istella/rows=full fit ours | 9954 | 9412 | 0.95 | SAME | auc=0.9801 logloss=0.1868 | auc=0.9801 logloss=0.1868 | PASS | catboost-gpu (7448) | 1.34 | 1.26 |  |
| trees/gbdt-symmetric/istella/rows=full infer ours test | 33.07 | 33.21 | 1.00 | SAME | auc=0.9801 logloss=0.1868 | auc=0.9801 logloss=0.1868 | PASS | catboost-gpu (729) | 0.05 | 0.05 |  |
| trees/gbdt-symmetric/istella/rows=full infer ours large | 60.41 | 61.30 | 1.01 | SAME | - | - | INSUFFICIENT | catboost-gpu (1282) | 0.05 | 0.05 |  |
| trees/gbdt-symmetric/taxi/rows=full fit ours | 6151 | 5863 | 0.95 | SAME | auc=0.6302 logloss=0.5286 | auc=0.6302 logloss=0.5286 | PASS | catboost-gpu (5482) | 1.12 | 1.07 |  |
| trees/gbdt-symmetric/taxi/rows=full infer ours test | 6.06 | 6.13 | 1.01 | SAME | auc=0.6302 logloss=0.5286 | auc=0.6302 logloss=0.5286 | PASS | catboost-gpu (67.38) | 0.09 | 0.09 |  |
| trees/gbdt-symmetric/taxi/rows=full infer ours large | 11.13 | 10.99 | 0.99 | SAME | - | - | INSUFFICIENT | catboost-gpu (127) | 0.09 | 0.09 |  |
| trees/iforest/istella/rows=full fit ours | 197 | 8422 | 42.72 | REGRESSION NOW-LOSES | auc=0.8293 | auc=0.8293 | PASS | cuml-iforest-gpu (2290) | 0.09 | 3.68 |  |
| trees/iforest/istella/rows=full infer ours test | 60.37 | 1820 | 30.15 | REGRESSION NOW-LOSES | auc=0.8293 | auc=0.8293 | PASS | cuml-iforest-gpu (77.98) | 0.77 | 23.34 |  |
| trees/iforest/istella/rows=full infer ours large | 123 | 3711 | 30.09 | REGRESSION NOW-LOSES | - | - | INSUFFICIENT | cuml-iforest-gpu (157) | 0.78 | 23.57 |  |
| trees/iforest/taxi/rows=full fit ours | 149 | 1324 | 8.88 | REGRESSION | auc=0.5475 | auc=0.5475 | PASS | cuml-iforest-gpu (114) | 1.31 | 11.66 |  |
| trees/iforest/taxi/rows=full infer ours test | 3.27 | 137 | 41.80 | REGRESSION NOW-LOSES | auc=0.5475 | auc=0.5475 | PASS | cuml-iforest-gpu (8.98) | 0.36 | 15.23 |  |
| trees/iforest/taxi/rows=full infer ours large | 5.61 | 273 | 48.63 | REGRESSION NOW-LOSES | - | - | INSUFFICIENT | cuml-iforest-gpu (10.20) | 0.55 | 26.75 |  |
| trees/rf/istella/rows=full fit ours | 30769 | 37852 | 1.23 | REGRESSION | auc=0.9454 logloss=0.182 | auc=0.9454 logloss=0.182 | PASS | cuml-rf-gpu (23672) | 1.30 | 1.60 |  |
| trees/rf/istella/rows=full infer ours test | 52.31 | 1797 | 34.34 | REGRESSION | auc=0.9454 logloss=0.182 | auc=0.9454 logloss=0.182 | PASS | cuml-rf-gpu (47.80) | 1.09 | 37.59 |  |
| trees/rf/istella/rows=full infer ours large | 105 | 3603 | 34.29 | REGRESSION | - | - | INSUFFICIENT | cuml-rf-gpu (92.31) | 1.14 | 39.04 |  |
| trees/rf/taxi/rows=full fit ours | 9443 | 10451 | 1.11 | REGRESSION | auc=0.6178 logloss=0.526 | auc=0.6178 logloss=0.526 | PASS | cuml-rf-gpu (20310) | 0.46 | 0.51 |  |
| trees/rf/taxi/rows=full infer ours test | 8.13 | 134 | 16.43 | REGRESSION NOW-LOSES | auc=0.6178 logloss=0.526 | auc=0.6178 logloss=0.526 | PASS | cuml-rf-gpu (9.36) | 0.87 | 14.28 |  |
| trees/rf/taxi/rows=full infer ours large | 16.01 | 266 | 16.61 | REGRESSION NOW-LOSES | - | - | INSUFFICIENT | cuml-rf-gpu (17.47) | 0.92 | 15.23 |  |
