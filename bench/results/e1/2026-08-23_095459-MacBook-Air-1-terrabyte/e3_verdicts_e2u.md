# E2 verdicts, reference = APPLE @ fe00e8aae203b53d6d7bd2acb92dfc3669f97107

- NVIDIA: commit fe00e8aae203b53d6d7bd2acb92dfc3669f97107, inputs IDENTICAL (80 cells)
- AMD: commit fe00e8aae203b53d6d7bd2acb92dfc3669f97107, inputs IDENTICAL (80 cells)

| cell | NVIDIA | AMD |
|---|---|---|
| dbscan_algorithm_bad | REFUSED= | REFUSED= |
| dbscan_brute | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_brute_budget1mb | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_chain_default | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_chain_iter200 | REFUSED= | REFUSED= |
| dbscan_chain_iter5000 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_chain_iter5000_brute | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_eps0.15 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_eps1.0 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_maxiter1 | REFUSED= | REFUSED= |
| dbscan_metric_manhattan | REFUSED= | REFUSED= |
| dbscan_min2 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_min200 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_min50 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_min50_brute | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_rbc | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_rbc_budget1mb | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_sw | REFUSED= | REFUSED= |
| dbscan_uniform24 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| dbscan_uniform24_brute | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| kmeans_init_bad | REFUSED= | REFUSED= |
| kmeans_k3 | IDENTICAL (490 stages) | IDENTICAL (490 stages) |
| kmeans_k32 | IDENTICAL (889 stages) | IDENTICAL (889 stages) |
| kmeans_k32_random | IDENTICAL (833 stages) | IDENTICAL (833 stages) |
| kmeans_k8 | IDENTICAL (602 stages) | IDENTICAL (602 stages) |
| kmeans_k8_200k | IDENTICAL (602 stages) | IDENTICAL (602 stages) |
| kmeans_k8_array | IDENTICAL (581 stages) | IDENTICAL (581 stages) |
| kmeans_k8_maxiter2 | IDENTICAL (49 stages) | IDENTICAL (49 stages) |
| kmeans_k8_ninit3 | IDENTICAL (1884 stages) | IDENTICAL (1884 stages) |
| kmeans_k8_random | IDENTICAL (497 stages) | IDENTICAL (497 stages) |
| kmeans_k8_seed11 | IDENTICAL (518 stages) | IDENTICAL (518 stages) |
| kmeans_k8_sw | IDENTICAL (427 stages) | IDENTICAL (427 stages) |
| kmeans_k8_ties | IDENTICAL (567 stages) | IDENTICAL (567 stages) |
| kmeans_k8_tol1e-2 | IDENTICAL (56 stages) | IDENTICAL (56 stages) |
| kmeans_k8_wide | IDENTICAL (266 stages) | IDENTICAL (266 stages) |
| knn_algorithm_kd_tree | REFUSED= | REFUSED= |
| knn_clf_k15_3class | IDENTICAL (9 stages) | IDENTICAL (9 stages) |
| knn_clf_k5 | IDENTICAL (9 stages) | IDENTICAL (9 stages) |
| knn_clf_ties | IDENTICAL (9 stages) | IDENTICAL (9 stages) |
| knn_clf_weights_distance_refused | REFUSED= | REFUSED= |
| knn_k1 | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k10 | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k100 | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k100_ties | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k10_200k | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k10_self | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k10_ties | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k10_tile64 | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_k300 | REFUSED= | REFUSED= |
| knn_k64_ties | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_metric_cosine | REFUSED= | REFUSED= |
| knn_metric_l2 | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| knn_metric_minkowski_p1 | REFUSED= | REFUSED= |
| knn_reg_k5 | IDENTICAL (7 stages) | IDENTICAL (7 stages) |
| knn_reg_k50 | IDENTICAL (7 stages) | IDENTICAL (7 stages) |
| logreg_c1 | IDENTICAL (74 stages) | IDENTICAL (74 stages) |
| logreg_c100 | IDENTICAL (110 stages) | IDENTICAL (110 stages) |
| logreg_l1_refused | REFUSED= | REFUSED= |
| logreg_nointercept | IDENTICAL (47 stages) | IDENTICAL (47 stages) |
| ols_200k | IDENTICAL (9 stages) | IDENTICAL (9 stages) |
| ols_intercept | IDENTICAL (9 stages) | IDENTICAL (9 stages) |
| ols_narrow4 | IDENTICAL (9 stages) | IDENTICAL (9 stages) |
| ols_nointercept | IDENTICAL (9 stages) | IDENTICAL (9 stages) |
| ols_onecol_refused | REFUSED= | REFUSED= |
| ols_sw_refused | REFUSED= | REFUSED= |
| ols_wide_refused | REFUSED= | REFUSED= |
| pca_all | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| pca_c2 | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| pca_c8 | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| pca_c8_200k | IDENTICAL (6 stages) | IDENTICAL (6 stages) |
| pca_c8_wide | REFUSED= | REFUSED= |
| pca_solver_full | REFUSED= | REFUSED= |
| pca_whiten | REFUSED= | REFUSED= |
| ridge_a1 | IDENTICAL (14 stages) | IDENTICAL (14 stages) |
| ridge_a100 | IDENTICAL (14 stages) | IDENTICAL (14 stages) |
| ridge_nointercept | IDENTICAL (14 stages) | IDENTICAL (14 stages) |
| tsvd_c2 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| tsvd_c8 | IDENTICAL (3 stages) | IDENTICAL (3 stages) |
| tsvd_c8_wide | REFUSED= | REFUSED= |
| tsvd_randomized | REFUSED= | REFUSED= |

- NVIDIA: IDENTICAL 60, REFUSED= 20
- AMD: IDENTICAL 60, REFUSED= 20
