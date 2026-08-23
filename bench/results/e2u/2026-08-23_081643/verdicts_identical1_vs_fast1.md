# E2 verdicts, reference = ID1 @ 96dc2840b9e227638e7a9134536536e525487c22

- FAST1: commit 96dc2840b9e227638e7a9134536536e525487c22, inputs IDENTICAL (67 cells)

| cell | FAST1 |
|---|---|
| dbscan_algorithm_bad | REFUSED= |
| dbscan_brute | IDENTICAL (3 stages) |
| dbscan_brute_budget1mb | IDENTICAL (3 stages) |
| dbscan_chain_default | IDENTICAL (3 stages) |
| dbscan_chain_iter200 | **REFUSED!=** (ref=Exception: weak_cc_batched: label propagation did  | other=None) |
| dbscan_chain_iter5000 | IDENTICAL (3 stages) |
| dbscan_chain_iter5000_brute | IDENTICAL (3 stages) |
| dbscan_eps0.15 | IDENTICAL (3 stages) |
| dbscan_eps1.0 | IDENTICAL (3 stages) |
| dbscan_maxiter1 | **REFUSED!=** (ref=Exception: weak_cc_batched: label propagation did  | other=None) |
| dbscan_metric_manhattan | REFUSED= |
| dbscan_min2 | IDENTICAL (3 stages) |
| dbscan_min200 | IDENTICAL (3 stages) |
| dbscan_min50 | IDENTICAL (3 stages) |
| dbscan_min50_brute | IDENTICAL (3 stages) |
| dbscan_rbc | IDENTICAL (3 stages) |
| dbscan_rbc_budget1mb | IDENTICAL (3 stages) |
| dbscan_sw | REFUSED= |
| dbscan_uniform24 | IDENTICAL (3 stages) |
| dbscan_uniform24_brute | IDENTICAL (3 stages) |
| kmeans_init_bad | REFUSED= |
| kmeans_k3 | **OUTPUT-ONLY@restart00.init.par.restart00.iter01.shift** |
| kmeans_k32 | **OUTPUT-ONLY@restart00.init.par.restart00.final.inertia** |
| kmeans_k32_random | **DIVERGENT@restart00.iter02.shift** |
| kmeans_k8 | **DIVERGENT@restart00.init.par.restart00.iter02.shift** |
| kmeans_k8_200k | **DIVERGENT@restart00.init.par.restart00.iter02.shift** |
| kmeans_k8_array | **DIVERGENT@restart00.iter05.shift** |
| kmeans_k8_maxiter2 | **OUTPUT-ONLY@restart00.init.par.restart00.iter02.shift** |
| kmeans_k8_ninit3 | **OUTPUT-ONLY@restart00.init.par.restart00.iter02.shift** |
| kmeans_k8_random | **OUTPUT-ONLY@restart00.iter06.shift** |
| kmeans_k8_seed11 | **OUTPUT-ONLY@restart00.init.par.restart00.iter01.shift** |
| kmeans_k8_sw | **DIVERGENT@restart00.init.par.restart00.iter02.shift** |
| kmeans_k8_ties | **OUTPUT-ONLY@restart00.init.par.restart00.iter02.shift** |
| kmeans_k8_tol1e-2 | **DIVERGENT@restart00.init.par.restart00.iter02.shift** |
| kmeans_k8_wide | **DIVERGENT@fit.x_norm** |
| knn_algorithm_kd_tree | REFUSED= |
| knn_k1 | IDENTICAL (6 stages) |
| knn_k10 | IDENTICAL (6 stages) |
| knn_k100 | **OUTPUT-ONLY@knn.out_dist** |
| knn_k100_ties | **DIVERGENT@knn.out_dist** |
| knn_k10_200k | IDENTICAL (6 stages) |
| knn_k10_self | **DIVERGENT@knn.out_dist** |
| knn_k10_ties | **OUTPUT-ONLY@knn.out_dist** |
| knn_k10_tile64 | IDENTICAL (6 stages) |
| knn_k300 | **REFUSED!=** (ref=Exception: select_radix (IDENTICAL): k > 256 is re | other=None) |
| knn_k64_ties | **DIVERGENT@knn.out_dist** |
| knn_metric_cosine | REFUSED= |
| knn_metric_l2 | IDENTICAL (6 stages) |
| knn_metric_minkowski_p1 | REFUSED= |
| ols_200k | **DIVERGENT@ols.step1.covA** |
| ols_intercept | **DIVERGENT@ols.step1.covA** |
| ols_narrow4 | **DIVERGENT@ols.step1.covA** |
| ols_nointercept | **DIVERGENT@ols.step1.covA** |
| ols_onecol_refused | REFUSED= |
| ols_sw_refused | REFUSED= |
| ols_wide_refused | REFUSED= |
| pca_all | **DIVERGENT@pca.mean** |
| pca_c2 | **DIVERGENT@pca.mean** |
| pca_c8 | **DIVERGENT@pca.mean** |
| pca_c8_200k | **DIVERGENT@pca.mean** |
| pca_c8_wide | **REFUSED!=** (ref=Exception: gemm_tn: NUMERIC_IDENTICAL refuses the  | other=None) |
| pca_solver_full | REFUSED= |
| pca_whiten | REFUSED= |
| tsvd_c2 | **DIVERGENT@tsvd.jacobi.a** |
| tsvd_c8 | **DIVERGENT@tsvd.jacobi.a** |
| tsvd_c8_wide | **REFUSED!=** (ref=Exception: gemm_tn: NUMERIC_IDENTICAL refuses the  | other=None) |
| tsvd_randomized | REFUSED= |

- FAST1: DIVERGENT 20, IDENTICAL 20, OUTPUT-ONLY 9, REFUSED!= 5, REFUSED= 13
