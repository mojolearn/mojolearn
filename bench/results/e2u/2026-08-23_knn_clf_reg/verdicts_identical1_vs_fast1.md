# E2 verdicts, reference = ID1 @ ade0dc40c4866b35d6d6b7bb91a7d0155d21ccc5

- FAST1: commit ade0dc40c4866b35d6d6b7bb91a7d0155d21ccc5, inputs IDENTICAL (20 cells)

| cell | FAST1 |
|---|---|
| knn_algorithm_kd_tree | REFUSED= |
| knn_clf_k15_3class | IDENTICAL (9 stages) |
| knn_clf_k5 | IDENTICAL (9 stages) |
| knn_clf_ties | **DIVERGENT@knn.out_dist** |
| knn_clf_weights_distance_refused | REFUSED= |
| knn_k1 | IDENTICAL (6 stages) |
| knn_k10 | IDENTICAL (6 stages) |
| knn_k100 | **OUTPUT-ONLY@knn.out_dist** |
| knn_k100_ties | **DIVERGENT@knn.out_dist** |
| knn_k10_200k | IDENTICAL (6 stages) |
| knn_k10_self | **DIVERGENT@knn.out_dist** |
| knn_k10_ties | **DIVERGENT@knn.out_dist** |
| knn_k10_tile64 | IDENTICAL (6 stages) |
| knn_k300 | **REFUSED!=** (ref=Exception: select_radix (IDENTICAL): k > 256 is re | other=None) |
| knn_k64_ties | **DIVERGENT@knn.out_dist** |
| knn_metric_cosine | REFUSED= |
| knn_metric_l2 | IDENTICAL (6 stages) |
| knn_metric_minkowski_p1 | REFUSED= |
| knn_reg_k5 | IDENTICAL (7 stages) |
| knn_reg_k50 | **OUTPUT-ONLY@knn.out_idx** |

- FAST1: DIVERGENT 5, IDENTICAL 8, OUTPUT-ONLY 2, REFUSED!= 1, REFUSED= 4
