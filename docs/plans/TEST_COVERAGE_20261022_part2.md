# Test coverage of changed public endpoints, part 2 (2026-10-22)

Release gate (Andrew, 2026-10-21): every public endpoint new or changed since 0.8.55 (`fc8d425fb`) has a test
on the real binding path and, where a failure path exists, a test that expects the refusal. This ledger covers
the 105 endpoints of `~/mojolearn-evidence/audit-20261022/test_gaps_part2.tsv` (scan of main `551dee1ce` with
`real=0` or `fault=0`), on lane `lane/tests-gaps-p2`. Method and conventions as in
`docs/plans/TEST_COVERAGE_20261021.md`.

- `before` is the scan's `real/fault` count at `551dee1ce`; `after` is the same scan
  (`tools/changed_public_coverage.py ~/mojolearn-evidence/audit-20261022/endpoints.tsv`) on this lane,
  saved as `~/mojolearn-evidence/audit-20261022/coverage_after_p2.tsv`. The scan anchors on the class name
  anywhere in a test file, so `after` counts are loose upper bounds; the action column names the test that
  actually exercises the endpoint.
- "existing real" names the test that already calls the endpoint without a fake. "indirect" marks a test the
  scan cannot see (a call through `obj(x)`, or a metric reached by name string).
- New tests compare against an independent restatement (NumPy, or the model's own `fit` + `transform` /
  `predict`) on small data, and each refusal test expects the named exception of the endpoint's own check
  (unfitted model, wrong width, invalid option, mismatched row count, float64 or float ids).
- No endpoint in this share lacks a failure path, so there is no `fault n/a` row.
- The tests were written from the source and not executed: no install of mojolearn imports on this Mac
  (the main checkout's built bindings are stale: `_mojolearn_tokenizer_host` has no `arena_rows_native_call`),
  and lanes do not build bindings.

## Result

| | endpoints |
|---|---|
| rows | 105 |
| real test added (row had `real=0`) | 46 |
| fault test added (row had `fault=0`) | 99 |
| existing real test (59 direct, 7 indirect) | 66 |
| n/a | 0 |
| rows with `real=0` or `fault=0` after | 0 |

New test files under `python/mojolearn/tests/`: `test_prep_endpoints_p2.py`, `test_estimator_score_p2.py`,
`test_metrics_selection_host_p2.py`, `test_sequence_layers_p2.py`.

## Ledger

| endpoint | before (real/fault) | after (real/fault) | action |
|---|---|---|---|
| `_expansion_prep.LabelBinarizer.fit` | 14/0 | 35/21 | existing real: test_label_binarizer_inverse_window.py::test_binary_one_and_two_columns; added fault: test_prep_endpoints_p2.py::test_label_binarizer_fit_refuses_bad_labels_and_sparse |
| `_expansion_prep.LabelBinarizer.transform` | 7/0 | 21/14 | existing real: test_label_binarizer_inverse_window.py::test_binary_one_and_two_columns; added fault: test_prep_endpoints_p2.py::test_label_binarizer_transform_refuses_unfitted |
| `_expansion_prep.LabelBinarizer.fit_transform` | 2/0 | 18/16 | existing real: test_x_prep_labels.py::test_multilabel_binarizer; added fault: test_prep_endpoints_p2.py::test_label_binarizer_fit_transform_refuses_bad_labels |
| `_expansion_prep.LabelBinarizer.inverse_transform` | 8/0 | 14/6 | existing real: test_label_binarizer_inverse_window.py::test_binary_one_and_two_columns; added fault: test_prep_endpoints_p2.py::test_label_binarizer_inverse_transform_refuses_wrong_width |
| `_expansion_prep.LabelEncoder.fit` | 3/0 | 24/21 | existing real: test_x_prep_labels.py::test_label_encoder; added fault: test_prep_endpoints_p2.py::test_label_encoder_fit_and_fit_transform_refuse_a_string |
| `_expansion_prep.LabelEncoder.transform` | 3/0 | 17/14 | existing real: test_x_prep_labels.py::test_label_encoder; added fault: test_prep_endpoints_p2.py::test_label_encoder_transform_refuses_unseen_and_unfitted |
| `_expansion_prep.LabelEncoder.fit_transform` | 1/0 | 17/16 | existing real: test_x_prep_labels.py; added real: test_prep_endpoints_p2.py::test_encoder_and_imputer_fit_transform_equal_fit_then_transform; added fault: test_prep_endpoints_p2.py::test_label_encoder_fit_and_fit_transform_refuse_a_string |
| `_expansion_prep.LabelEncoder.inverse_transform` | 2/0 | 8/6 | existing real: test_x_prep_labels.py::test_label_encoder; added real+fault: test_prep_endpoints_p2.py::test_label_encoder_inverse_transform_refuses_out_of_range_code |
| `_expansion_prep.LinearDiscriminantAnalysis.fit` | 15/0 | 38/21 | existing real: test_x_prep_discriminant.py::test_lda; added fault: test_prep_endpoints_p2.py::test_lda_fit_and_fit_transform_refuse_bad_options |
| `_expansion_prep.LinearDiscriminantAnalysis.decision_function` | 2/0 | 4/2 | existing real: test_x_prep_discriminant.py::test_lda; added fault: test_prep_endpoints_p2.py::test_lda_decision_function_and_transform_refusals |
| `_expansion_prep.LinearDiscriminantAnalysis.transform` | 6/0 | 20/14 | existing real: test_x_prep_discriminant.py::test_lda; added fault: test_prep_endpoints_p2.py::test_lda_decision_function_and_transform_refusals |
| `_expansion_prep.LinearDiscriminantAnalysis.fit_transform` | 1/0 | 17/16 | existing real: test_x_prep_parity.py; added real: test_prep_endpoints_p2.py::test_lda_score_and_fit_transform; added fault: test_prep_endpoints_p2.py::test_lda_fit_and_fit_transform_refuse_bad_options |
| `_expansion_prep.LinearDiscriminantAnalysis.score` | 0/0 | 4/3 | added real+fault: test_prep_endpoints_p2.py::test_lda_score_and_fit_transform |
| `_expansion_prep.MultiLabelBinarizer.fit` | 3/0 | 24/21 | existing real: test_x_prep_labels.py::test_multilabel_binarizer; added fault: test_prep_endpoints_p2.py::test_multilabel_binarizer_refusals |
| `_expansion_prep.MultiLabelBinarizer.transform` | 3/0 | 17/14 | existing real: test_x_prep_labels.py::test_multilabel_binarizer; added fault: test_prep_endpoints_p2.py::test_multilabel_binarizer_refusals |
| `_expansion_prep.MultiLabelBinarizer.fit_transform` | 1/0 | 17/16 | existing real: test_x_prep_labels.py::test_multilabel_binarizer; added fault: test_prep_endpoints_p2.py::test_multilabel_binarizer_refusals |
| `_expansion_prep.MultinomialNB.score` | 0/0 | 3/3 | added real+fault: test_prep_endpoints_p2.py::test_multinomial_nb_score |
| `_expansion_prep.Normalizer.transform` | 1/0 | 15/14 | existing real: test_x_prep_normalizer.py::test_normalizer; added fault: test_prep_endpoints_p2.py::test_normalizer_and_polynomial_transform_refusals |
| `_expansion_prep.OneHotEncoder.fit` | 11/0 | 32/21 | existing real: test_x_prep_encoders.py::test_onehot; added fault: test_prep_endpoints_p2.py::test_one_hot_encoder_refusals |
| `_expansion_prep.OneHotEncoder.transform` | 7/0 | 21/14 | existing real: test_x_prep_encoders.py::test_onehot; added fault: test_prep_endpoints_p2.py::test_one_hot_encoder_refusals |
| `_expansion_prep.OneHotEncoder.fit_transform` | 1/0 | 17/16 | existing real: test_x_prep_parity.py; added real: test_prep_endpoints_p2.py::test_encoder_and_imputer_fit_transform_equal_fit_then_transform; added fault: test_prep_endpoints_p2.py::test_one_hot_encoder_refusals |
| `_expansion_prep.OneHotEncoder.inverse_transform` | 4/0 | 10/6 | existing real: test_x_prep_encoders.py::test_onehot; added fault: test_prep_endpoints_p2.py::test_one_hot_encoder_refusals |
| `_expansion_prep.OrdinalEncoder.fit` | 11/0 | 32/21 | existing real: test_x_prep_encoders.py::test_ordinal; added fault: test_prep_endpoints_p2.py::test_ordinal_encoder_refusals |
| `_expansion_prep.OrdinalEncoder.transform` | 7/0 | 21/14 | existing real: test_x_prep_encoders.py::test_ordinal; added fault: test_prep_endpoints_p2.py::test_ordinal_encoder_refusals |
| `_expansion_prep.OrdinalEncoder.fit_transform` | 1/0 | 17/16 | existing real: test_x_prep_parity.py; added real: test_prep_endpoints_p2.py::test_encoder_and_imputer_fit_transform_equal_fit_then_transform; added fault: test_prep_endpoints_p2.py::test_ordinal_encoder_refusals |
| `_expansion_prep.OrdinalEncoder.inverse_transform` | 4/0 | 10/6 | existing real: test_x_prep_encoders.py::test_ordinal_missing_and_inverse; added fault: test_prep_endpoints_p2.py::test_ordinal_encoder_refusals |
| `_expansion_prep.PolynomialFeatures.transform` | 1/0 | 15/14 | existing real: test_x_prep_polynomial.py::test_poly; added fault: test_prep_endpoints_p2.py::test_normalizer_and_polynomial_transform_refusals |
| `_expansion_prep.PowerTransformer.fit` | 1/0 | 22/21 | existing real: test_x_prep_power.py::test_power; added fault: test_prep_endpoints_p2.py::test_power_transformer_fit_transform_and_refusals |
| `_expansion_prep.PowerTransformer.transform` | 1/0 | 15/14 | existing real: test_x_prep_power.py::test_power; added fault: test_prep_endpoints_p2.py::test_power_transformer_fit_transform_and_refusals |
| `_expansion_prep.PowerTransformer.fit_transform` | 0/0 | 16/16 | added real+fault: test_prep_endpoints_p2.py::test_power_transformer_fit_transform_and_refusals |
| `_expansion_prep.PowerTransformer.inverse_transform` | 1/0 | 7/6 | existing real: test_x_prep_power.py::test_power; added fault: test_prep_endpoints_p2.py::test_power_transformer_fit_transform_and_refusals |
| `_expansion_prep.QuadraticDiscriminantAnalysis.fit` | 11/0 | 32/21 | existing real: test_x_prep_discriminant.py::test_qda; added fault: test_prep_endpoints_p2.py::test_qda_refusals_and_score |
| `_expansion_prep.QuadraticDiscriminantAnalysis.decision_function` | 2/0 | 4/2 | existing real: test_x_prep_discriminant.py::test_qda; added fault: test_prep_endpoints_p2.py::test_qda_refusals_and_score |
| `_expansion_prep.QuadraticDiscriminantAnalysis.fit_transform` | 1/0 | 17/16 | existing real: test_x_prep_parity.py; added fault: test_prep_endpoints_p2.py::test_qda_has_no_fit_transform |
| `_expansion_prep.QuadraticDiscriminantAnalysis.score` | 0/0 | 3/3 | added real+fault: test_prep_endpoints_p2.py::test_qda_refusals_and_score |
| `_expansion_prep.QuantileTransformer.fit` | 1/0 | 22/21 | existing real: test_x_prep_quantile.py::test_quantile; added fault: test_prep_endpoints_p2.py::test_quantile_transformer_fit_transform_and_refusals |
| `_expansion_prep.QuantileTransformer.transform` | 1/0 | 15/14 | existing real: test_x_prep_quantile.py::test_quantile; added fault: test_prep_endpoints_p2.py::test_quantile_transformer_fit_transform_and_refusals |
| `_expansion_prep.QuantileTransformer.fit_transform` | 0/0 | 16/16 | added real+fault: test_prep_endpoints_p2.py::test_quantile_transformer_fit_transform_and_refusals |
| `_expansion_prep.QuantileTransformer.inverse_transform` | 1/0 | 7/6 | existing real: test_x_prep_quantile.py::test_quantile; added fault: test_prep_endpoints_p2.py::test_quantile_transformer_fit_transform_and_refusals |
| `_expansion_prep.RFE.fit` | 5/0 | 26/21 | existing real: test_x_prep_selection.py::test_rfe; added fault: test_prep_endpoints_p2.py::test_rfe_fit_transform_and_refusals |
| `_expansion_prep.RFE.fit_transform` | 0/0 | 16/16 | added real+fault: test_prep_endpoints_p2.py::test_rfe_fit_transform_and_refusals |
| `_expansion_prep.SelectKBest.fit` | 5/0 | 26/21 | existing real: test_x_prep_selection.py::test_scores_and_kbest; added fault: test_prep_endpoints_p2.py::test_select_k_best_fit_transform_and_refusals |
| `_expansion_prep.SelectKBest.fit_transform` | 0/0 | 16/16 | added real+fault: test_prep_endpoints_p2.py::test_select_k_best_fit_transform_and_refusals |
| `_expansion_prep.SimpleImputer.fit` | 9/0 | 30/21 | existing real: test_x_prep_imputer.py::test_strategies; added fault: test_prep_endpoints_p2.py::test_simple_imputer_refusals |
| `_expansion_prep.SimpleImputer.transform` | 5/0 | 19/14 | existing real: test_x_prep_imputer.py::test_strategies; added fault: test_prep_endpoints_p2.py::test_simple_imputer_refusals |
| `_expansion_prep.SimpleImputer.fit_transform` | 1/0 | 17/16 | existing real: test_x_prep_parity.py; added real: test_prep_endpoints_p2.py::test_encoder_and_imputer_fit_transform_equal_fit_then_transform; added fault: test_prep_endpoints_p2.py::test_simple_imputer_refusals |
| `_expansion_prep.SplineTransformer.fit` | 2/0 | 23/21 | existing real: test_x_prep_spline.py::test_spline; added fault: test_prep_endpoints_p2.py::test_spline_transformer_fit_transform_and_refusals |
| `_expansion_prep.SplineTransformer.transform` | 2/0 | 16/14 | existing real: test_x_prep_spline.py::test_spline; added fault: test_prep_endpoints_p2.py::test_spline_transformer_fit_transform_and_refusals |
| `_expansion_prep.SplineTransformer.fit_transform` | 0/0 | 16/16 | added real+fault: test_prep_endpoints_p2.py::test_spline_transformer_fit_transform_and_refusals |
| `_expansion_prep.TargetEncoder.fit` | 9/0 | 30/21 | existing real: test_x_prep_parity.py::test_target_encoder_cv_categories; added fault: test_prep_endpoints_p2.py::test_target_encoder_refusals |
| `_expansion_prep.TargetEncoder.transform` | 5/0 | 19/14 | existing real: test_x_prep_parity.py::test_target_encoder_cv_categories; added fault: test_prep_endpoints_p2.py::test_target_encoder_refusals |
| `_expansion_prep.TargetEncoder.fit_transform` | 3/0 | 19/16 | existing real: test_x_prep_parity.py::test_target_encoder_cv_categories; added fault: test_prep_endpoints_p2.py::test_target_encoder_refusals |
| `_expansion_prep.VarianceThreshold.fit` | 4/0 | 25/21 | existing real: test_x_prep_selection.py::test_variance_threshold; added fault: test_prep_endpoints_p2.py::test_variance_threshold_fit_transform_and_refusals |
| `_expansion_prep.VarianceThreshold.transform` | 1/0 | 15/14 | existing real: test_x_prep_selection.py::test_variance_threshold; added fault: test_prep_endpoints_p2.py::test_variance_threshold_fit_transform_and_refusals |
| `_expansion_prep.VarianceThreshold.fit_transform` | 0/0 | 16/16 | added real+fault: test_prep_endpoints_p2.py::test_variance_threshold_fit_transform_and_refusals |
| `_expansion_prep.f_classif` | 2/0 | 3/1 | existing real: test_x_prep_selection.py::test_scores_and_kbest; added fault: test_prep_endpoints_p2.py::test_univariate_score_functions_refuse_mismatched_rows |
| `_expansion_prep.f_regression` | 2/0 | 3/1 | existing real: test_x_prep_selection.py::test_scores_and_kbest; added fault: test_prep_endpoints_p2.py::test_univariate_score_functions_refuse_mismatched_rows |
| `_expansion_prep.mutual_info_classif` | 2/0 | 4/2 | existing real: test_x_prep_selection.py::test_mutual_info; added fault: test_prep_endpoints_p2.py::test_univariate_score_functions_refuse_mismatched_rows; added fault: test_prep_endpoints_p2.py::test_mutual_info_refuses_neighbour_count |
| `_expansion_prep.mutual_info_regression` | 2/0 | 4/2 | existing real: test_x_prep_selection.py::test_mutual_info; added fault: test_prep_endpoints_p2.py::test_univariate_score_functions_refuse_mismatched_rows; added fault: test_prep_endpoints_p2.py::test_mutual_info_refuses_neighbour_count |
| `_expansion_prep.r_regression` | 1/0 | 2/1 | existing real: test_x_prep_selection.py::test_score_edges_and_rfe_getter; added fault: test_prep_endpoints_p2.py::test_univariate_score_functions_refuse_mismatched_rows |
| `_expansion_trees.AdaBoostClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `_expansion_trees.AdaBoostRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `_expansion_trees.BaggingClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `_expansion_trees.BaggingRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `_expansion_trees.DARTClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `_expansion_trees.DARTRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `_expansion_trees.StackingRegressor.fit_transform` | 0/0 | 2/2 | added real+fault: test_estimator_score_p2.py::test_stacking_regressor_fit_transform_stacks_member_predictions; added fault: test_estimator_score_p2.py::test_stacking_regressor_fit_transform_refuses_parallel_jobs |
| `_gp_impl.GaussianProcessRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `_metrics_impl.f1_score` | 1/0 | 3/1 | existing real: test_classical_pip_smoke_sep22.py::test_sklearn_metrics_accept_an_array_of_labels; added real: test_metrics_selection_host_p2.py::test_precision_recall_f1_match_counts; added fault: test_metrics_selection_host_p2.py::test_precision_recall_f1_refuse_bad_options |
| `_metrics_impl.mean_absolute_error` | 0/0 | 2/1 | existing real (indirect): test_x_metrics_sanity.py::test_metrics_sanity (by name string against scikit-learn); added real: test_metrics_selection_host_p2.py::test_mean_absolute_error_and_rmse_match_numpy; added fault: test_metrics_selection_host_p2.py::test_mean_absolute_error_and_rmse_refusals |
| `_metrics_impl.precision_recall_curve` | 0/0 | 2/1 | existing real (indirect): test_x_metrics_sanity.py::test_metrics_sanity (by name string); added real: test_metrics_selection_host_p2.py::test_precision_recall_curve_matches_counts_at_each_threshold; added fault: test_metrics_selection_host_p2.py::test_precision_recall_curve_refusals |
| `_metrics_impl.precision_score` | 0/0 | 2/1 | existing real (indirect): test_x_metrics_sanity.py::test_metrics_sanity (by name string); added real: test_metrics_selection_host_p2.py::test_precision_recall_f1_match_counts; added fault: test_metrics_selection_host_p2.py::test_precision_recall_f1_refuse_bad_options |
| `_metrics_impl.recall_score` | 0/0 | 2/1 | added real: test_metrics_selection_host_p2.py::test_precision_recall_f1_match_counts; added fault: test_metrics_selection_host_p2.py::test_precision_recall_f1_refuse_bad_options |
| `_metrics_impl.root_mean_squared_error` | 0/0 | 2/1 | existing real (indirect): test_x_metrics_sanity.py::test_metrics_sanity (by name string); added real: test_metrics_selection_host_p2.py::test_mean_absolute_error_and_rmse_match_numpy; added fault: test_metrics_selection_host_p2.py::test_mean_absolute_error_and_rmse_refusals |
| `_solver_impl.ElasticNet.score` | 0/1 | 4/5 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `_solver_impl.Lasso.score` | 1/0 | 5/4 | existing real: test_classical_pip_smoke_sep22.py; added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `_x_sequence_mlp.MLPClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `_x_sequence_mlp.MLPRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `_x_sequence_moe.MoEBlock.forward` | 0/0 | 6/3 | added real: test_sequence_layers_p2.py::test_moe_block_forward_and_call_by_name; added fault: test_sequence_layers_p2.py::test_moe_block_forward_and_call_refuse_float64 |
| `_x_sequence_moe.MoEBlock.__call__` | 0/0 | 6/3 | existing real (indirect): test_x_sequence_moe.py::test_forward_and_ties (calls `m(x)`, invisible to the scan); added real: test_sequence_layers_p2.py::test_moe_block_forward_and_call_by_name; added fault: test_sequence_layers_p2.py::test_moe_block_forward_and_call_refuse_float64 |
| `_x_sequence_norm.LayerNorm.forward` | 0/0 | 6/3 | added real: test_sequence_layers_p2.py::test_layer_norm_module_forward_and_call_by_name; added fault: test_sequence_layers_p2.py::test_layer_norm_module_refuses_float64_and_wrong_trailing_shape |
| `_x_sequence_norm.LayerNorm.__call__` | 0/0 | 6/3 | existing real (indirect): test_x_sequence_norm.py::test_module (calls `ln(x)`); added real: test_sequence_layers_p2.py::test_layer_norm_module_forward_and_call_by_name; added fault: test_sequence_layers_p2.py::test_layer_norm_module_refuses_float64_and_wrong_trailing_shape |
| `_x_sequence_norm.layer_norm_backward` | 1/0 | 2/1 | existing real: test_x_sequence_norm.py::test_forward_backward; added fault: test_sequence_layers_p2.py::test_layer_norm_functions_refuse_bad_inputs |
| `_x_sequence_norm.layer_norm_forward` | 1/0 | 2/1 | existing real: test_x_sequence_norm.py::test_forward_backward; added fault: test_sequence_layers_p2.py::test_layer_norm_functions_refuse_bad_inputs |
| `_x_sequence_rnn.GRUClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_recurrent_classifier_score |
| `_x_sequence_rnn.GRURegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_recurrent_regressor_score |
| `_x_sequence_rnn.LSTMClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_recurrent_classifier_score |
| `_x_sequence_rnn.LSTMRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_recurrent_regressor_score |
| `_x_sequence_rnn.RNNClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_recurrent_classifier_score |
| `_x_sequence_rnn.RNNRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_recurrent_regressor_score |
| `embedding.Embedding.__call__` | 0/0 | 6/3 | added real: test_sequence_layers_p2.py::test_embedding_forward_and_call_by_name; added fault: test_sequence_layers_p2.py::test_embedding_forward_and_call_refuse_float_ids |
| `extratrees.ExtraTreesClassifier.score` | 0/1 | 4/5 | added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `extratrees.ExtraTreesRegressor.score` | 0/1 | 4/5 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `host_predict` | 2/0 | 4/1 | existing real: test_classical_numpy_free.py::test_numpy_free_methods_return_the_numpy_bits; added fault: test_metrics_selection_host_p2.py::test_host_predict_and_proba_refuse_an_unknown_format |
| `host_predict_proba` | 0/0 | 2/1 | existing real (indirect): test_trees_repeat.py (helper, not a test function); added real: test_metrics_selection_host_p2.py::test_host_predict_and_proba_match_the_fitted_forest; added fault: test_metrics_selection_host_p2.py::test_host_predict_and_proba_refuse_an_unknown_format |
| `kernel_methods.KernelRidge.score` | 0/1 | 4/5 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `linear_model.QNRegressor.score` | 3/0 | 7/4 | existing real: test_qn_objectives_surface.py::test_qn_regressor_fits_the_documented_objective; added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `model_selection.cross_val_predict` | 0/0 | 2/1 | added real: test_metrics_selection_host_p2.py::test_cross_val_predict_matches_per_fold_fits; added fault: test_metrics_selection_host_p2.py::test_cross_val_predict_refuses_parallel_jobs |
| `model_selection.permutation_test_score` | 0/0 | 2/1 | added real: test_metrics_selection_host_p2.py::test_permutation_test_score_counts_its_permutations; added fault: test_metrics_selection_host_p2.py::test_permutation_test_score_refuses_unknown_scoring |
| `neighbors.KNeighborsClassifier.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `neighbors.KNeighborsRegressor.score` | 0/0 | 4/4 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `randomforest.RandomForestClassifier.score` | 0/1 | 5/5 | added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `randomforest.RandomForestRegressor.score` | 0/1 | 4/5 | added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
| `svm.LinearSVC.score` | 3/0 | 7/4 | existing real: test_qn_objectives_surface.py::test_linear_svc_fits_the_documented_objective; added real+fault: test_estimator_score_p2.py::test_classifier_score_matches_its_predictions_and_refuses_short_y |
| `svm.LinearSVR.score` | 3/0 | 7/4 | existing real: test_qn_objectives_surface.py::test_linear_svr_fits_the_documented_objective; added real+fault: test_estimator_score_p2.py::test_regressor_score_is_r2_of_its_predictions_and_refuses_short_y |
