import unittest

import board_quality as Q


def pair(a, b):
    return Q.compare(dict(arm="ours", quality=a), dict(arm="library", quality=b))


class QualityParity(unittest.TestCase):
    def test_density_partition_proof_is_exact_and_does_not_replace_intrinsic_quality(self):
        a = dict(arm="ours", lane="dbscan", quality={"rows": 12})
        b = dict(arm="library", lane="dbscan", quality={"rows": 12,
            "ari_vs_ours": 1.0, "noise_agreement_vs_ours": 1.0})
        self.assertEqual(Q.compare(a, b)["verdict"], "PASS")
        self.assertEqual(Q.compare(b, a)["verdict"], "PASS")
        for field, value in (("ari_vs_ours", .99), ("ari_vs_ours", True),
                             ("noise_agreement_vs_ours", .99), ("rows", 11)):
            bad = dict(b, quality=dict(b["quality"], **{field: value}))
            self.assertEqual(Q.compare(a, bad)["verdict"], "INSUFFICIENT")
        one = dict(a, quality={"rows": 1})
        boolean_rows = dict(b, quality=dict(b["quality"], rows=True))
        self.assertEqual(Q.compare(one, boolean_rows)["verdict"], "INSUFFICIENT")
        self.assertEqual(Q.compare(boolean_rows, one)["verdict"], "INSUFFICIENT")
        a["quality"]["silhouette"] = .5
        b["quality"].update(silhouette=.52, ari_vs_ours=.7)
        self.assertEqual(Q.compare(a, b)["verdict"], "FAIL")
        b["quality"]["silhouette"] = .505
        self.assertEqual(Q.compare(a, b)["verdict"], "PASS")

    def test_saved_svr_scores_pass_despite_different_iterations(self):
        result = pair(dict(r2=.31825814626514914, rmse=.6808156927626244, n_iter=10360),
                      dict(r2=.31823207274760623, rmse=.6808287116862013, n_iter=10625))
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual({c["metric"] for c in result["checks"]}, {"r2", "rmse"})

    def test_iteration_counts_finiteness_and_matching_shapes_are_not_quality(self):
        for q in ({"n_iter": 42}, {"finite": True}, {"output_shape": [100, 10]}, {}):
            self.assertEqual(pair(q, q)["verdict"], "INSUFFICIENT")

    def test_all_metrics_must_pass_and_rule_is_symmetric(self):
        a, b = dict(r2=.8, rmse=1.0), dict(r2=.8, rmse=1.2)
        self.assertEqual(pair(a, b)["verdict"], "FAIL")
        self.assertEqual(pair(b, a)["verdict"], "FAIL")
        self.assertEqual(pair(dict(rmse=0), dict(rmse=0))["verdict"], "PASS")
        self.assertEqual(pair(dict(rmse=0), dict(rmse=.01))["verdict"], "FAIL")

    def test_missing_or_nonfinite_metric_cannot_pass(self):
        for value in (None, float("nan"), float("inf"), True):
            self.assertEqual(pair(dict(r2=.7), dict(r2=value))["verdict"], "INSUFFICIENT")
        self.assertEqual(pair(dict(rmse=1), dict(rmse=1, r2=.7))["verdict"], "INSUFFICIENT")

    def test_log_density_uses_absolute_nats(self):
        self.assertEqual(pair(dict(mean_log_likelihood=200.79), dict(mean_log_likelihood=200.77))["verdict"], "PASS")
        self.assertEqual(pair(dict(mean_log_likelihood=200), dict(mean_log_likelihood=201))["verdict"], "FAIL")

    def test_fast_precision_compares_to_reference_bound(self):
        a, b = dict(max_rel_err_vs_fp64=2.4e-7), dict(max_rel_err_vs_fp64=.00376)
        self.assertEqual(pair(a, b)["verdict"], "PASS")
        self.assertEqual(pair(a, dict(max_rel_err_vs_fp64=.1))["verdict"], "FAIL")
        self.assertEqual(pair(dict(rmse=1, error="scorer crashed"), dict(rmse=1))["verdict"], "FAIL")

    def test_reference_quality_limits_and_missing_optimizer_step(self):
        for metric, margin in (("rel_diff_vs_sklearn", 1e-4),
                               ("rel_fro_vs_torch_eager_fp32", .01),
                               ("optimizer_update_error_vs_torch_fp32", 1e-4)):
            self.assertEqual(pair({metric: margin}, {metric: 0})["verdict"], "PASS")
            self.assertEqual(pair({metric: margin * 2}, {metric: 0})["verdict"], "FAIL")
            self.assertEqual(pair({metric: 0}, {})["verdict"], "INSUFFICIENT")
        self.assertEqual(pair(dict(rel_fro_vs_torch_eager_fp32=.00001,
                                   optimizer_update_error_vs_torch_fp32=1),
                              dict(rel_fro_vs_torch_eager_fp32=0,
                                   optimizer_update_error_vs_torch_fp32=0))["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
