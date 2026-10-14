import unittest

import board_quality as Q


def pair(a, b):
    return Q.compare(dict(arm="ours", quality=a), dict(arm="library", quality=b))


class QualityParity(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
