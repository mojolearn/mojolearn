"""Source-only CPU ranking verifier wiring; never load a native binding."""

import ast
import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools/identity_break.py"
LANES = {
    "gbdt-query-rmse-pointwise": "QueryRMSE",
    "gbdt-pair-logit-pointwise": "PairLogit",
    "gbdt-yeti-rank-pointwise": "YetiRank",
}
DEVICE_LANES = {
    "optim-resident-io",
    "sequence-device-release",
    "training-device-cross-entropy",
}


def manifest():
    spec = importlib.util.spec_from_file_location(
        "ranking_test_host_surface", ROOT / "python/mojolearn/host_surface.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_only_lanes():
    """Read the registry and its fragments without importing the harness."""
    result = {}
    found_registry = False
    paths = [HARNESS, *sorted((ROOT / "tools/identity_lanes").glob("*.py"))]
    for path in paths:
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GPU_ONLY_LANES":
                    if path != HARNESS or found_registry:
                        raise AssertionError("GPU_ONLY_LANES was unexpectedly rebound")
                    result.update(ast.literal_eval(node.value))
                    found_registry = True
                elif (isinstance(target, ast.Subscript)
                      and isinstance(target.value, ast.Name)
                      and target.value.id == "GPU_ONLY_LANES"):
                    result[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
    if not found_registry:
        raise AssertionError("the source declares no GPU_ONLY_LANES registry")
    return result


def mojo_function(path, name):
    source = (ROOT / path).read_text()
    match = re.search(rf"^def {re.escape(name)}\(", source, re.M)
    if not match:
        raise AssertionError(f"missing native function {name} in {path}")
    tail = source[match.start():]
    following = re.search(r"\n(?:def |struct |@fieldwise_init)", tail)
    return tail[:following.start()] if following else tail


class RankingVerifierRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = manifest()
        cls.tree = ast.parse(HARNESS.read_text())

    def test_cpu_gate_and_default_harness_selection_keep_all_three_routes(self):
        host = self.host
        covered = set(host.covered_lanes())
        family = host.family("gbdt")
        for lane in LANES:
            self.assertIn(lane, covered)
            self.assertIn(lane, host.record_covered_lanes())
            self.assertIn(lane, family["training_lanes"])
            self.assertIn(lane, host.TRAINING_LANE_NAMES)
            self.assertNotIn(lane, gpu_only_lanes())
            self.assertIsNone(host.inapplicable_prefix_reason(lane))
        self.assertIn("gbdt/host/gbdt_oracle_pointwise_rank.mojo", family["host_modules"])

    def test_existing_gpu_references_are_not_relabelled_as_missing(self):
        host = self.host
        for lane in ("gbdt-pair-logit-pointwise", "gbdt-yeti-rank-pointwise"):
            self.assertEqual(host.PUBLIC_PENDING_LANES[lane], "unwatched")
            for device in ("cpu", "nvidia", "amd", "apple"):
                self.assertIsNone(host.pending_blocks_comparison(lane, device))
            self.assertEqual(host.lane_exposure([lane])[lane]["note"], "unwatched")
        query = "gbdt-query-rmse-pointwise"
        self.assertEqual(host.PUBLIC_PENDING_LANES[query], "no reference")
        self.assertEqual(host.lane_exposure([query])[query]["status"], host.LANE_OWED)
        self.assertIsNone(host.pending_blocks_comparison("gbdt-query-rmse"))

    def test_device_residency_exclusions_are_preserved(self):
        exclusions = gpu_only_lanes()
        self.assertEqual(set(exclusions), DEVICE_LANES)
        pending = {
            lane for lane, reason in self.host.PUBLIC_PENDING_LANES.items()
            if reason == "no cpu route"
        }
        self.assertEqual(pending, DEVICE_LANES)
        for lane in DEVICE_LANES:
            self.assertNotIn(lane, self.host.covered_lanes())

    def test_lane_fixtures_select_the_real_pointwise_searcher(self):
        registered = {lane: [] for lane in LANES}
        helper = None
        for node in self.tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "_querywise_pointwise_identity":
                helper = node
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "lane"
                        and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)):
                    name = decorator.args[0].value
                    if name in registered:
                        registered[name].append(node)
        self.assertIsNotNone(helper)
        for name, nodes in registered.items():
            self.assertEqual(len(nodes), 1, name)
            calls = [n for n in ast.walk(nodes[0]) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)
                     and n.func.id == "_querywise_pointwise_identity"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.literal_eval(calls[0].args[-1]), LANES[name])
        constructors = [n for n in ast.walk(helper) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "GradientBoosting"]
        self.assertEqual(len(constructors), 1)
        options = {k.arg: k.value for k in constructors[0].keywords}
        self.assertIs(ast.literal_eval(options["use_pointwise_searcher"]), True)
        self.assertEqual(ast.literal_eval(options["boosting_type"]), "Plain")
        self.assertEqual(ast.literal_eval(options["leaf_estimation_method"]), "Newton")
        self.assertIn("Cosine", ast.get_source_segment(HARNESS.read_text(), helper))
        self.assertIn("NewtonCosine", ast.get_source_segment(HARNESS.read_text(), helper))

    def test_each_lane_has_batch_prediction_registration(self):
        batch_lanes = set()
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_batch_decl"):
                batch_lanes.update(arg.value for arg in node.args
                                   if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
        self.assertTrue(set(LANES) <= batch_lanes)

    def test_source_route_reaches_existing_host_negative_control(self):
        # This checks source reachability only. A caught numerical sabotage
        # result still requires separately authorized compiled execution.
        self.assertIn("-D MOJOLEARN_HOST_SABOTAGE=1",
                      self.host.sabotage_build_defines("gbdt"))
        fit = mojo_function("gbdt/host/gbdt_oracle_pointwise_rank.mojo",
                            "gbdt_ranking_pointwise_host_fit")
        self.assertIn("_estimate_leaves_for_loss(", fit)
        estimator = mojo_function("gbdt/host/gbdt_oracle_losses.mojo", "_estimate_leaves_for_loss")
        self.assertIn("return _walker_estimate(", estimator)
        walker = mojo_function("gbdt/host/gbdt_oracle_losses.mojo", "_walker_estimate")
        self.assertRegex(walker, r"comptime if GBDT_ORACLE_HOST_SABOTAGE:\s+lambda_reg = lambda_reg \+ 1\.0")


if __name__ == "__main__":
    unittest.main()
