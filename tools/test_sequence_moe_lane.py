"""The tie fixture must replace MoE weights through their public setter."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_moe_tie_lane_respects_real_readonly_weight_setter():
    model_tree = ast.parse((ROOT / 'python/mojolearn/_x_sequence_moe.py').read_text())
    model_class = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoEBlock')
    scope = dict(np=np, _WEIGHTS=('router', 'gate_up_proj', 'down_proj'))
    exec(compile(ast.Module(body=[model_class], type_ignores=[]), 'actual-moe-model', 'exec'), scope)
    instances = []

    class ProbeMoE(scope['MoEBlock']):
        # Only initialization/forward are replaced: the real public weight
        # setter still enforces read-only ownership and device-cache versions.
        def __init__(self, hidden_size, intermediate_size, *, num_experts, top_k, **kwargs):
            self.D, self.E, self.k = hidden_size, num_experts, top_k
            self.router = np.arange(self.E * self.D, dtype=np.float32).reshape(self.E, self.D)
            self.initial_router = self.router
            instances.append(self)

        def __call__(self, x):
            self.router_logits_ = np.zeros((len(x), self.E), dtype=np.float32)
            self.selected_experts_ = np.zeros((len(x), self.k), dtype=np.int64)
            self.routing_weights_ = np.zeros((len(x), self.k), dtype=np.float32)
            return np.zeros_like(x)

    tree = ast.parse((ROOT / 'tools/identity_lanes/sequence.py').read_text())
    lane = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and any(isinstance(d, ast.Call) and d.args
                        and isinstance(d.args[0], ast.Constant) and d.args[0].value == 'sequence-moe'
                        for d in n.decorator_list))
    lane.decorator_list = []
    namespace = dict(np=np, _h=lambda *args: None, _fit=lambda result, *args: result)
    exec(compile(ast.Module(body=[lane], type_ignores=[]), 'actual-moe-lane', 'exec'), namespace)
    namespace[lane.name](SimpleNamespace(MoEBlock=ProbeMoE), np.zeros((256, 16), np.float32), None, None)
    tied = instances[1]
    np.testing.assert_array_equal(tied.router[3], tied.router[1])
    assert not np.array_equal(tied.initial_router[3], tied.initial_router[1])
    assert not tied.router.flags.writeable
    assert tied._wver == 2
