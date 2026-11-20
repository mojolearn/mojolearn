"""The tie fixture must replace MoE weights through their public setter."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_moe_tie_lane_respects_real_readonly_weight_setter():
    # MoEBlock's attribute writes, routing outputs and release are native
    # since b30df05ea8 (`seqmoe_setattr`, `seqmoe_routing`, ...): the
    # read-only flag and the device-cache version now live in Mojo and need
    # built bindings. This source test checks the lane's side of the
    # contract on the real class body: the initial weights are read-only, so
    # the tie must copy the router and assign it through the public setter.
    model_tree = ast.parse((ROOT / 'python/mojolearn/_x_sequence_moe.py').read_text())
    model_class = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoEBlock')
    assignments = []

    def native(operation, model, args, mode, context):
        if operation == 'seqmoe_setattr':
            name, value = args
            assignments.append((model, name, value))
            object.__setattr__(model, name, value)
            return None
        if operation == 'seqmoe_routing':
            return model._routing[args[0]]
        if operation == 'seqmoe_selected':
            return model._routing[1]
        if operation == 'seqmoe_release':
            return None
        raise AssertionError('unexpected native MoE operation ' + operation)

    scope = dict(np=np, _WEIGHTS=('router', 'gate_up_proj', 'down_proj'), is_seq_tensor=lambda v: False,
                 _SEQ_NATIVE=native, _SEQ_CONTEXT={})
    exec(compile(ast.Module(body=[model_class], type_ignores=[]), 'actual-moe-model', 'exec'), scope)
    instances = []

    class ProbeMoE(scope['MoEBlock']):
        # Only initialization/forward are replaced; every attribute write
        # still goes through the real MoEBlock.__setattr__ door.
        def __init__(self, hidden_size, intermediate_size, *, num_experts, top_k, **kwargs):
            self.D, self.E, self.k = hidden_size, num_experts, top_k
            router = np.arange(self.E * self.D, dtype=np.float32).reshape(self.E, self.D)
            router.setflags(write=False)
            self.router = router
            self.initial_router = router
            instances.append(self)

        def __call__(self, x):
            # forward keeps its routing outputs as [logits, picks, weights,
            # int64 picks] behind the read-only properties (51fc53dfb)
            object.__setattr__(self, '_routing', [np.zeros((len(x), self.E), dtype=np.float32),
                                                  np.zeros((len(x), self.k), dtype=np.float32),
                                                  np.zeros((len(x), self.k), dtype=np.float32), None])
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
    replaced = [value for model, name, value in assignments if model is tied and name == 'router']
    assert len(replaced) == 2 and replaced[0] is tied.initial_router
    assert tied.router is replaced[1] and tied.router is not tied.initial_router
    np.testing.assert_array_equal(tied.router[3], tied.router[1])
    assert not np.array_equal(tied.initial_router[3], tied.initial_router[1])
    assert not tied.initial_router.flags.writeable
