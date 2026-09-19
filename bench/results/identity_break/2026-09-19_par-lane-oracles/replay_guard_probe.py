"""Watch _byte_lm_replay FAIL, with no binding: two stub trainers whose
per-step losses / state / gradients are made to disagree one at a time."""
import sys, numpy as np
sys.path.insert(0, "tools")
sys.argv = ["x"]
import identity_break as ib

N = 8
def state(bias=0.0):
    return {k: np.arange(N, dtype=np.float32) + bias for k in ("parameters", "m", "v", "flags")}

class Stub:
    def __init__(self, loss=0.5, bias=0.0, grad=1.0):
        self.loss, self.bias, self.grad = loss, bias, grad
    def train_step(self, shards):
        return {"losses": (self.loss, self.loss + 1.0)}
    def state_dict(self):
        return state(self.bias)
    def export_gradients(self):
        return np.full(N, self.grad, dtype=np.float32)

ids = np.zeros((16, 4), dtype=np.int32)

def run(a, b):
    try:
        ib._byte_lm_replay(a, b, ids)
        return "PASSED"
    except ValueError as e:
        return "RAISED: " + str(e)

print("agreeing pair                :", run(Stub(), Stub()))
print("losses perturbed (+1 ulp)    :", run(Stub(loss=np.float32(0.5) + np.float32(1e-7)), Stub()))
print("state perturbed              :", run(Stub(bias=1.0), Stub()))
print("gradient perturbed           :", run(Stub(grad=1.5), Stub()))
