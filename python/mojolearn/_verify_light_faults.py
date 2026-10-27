"""Test-only public-operation faults, never imported by product execution.

The faulty calls still execute the real native kernels. They deliberately
choose the wrong operation, derivative seed, update or restored parameter.
They prove the verifier consumes those numerical results; they are NOT
claimed as native-kernel source mutations. Retained kernel fault evidence
continues to qualify the shared arithmetic separately.
"""
from contextlib import contextmanager

FAULTS = {
    'forward-operation': ('tape_elementwise', 'mul'),
    'backward-seed': ('tape_gradients', 'backward'),
    'optimizer-rate': ('tape_optimizer', 'optimizer'),
    'checkpoint-parameter': ('tape_optimizer', 'load_state'),
    'program-product': ('tape_program', 'matmul'),
    'layout-result': ('tape_layout', 'transpose'),
    'convolution-result': ('tape_convolution', 'conv2d'),
    'numerical-copy': ('numerical_edges', 'copy'),
    'loss-result': ('tape_losses', 'mse'),
}


@contextmanager
def inject(name):
    if name is None:
        yield
        return
    from mojolearn import ops
    _, method = FAULTS[name]
    original = getattr(ops.Session, method)
    if name == 'forward-operation':
        def broken(self, a, b):
            return self.add(a, b)
    elif name == 'backward-seed':
        def broken(self, loss, *args, **kwargs):
            return original(self, self.scale(loss, 2), *args, **kwargs)
    elif name == 'optimizer-rate':
        def broken(self, kind='adam', lr=1e-3, **kwargs):
            return original(self, kind, lr=2 * lr, **kwargs)
    elif name == 'checkpoint-parameter':
        def broken(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            result['params'] = [self.scale(h, 2) for h in result['params']]
            return result
    elif name == 'program-product':
        def broken(self, a, b):
            # A fixed-shape constant also broadcasts run_dynamic back to the
            # capture shape. Scale preserves shape, isolating numerical bits.
            return self.scale(original(self, a, b), 2)
    else:
        def broken(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            return self.add(value, self.full(self.shape(value), 0.25))
    setattr(ops.Session, method, broken)
    try:
        yield
    finally:
        setattr(ops.Session, method, original)
