# SPDX-License-Identifier: Apache-2.0
"""Process-local loaded-model layers. Transport is trusted local DevicePool RPC."""
_blocks = {}
_states = {}
_tensors = {}
#: THE ROUTE THIS WORKER BUILDS ITS LAYER ON, set by the `route` operation
#: the driver sends before anything else (lane/cpu-routes-gpu-only-four,
#: 2026-09-20). It is NOT read from the environment and NOT defaulted: a
#: worker that was never told refuses, because a worker that guessed `gpu` on
#: a host box would refuse deep inside a binding and a worker that guessed
#: `cpu` on a GPU box would quietly compute on the wrong route.
_route = None


def execute(operation, args):
    global _route
    from .models.causal_lm import _block_classes, _CpuPrimitives, _GpuPrimitives
    if operation == 'route':
        if args not in ('cpu', 'gpu'):
            raise ValueError('loaded-model route must be cpu or gpu')
        if _route is not None and _route != args:
            raise ValueError('loaded-model worker already owns a different route')
        _route = args
        return True
    if _route is None:
        raise RuntimeError('loaded-model worker was not told its route')
    if operation == 'block':
        index, kind, weights, kwargs = args
        if index in _blocks:
            raise ValueError('layer already initialized')
        block = _block_classes(_route)[kind](weights, **kwargs)
        _blocks[index] = block
        return block.weight_format
    if operation == 'tensors':
        _tensors.update(args)
        return True
    if operation == 'allocate':
        import uuid
        index, batch, capacity, kind = args
        block = _blocks[index]
        state = block.allocate_state(batch, capacity) if kind == 'transformer' else block.allocate_state(batch)
        token = uuid.uuid4().hex
        _states[token] = (index, state)
        return token
    if operation == 'state':
        return _states[args][1]
    if operation == 'release':
        _states.pop(args, None)
        return True
    if operation == 'run':
        index, method, x, token = args
        block = _blocks[index]
        state = None
        if token is not None:
            owner, state = _states[token]
            if owner != index:
                raise ValueError('state belongs to another layer')
        return getattr(block, method)(x) if state is None else getattr(block, method)(x, state)
    if operation == 'parameters':
        block = _blocks[args]
        return dict(zip(block._W_NAMES, block._w))
    prims = _GpuPrimitives() if _route == 'gpu' else _CpuPrimitives()
    if operation == 'embedding':
        return prims.embedding(_tensors['embed'], args)
    if operation == 'norm':
        x, eps = args
        return prims.rms_norm(x, _tensors['norm'], eps)
    if operation == 'head':
        return prims.linear(args, _tensors['head'])
    raise ValueError('unknown loaded-model operation: ' + operation)
