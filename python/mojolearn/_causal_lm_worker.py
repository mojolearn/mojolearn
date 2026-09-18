# SPDX-License-Identifier: Apache-2.0
"""Process-local loaded-model layers. Transport is trusted local DevicePool RPC."""
_blocks = {}
_states = {}
_tensors = {}


def execute(operation, args):
    from .models.causal_lm import _block_classes, _GpuPrimitives
    if operation == 'block':
        index, kind, weights, kwargs = args
        if index in _blocks:
            raise ValueError('layer already initialized')
        block = _block_classes('gpu')[kind](weights, **kwargs)
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
    prims = _GpuPrimitives()
    if operation == 'embedding':
        return prims.embedding(_tensors['embed'], args)
    if operation == 'norm':
        x, eps = args
        return prims.rms_norm(x, _tensors['norm'], eps)
    if operation == 'head':
        return prims.linear(args, _tensors['head'])
    raise ValueError('unknown loaded-model operation: ' + operation)
