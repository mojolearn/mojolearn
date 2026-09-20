# SPDX-License-Identifier: Apache-2.0
"""Explicit layer-owner inference, using canonical CausalLM arithmetic.

Each selected CUDA/HIP device has one persistent isolated worker. Metal
supports one worker on device zero, with every layer assigned to that worker. Only its
assigned layers are constructed there. Hidden activations cross boundaries via
host memory; this is sequential model parallelism, not tensor parallelism or a
throughput claim. Standard block states remain host-backed in their owner
process. Transformer native sessions retain weights where supported, but this
API does not promise resident KV or support a layer larger than one device.

Checkpoint loading currently materializes all weights in parent RAM before
sending each layer to its owner; that transient host-memory requirement remains.
Physical two-GPU qualification is outstanding. Explicit experimental API.

THE CPU HOST ROUTE (lane/cpu-routes-gpu-only-four, 2026-09-20). On a CPU-only
install there are no devices to isolate, and the class says so rather than
pretending otherwise: a "device index" is then one WORKER PROCESS, each layer
is built there from `CausalLM`'s own CPU route (`_block_classes("cpu")`, the
`neural_inference` block classes and `_CpuPrimitives`), and the hidden
activations cross process boundaries exactly as they cross device boundaries
on a GPU column. WHAT THAT COLUMN CHECKS is the layer PARTITION and the
ordered hand-off -- one `_RemoteBlock` per layer, `_RemotePrimitives` for the
embedding on the first owner and the norm and head on the last -- held byte
for byte to a plain in-process `CausalLM` on the same weights. WHAT IT DOES
NOT CHECK is device isolation, memory residency or anything physical: one
process per index is the degenerate case of the device axis and two GPUs
remain owed. It is AGREEMENT with the plain path, not a claim that either is
right. Metal admits only device zero: its worker uses the ordinary GPU
route, and no visibility mask or physical multi-device isolation is claimed.
"""
from .. import _backend
from .._parallel_pool import DevicePool, _cpu_refusal
from .causal_lm import CausalLM

__all__ = ['ParallelCausalLM']


def _admit_route(layer_devices=()):
    """`gpu` on CUDA/HIP or single-device Metal; `cpu` on a CPU-only install.

    Called in the same two places the vendor check used to sit, and before a
    layer is built or a checkpoint is read."""
    vendor = _backend.vendor()
    if vendor == 'metal':
        if not layer_devices or any(type(d) is not int or d != 0 for d in layer_devices):
            raise ValueError('Metal ParallelCausalLM requires every layer owner to be device 0')
        return 'gpu'
    if vendor in ('cuda', 'hip'):
        return 'gpu'
    if _backend._CPU_ONLY is not None:
        return 'cpu'
    raise NotImplementedError(
        'ParallelCausalLM requires CUDA or HIP device isolation, or a CPU-only '
        'install where one owner is one worker process')


class _RemoteState:
    def __init__(self, layer, token):
        self.layer = layer
        self.token = token

    def snapshot(self):
        if self.token is None:
            raise ValueError('state released')
        return self.layer.call('state', self.token)

    def close(self):
        if self.token is not None and not self.layer.model._closed:
            self.layer.call('release', self.token)
            self.token = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _RemoteBlock:
    def __init__(self, model, index, weights, kwargs):
        self.model, self.index = model, index
        self.weight_format = self.call('block', (index, model.kind, weights, kwargs))

    def call(self, operation, args):
        return self.model._rpc(self.model.layer_devices[self.index], operation, args)

    def allocate_state(self, batch, capacity=None):
        return _RemoteState(self, self.call('allocate', (self.index, batch, capacity, self.model.kind)))

    def _run(self, method, x, state):
        if state is not None and (state.layer is not self or state.token is None):
            raise ValueError('state belongs to another layer or has been released')
        return self.call('run', (self.index, method, x, None if state is None else state.token))

    def forward(self, x, state=None):
        return self._run('forward', x, state)

    def step(self, x, state):
        return self._run('step', x, state)


class _RemotePrimitives:
    def __init__(self, model):
        self.model = model

    def embedding(self, table, ids):
        return self.model._rpc(self.model.layer_devices[0], 'embedding', ids)

    def rms_norm(self, x, weight, eps):
        return self.model._rpc(self.model.layer_devices[-1], 'norm', (x, eps))

    def linear(self, x, weight):
        return self.model._rpc(self.model.layer_devices[-1], 'head', x)


class ParallelCausalLM(CausalLM):
    """Experimental layer ownership. Use ``load(path, layer_devices=(0, 1))``.

    One device index per checkpoint layer; repeated indices are allowed.
    Metal requires every index to be zero (one worker on one GPU). Close
    explicitly or use a context manager. Calls and states must not be used
    concurrently. Output and state mathematics are the ordinary CausalLM path.
    """
    def __init__(self, plan, weights, *, layer_devices, **kwargs):
        self.layer_devices = tuple(layer_devices)
        if (len(self.layer_devices) != plan.n_layers or
                any(type(d) is not int or d < 0 for d in self.layer_devices)):
            raise ValueError('layer_devices requires one nonnegative device index per layer')
        self.route = _admit_route(self.layer_devices)
        self._closed = False
        self._pools = {}
        try:
            devices = tuple(dict.fromkeys(self.layer_devices))
            pool = DevicePool(devices)
            self._pools = dict.fromkeys(devices, pool)
            self._worker_index = {device: i for i, device in enumerate(devices)}
            pool._start()
            # EVERY WORKER LEARNS THE ROUTE BEFORE IT BUILDS ANYTHING, so no
            # worker can silently take the GPU block classes on a host box.
            for device in devices:
                self._rpc(device, 'route', self.route)
            super().__init__(plan, weights, device=self.route, **kwargs)
            self._rpc(self.layer_devices[0], 'tensors', {'embed': self._embed})
            self._rpc(self.layer_devices[-1], 'tensors', {'norm': self._norm, 'head': self._head})
            self._prims = _RemotePrimitives(self)
        except BaseException:
            self.close()
            raise

    @classmethod
    def load(cls, path, *, layer_devices, weight_format='float32', max_positions=None):
        from .config import HFConfig, plan_for
        from .safetensors import Checkpoint
        if weight_format not in ('float32', 'bfloat16', 'int8'):
            raise ValueError('unsupported weight_format')
        layer_devices = tuple(layer_devices)
        if not layer_devices or any(type(d) is not int or d < 0 for d in layer_devices):
            raise ValueError('layer_devices requires one nonnegative device index per layer')
        _admit_route(layer_devices)
        plan = plan_for(HFConfig.from_json(path))
        if len(layer_devices) != plan.n_layers:
            raise ValueError('layer_devices requires one nonnegative device index per layer')
        ckpt = Checkpoint.open(path)
        try:
            weights = cls._read_weights(ckpt, plan, weight_format)
        finally:
            ckpt.close()
        return cls(plan, weights, layer_devices=layer_devices,
                   weight_format=weight_format, max_positions=max_positions)

    def _make_blocks(self, cls, layers, kwargs):
        return [_RemoteBlock(self, i, weights, kwargs) for i, weights in enumerate(layers)]

    def _rpc(self, device, operation, args):
        if self._closed:
            raise RuntimeError('ParallelCausalLM is closed')
        pool = self._pools[device]
        request = ('causal_lm_layer', operation, args)
        # `_rpc` addresses ONE worker and so cannot go through `pool.map`,
        # which is where the CPU admission normally sits. State it here rather
        # than let the operation past that gate by the back door.
        # The pool is never cooperative here and the device count is not part
        # of the question, which is only whether `causal_lm_layer` is in
        # `CPU_OPERATIONS`.
        if _backend._CPU_ONLY is not None:
            refusal = _cpu_refusal([request], False)
            if refusal is not None:
                raise refusal
        return pool._call(pool._workers[self._worker_index[device]], request)

    def parameters(self):
        out = {self.plan.embed_name: self._embed, self.plan.norm_name: self._norm}
        if self.plan.head_name is not None:
            out[self.plan.head_name] = self._head
        for i, block in enumerate(self._blocks):
            names = {k: n for k, n, _ in self.plan.layer_weights(i)}
            for key, value in block.call('parameters', i).items():
                out.setdefault(names[key], value)
        return out

    def close(self):
        self._closed = True
        for pool in set(getattr(self, '_pools', {}).values()):
            pool.close()
        self._pools = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
