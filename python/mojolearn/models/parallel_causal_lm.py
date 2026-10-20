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
        if not layer_devices or any(type(d) is not int or d != 0 for d in layer_devices):  # glue: validates the layer_devices argument
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

    def linear_int15(self, x, planes):
        # the head under fixed15_v1 runs on the head's owner, as `linear` does
        return self.model._rpc(self.model.layer_devices[-1], 'head_int15', x)


from .causal_lm import CausalLMState, _argmax_last, _prompt_then_new
from .._buffer import empty, addr, addr_ro
from .._bufcheck import memcopy


class ParallelCausalLM(CausalLM):
    """Experimental layer ownership. Use ``load(path, layer_devices=(0, 1))``.

    One device index per checkpoint layer; repeated indices are allowed.
    Metal requires every index to be zero (one worker on one GPU). Close
    explicitly or use a context manager. Calls and states must not be used
    concurrently. Output and state mathematics are the ordinary CausalLM path.
    """
    # Existing RPC owner remains here until its transport/orchestration is
    # native. It must not enter CausalLM's native-only coordinator or masquerade
    # as a repaired product path. These are the pre-migration methods unchanged.
    def _initialize_native_runtime(self):
        pass

    @property
    def blocks(self):
        return list(self._blocks)

    def _run(self, ids, state, step, what):
        """The stack: embedding, the blocks, the final norm, the head. ONE
        spelling for the stateless forward, the stateful forward and the
        decode step; only what each block is handed differs."""
        b, l = int(ids.shape[0]), int(ids.shape[1])
        if state is not None:
            if not isinstance(state, CausalLMState):
                raise TypeError(f"mojolearn {what}: state must be a CausalLMState (allocate_state)")
            if state._owner is not self:
                raise ValueError(f"mojolearn {what}: state belongs to another model; allocate a state on this model")
            if state.batch_size != b or len(state.layers) != self.n_layers:
                raise ValueError(
                    f"mojolearn {what}: the state holds {state.batch_size} rows and {len(state.layers)} layers, "
                    f"the call has B = {b} and the model {self.n_layers} layers")
        self._check_positions(what, state, l)
        n, d = b * l, self.d_model
        x = self._prims.embedding(self._embed, ids.reshape((n,))).reshape((b, l, d))
        for i, blk in enumerate(self._blocks):  # glue: drives one native forward per layer
            if state is None:
                x = blk.forward(x)
            elif step:
                x = blk.step(x, state.layers[i])
            else:
                x = blk.forward(x, state.layers[i])
        hn = x.reshape((n, d))
        if self._norm is not None:          # None: no final norm (the byte-LM format)
            hn = self._prims.rms_norm(hn, self._norm, self.norm_eps)
        if self._head_int15 is not None:
            logits = self._prims.linear_int15(hn, self._head_int15)
        else:
            logits = self._prims.linear(hn, self._head)
        if state is not None:
            state.positions += l
        return logits.reshape((b, l, self.vocab_size))

    def forward(self, ids, state=None):
        """`(B, L)` int32 ids in, `(B, L, vocab)` float32 logits out. With
        `state=None` every block runs from a zero state and nothing is kept;
        with a `CausalLMState` the blocks read and update it in place."""
        what = "CausalLM.forward"
        return self._run(self._ids(ids, what), state, False, what)

    __call__ = forward

    def allocate_state(self, batch_size, max_tokens):
        """The zero decode state for `batch_size` rows of up to `max_tokens`
        positions each (`Mamba*Block.allocate_state(B)`,
        `TransformerBlock.allocate_state(B, max_tokens)`)."""
        b, smax = int(batch_size), int(max_tokens)
        if b < 1 or smax < 1:
            raise ValueError("mojolearn CausalLM.allocate_state: batch_size and max_tokens must be positive")
        if self.max_positions is not None and smax > self.max_positions:
            raise ValueError(
                f"mojolearn CausalLM.allocate_state: max_tokens={smax} exceeds max_positions={self.max_positions}")
        layers = []
        for blk in self._blocks:  # glue: allocates native state per layer
            layers.append(blk.allocate_state(b, smax) if self.kind == "transformer" else blk.allocate_state(b))
        return CausalLMState(b, smax, layers, owner=self)

    def reset_state(self, state):
        """Replace caches with fresh zero state, preserving capacity and batch size.

        States are owned by their allocating model; cross-model reuse is refused.
        """
        if not isinstance(state, CausalLMState) or state._owner is not self:
            raise ValueError("mojolearn CausalLM.reset_state: state belongs to another model")
        fresh = self.allocate_state(state.batch_size, state.max_tokens)
        state.layers = fresh.layers
        state.positions = 0
        return state

    def step(self, ids, state):
        """One decode token per row: `(B,)` or `(B, 1)` ids in, `(B, vocab)`
        logits out, `state` updated in place. It is `forward` at L = 1 with
        each block's `step` (their one spelling for decode)."""
        what = "CausalLM.step"
        if state is None:
            raise ValueError("mojolearn CausalLM.step: state is required (allocate_state(B, max_tokens) makes the fresh one)")
        ids = self._ids(ids, what, step=True)
        out = self._run(ids, state, True, what)
        return out.reshape((int(ids.shape[0]), self.vocab_size))

    def generate(self, ids, max_new_tokens, *, greedy=True):
        """`(B, L)` ids in, `(B, L + max_new_tokens)` int32 ids out: the
        prompt followed by greedy continuations (argmax of the last
        position, ties to the lowest index), one carried state, the prompt
        as one prefill and every later token as one `step`. No stop token:
        exactly `max_new_tokens` are produced per row."""
        what = "CausalLM.generate"
        if greedy is not True:
            raise NotImplementedError(f"mojolearn {what}: greedy=True is the only decoding implemented; got greedy={greedy!r}")
        n_new = int(max_new_tokens)
        if n_new < 0:
            raise ValueError(f"mojolearn {what}: max_new_tokens must be >= 0, got {max_new_tokens!r}")
        ids = self._ids(ids, what)
        b, l = int(ids.shape[0]), int(ids.shape[1])
        if n_new == 0:
            return ids.copy()
        total = l + n_new
        if self.max_positions is not None and total > self.max_positions:
            raise ValueError(f"mojolearn {what}: {l} prompt + {n_new} new tokens exceed max_positions={self.max_positions}")
        fast = self._generate_resident(ids, n_new, total)
        if fast is not None:
            return fast
        state = self.allocate_state(b, total)
        logits = self._run(ids, state, False, what)
        nxt = _argmax_last(logits, b, l, self.vocab_size)
        new = empty((n_new, b), "<i4")
        at = addr(new, name="new ids")
        for k in range(n_new):  # decode steps: one native forward each, no per-row work  # glue: drives one native forward per decode step
            memcopy(at + k * b * 4, addr_ro(nxt, name="next ids"), b * 4)
            if k == n_new - 1:
                break
            lg = self._run(nxt.reshape((b, 1)), state, True, what)
            nxt = _argmax_last(lg, b, 1, self.vocab_size)
        return _prompt_then_new(ids, b, l, n_new, at, True)

    def _generate_resident(self, ids, n_new, total, last_logits=None):
        # Blocks and decode handles belong to remote worker processes. The
        # parent's single-process resident loop cannot consume those handles;
        # generate must use the existing per-layer RPC forward/step route.
        return None

    def __init__(self, plan, weights, *, layer_devices, **kwargs):
        self.layer_devices = tuple(layer_devices)
        if (len(self.layer_devices) != plan.n_layers or
                any(type(d) is not int or d < 0 for d in self.layer_devices)):  # glue: validates the layer_devices argument
            raise ValueError('layer_devices requires one nonnegative device index per layer')
        self.route = _admit_route(self.layer_devices)
        self._closed = False
        self._pools = {}
        try:
            devices = tuple(dict.fromkeys(self.layer_devices))
            pool = DevicePool(devices)
            self._pools = dict.fromkeys(devices, pool)
            self._worker_index = {device: i for i, device in enumerate(devices)}  # glue: worker index per device
            pool._start()
            # EVERY WORKER LEARNS THE ROUTE BEFORE IT BUILDS ANYTHING, so no
            # worker can silently take the GPU block classes on a host box.
            for device in devices:  # glue: starts one worker per device
                self._rpc(device, 'route', self.route)
            super()._initialize_legacy(plan, weights, device=self.route, **kwargs)
            self._rpc(self.layer_devices[0], 'tensors', {'embed': self._embed})
            self._rpc(self.layer_devices[-1], 'tensors', {'norm': self._norm, 'head': self._head,
                                                          'head_int15': self._head_int15})
            self._prims = _RemotePrimitives(self)
        except BaseException:
            self.close()
            raise

    @classmethod
    def load(cls, path, *, layer_devices, weight_format='float32', max_positions=None,
             numeric_profile=None):
        from .. import _numeric_profile
        from .config import HFConfig, plan_for
        from .safetensors import Checkpoint
        if numeric_profile is not None:  # None: the model's family decides (CausalLM.__init__)
            numeric_profile = _numeric_profile.resolve(numeric_profile, 'mojolearn.models.ParallelCausalLM.load numeric_profile')
        if weight_format not in ('float32', 'bfloat16', 'int8'):
            raise ValueError('unsupported weight_format')
        layer_devices = tuple(layer_devices)
        if not layer_devices or any(type(d) is not int or d < 0 for d in layer_devices):  # glue: validates the layer_devices argument
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
                   weight_format=weight_format, max_positions=max_positions,
                   numeric_profile=numeric_profile)

    def _make_blocks(self, cls, layers, kwargs):
        return [_RemoteBlock(self, i, weights, kwargs) for i, weights in enumerate(layers)]  # glue: one remote block per model layer

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
        for i, block in enumerate(self._blocks):  # glue: walks the model layers
            names = {k: n for k, n, _ in self.plan.layer_weights(i)}  # glue: names the weights of one layer
            for key, value in block.call('parameters', i).items():  # glue: walks the named weights of one layer
                out.setdefault(names[key], value)
        return out

    def close(self):
        self._closed = True
        for pool in set(getattr(self, '_pools', {}).values()):  # glue: closes the device worker pools
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
