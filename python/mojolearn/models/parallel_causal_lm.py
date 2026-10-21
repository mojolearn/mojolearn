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
from .causal_lm import CausalLM, _NATIVE as _CAUSAL_RUNTIME, _NATIVE_CONTEXT as _CAUSAL_CONTEXT

__all__ = ['ParallelCausalLM']


def _admit_route(layer_devices=()):
    """`gpu` on CUDA/HIP or single-device Metal; `cpu` on a CPU-only install.

    Called in the same two places the vendor check used to sit, and before a
    layer is built or a checkpoint is read."""
    return _native_parallel_causal("route", (layer_devices,))


class _RemoteState:
    def __init__(self, layer, token):
        return _native_parallel_causal("state_init", (self, layer, token))

    def snapshot(self):
        return _native_parallel_causal("state_snapshot", (self,))

    def close(self):
        return _native_parallel_causal("state_close", (self,))

    def __del__(self):
        return _native_parallel_causal("state_close_quiet", (self,))


class _RemoteBlock:
    def __init__(self, model, index, weights, kwargs):
        return _native_parallel_causal("block_init", (self, model, index, weights, kwargs))

    def call(self, operation, args):
        return _native_parallel_causal("block_call", (self, operation, args))

    def allocate_state(self, batch, capacity=None):
        return _native_parallel_causal("block_allocate", (self, batch, capacity))

    def _run(self, method, x, state):
        return _native_parallel_causal("block_run", (self, method, x, state))

    def forward(self, x, state=None):
        return _native_parallel_causal("block_run", (self, "forward", x, state))

    def step(self, x, state):
        return _native_parallel_causal("block_run", (self, "step", x, state))


class _RemotePrimitives:
    def __init__(self, model):
        return _native_parallel_causal("primitives_init", (self, model))

    def embedding(self, table, ids):
        return _native_parallel_causal("primitive", (self, "embedding", ids))

    def rms_norm(self, x, weight, eps):
        return _native_parallel_causal("primitive", (self, "norm", (x, eps)))

    def linear(self, x, weight):
        return _native_parallel_causal("primitive", (self, "head", x))

    def linear_int15(self, x, planes):
        # the head under fixed15_v1 runs on the head's owner, as `linear` does
        return _native_parallel_causal("primitive", (self, "head_int15", x))


class ParallelCausalLM(CausalLM):
    """Experimental layer ownership. Use ``load(path, layer_devices=(0, 1))``.

    One device index per checkpoint layer; repeated indices are allowed.
    Metal requires every index to be zero (one worker on one GPU). Close
    explicitly or use a context manager. Calls and states must not be used
    concurrently. Output and state mathematics are the ordinary CausalLM path.
    """
    # The inherited public methods enter the shared native owner; remote
    # layer dispatch is cached as a compiled entry in _native_entries.

    def _generate_resident(self, ids, n_new, total, last_logits=None):
        # Blocks and decode handles belong to remote worker processes. The
        # parent's single-process resident loop cannot consume those handles;
        # The shared native generation loop drives the remote owners.
        return _native_parallel_causal("resident", (self, ids, n_new, total, last_logits))

    def __init__(self, plan, weights, *, layer_devices, **kwargs):
        return _native_parallel_causal("init", (self, plan, weights, layer_devices, kwargs), _CAUSAL_CONTEXT)

    @classmethod
    def load(cls, path, *, layer_devices, weight_format='float32', max_positions=None,
             numeric_profile=None):
        return _native_parallel_causal("load", (cls, path, layer_devices, (weight_format, max_positions, numeric_profile)), _CAUSAL_CONTEXT)

    def _make_blocks(self, cls, layers, kwargs):
        return _native_parallel_causal("make_blocks", (self, layers, kwargs))

    def _rpc(self, device, operation, args):
        return _native_parallel_causal("rpc", (self, device, operation, args))

    def parameters(self):
        return _native_parallel_causal("parameters", (self,))

    def close(self):
        return _native_parallel_causal("close", (self,))

    def __enter__(self):
        return _native_parallel_causal("enter", (self,))

    def __exit__(self, *exc):
        return _native_parallel_causal("exit", (self,))

    def __del__(self):
        return _native_parallel_causal("close_quiet", (self,))


# Connection metadata only; native owners perform every product operation.
from .. import _native_process as _process_connection
_native_parallel_causal = _CAUSAL_RUNTIME.parallel_causal_call
_native_parallel_causal("initialize", (_CAUSAL_CONTEXT, {
    "model": ParallelCausalLM, "block": _RemoteBlock,
    "state": _RemoteState, "primitives": _RemotePrimitives,
}, _native_parallel_causal))
