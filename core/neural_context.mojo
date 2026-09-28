# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE PROCESS-LIFETIME DeviceContext PER NEURAL BINDING (lane/neural,
2026-09-28; CURRENT DIRECTIVES "one process-lifetime DeviceContext").

The training, transformer, mamba, embedding and byte LM bindings built a new
`DeviceContext()` in every entry (and in every decode or training session).
Two failures traced to that:

  * do-amd (MI325X), request 1790542263165-neural: `samba/negative ragged`
    (the ragged stage opens and closes transformer and Mamba sessions over
    and over) never returned. The main thread sat in `sched_yield` inside
    libamdhip64 under `DeviceBuffer::~DeviceBuffer` in
    `fused_attention.device_absmax`, after every stream in it had already
    synchronized, with no wave resident on the GPU. DEVIATION 2513 saw the
    same class on an RTX 4090: a context created after another was destroyed
    in the same process never returned from its first use.
  * M2 Pro and M3 Ultra: every `training-primitives batchgrad` cell (one
    `linear_backward` call PER ROW, each on a new context, so a new Metal
    queue and a fresh pipeline compile per call) REFUSED with
    `XPC_ERROR_CONNECTION_INTERRUPTED` from the Metal compiler service.
    Metal's command queues are also a per-process budget (memory: METAL
    QUEUE LIMIT IS PER-PROCESS).

Every entry and session now takes the binding's one context, created on first
use and kept for the process. Same kernels, same launches, same order on one
stream, every entry still synchronizes before it returns, so no bit moves.

Storage: `std.ffi._Global` (x_cnn/device.mojo's pattern). The caller passes
the slot NAME, one per binding and numeric tier, so a FAST and an IDENTICAL
.so in one process never share a slot and no two bindings share one.

THE ONE ACCESSOR FOR EVERY BINDING (lane/devctx-lifetime, 2026-09-28).
`process_ctx[name]()` is the process-lifetime context of every binding entry,
not only the neural ones; `neural_ctx` is the same slot under its first name.
The merged-m2-fix lane found the same Metal queue exhaustion in the GP, SVM,
KernelRidge, GMM and Cholesky bindings, and this lane moved every remaining
per-call `DeviceContext()` in a binding entry onto this accessor (the audit is
docs/lanes/progress/devctx-lifetime.md). Rules for a caller:

  * One slot name per binding .so and numeric tier, spelled
    `"Mojo<Binding>ContextIdentical" if <mode> == IDENTICAL else
    "Mojo<Binding>ContextFast"`. A library module reached by one GPU binding
    takes THAT binding's name, so the binding holds one context in all.
  * Call it only on a device path. A CPU-only install never reaches it: the
    host bindings do not call it, and nothing here runs at import, so no
    context is created where the path is host-only.
  * Every entry still synchronizes before it returns. The context is no
    longer destroyed at the end of the call, so nothing may lean on that.
  * Multi-GPU drivers (`DeviceContext(device_id=rank)`) are not on this slot.
"""
from std.ffi import _Global
from max.gpu.host import DeviceContext


struct _NeuralContext(Defaultable, Movable):
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


def process_ctx[name: StaticString]() raises -> DeviceContext:
    """The binding's shared context (slot `name`), created on first use and
    kept for the process."""
    comptime SLOT = _Global[StorageType=_NeuralContext, name=name, init_fn=_NeuralContext.__init__]
    var slot = SLOT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


def neural_ctx[name: StaticString]() raises -> DeviceContext:
    """`process_ctx` under the name the neural bindings first used."""
    return process_ctx[name]()
