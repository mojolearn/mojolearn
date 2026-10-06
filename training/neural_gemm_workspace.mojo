# SPDX-License-Identifier: Apache-2.0
"""NI01 caller integration: bounded scratch for synchronous training tensors.

Source-only, default OFF, no validation or measurement in this revision.
Only scratch is retained, never operands or outputs. Each entry owns its
device context; the public training tensor operation drains before returning.
"""
from std.ffi import _Global
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.ctx_key import ctx_cache_key, ctx_cache_slot
from gemm.experiments.bounded_workspace import BoundedGemmWorkspace

comptime IDN_TRAINING_GEMM_WORKSPACE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NI01_TRAINING_WORKSPACE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
# A 16 MiB per-context retained-memory budget, independent of matrix shapes.
# Larger calls use the existing bounded owner's temporary-and-wait path.
comptime TRAINING_GEMM_RETAINED_FLOATS = (16 * 1024 * 1024) // 4

struct _TrainingGemmScratch(Defaultable, Movable):
    var ids: List[Int]
    var workspaces: List[BoundedGemmWorkspace]

    def __init__(out self):
        self.ids = List[Int]()
        self.workspaces = List[BoundedGemmWorkspace]()

comptime _TRAINING_GEMM_SCRATCH = _Global[StorageType=_TrainingGemmScratch,
    name="MojolearnNI01TrainingGemmScratch", init_fn=_TrainingGemmScratch.__init__]

def training_gemm_cached_into(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int, op: Int,
) raises:
    var pool = _TRAINING_GEMM_SCRATCH.get_or_create_ptr()
    var slot = ctx_cache_slot(pool[].ids, ctx)
    if slot < 0:
        pool[].workspaces.append(BoundedGemmWorkspace(ctx, TRAINING_GEMM_RETAINED_FLOATS))
        pool[].ids.append(ctx_cache_key(ctx))
        slot = len(pool[].ids) - 1
    pool[].workspaces[slot].run(ctx, c, a, b, m, n, k, op)

def training_gemm_cached_close(ctx: DeviceContext) raises:
    """Release retained storage for a caller ending its training session."""
    var pool = _TRAINING_GEMM_SCRATCH.get_or_create_ptr()
    var slot = ctx_cache_slot(pool[].ids, ctx)
    if slot >= 0:
        pool[].workspaces[slot].close(ctx)
