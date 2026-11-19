# SPDX-License-Identifier: Apache-2.0
# Shared stateless logits owner, extracted without changing numerical transport.
from std.sys.compile import is_defined
from core.hostptr import f32_ptr
from core.neural_context import neural_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.byte_lm_binding_impl import ByteLMSession, _validate_slot_table, _read_f32, _read_ids
from training.byte_lm_config import ByteConfig
from training.byte_lm_logits import byte_logits_validate, byte_logits_validate_params, byte_logits_from_params_into
comptime _NEURAL_CTX = "MojolearnNeuralByteLMContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralByteLMContextFastV3"
# TOMBSTONE: MOJOLEARN_BYTE_LM_KEEP_CONTEXT (opt-in arm, no A/B recorded in four weeks: the run-time switch since 2026-10-21, a build define since 2026-11-15) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
# Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_KEEP_CONTEXT.patch; record in docs/TOMBSTONES.md.


def byte_lm_stateless_logits_native(addr: List[Int], batch: Int, length: Int, cfg: ByteConfig) raises -> Int:
    if len(addr) != 3:
        raise Error("byte LM: expected 3 addresses")
    var m = batch * length
    var cells_out = m * cfg.vocab_size
    var n = cfg.n_total()
    var cells: List[Int] = [n, m, cells_out]
    _validate_slot_table(addr, cells, 2)
    var params = _read_f32(addr[0], n)
    var ids = _read_ids(addr[1], m)
    byte_logits_validate(ids, batch, length, cfg)
    byte_logits_validate_params(params, cfg)
    # lane/neural-net-experiment: the logits land in the caller's array by
    # one device-to-host copy (`byte_logits_from_params_into`), no host
    # List, no second copy; the caller's array is alive for the whole call.
    var out = f32_ptr(addr[2])
    var session = ByteLMSession()
    session.busy = True
    try:
        session.ctx = neural_ctx[_NEURAL_CTX]()
        byte_logits_from_params_into(session.ctx.value(), params, ids, batch, length, cfg, out)
        session.ctx.value().synchronize()
        session.ctx = None
    except error:
        session.busy = False
        raise error
    session.busy = False
    session.usable = False
    return cells_out
