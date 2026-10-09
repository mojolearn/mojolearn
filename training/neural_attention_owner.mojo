# SPDX-License-Identifier: Apache-2.0
"""NN31/NN32 native retained-forward and bounded checkpoint components.

SOURCE DRAFT ONLY: no compilation, verification, quality or timing performed.
The public TransformerBlock tape API now uses BorrowedAttentionTape below.
Direct ByteTrainer/Samba optimizer ownership remains an optional extension.

The owner MOVES IN a context, weights and RoPE table. It snapshots the input
on device and owns every forward/backward buffer. It returns a single-use
owner/generation ticket, never a pointer-based cache guess. Parameter and
configuration replacement synchronizes and invalidates outstanding tickets.
A retained A arm calls the actual existing transformer backward on the saved
LlamaDeviceStages; B discards those stages after forward and recomputes them
from the owned input snapshot before calling the SAME backward.

Only the default FP32 transformer training profile is supported here. There
is no dropout, prefix/decode state, changed reduction graph, Python data
comparison, host tensor arithmetic, graph capture or asynchronous offload.
The explicit full-prefill restriction prevents guessing how a cache prefix
or a new option changes replay. Any broader training integration must supply its
real weight/config mutation hooks and retain model-wide RNG metadata before
broadening that scope.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from core.step_phase import step_count_d2d, step_count_device_alloc, step_count_sync
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceWeights, LlamaDeviceStages, LlamaDims, LlamaKVCache,
    LlamaRopeTable, llama_decoder_layer_forward,
)
from transformer.checks.transformer_backward import (
    LlamaBackwardStages, llama_decoder_layer_backward_device,
)

# Both are OFF: no new source has compile/identity/quality/timing evidence.
# NN32 isolates retention vs replay in one native layer owner. NN31 admits
# retention by actual retained bytes and replay cost; budget misses replay.
# Never choose a rule from a dataset, board size or hardware vendor.
from transformer.experiments.checkpoint_contract import (
    NN32_RETAIN_FORWARD, NN31_BOUNDED_CHECKPOINTS, attention_checkpoint_retain,
)

comptime OWNER_EMPTY = 0
comptime OWNER_FORWARD_READY = 1
comptime OWNER_GRADIENT_READY = 2
comptime OWNER_BUSY = 3
comptime OWNER_FAILED = 4
comptime OWNER_CLOSED = 5
comptime CHECKPOINT_REPLAY = 0
comptime CHECKPOINT_RETAIN = 1


@fieldwise_init
struct AttentionForwardTicket(Copyable, Movable):
    var session: Int
    var owner: Int
    var forward_generation: Int
    var input_generation: Int
    var weight_generation: Int
    var config_generation: Int
    var budget_generation: Int
    var retained: Bool


@fieldwise_init
struct CheckpointLease(Copyable, Movable):
    var number: Int
    var owner: Int
    var bytes: Int
    var replay_operations: Int


struct AttentionCheckpointBudget(Movable):
    """NN31 budget for live retained forward stages, not total device memory.

    `limit_bytes` covers all explicit DeviceBuffers of each retained
    LlamaDeviceStages, including its retained GEMM workspace. Input snapshots,
    weights, backward buffers, contexts and shared process caches must be
    budgeted separately by the complete model. The transient current forward
    is also separate: we count the allocated result before deciding to keep
    it, so the policy never claims to prevent a forward allocation failure.

    Calls arrive in model layer order. Eligible layers reserve bytes until
    the bound is reached; later ones replay. `minimum_ops_per_byte` can make
    low-recompute-cost layers replay even when room remains. All inputs are
    byte/cost quantities with neighbor-shape meaning. No timings or board
    constants are encoded. For a deliberate layer group, give every owner
    the same group policy and retain the group's saved inputs independently.
    """
    var session: Int
    var generation: Int
    var limit_bytes: Int
    var live_bytes: Int
    var peak_bytes: Int
    var minimum_ops_per_byte: Int
    var next_owner: Int
    var next_lease: Int
    var leases: List[CheckpointLease]
    var outstanding_owners: List[Int]

    def __init__(out self, session: Int, limit_bytes: Int,
                 minimum_ops_per_byte: Int = 0) raises:
        if session <= 0 or limit_bytes < 0 or minimum_ops_per_byte < 0:
            raise Error("NN31: positive unique session and nonnegative budget/cost required")
        self.session = session
        self.generation = 1
        self.limit_bytes = limit_bytes
        self.live_bytes = 0
        self.peak_bytes = 0
        self.minimum_ops_per_byte = minimum_ops_per_byte
        self.next_owner = 1
        self.next_lease = 1
        self.leases = List[CheckpointLease]()
        self.outstanding_owners = List[Int]()

    def register_owner(mut self) -> Int:
        var owner = self.next_owner
        self.next_owner += 1
        return owner

    def begin_forward(mut self, owner: Int) raises:
        for i in range(len(self.outstanding_owners)):  # small-loop(outstanding_owners: open forward owners): walks owner handles, not data
            if self.outstanding_owners[i] == owner:
                raise Error("NN31: owner already has an outstanding forward")
        self.outstanding_owners.append(owner)

    def end_forward(mut self, owner: Int) raises:
        for i in range(len(self.outstanding_owners)):  # small-loop(outstanding_owners: open forward owners): walks owner handles, not data
            if self.outstanding_owners[i] == owner:
                _ = self.outstanding_owners.pop(i)
                return
        raise Error("NN31: forward was already consumed or not registered")

    def try_reserve(mut self, owner: Int, bytes: Int,
                    replay_operations: Int, apply_cost_policy: Bool) raises -> Int:
        if owner <= 0 or bytes <= 0 or replay_operations < 0:
            raise Error("NN31: invalid retained-state reservation")
        for i in range(len(self.leases)):  # small-loop(leases: retained-state leases): walks lease records, not data
            if self.leases[i].owner == owner:
                raise Error("NN31: owner already holds a retained forward")
        if bytes > self.limit_bytes - self.live_bytes:
            return 0
        if apply_cost_policy and self.minimum_ops_per_byte > 0:
            # Division avoids multiplying a potentially large operation
            # estimate by a byte count; this is conservative admission.
            if replay_operations // bytes < self.minimum_ops_per_byte:
                return 0
        var number = self.next_lease
        self.next_lease += 1
        self.leases.append(CheckpointLease(number, owner, bytes, replay_operations))
        self.live_bytes += bytes
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        return number

    def release(mut self, owner: Int, lease: Int) raises:
        if lease == 0:
            return
        for i in range(len(self.leases)):  # small-loop(leases: retained-state leases): walks lease records, not data
            if self.leases[i].number == lease:
                if self.leases[i].owner != owner:
                    raise Error("NN31: lease belongs to another owner")
                var released = self.leases.pop(i)
                self.live_bytes -= released.bytes
                return
        raise Error("NN31: unknown or already released lease")

    def reset(mut self, limit_bytes: Int) raises:
        if len(self.leases) != 0 or self.live_bytes != 0 or len(self.outstanding_owners) != 0:
            raise Error("NN31: cannot reset budget while forward tickets are live")
        if limit_bytes < 0:
            raise Error("NN31: negative budget")
        self.limit_bytes = limit_bytes
        self.peak_bytes = 0
        self.generation += 1


def _forward_bytes(st: LlamaDeviceStages) -> Int:
    """Actual retained explicit allocation lengths, after route-dependent growth.

    These are distinct owned forward buffers in the default FP32 profile.
    There are no backwards' aliased residual views here. Persistent shared
    attention process caches are not owned by this stage and are excluded.
    """
    var cells = (
        len(st.norm1_sumsq) + len(st.norm1_out) + len(st.q_proj)
        + len(st.k_proj) + len(st.v_proj) + len(st.q_rope) + len(st.k_rope)
        + len(st.k_cache) + len(st.v_cache) + len(st.scores) + len(st.masked)
        + len(st.amax) + len(st.aexp) + len(st.denom) + len(st.weights)
        + len(st.ctxv) + len(st.o_proj) + len(st.residual1)
        + len(st.norm2_sumsq) + len(st.norm2_out) + len(st.gate_proj)
        + len(st.up_proj) + len(st.silu_out) + len(st.gated)
        + len(st.down_proj) + len(st.residual2) + len(st.qbh)
        + len(st.kbh) + len(st.sbh) + len(st.qk_sumsq)
        + len(st.gemm_workspace.buffer)
    )
    return 4 * cells


def attention_replay_operation_estimate(b: Int, l: Int, dims: LlamaDims) -> Int:
    """Shape-derived FP operation cost model, not a benchmark threshold.

    Counts seven projection GEMMs and dense QK/PV multiply-add work. A
    window/fused-mask route may do less than the dense bound, so the native
    component uses full-causal input only and declares this a cost estimate.
    Transcendentals/normalization are omitted rather than given fake prices.
    """
    var m = b * l
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var projections = 2 * m * (dm * (2 * qw + 2 * kw) + 3 * dm * it)
    var attention = 4 * b * dims.n_heads * l * l * dims.head_dim
    return projections + attention


def _same_dims(a: LlamaDims, b: LlamaDims) -> Bool:
    return (a.d_model == b.d_model and a.n_heads == b.n_heads
            and a.n_kv == b.n_kv and a.head_dim == b.head_dim
            and a.intermediate == b.intermediate)


def _validate_profile(w: LlamaDeviceWeights, rope: LlamaRopeTable, l: Int) raises:
    w.dims.validate()
    if not w.opts.is_default() or w.int15:
        raise Error("NN31/NN32: only the default FP32 transformer training profile is defined")
    if rope.p_max < l or rope.rope_dim != w.dims.head_dim:
        raise Error("NN31/NN32: rotary table does not cover the full prefill")
    if bitcast[DType.uint32](rope.theta) != bitcast[DType.uint32](w.opts.rope_theta):
        raise Error("NN31/NN32: rotary configuration differs from the owned weights")


def _copy_parameter_piece(ctx: DeviceContext,
                          mut dst: DeviceBuffer[DType.float32],
                          mut src: DeviceBuffer[DType.float32], first: Int) raises:
    var target = dst.create_sub_buffer[DType.float32](first, len(src))
    step_count_d2d()
    ctx.enqueue_copy(dst_buf=target, src_buf=src)
    # A sub-buffer owner may be released only after its enqueued write.
    step_count_sync()
    ctx.synchronize()


struct NeuralAttentionOwner(Movable):
    """Native single-live-forward owner usable by a future Samba/LM session.

    Construction transfers ownership of context/weights/RoPE. This internal
    API does not export mutable weight or saved-stage views. Callers must
    replace weights through replace_weights, not mutate public struct fields
    behind the owner. A complete binding must enforce this ownership rule.
    All boundary copies finish before return, making source/output lifetime
    explicit. Later stream-aware integration can remove waits only with an
    explicit native event/lifetime contract and supported Modular APIs.
    """
    var context: DeviceContext
    var weights: LlamaDeviceWeights
    var rope: LlamaRopeTable
    var cache: LlamaKVCache
    var forward: Optional[LlamaDeviceStages]
    var backward: LlamaBackwardStages
    var input: DeviceBuffer[DType.float32]
    var cotangent: DeviceBuffer[DType.float32]
    var dims: LlamaDims
    var b: Int
    var l: Int
    var session: Int
    var owner: Int
    var budget_generation: Int
    var forward_generation: Int
    var input_generation: Int
    var weight_generation: Int
    var config_generation: Int
    var lease: Int
    var registered_forward: Bool
    var state: Int
    var last_decision: Int
    var last_retained_bytes: Int
    var retained_forwards: Int
    var replayed_forwards: Int

    def __init__(out self, var ctx: DeviceContext,
                 var weights: LlamaDeviceWeights, var rope: LlamaRopeTable,
                 b: Int, l: Int, mut budget: AttentionCheckpointBudget) raises:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("NN31/NN32 owner is an IDENTICAL-only component")
        if b <= 0 or l <= 0:
            raise Error("NN31/NN32: positive full-prefill dimensions required")
        _validate_profile(weights, rope, l)
        self.dims = weights.dims.copy()
        self.b = b
        self.l = l
        self.session = budget.session
        self.owner = budget.register_owner()
        self.budget_generation = budget.generation
        self.forward_generation = 0
        self.input_generation = 0
        self.weight_generation = 1
        self.config_generation = 1
        self.lease = 0
        self.registered_forward = False
        self.state = OWNER_EMPTY
        self.last_decision = CHECKPOINT_REPLAY
        self.last_retained_bytes = 0
        self.retained_forwards = 0
        self.replayed_forwards = 0
        self.forward = None
        self.cache = LlamaKVCache(ctx, b, self.dims, l)
        self.backward = LlamaBackwardStages(ctx, b, l, l, self.dims, lean=True)
        step_count_device_alloc()
        self.input = ctx.enqueue_create_buffer[DType.float32](b * l * self.dims.d_model)
        step_count_device_alloc()
        self.cotangent = ctx.enqueue_create_buffer[DType.float32](b * l * self.dims.d_model)
        step_count_sync()
        ctx.synchronize()
        self.weights = weights^
        self.rope = rope^
        self.context = ctx^

    def _require_budget(self, budget: AttentionCheckpointBudget) raises:
        if budget.session != self.session:
            raise Error("NN31/NN32: owner belongs to another budget/session")
        if budget.generation != self.budget_generation:
            raise Error("NN31/NN32: budget generation changed; explicitly rebind an empty owner")

    def _require_live(self) raises:
        if self.state == OWNER_FAILED or self.state == OWNER_CLOSED or self.state == OWNER_BUSY:
            raise Error("NN31/NN32: owner is failed, closed or already executing")

    def _require_ticket(self, ticket: AttentionForwardTicket,
                        budget: AttentionCheckpointBudget) raises:
        self._require_budget(budget)
        self._require_live()
        if self.state != OWNER_FORWARD_READY:
            raise Error("NN32: backward requires an unconsumed forward ticket")
        if (ticket.session != self.session or ticket.owner != self.owner
                or ticket.forward_generation != self.forward_generation
                or ticket.input_generation != self.input_generation
                or ticket.weight_generation != self.weight_generation
                or ticket.config_generation != self.config_generation
                or ticket.budget_generation != self.budget_generation
                or ticket.retained != (self.last_decision == CHECKPOINT_RETAIN)):
            raise Error("NN32: stale or foreign forward ticket")

    def invalidate(mut self, mut budget: AttentionCheckpointBudget) raises:
        """Synchronize before dropping state; all previous tickets become stale."""
        self._require_budget(budget)
        self._require_live()
        step_count_sync()
        self.context.synchronize()
        budget.release(self.owner, self.lease)
        self.lease = 0
        if self.registered_forward:
            budget.end_forward(self.owner)
            self.registered_forward = False
        self.forward = None
        self.forward_generation += 1
        self.last_retained_bytes = 0
        self.state = OWNER_EMPTY

    def rebind_budget(mut self, budget: AttentionCheckpointBudget) raises:
        self._require_live()
        if budget.session != self.session or self.state == OWNER_FORWARD_READY or self.lease != 0:
            raise Error("NN31: rebind requires the same session and no live forward")
        self.budget_generation = budget.generation

    def replace_weights(mut self, var replacement: LlamaDeviceWeights,
                        mut budget: AttentionCheckpointBudget) raises:
        """Replacement must be built on this owner's context before transfer."""
        _validate_profile(replacement, self.rope, self.l)
        if not _same_dims(replacement.dims, self.dims):
            raise Error("NN32: changed model dimensions require a new native owner")
        self.invalidate(budget)
        self.weights = replacement^
        self.weight_generation += 1

    def replace_rope(mut self, var replacement: LlamaRopeTable,
                     mut budget: AttentionCheckpointBudget) raises:
        _validate_profile(self.weights, replacement, self.l)
        self.invalidate(budget)
        self.rope = replacement^
        self.config_generation += 1

    def _run_forward(mut self, mut trace: IdentityTrace, prefix: String) raises:
        self.forward = Optional[LlamaDeviceStages](
            LlamaDeviceStages(self.context, self.b, self.l, self.l, self.dims, lean=True)
        )
        self.cache.s = 0
        llama_decoder_layer_forward(
            self.context, self.forward.value(), self.cache, self.rope,
            self.weights, self.input, self.b, self.l, 0, trace, prefix,
            forward_only=False,
        )
        step_count_sync()
        self.context.synchronize()

    def _fail(mut self, mut budget: AttentionCheckpointBudget) raises:
        self.state = OWNER_FAILED
        # Keep allocations alive if synchronization itself fails. An owner
        # in FAILED state never accepts another numerical call.
        step_count_sync()
        self.context.synchronize()
        budget.release(self.owner, self.lease)
        self.lease = 0
        if self.registered_forward:
            budget.end_forward(self.owner)
            self.registered_forward = False
        self.forward = None
        self.forward_generation += 1

    def forward_into(mut self, source_ctx: DeviceContext,
                     mut source: DeviceBuffer[DType.float32],
                     mut output: DeviceBuffer[DType.float32],
                     source_generation: Int, mut budget: AttentionCheckpointBudget,
                     mut trace: IdentityTrace, prefix: String,
                     deliberate_checkpoint: Bool = False) raises -> AttentionForwardTicket:
        self._require_live()
        self._require_budget(budget)
        if self.state == OWNER_FORWARD_READY:
            raise Error("NN32: consume or invalidate the previous forward before overwriting it")
        var n = self.b * self.l * self.dims.d_model
        if len(source) != n or len(output) != n or source_generation < 0:
            raise Error("NN32: input/output shape or generation mismatch")
        budget.begin_forward(self.owner)
        self.registered_forward = True
        self.state = OWNER_BUSY
        self.forward_generation += 1
        self.input_generation = source_generation
        self.last_retained_bytes = 0
        self.last_decision = CHECKPOINT_REPLAY
        try:
            # Snapshot actual input bytes; input-generation equality is not
            # used to infer that externally mutable data remained unchanged.
            step_count_sync()
            source_ctx.synchronize()
            step_count_d2d()
            source.enqueue_copy_to(self.input)
            step_count_sync()
            source_ctx.synchronize()
            step_count_sync()
            self.context.synchronize()
            self._run_forward(trace, prefix)
            step_count_d2d()
            self.forward.value().residual2.enqueue_copy_to(output)
            step_count_sync()
            self.context.synchronize()
            step_count_sync()
            source_ctx.synchronize()
            comptime if NN32_RETAIN_FORWARD or NN31_BOUNDED_CHECKPOINTS:
                if not deliberate_checkpoint:
                    var bytes = _forward_bytes(self.forward.value())
                    var cost = attention_replay_operation_estimate(self.b, self.l, self.dims)
                    self.lease = budget.try_reserve(self.owner, bytes, cost, NN31_BOUNDED_CHECKPOINTS)
                    if self.lease != 0:
                        self.last_decision = CHECKPOINT_RETAIN
                        self.last_retained_bytes = bytes
                        self.retained_forwards += 1
                    elif not NN31_BOUNDED_CHECKPOINTS:
                        # NN32's isolated retention A must not silently run B.
                        raise Error("NN32: retention arm lacks the declared forward-state budget")
            if self.last_decision == CHECKPOINT_REPLAY:
                # All consumers have completed before releasing the real
                # forward buffers. This is a memory-lifetime change, not a flag.
                self.forward = None
            self.state = OWNER_FORWARD_READY
            return AttentionForwardTicket(
                self.session, self.owner, self.forward_generation,
                self.input_generation, self.weight_generation,
                self.config_generation, self.budget_generation,
                self.last_decision == CHECKPOINT_RETAIN,
            )
        except error:
            self._fail(budget)
            raise error

    def backward_into(mut self, ticket: AttentionForwardTicket,
                      source_ctx: DeviceContext,
                      mut cotangent: DeviceBuffer[DType.float32],
                      mut d_input: DeviceBuffer[DType.float32],
                      mut budget: AttentionCheckpointBudget,
                      mut trace: IdentityTrace, prefix: String) raises:
        self._require_ticket(ticket, budget)
        var n = self.b * self.l * self.dims.d_model
        if len(cotangent) != n or len(d_input) != n:
            raise Error("NN32: cotangent/input-gradient shape mismatch")
        # Consumed BEFORE enqueuing: failure cannot replay a partial backward.
        self.state = OWNER_BUSY
        try:
            step_count_sync()
            source_ctx.synchronize()
            step_count_d2d()
            cotangent.enqueue_copy_to(self.cotangent)
            step_count_sync()
            source_ctx.synchronize()
            step_count_sync()
            self.context.synchronize()
            if not self.forward:
                # Same input snapshot, parameter owner, position table and
                # forward graph. No loss/RNG/optimizer work is repeated here.
                var replay_trace = IdentityTrace.disabled()
                self._run_forward(replay_trace, prefix + ".checkpoint_replay")
                self.replayed_forwards += 1
            llama_decoder_layer_backward_device(
                self.context, self.backward, self.forward.value(), self.weights,
                self.rope.cos, self.rope.sin, self.input, self.cotangent,
                self.b, self.l, 0, trace, prefix,
            )
            step_count_sync()
            self.context.synchronize()
            step_count_d2d()
            self.backward.d_x.enqueue_copy_to(d_input)
            step_count_sync()
            self.context.synchronize()
            step_count_sync()
            source_ctx.synchronize()
            budget.release(self.owner, self.lease)
            self.lease = 0
            budget.end_forward(self.owner)
            self.registered_forward = False
            self.forward = None
            self.state = OWNER_GRADIENT_READY
        except error:
            self._fail(budget)
            raise error

    def parameter_gradient_elements(self) -> Int:
        return (len(self.backward.dw_norm1) + len(self.backward.dw_q)
                + len(self.backward.dw_k) + len(self.backward.dw_v)
                + len(self.backward.dw_o) + len(self.backward.dw_norm2)
                + len(self.backward.dw_gate) + len(self.backward.dw_up)
                + len(self.backward.dw_down))

    def copy_parameter_gradients(mut self, mut output: DeviceBuffer[DType.float32]) raises:
        """Owned-context output in norm1,q,k,v,o,norm2,gate,up,down order."""
        self._require_live()
        if self.state != OWNER_GRADIENT_READY or len(output) != self.parameter_gradient_elements():
            raise Error("NN32: fresh gradients and exact canonical output length required")
        var first = 0
        _copy_parameter_piece(self.context, output, self.backward.dw_norm1, first)
        first += len(self.backward.dw_norm1)
        _copy_parameter_piece(self.context, output, self.backward.dw_q, first)
        first += len(self.backward.dw_q)
        _copy_parameter_piece(self.context, output, self.backward.dw_k, first)
        first += len(self.backward.dw_k)
        _copy_parameter_piece(self.context, output, self.backward.dw_v, first)
        first += len(self.backward.dw_v)
        _copy_parameter_piece(self.context, output, self.backward.dw_o, first)
        first += len(self.backward.dw_o)
        _copy_parameter_piece(self.context, output, self.backward.dw_norm2, first)
        first += len(self.backward.dw_norm2)
        _copy_parameter_piece(self.context, output, self.backward.dw_gate, first)
        first += len(self.backward.dw_gate)
        _copy_parameter_piece(self.context, output, self.backward.dw_up, first)
        first += len(self.backward.dw_up)
        _copy_parameter_piece(self.context, output, self.backward.dw_down, first)

    def abort(mut self, mut budget: AttentionCheckpointBudget) raises:
        """Retry cleanup after a failed call; never make that owner reusable."""
        self._require_budget(budget)
        step_count_sync()
        self.context.synchronize()
        budget.release(self.owner, self.lease)
        self.lease = 0
        if self.registered_forward:
            budget.end_forward(self.owner)
            self.registered_forward = False
        self.forward = None
        self.forward_generation += 1
        self.state = OWNER_CLOSED

    def close(mut self, mut budget: AttentionCheckpointBudget) raises:
        self.invalidate(budget)
        self.state = OWNER_CLOSED


struct BorrowedAttentionTape(Movable):
    """Actual public-session tape; borrows context/weights only during calls.

    The binding owns the context and uploaded weight snapshot. It invalidates
    this tape BEFORE any weight/config/session operation. No saved host or
    Python pointers exist: input, rotary table, cache and stages are owned.
    One epoch labels forward/input/weight/config generations; backward checks
    each and consumes the ticket before enqueueing work. Failure is terminal.
    """
    var dims: LlamaDims
    var b: Int
    var l: Int
    var window: Int
    var forward_generation: Int
    var input_generation: Int
    var weight_generation: Int
    var config_generation: Int
    var live: Bool
    var retained: Bool
    var retained_bytes: Int
    var input: DeviceBuffer[DType.float32]
    var rope: LlamaRopeTable
    var cache: LlamaKVCache
    var stages: Optional[LlamaDeviceStages]

    def __init__(out self, ctx: DeviceContext, mut w: LlamaDeviceWeights,
                 var input: DeviceBuffer[DType.float32], b: Int, l: Int,
                 window: Int, generation: Int) raises:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL or b <= 0 or l <= 0 or window < 0:
            raise Error("NN31/NN32 tape requires IDENTICAL and positive full-prefill shape")
        if not w.opts.is_default() or w.int15:
            raise Error("NN31/NN32 tape requires default FP32 transformer options")
        if len(input) != b*l*w.dims.d_model or generation <= 0:
            raise Error("NN31/NN32 tape input shape or generation mismatch")
        self.dims = w.dims.copy()
        self.b = b
        self.l = l
        self.window = window
        self.forward_generation = generation
        self.input_generation = generation
        self.weight_generation = generation
        self.config_generation = generation
        self.live = True
        self.retained = False
        self.retained_bytes = 0
        self.input = input^
        self.rope = LlamaRopeTable(ctx,self.dims,w.opts,l)
        self.cache = LlamaKVCache(ctx,b,self.dims,l,window,w.opts.max_positions)
        self.stages = None
        self.replay(ctx,w)

    def replay(mut self, ctx: DeviceContext, mut w: LlamaDeviceWeights) raises:
        self.cache.s = 0
        self.cache.k.enqueue_fill(Float32(0.0))
        self.cache.v.enqueue_fill(Float32(0.0))
        self.stages = LlamaDeviceStages(ctx,self.b,self.l,self.l,self.dims,self.window,lean=True)
        var trace = IdentityTrace.disabled()
        llama_decoder_layer_forward(ctx,self.stages.value(),self.cache,self.rope,
            w,self.input,self.b,self.l,0,trace,String("tape"),forward_only=False)
        ctx.synchronize()

    def seal(mut self, ctx: DeviceContext, budget: Int, minimum_ops_per_byte: Int) raises:
        if budget < 0 or minimum_ops_per_byte < 0:
            raise Error("NN31 tape budget and cost threshold must be nonnegative")
        self.retained_bytes = _forward_bytes(self.stages.value())
        self.retained = attention_checkpoint_retain(self.retained_bytes,budget,
            attention_replay_operation_estimate(self.b,self.l,self.dims),minimum_ops_per_byte)
        ctx.synchronize()
        if not self.retained:
            self.stages = None
            ctx.synchronize()

    def backward_into(mut self, ctx: DeviceContext, mut w: LlamaDeviceWeights,
                      mut bst: LlamaBackwardStages, mut dy: DeviceBuffer[DType.float32],
                      generation: Int, input_generation: Int,
                      weight_generation: Int, config_generation: Int) raises:
        if (not self.live or generation != self.forward_generation
                or input_generation != self.input_generation
                or weight_generation != self.weight_generation
                or config_generation != self.config_generation):
            raise Error("NN31/NN32 stale, foreign or already consumed forward tape")
        self.live = False
        if not self.stages:
            self.replay(ctx,w)
        var trace = IdentityTrace.disabled()
        llama_decoder_layer_backward_device(ctx,bst,self.stages.value(),w,
            self.rope.cos,self.rope.sin,self.input,dy,self.b,self.l,0,trace,String("tape.bwd"))
        ctx.synchronize()
        self.stages = None
        ctx.synchronize()
