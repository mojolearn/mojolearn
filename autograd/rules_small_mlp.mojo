# SPDX-License-Identifier: Apache-2.0
# The small MLP train step's backward on the tape (the step's only backward).
# TAPE_EVERYWHERE_PLAN.md path 8, lane ag-smallmlp-tape.
#
# ---------------------------------------------------------------- launch inventory
# The hand path, training/mlp_ops.mojo::mlp_train_step_host after the loss
# (identical_ce_loss_resident already wrote dlogits), in launch order:
#   H1  dw2 = dlogits^T . act              _gemm[ROLE_WGRAD] (3, 16, rows) OP_TN
#   H2  db2 = sum_rows(dlogits)            _mlp_kernel op 3 (rows, 3)
#   H3  incoming = dlogits . w2            _gemm (rows, 16, 3) OP_NN
#   H4  dhidden = relu'(act) incoming      _mlp_kernel op 2 (rows, 16)
#   H5  dw1 = dhidden^T . x                _gemm[ROLE_WGRAD] (16, 8, rows) OP_TN
#   H6  db1 = sum_rows(dhidden)            _mlp_kernel op 3 (rows, 16)
#       refusal scan of the flat gradient (device_first_nonfinite, one wait)
#   H7  dx = dhidden . w1                  _gemm (rows, 8, 16) OP_NN, want_input_grad only
#       refusal scan of dx (one wait)
#
# The tape: the same seven kernels with the same arguments on the same
# buffers, in reverse creation order of the graph below:
#   T1 = H2 (bias node of the head), T2 = H1, T3 = H3 (head linear node),
#   T4 = H4, T5 = H6 (bias + ReLU node of the hidden layer), T6 = H5,
#   T7 = H7 (hidden linear node), then the two refusal scans.
# Each kernel writes its own destination from operands no other kernel of
# the backward writes after it is read, so the order change (db2 before
# dw2, db1 before dw1, dx before the gradient scan) moves no bit. The tape
# itself launches nothing: every node holds its gradient in the trainer's
# buffer (variable_into / custom_into / custom_into_shared), no input has two
# consumers, zeroed_destination is False (every kernel overwrites its whole
# destination) and the seed is the trainer's dlogits read in place.
# Launches: hand 6 (+1 dx) + 2 scans; tape 6 (+1 dx) + 2 scans. Waits: the
# same (the scans'). The tape records on SmallMlpRecordExec, which raises on
# any launch, allocation, copy or wait: so the tape provably adds none, and
# no executor drain (DeviceExec's destructor) enters the step.
#
# ReLU. The tape's own RELU op (OP_AG_UNARY_VJP: `g if x > 0 else 0`, g
# loaded through ftz) gives the same word as op 2 (`ftz(g) if act > 0 else
# 0`; act > 0 exactly when the pre-activation is), but its forward is a
# separate launch over a pre-activation buffer the fused bias + ReLU forward
# never writes. So the ReLU stays in the hand kernel, as the backward of the
# bias + ReLU node (SM_BIAS_ACT with i1 = 1).
#
# Loss. identical_ce_loss_resident writes dlogits in the forward (the LM
# rules' AFN_LM_HEAD_FUSE case, where LM_CE does nothing). The graph's root
# is the logits node and its seed is dlogits itself, held in place: no loss
# node, no seed buffer, no copy.
#
# ---------------------------------------------------------------- the tape graph
#   [x (variable_into dx, only when the input gradient is wanted)]
#   w1 (variable_into dw1)
#     -> P   SM_LINEAR layer 0   pre1 = x w1^T        grad storage dhidden
#   b1 (variable_into db1)
#     -> A   SM_BIAS_ACT relu    act = relu(pre1 + b1) grad storage incoming
#   w2 (variable_into dw2)
#     -> P2  SM_LINEAR layer 1   pre2 = act w2^T      grad storage dlogits
#   b2 (variable_into db2)
#     -> L   SM_BIAS_ACT plain   logits = pre2 + b2   grad shared with P2
#   seed: dlogits (the loss's gradient, written by the forward)
#
# Bits: same kernels, same operands, same buffers => the hand path's bits.
from max.gpu.host import DeviceBuffer

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from gemm.contract import OP_NN, OP_TN

# TOMBSTONE: MOJOLEARN_AUTOGRAD_SMALL_MLP_OFF (removed: hand glue superseded by the tape) deleted 2026-10-19 by 3cd0063d7; code recoverable at 3cd0063d7^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_SMALL_MLP_OFF.patch
# (the AUTOGRAD_SMALL_MLP switch and its opt-in spelling MOJOLEARN_AUTOGRAD_SMALL_MLP;
# training/checks/mlp_tape_step_check.mojo compares the tape against stored digests).

# Tags 120-121 (LayerNorm 1, recurrent 16-18, transformer 32-47, Mamba 48-63,
# LM 64-71, Mamba-3 72-86, CNN 96-111, MoE 112-115).
comptime SM_LINEAR = 120
comptime SM_BIAS_ACT = 121

comptime SmallMlpBuffer = MutPointer[DeviceBuffer[DType.float32], MutUntrackedOrigin]


@always_inline
def small_mlp_buffer(mut buf: DeviceBuffer[DType.float32]) -> SmallMlpBuffer:
    """A pointer to a trainer-owned buffer; the buffer must outlive its use."""
    return SmallMlpBuffer(unsafe_from_address=Int(MutPointer(to=buf)))


@always_inline
def _fp(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


def _same(p: FP, mut buf: DeviceBuffer[DType.float32], what: String) raises:
    # The rules launch on the trainer's own buffers; the tape must hand them
    # exactly those (the driver records them so). Anything else is a driver bug.
    if Int(p) != Int(buf.unsafe_ptr()):
        raise Error(String("small MLP backward (autograd): ") + what + " is not the trainer's buffer")


trait SmallMlpKernels(Movable & Deinitable):
    """The two launches the small MLP's hand path makes, as the trainer
    makes them (its GEMM dispatch, its workspace and its context), so the
    rules launch exactly the hand kernels without importing the trainer."""

    def gemm(
        self, mut c: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
        mut b: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int, op: Int, wgrad: Bool,
    ) raises:
        """C = op(A, B) through the trainer's GEMM; `wgrad` selects its
        weight-gradient role."""
        ...

    def cells(self, source: FP, other: FP, output: FP, rows: Int, cols: Int, operation: Int) raises:
        """The trainer's elementwise kernel (2: ReLU backward, 3: row sum)."""
        ...


struct SmallMlpRules[K: SmallMlpKernels](BackwardRules):
    """Backward adapters for the small MLP's two node kinds: a linear layer
    without bias (dW, and dX when its input needs one) and a bias add with
    optional ReLU (the ReLU backward, and the bias row sum).

    Per layer (0 hidden, 1 head) it holds pointers to the layer's input,
    weight, output gradient, weight gradient and input gradient buffers;
    all must outlive the backward."""

    var k: Self.K
    var batch: Int
    var xin: List[SmallMlpBuffer]
    var w: List[SmallMlpBuffer]
    var dz: List[SmallMlpBuffer]
    var dw: List[SmallMlpBuffer]
    var dx: List[SmallMlpBuffer]

    def __init__(
        out self, var k: Self.K, batch: Int, var xin: List[SmallMlpBuffer], var w: List[SmallMlpBuffer],
        var dz: List[SmallMlpBuffer], var dw: List[SmallMlpBuffer], var dx: List[SmallMlpBuffer],
    ):
        self.k = k^
        self.batch = batch
        self.xin = xin^
        self.w = w^
        self.dz = dz^
        self.dw = dw^
        self.dx = dx^

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every kernel overwrites every element of its destination
        # (the hand path hands them unfilled buffers), and the shared edge of
        # the plain bias node is written by nobody (it IS the upstream).
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # No input has two consumers: every gradient is one contribution,
        # written in place as a plain first contribution.
        return False

    def supports(self, tag: Int) -> Bool:
        return tag == SM_LINEAR or tag == SM_BIAS_ACT

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on the trainer's context, the one `ex` wraps.
        var layer = saved.i0
        if layer < 0 or layer >= len(self.dz):
            raise Error("small MLP backward (autograd): node carries no valid layer index")
        if tag == SM_LINEAR:
            self._linear(layer, upstream, gradients, rows, cols)
        elif tag == SM_BIAS_ACT:
            self._bias_act(layer, saved.i1, output, upstream, gradients, cols)
        else:
            raise Error("unknown small MLP backward rule")

    def _linear(self, layer: Int, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # forward Z [B, out] = X [B, in] . W [out, in]^T (OP_NT): the hand
        # path's dW (OP_TN, weight-gradient role) then dX (OP_NN)
        var last = len(gradients) - 1
        var n_out = rows[last]
        var n_in = cols[last]
        _same(upstream, self.dz[layer][], "the layer output gradient")
        _same(gradients[last], self.dw[layer][], "the weight gradient")
        self.k.gemm(self.dw[layer][], self.dz[layer][], self.xin[layer][], n_out, n_in, self.batch, OP_TN, True)
        if last == 1:
            _same(gradients[0], self.dx[layer][], "the layer input gradient")
            self.k.gemm(self.dx[layer][], self.dz[layer][], self.w[layer][], self.batch, n_in, n_out, OP_NN, False)

    def _bias_act(
        self, layer: Int, relu: Int, output: FP, upstream: FP, gradients: List[FP], cols: List[Int],
    ) raises:
        var n = cols[1]
        if relu == 1:
            # dZ = relu'(act) upstream, read from the forward output act
            # (op 2), then the bias row sum of dZ (op 3; its `other` operand
            # is never read: the upstream, as the hand path passes)
            _same(gradients[0], self.dz[layer][], "the pre-activation gradient")
            self.k.cells(output, upstream, gradients[0], self.batch, n, 2)
            self.k.cells(gradients[0], upstream, gradients[1], self.batch, n, 3)
        else:
            # the identity edge shares the upstream's storage (written by
            # the forward's loss); only the bias row sum runs. Its unread
            # `other` operand is the next layer down's input gradient, as
            # the hand path passes.
            _same(gradients[0], self.dz[layer][], "the shared output gradient")
            self.k.cells(upstream, _fp(self.dx[layer][]), gradients[1], self.batch, n, 3)


struct SmallMlpRecordExec(Exec, Movable):
    """The executor the small MLP's tape records on. The graph holds only
    caller storage and its rules launch on the trainer's context, so the
    tape needs no storage, launch, copy or wait of its own: every method
    raises, which keeps it that way."""

    def __init__(out self):
        pass

    def alloc(mut self, n: Int) raises -> FP:
        raise Error(_RECORD_ONLY)

    def free(mut self, p: FP) raises:
        raise Error(_RECORD_ONLY)

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        raise Error(_RECORD_ONLY)

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        raise Error(_RECORD_ONLY)

    def download_async(mut self, dst: FP, src: FP, n: Int) raises:
        raise Error(_RECORD_ONLY)

    def bind(mut self, src: FP, n: Int) raises -> FP:
        raise Error(_RECORD_ONLY)

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        raise Error(_RECORD_ONLY)

    def sync(mut self) raises:
        raise Error(_RECORD_ONLY)


comptime _RECORD_ONLY = "small MLP backward (autograd): the tape asked its record-only executor for storage, a launch, a copy or a wait"


@always_inline
def _at(layer: Int, relu: Int) -> Args:
    var a = Args()
    a.i0 = layer
    a.i1 = relu
    return a


def small_mlp_backward_autograd[E: Exec, K: SmallMlpKernels](
    mut ex: E,
    var kernels: K,
    mut x: DeviceBuffer[DType.float32],
    mut w1: DeviceBuffer[DType.float32],
    mut b1: DeviceBuffer[DType.float32],
    mut w2: DeviceBuffer[DType.float32],
    mut b2: DeviceBuffer[DType.float32],
    mut pre1: DeviceBuffer[DType.float32],
    mut act: DeviceBuffer[DType.float32],
    mut pre2: DeviceBuffer[DType.float32],
    mut logits: DeviceBuffer[DType.float32],
    mut dlogits: DeviceBuffer[DType.float32],
    mut dw1: DeviceBuffer[DType.float32],
    mut db1: DeviceBuffer[DType.float32],
    mut dw2: DeviceBuffer[DType.float32],
    mut db2: DeviceBuffer[DType.float32],
    mut incoming: DeviceBuffer[DType.float32],
    mut dhidden: DeviceBuffer[DType.float32],
    mut dx: DeviceBuffer[DType.float32],
    batch: Int,
    n_in: Int,
    n_hid: Int,
    n_out: Int,
    want_input_grad: Bool,
) raises:
    """The small MLP's backward on one tape, recorded over the forward that
    already ran (`dlogits` holds the loss's gradient), then one backward.

    Writes exactly what the hand chain writes, with the same kernels on the
    same buffers: dw2, db2, incoming, dhidden, dw1, db1 and, when
    `want_input_grad`, dx. Queues the launches on the executor's context
    and does not wait."""
    if batch <= 0 or n_in <= 0 or n_hid <= 0 or n_out <= 0:
        raise Error("small MLP backward (autograd): dimensions must be positive")
    if len(dw1) != n_hid * n_in or len(db1) != n_hid or len(dw2) != n_out * n_hid or len(db2) != n_out:
        raise Error("small MLP backward (autograd): the gradient buffers do not match the layer shapes")
    if len(dlogits) != batch * n_out or len(incoming) != batch * n_hid or len(dhidden) != batch * n_hid:
        raise Error("small MLP backward (autograd): the stage buffers do not match the batch")
    if want_input_grad and len(dx) != batch * n_in:
        raise Error("small MLP backward (autograd): the input gradient buffer does not match the batch")

    var tape = Tape()
    # ---- hidden layer: pre1 = x w1^T, act = relu(pre1 + b1)
    var hidden_in = List[Tensor]()
    if want_input_grad:
        hidden_in.append(tape.variable_into(ex, _fp(x), _fp(dx), batch, n_in))
    hidden_in.append(tape.variable_into(ex, _fp(w1), _fp(dw1), n_hid, n_in))
    var p = tape.custom_into(ex, hidden_in, _fp(pre1), _fp(dhidden), batch, n_hid, SM_LINEAR, _at(0, 0))
    var g_b1 = tape.variable_into(ex, _fp(b1), _fp(db1), 1, n_hid)
    var a = tape.custom_into(ex, [p, g_b1], _fp(act), _fp(incoming), batch, n_hid, SM_BIAS_ACT, _at(0, 1))
    # ---- head: pre2 = act w2^T, logits = pre2 + b2 (gradient shared with pre2)
    var g_w2 = tape.variable_into(ex, _fp(w2), _fp(dw2), n_out, n_hid)
    var p2 = tape.custom_into(ex, [a, g_w2], _fp(pre2), _fp(dlogits), batch, n_out, SM_LINEAR, _at(1, 0))
    var g_b2 = tape.variable_into(ex, _fp(b2), _fp(db2), 1, n_out)
    var lg = tape.custom_into_shared(ex, [p2, g_b2], _fp(logits), p2, batch, n_out, SM_BIAS_ACT, _at(1, 0))
    # ---- the seed: dlogits in place (the root's own gradient storage)
    var seed = tape.variable(ex, _fp(dlogits), batch, n_out, False)

    var xin = List[SmallMlpBuffer]()
    xin.append(small_mlp_buffer(x))
    xin.append(small_mlp_buffer(act))
    var ws = List[SmallMlpBuffer]()
    ws.append(small_mlp_buffer(w1))
    ws.append(small_mlp_buffer(w2))
    var dz = List[SmallMlpBuffer]()
    dz.append(small_mlp_buffer(dhidden))
    dz.append(small_mlp_buffer(dlogits))
    var dws = List[SmallMlpBuffer]()
    dws.append(small_mlp_buffer(dw1))
    dws.append(small_mlp_buffer(dw2))
    var dxs = List[SmallMlpBuffer]()
    dxs.append(small_mlp_buffer(dx))
    dxs.append(small_mlp_buffer(incoming))
    var rules = SmallMlpRules[K](kernels^, batch, xin^, ws^, dz^, dws^, dxs^)
    tape.backward_with_rules(ex, lg, seed, rules)
    tape.release(ex)
