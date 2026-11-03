# SPDX-License-Identifier: Apache-2.0
# One device's executor for captured programs. The distributed
# (autograd/distributed_program.mojo) and partitioned
# (autograd/partition_stage.mojo) programs compose one ProgramExecutor per
# physical device and reach the session only through the methods below; the
# executor owns its OpsSession and the temporary handles of the graph it is
# building. Every method forwards to the same tape/session call the programs
# made before, with the same arguments in the same order: no arithmetic,
# launch or readback is added here. Raw device work (allocation, byte
# transfers, fold launches, optimizer steps over owned buffers) stays on the
# device executor `session.ex`; nothing else of the session is touched from
# outside this file.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_FILL
from autograd.ops_api import OpsSession
from autograd.program import OpsProgram
from autograd.program_runtime import replay_values, replay_values_prepared, program_loss
from autograd.program_replay_metadata import PreparedProgramReplay
from autograd.rankn import ShapeN
from autograd.tape import Tensor
from autograd.import_api import ImportRules
from core.cast_in import CastIn


struct ProgramExecutor[E: Exec & Movable & Deinitable](Movable):
    """One device: an inference-mode OpsSession over `E` and the temporary
    handles of the graph currently built on it."""
    var session: OpsSession[Self.E]
    var temps: List[Int]

    def __init__(out self, var ex: Self.E):
        self.session = OpsSession[Self.E](ex^, False)
        self.temps = List[Int]()

    # ------------------------------------------------------------ parameters
    def load_param(mut self, src: CastIn, shape: ShapeN) raises -> Int:
        """Upload one parameter (no gradient on the session itself); returns
        its session handle."""
        return self.session.upload_nd(src, shape, False)

    def param_buffer(self, h: Int) raises -> FP:
        """The device buffer that holds uploaded parameter h."""
        return self.session.buffer(h)

    # ------------------------------------------------------------ graph lifetime
    def clear_temps(mut self):
        """Retire the temporary handles of the last graph."""
        self.session.drop_temps(self.temps)
        self.temps.clear()

    def clear(mut self):
        """Retire the last graph's temporaries and reset the tape."""
        self.clear_temps()
        self.session.tape.reset()

    def clear_log(mut self):
        """Forget the inference operation log."""
        self.session.oplog.clear()

    def record(mut self):
        self.session.tape.record()

    def no_grad(mut self):
        self.session.tape.no_grad()

    # ------------------------------------------------------------ building
    def forward(
        mut self, p: OpsProgram, mut replay: PreparedProgramReplay, inputs: List[FP],
        shapes: List[ShapeN], params: List[FP], trainable: List[Bool],
        mut leaves: List[Int], stream: Int,
    ) raises -> List[Int]:
        """Replay prepared program p on the current tape; `leaves` receives
        one parameter leaf handle per parameter. Returns the output handles."""
        return replay_values_prepared(self.session, p, replay, inputs, shapes, params,
                                      trainable, self.temps, leaves, stream)

    def evaluate(
        mut self, p: OpsProgram, inputs: List[FP], shapes: List[ShapeN],
        params: List[FP], trainable: List[Bool], mut leaves: List[Int],
    ) raises -> List[Int]:
        """One-shot replay of program p (inference). Returns the output handles."""
        return replay_values(self.session, p, inputs, shapes, params, trainable,
                             self.temps, leaves)

    def evaluate_node(mut self, code: Int, hs: List[Int], ints: List[Int],
                      fs: List[Float32]) raises -> Int:
        """Run one captured operation on handles hs (not logged)."""
        var h = self.session.run_op(code, hs, ints, fs)
        self.temps.append(h)
        return h

    def leaf(mut self, buf: FP, shape: ShapeN, requires_grad: Bool,
             integer_words: Bool) raises -> Int:
        """A temporary leaf over the caller's device buffer."""
        var h = self.session.leaf_nd(buf, shape, requires_grad, integer_words)
        self.temps.append(h)
        return h

    def target(mut self, buf: FP, rows: Int, cols: Int) raises -> Tensor:
        """A constant (rows, cols) target leaf over the caller's buffer."""
        return self.session.tape.variable(self.session.ex, buf, rows, cols, False)

    # ------------------------------------------------------------ inspection
    def tensor(self, h: Int) raises -> Tensor:
        return self.session.tensor(h)

    def shape(self, h: Int) raises -> ShapeN:
        return self.session.shape_nd(h)

    def is_integer(self, h: Int) raises -> Bool:
        return self.session.is_integer(h)

    def requires_grad(self, t: Tensor) raises -> Bool:
        return self.session.tape.requires_grad(t)

    def read(mut self, h: Int, dst: FP) raises:
        """Tensor h's values into host memory `dst`."""
        self.session.download(h, dst)

    # ------------------------------------------------------------ loss and gradients
    def loss(mut self, output: Tensor, target: Tensor, kind: Int, delta: Float32) raises -> Tensor:
        return program_loss(self.session.tape, self.session.ex, output, target, kind, delta)

    def read_loss(mut self, loss: Tensor, dst: FP) raises:
        """The scalar loss value into host memory `dst` (one word)."""
        self.session.ex.download(dst, self.session.tape.value(loss), 1)

    def backward(mut self, loss: Tensor) raises:
        """Differentiate a scalar loss."""
        self.session.tape.backward_scalar_with_rules(self.session.ex, loss, ImportRules())

    def backward_seeded(mut self, output: Tensor, seed: FP, shape: ShapeN) raises:
        """Differentiate `output` against the adjoint held in device buffer `seed`."""
        var seed_tensor = self.session.tape.variable_nd(self.session.ex, seed, shape, False)
        self.session.tape.backward_with_rules(self.session.ex, output, seed_tensor, ImportRules(), False)

    def has_gradient(self, h: Int) raises -> Bool:
        """Whether the last backward reached tensor h."""
        return self.session.has_gradient(h)

    def gradient(self, t: Tensor) raises -> FP:
        return self.session.tape.gradient(t)

    def parameter_gradient(mut self, h: Int, n: Int) raises -> FP:
        """Parameter leaf h's gradient, or an explicit zero gradient of n
        words when no differentiable path reached it (detach/copy)."""
        var parameter = self.session.tensor(h)
        if self.session.has_gradient(h):
            return self.session.tape.gradient(parameter)
        var zero = self.session.tape.reserve(self.session.ex, n)
        var fill = Args()
        fill.p0 = zero
        fill.f0 = Float32(0)
        self.session.ex.launch[OP_FILL](fill, n)
        return zero

    # ------------------------------------------------------------ lifetime
    def close(mut self) raises:
        """Return every session buffer to the device."""
        self.session.close()
