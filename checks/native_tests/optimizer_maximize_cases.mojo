# SPDX-License-Identifier: Apache-2.0
"""Native maximize correctness seam, independent sign-bit oracle.

Four optimizer configurations, clipping off/on, two seed labels and three
steps preserve test_optim_maximize_seam.py's semantic matrix. Native dyadic
fixtures retain all three tensor extents and planted signed zeros. The
existing final-verifier driver/registration remains until a coordinated switch;
this check is correctness coverage, not an identity run or release verdict.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from training.optimizer_public_common import PublicStep, PublicStepOut, OptHyper

comptime MAXIMIZE_FIXTURES = 16
comptime MX_N = 192
comptime MX_STEPS = 3
comptime MX_GUARD = Float32(31415.5)


trait OptimizerTestKernels:
    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        ...

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        ...

    @staticmethod
    def close(handle: Int) raises -> Int:
        ...


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def sign_flip(value: Float32) -> Float32:
    # Independent definition; never calls training.maximize or another
    # product numerical helper to manufacture the expected gradient.
    return bitcast[DType.float32](bitcast[DType.uint32](value) ^ UInt32(0x80000000))


@fieldwise_init
struct TensorOwner(Movable):
    var cells: List[Float32]
    var count: Int

    def address(self) -> Int:
        return Int(self.cells.unsafe_ptr()) + 8

    def check_guards(self) raises:
        require(self.cells[0] == MX_GUARD and self.cells[1] == MX_GUARD and
            self.cells[self.count + 2] == MX_GUARD and self.cells[self.count + 3] == MX_GUARD,
            "maximize wrote outside caller tensor")


def tensor_values(seed: Int, tensor: Int, count: Int, epoch: Int, parameter: Bool, spelling: Int) -> TensorOwner:
    var cells = List[Float32](length=count + 4, fill=MX_GUARD)
    var cp = cells.unsafe_ptr()
    def cell(index: Int) {imm cp, imm seed, imm tensor, imm epoch, imm parameter, imm spelling}:
        # Independent native fixture preparation, not product RNG. The two
        # original seed labels select distinct data with the same sign/range
        # and signed-zero conditions; no old random bytes/hashes are required.
        var word = ((index + 1) * 37 + seed * 19 + tensor * 53 + (epoch + 1) * 71) % 257 - 128
        var value = Float32(word) / (Float32(256) if parameter else Float32(128))
        if index < 3:
            value = bitcast[DType.float32](UInt32(0x80000000)) if parameter or index == 1 else Float32(0)
        if not parameter:
            if spelling == 1:
                value = sign_flip(value)
            elif spelling == 2:
                # IEEE +0 - (+/-0) is +0; all other finite inputs equal -g.
                # State the alternative in words so the test compiler cannot
                # erase the signed-zero distinction by rewriting subtraction.
                value = Float32(0) if (bitcast[DType.uint32](value) & UInt32(0x7fffffff)) == 0 else sign_flip(value)
        cp.unsafe_store(index + 2, value)
    host_parallelize(cell, count)
    return TensorOwner(cells^, count)


def hyperparameters(config: Int, maximize: Bool, clip: Bool) -> OptHyper:
    # Original SGD Nesterov+decay, plain SGD, Adam decay, AdamW settings.
    if config == 0:
        return OptHyper(0, 1, maximize, 0.01, 0.9, 0.999, 1e-8, 0.01, 0.9, 0, Float64(1 if clip else 0))
    if config == 1:
        return OptHyper(0, 0, maximize, 0.01, 0.9, 0.999, 1e-8, 0, 0, 0, Float64(1 if clip else 0))
    return OptHyper(1 if config == 2 else 2, 0, maximize, 0.001, 0.9, 0.999, 1e-8, 0.01, 0, 0, Float64(1 if clip else 0))


@fieldwise_init
struct RunResult(Movable):
    var state: List[UInt32]
    var gradients: List[UInt32]
    var norms: List[Float32]


def run_optimizer[K: OptimizerTestKernels](config: Int, seed: Int, maximize: Bool, spelling: Int, clip: Bool) raises -> RunResult:
    var sizes: List[Int] = [128, 16, 48]  # (16,8), (16,), (3,16)
    var offsets: List[Int32] = [0, 128, 144, 192]
    var initialized = List[Int32](length=3, fill=0)
    var first = List[Float32](length=MX_N, fill=0)
    var second = List[Float32](length=MX_N, fill=0)
    var parameters = List[TensorOwner]()
    for tensor in range(3):  # driver-loop: three independent parameter owners
        parameters.append(tensor_values(seed, tensor, sizes[tensor], 0, True, 0))
    var after = List[UInt32](length=MX_STEPS * MX_N, fill=0)
    var norms = List[Float32](length=MX_STEPS, fill=0)
    var state = List[UInt32](length=3 * MX_N + 3, fill=0)
    var handle = -1
    var executed = 0
    try:
        for epoch in range(MX_STEPS):  # dependent optimizer state recurrence
            var gradients = List[TensorOwner]()
            for tensor in range(3):
                gradients.append(tensor_values(seed, tensor, sizes[tensor], epoch, False, spelling))
            var req = PublicStep()
            req.j = 3
            req.t = executed
            req.hyper = hyperparameters(config, maximize, clip)
            req.n_total = MX_N
            req.offsets = Int(offsets.unsafe_ptr())
            req.init = Int(initialized.unsafe_ptr())
            req.m_addr = Int(first.unsafe_ptr())
            req.v_addr = Int(second.unsafe_ptr())
            req.handle = handle
            req.pristine = epoch == 0
            # Keep the original automatic resident choice; the harness closes
            # any returned native owner. Do not hide pool races by serialization.
            req.resident_wanted = -1
            for tensor in range(3):
                req.p_addr.append(parameters[tensor].address())
                req.p_bytes.append(4 * sizes[tensor])
                req.g_addr.append(gradients[tensor].address())
                req.g_bytes.append(4 * sizes[tensor])
            var result = K.step(req)
            handle = result.handle
            executed = result.t
            require(executed == epoch + 1, "maximize step count")
            require(result.clip_ran == clip, "maximize clipping report")
            norms[epoch] = result.total_norm
            for tensor in range(3):
                gradients[tensor].check_guards()
                parameters[tensor].check_guards()
                for index in range(sizes[tensor]):
                    after[epoch * MX_N + Int(offsets[tensor]) + index] = bitcast[DType.uint32](gradients[tensor].cells[index + 2])
            _ = gradients
        if handle >= 0:
            _ = K.download(handle, Int(first.unsafe_ptr()), Int(second.unsafe_ptr()), MX_N)
        for tensor in range(3):
            for index in range(sizes[tensor]):
                state[Int(offsets[tensor]) + index] = bitcast[DType.uint32](parameters[tensor].cells[index + 2])
            state[3 * MX_N + tensor] = UInt32(initialized[tensor])
        for index in range(MX_N):
            state[MX_N + index] = bitcast[DType.uint32](first[index])
            state[2 * MX_N + index] = bitcast[DType.uint32](second[index])
    except error:
        if handle >= 0:
            _ = K.close(handle)
        raise error
    if handle >= 0:
        _ = K.close(handle)
    _ = parameters
    _ = offsets
    _ = initialized
    _ = first
    _ = second
    return RunResult(state^, after^, norms^)


def same_words(first: List[UInt32], second: List[UInt32]) -> Bool:
    if len(first) != len(second):
        return False
    for index in range(len(first)):
        if first[index] != second[index]:
            return False
    return True


def run_maximize_fixture[K: OptimizerTestKernels](fixture_id: Int) raises:
    var config = fixture_id // 4
    var clip = (fixture_id // 2) % 2 != 0
    var seed = 7 if fixture_id % 2 == 0 else 11
    var got = run_optimizer[K](config, seed, True, 0, clip)
    var wanted = run_optimizer[K](config, seed, False, 1, clip)
    require(same_words(got.state, wanted.state), "maximize state differs from independent sign-bit oracle")
    for epoch in range(MX_STEPS):
        require(got.norms[epoch] == wanted.norms[epoch], "maximize pre-clip norm differs")
    if config < 2 and not clip and seed == 7:
        var alternative = run_optimizer[K](config, seed, False, 2, False)
        require(not same_words(wanted.state, alternative.state), "VACUOUS: signed-zero fixture does not separate subtraction from sign flip")
    var sizes: List[Int] = [128, 16, 48]
    var offsets: List[Int] = [0, 128, 144]
    for epoch in range(MX_STEPS):
        for tensor in range(3):
            var original = tensor_values(seed, tensor, sizes[tensor], epoch, False, 0)
            for index in range(sizes[tensor]):
                var flat = epoch * MX_N + offsets[tensor] + index
                var expected = wanted.gradients[flat] ^ UInt32(0x80000000) if clip else bitcast[DType.uint32](original.cells[index + 2])
                require(got.gradients[flat] == expected, "maximize altered caller gradient beyond ordinary clipping")
