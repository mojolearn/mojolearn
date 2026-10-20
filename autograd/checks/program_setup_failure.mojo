# SPDX-License-Identifier: Apache-2.0
# Testing-only executor faults. Arithmetic and storage use the real HostExec;
# failures occur at its ordinary fallible allocation/upload/launch boundary.
from sequence.exec import HostExec
from sequence.exec_trait import Exec
from sequence.execution_profile import ExecutionProfile
from sequence.ops import FP, Args, OPT_SGD
from autograd.ops_api import OpsSession, OPS_MATMUL
from autograd.program_runtime import train_program_control
from autograd.rankn import ShapeN
from autograd.cast_ops import upload_cast, upload_cast_i32
from core.cast_in import CastIn, CAST_KIND_FLOAT, CAST_KIND_SIGNED


struct FaultExec(Exec, Movable):
    var host: HostExec
    var kind: Int
    var at: Int
    var seen: Int
    var hit: Bool

    def __init__(out self):
        self.host = HostExec()
        self.kind = 0
        self.at = 0
        self.seen = 0
        self.hit = False

    def arm(mut self, kind: Int, at: Int):
        self.kind = kind
        self.at = at
        self.seen = 0
        self.hit = False

    def fault(mut self, kind: Int) raises:
        if kind == self.kind:
            self.seen += 1
            if self.seen == self.at:
                self.hit = True
                raise Error("injected setup failure")

    def alloc(mut self, n: Int) raises -> FP:
        self.fault(1)
        return self.host.alloc(n)

    def free(mut self, p: FP) raises:
        self.host.free(p)

    def upload(mut self, dst: FP, src: FP, n: Int) raises:
        self.fault(2)
        self.host.upload(dst, src, n)

    def download(mut self, dst: FP, src: FP, n: Int) raises:
        self.host.download(dst, src, n)

    def download_async(mut self, dst: FP, src: FP, n: Int) raises:
        self.host.download_async(dst, src, n)

    def bind(mut self, src: FP, n: Int) raises -> FP:
        return self.host.bind(src, n)

    def launch[OP: Int](mut self, a: Args, n: Int) raises:
        self.fault(3)
        self.host.launch[OP](a, n)

    def sync(mut self) raises:
        self.host.sync()

    def execution_profile_begin(mut self) raises:
        self.host.execution_profile_begin()

    def execution_profile_end(mut self) raises -> ExecutionProfile:
        return self.host.execution_profile_end()


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def one(value: Int) -> List[Int]:
    var result = List[Int]()
    result.append(value)
    return result^


def check_training_fault(kind: Int, at: Int, conditional: Bool) raises:
    var s = OpsSession[FaultExec](FaultExec(), False)
    var input = List[Float32](length=2, fill=1.0)
    input[1] = 2.0
    var weight = List[Float32](length=1, fill=2.0)
    var target = List[Float32](length=2, fill=0.0)
    var curve = List[Float32](length=1, fill=0.0)
    var src = CastIn.f32_c(Int(input.unsafe_ptr()), 2, 1)
    var dst = CastIn.f32_c(Int(target.unsafe_ptr()), 2, 1)
    var x = s.upload_nd(src, ShapeN.of([2, 1]), False)
    var w = s.upload_nd(CastIn.f32_c(Int(weight.unsafe_ptr()), 1, 1), ShapeN.of([1, 1]), True)
    var operands = one(x)
    operands.append(w)
    var output = s.op(OPS_MATMUL, operands, List[Int](), List[Float32]())
    var pid = s.capture(one(x), one(output))
    var condition = s.capture(one(x), one(w)) if conditional else 0
    var otherwise = s.capture(one(x), one(output)) if conditional else 0
    var integers = one(OPT_SGD)
    integers.append(0)
    integers.append(0)
    integers.append(1)
    var scalars = List[Float32](length=7, fill=0.0)
    scalars[0] = 0.01
    scalars[2] = 0.9
    scalars[3] = 0.999
    scalars[4] = 1e-8
    scalars[6] = 1.0
    var opt = s.optimizer(integers, scalars)
    var sources = List[CastIn]()
    sources.append(src)
    var shapes = List[ShapeN]()
    shapes.append(ShapeN.of([2, 1]))
    var before = len(s.ex.host.bufs)
    s.ex.arm(kind, at)
    var failed = False
    try:
        _ = train_program_control(s, pid, sources, shapes, dst, 1, Float32(1.0), opt, one(w),
            1, 1, False, UInt64(0), FP(unsafe_from_address=Int(curve.unsafe_ptr())), condition, otherwise)
    except error:
        require(String(error) == "injected setup failure", "unexpected training error: " + String(error))
        failed = True
    require(failed and s.ex.hit, "requested setup failure was not reached")
    s.ex.arm(0, 0)
    # Fully registered optimizer state is owned by the session and reusable;
    # every dataset/scratch buffer and every partial state must be released.
    require(len(s.ex.host.bufs) == before + 5 * len(s.opt_states), "setup left an unowned executor buffer")
    require(not s.tape.is_recording(), "setup failure changed the caller's tape mode")
    for i in range(len(s.opt_states)):
        require(s.opt_states[i].t == 0, "setup failure advanced optimizer state")
    var result = List[Float32](length=2, fill=0.0)
    var outputs = List[FP]()
    outputs.append(FP(unsafe_from_address=Int(result.unsafe_ptr())))
    s.run_program(pid, sources, outputs)
    require(result[0] == 2.0 and result[1] == 4.0, "session replay changed after setup failure")
    s.close()
    require(len(s.ex.host.bufs) == 0, "session close retained native buffers")
    _ = input^
    _ = weight^
    _ = target^
    _ = curve^
    _ = result^


def check_cast_fault(integer: Bool, kind: Int) raises:
    var ex = FaultExec()
    var floats = List[Float64](length=2, fill=1.0)
    var integers = List[Int64](length=2, fill=1)
    var dst = ex.alloc(2)
    var source = CastIn(Int(integers.unsafe_ptr()) if integer else Int(floats.unsafe_ptr()),
                        CAST_KIND_SIGNED if integer else CAST_KIND_FLOAT, 8, 2, 1, 8, 8)
    ex.arm(kind, 1)
    var failed = False
    try:
        if integer:
            upload_cast_i32(ex, dst, source, "ids")
        else:
            upload_cast(ex, dst, source)
    except error:
        require(String(error) == "injected setup failure", "unexpected cast error: " + String(error))
        failed = True
    require(failed and ex.hit, "cast fault was not reached")
    require(len(ex.host.bufs) == 1, "failed cast retained its temporary raw buffer")
    ex.arm(0, 0)
    ex.free(dst)
    _ = floats^
    _ = integers^


def main() raises:
    var cases = 0
    for conditional in range(2):
        # Dataset setup plus all five optimizer-state allocations and the
        # final index/loss buffers; every failure precedes the first update.
        for at in range(1, 13):
            check_training_fault(1, at, conditional != 0)
            cases += 1
        for at in range(1, 3):
            check_training_fault(2, at, conditional != 0)
            cases += 1
    for integer in range(2):
        for kind in range(2, 4):
            check_cast_fault(integer != 0, kind)
            cases += 1
    print("PASS program setup exception ownership", cases, "fault cases")
