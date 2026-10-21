# SPDX-License-Identifier: Apache-2.0
# Typed entries behind program_pyapi.mojo: captured-program execution
# (uploads, the replay, host results and the optional profile counters) and
# the native-options fit's parameter selection. The Python door converts
# arguments and builds results only.
from std.time import perf_counter_ns
from sequence.exec_trait import Exec
from sequence.execution_profile import ExecutionProfile
from sequence.ops import FP
from core.cast_in import CastIn
from autograd.rankn import ShapeN
from autograd.tape import Tape
from autograd.cast_ops import upload_cast, upload_cast_i32
from autograd.ops_api import ops_session
from autograd.program import OpsProgram, PROGRAM_IN_IDS
from autograd.program_io import HostArray, ProgramInput
from autograd.program_tools import ProgramTrace
from autograd.program_optimize import ProgramOptimization, program_optimize
from autograd.program_runtime import replay_values, parameter_buffers, train_program_control


struct ProgramExecPlan(Movable):
    var program: OpsProgram
    var optimization: ProgramOptimization
    var original_nodes: Int

    def __init__(out self, var program: OpsProgram, var optimization: ProgramOptimization, original_nodes: Int):
        self.program = program^
        self.optimization = optimization^
        self.original_nodes = original_nodes


struct ProgramExecResult(Movable):
    var outputs: List[HostArray]
    var executor: ExecutionProfile
    var upload_bytes: Int
    var input_bytes: Int
    var download_bytes: Int
    var parameter_bytes: Int
    var upload_ns: Int
    var download_ns: Int

    def __init__(out self, var executor: ExecutionProfile):
        self.outputs = List[HostArray]()
        self.executor = executor^
        self.upload_bytes = 0
        self.input_bytes = 0
        self.download_bytes = 0
        self.parameter_bytes = 0
        self.upload_ns = 0
        self.download_ns = 0


def program_execute_prepare[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, owner_sid: Int, optimize: Bool, profile: Bool,
) raises -> ProgramExecPlan:
    if owner_sid != sid:
        raise Error("program must be a Program of this session")
    var s = ops_session[E, KIND](sid)
    var p = s[].program(pid)
    if s[].tape.is_recording():
        raise Error("program execution requires an inference session")
    var original_nodes = p.n_nodes()
    var optimization = ProgramOptimization(OpsProgram())
    if optimize or profile:
        optimization = program_optimize(p, optimize)
        p = optimization.program.copy()
    return ProgramExecPlan(p^, optimization^, original_nodes)


def program_execute_run[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, plan: ProgramExecPlan, var inputs: List[ProgramInput], profile: Bool,
) raises -> ProgramExecResult:
    """Upload `inputs`, replay the prepared program without recording and
    download every output to the host. With `profile`, the executor profile
    and the byte and time counters are filled; the per-node rows stay in
    the session's program_trace."""
    var s = ops_session[E, KIND](sid)
    var p = plan.program.copy()
    # Validate every descriptor and parameter before allocation. Earlier
    # dynamic replay leaked input buffers on a later input/parameter error.
    var params = parameter_buffers(s[], p)
    var trainable = List[Bool](length=len(params), fill=False)
    var buffers = List[FP]()
    var shapes = List[ShapeN]()
    var upload_bytes = 0
    var input_bytes = 0
    var parameter_bytes = 0
    for j in range(len(p.param_shapes)):  # driver-loop: over parameter storage accounting
        parameter_bytes += p.param_shapes[j].count() * 4
    s[].program_trace = ProgramTrace()
    s[].program_trace.enabled = profile
    if profile:
        try:
            s[].ex.execution_profile_begin()
        except error:
            s[].program_trace.enabled = False
            raise error
    var started = Int(perf_counter_ns())
    try:
        for k in range(len(inputs)):  # driver-loop: over native uploads
            var sh = inputs[k].shape
            var buf = s[].ex.alloc(sh.count())
            buffers.append(buf)
            shapes.append(sh)
            input_bytes += sh.count() * 4
            upload_bytes += sh.count() * inputs[k].src.itemsize
            if p.in_kinds[k] == PROGRAM_IN_IDS:
                upload_cast_i32(s[].ex, buf, inputs[k].src, "program ids")
            else:
                upload_cast(s[].ex, buf, inputs[k].src)
        if profile:
            s[].ex.sync()
    except err:
        s[].program_trace.enabled = False
        for k in range(len(buffers)):  # driver-loop: over owners acquired before a failed upload
            s[].ex.free(buffers[k])
        if profile:
            _ = s[].ex.execution_profile_end()
        raise err
    _ = inputs^  # raw addresses erase origins; retain scalar sequence owners through upload
    var upload_ns = Int(perf_counter_ns()) - started
    var held = Tape()
    swap(held, s[].prog_tape)
    swap(held, s[].tape)
    var temps = List[Int]()
    var phs = List[Int]()
    var outputs = List[HostArray]()
    var download_bytes = 0
    var download_ns = 0
    try:
        s[].tape.no_grad()
        s[].tape.reset()
        var outs = replay_values(s[], p, buffers, shapes, params, trainable, temps, phs)
        started = Int(perf_counter_ns())
        for k in range(len(outs)):  # driver-loop: over returned arrays
            var sh = s[].shapes[s[]._slot(outs[k])]
            var a = HostArray(sh)
            s[].download(outs[k], FP(unsafe_from_address=Int(a.data.unsafe_ptr())))
            download_bytes += sh.count() * 4
            outputs.append(a^)
        download_ns = Int(perf_counter_ns()) - started
    except err:
        s[].program_trace.enabled = False
        s[]._drop_temps(temps)
        swap(held, s[].tape)
        swap(held, s[].prog_tape)
        for k in range(len(buffers)):  # driver-loop: over input owners
            s[].ex.free(buffers[k])
        if profile:
            _ = s[].ex.execution_profile_end()
        raise err
    s[].program_trace.enabled = False
    s[]._drop_temps(temps)
    swap(held, s[].tape)
    swap(held, s[].prog_tape)
    for k in range(len(buffers)):  # driver-loop: over input owners
        s[].ex.free(buffers[k])
    var result = ProgramExecResult(s[].ex.execution_profile_end() if profile else ExecutionProfile())
    result.outputs = outputs^
    result.upload_bytes = upload_bytes
    result.input_bytes = input_bytes
    result.download_bytes = download_bytes
    result.parameter_bytes = parameter_bytes
    result.upload_ns = upload_ns
    result.download_ns = download_ns
    return result^


def program_fit_parameters[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, otherwise_id: Int, explicit: Bool, selected: List[Int],
) raises -> List[Int]:
    """The parameters a native-options fit trains: the program's, joined
    with the default branch's when `otherwise_id` > 0, or exactly
    `selected` when `explicit`."""
    var s = ops_session[E, KIND](sid)
    var p = s[].program(pid)
    var params = p.params.copy()
    if otherwise_id > 0:
        var other = s[].program(otherwise_id)
        for k in range(len(other.params)):  # driver-loop: over default branch parameter union
            var found = False
            for j in range(len(params)):  # driver-loop: over shared parameter identity
                found = found or params[j] == other.params[k]
            if not found:
                params.append(other.params[k])
    if explicit:
        params = selected.copy()
    return params^


def program_fit_native[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, var owners: List[ProgramInput], var target_owners: List[ProgramInput],
    kind: Int, delta: Float32, opt: Int, params: List[Int], count: Int, batch: Int, shuffle: Bool,
    seed_word: UInt64, curve: FP, condition_id: Int, otherwise_id: Int,
) raises -> Tuple[Int, Int]:
    """Train program `pid` over the converted inputs (and targets, unless
    kind is 0, loss='output'); the loss curve goes to `curve`. Returns the
    branch counts of a conditional program."""
    var srcs = List[CastIn]()
    var shapes = List[ShapeN]()
    for k in range(len(owners)):  # driver-loop: over borrowed input descriptors
        srcs.append(owners[k].src)
        shapes.append(owners[k].shape)
    var target = CastIn.f32_c(0, 1, 1)
    if kind != 0:
        target = target_owners[0].src
    var s = ops_session[E, KIND](sid)
    var branches = train_program_control(s[], pid, srcs, shapes, target, kind, delta,
                                        opt, params, count, batch, shuffle, seed_word, curve,
                                        condition_id, otherwise_id)
    _ = owners^
    _ = target_owners^
    return branches
