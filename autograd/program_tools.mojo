# SPDX-License-Identifier: Apache-2.0
# Graph metadata transformations and diagnostics; no tensor arithmetic here.
from std.memory import bitcast
from autograd.program import OpsProgram


def program_op_name(code: Int) -> String:
    if code == 0:
        return "invalid"
    if code == 1:
        return "matmul"
    if code == 2:
        return "add"
    if code == 3:
        return "subtract"
    if code == 4:
        return "multiply"
    if code == 5:
        return "divide"
    if code == 6:
        return "relu"
    if code == 7:
        return "sigmoid"
    if code == 8:
        return "tanh"
    if code == 9:
        return "exp"
    if code == 10:
        return "log"
    if code == 11:
        return "sqrt"
    if code == 12:
        return "scale"
    if code == 13:
        return "transpose"
    if code == 14:
        return "reshape"
    if code == 15:
        return "sum"
    if code == 16:
        return "mean"
    if code == 17:
        return "softmax"
    if code == 18:
        return "log_softmax"
    if code == 19:
        return "linear"
    if code == 20:
        return "layer_norm"
    if code == 21:
        return "dropout"
    if code == 22:
        return "gather_rows"
    if code == 23:
        return "permutation"
    if code == 24:
        return "mse"
    if code == 25:
        return "mae"
    if code == 26:
        return "huber"
    if code == 27:
        return "bce_with_logits"
    if code == 28:
        return "cross_entropy"
    if code == 29:
        return "detach"
    if code == 30:
        return "full"
    if code == 31:
        return "copy"
    if code == 32:
        return "embedding"
    if code == 33:
        return "reshape_nd"
    if code == 34:
        return "permute"
    if code == 35:
        return "sum_nd"
    if code == 36:
        return "mean_nd"
    if code == 37:
        return "max"
    if code == 38:
        return "matmul_nd"
    if code == 39:
        return "concat"
    if code == 40:
        return "narrow"
    if code == 41:
        return "take"
    if code == 42:
        return "softmax_nd"
    if code == 43:
        return "log_softmax_nd"
    if code == 44:
        return "conv2d"
    if code == 45:
        return "max_pool2d"
    if code == 46:
        return "avg_pool2d"
    if code == 47:
        return "reserved"
    if code == 48:
        return "erf"
    if code == 49:
        return "gelu"
    if code == 50:
        return "softplus"
    if code == 51:
        return "silu"
    if code == 52:
        return "leaky_relu"
    if code == 53:
        return "elu"
    if code == 54:
        return "hard_sigmoid"
    if code == 55:
        return "hard_swish"
    if code == 56:
        return "clip"
    if code == 57:
        return "abs"
    if code == 58:
        return "floor"
    if code == 59:
        return "ceil"
    if code == 60:
        return "round"
    if code == 61:
        return "sin"
    if code == 62:
        return "cos"
    if code == 63:
        return "sign"
    if code == 64:
        return "reciprocal"
    if code == 65:
        return "pow"
    if code == 66:
        return "minimum"
    if code == 67:
        return "maximum"
    if code == 68:
        return "where"
    if code == 69:
        return "compare"
    if code == 70:
        return "logical"
    if code == 71:
        return "expand"
    if code == 72:
        return "argmax"
    if code == 73:
        return "pad"
    if code == 74:
        return "any_all"
    if code == 75:
        return "cumsum"
    if code == 76:
        return "index"
    if code == 77:
        return "rms_norm"
    if code == 78:
        return "rotary"
    if code == 79:
        return "transformer_linear"
    return "op_" + String(code)


def _cse_safe(code: Int) -> Bool:
    # Random streams and copy/detach alias semantics are excluded. This is
    # inference-only: sharing nodes during training changes gradient sums.
    return (code > 0 and code <= 76 and code != 21 and code != 23
            and code != 29 and code != 31 and code != 47)


def _same_node(p: OpsProgram, k: Int, q: OpsProgram, j: Int, args: List[Int]) -> Bool:
    if p.code[k] != q.code[j] or len(args) != q.a_at[j + 1] - q.a_at[j]:
        return False
    if p.i_at[k + 1] - p.i_at[k] != q.i_at[j + 1] - q.i_at[j]:
        return False
    if p.f_at[k + 1] - p.f_at[k] != q.f_at[j + 1] - q.f_at[j]:
        return False
    for i in range(len(args)):  # driver-loop: over node input references
        if args[i] != q.args[q.a_at[j] + i]:
            return False
    for i in range(p.i_at[k + 1] - p.i_at[k]):  # driver-loop: over integer argument words
        if p.ints[p.i_at[k] + i] != q.ints[q.i_at[j] + i]:
            return False
    for i in range(p.f_at[k + 1] - p.f_at[k]):  # driver-loop: over float argument words
        # Bitwise words preserve signed zero and NaN payload distinctions.
        if bitcast[DType.uint32](p.fs[p.f_at[k] + i]) != bitcast[DType.uint32](q.fs[q.f_at[j] + i]):
            return False
    return True


def program_cse(p: OpsProgram) -> OpsProgram:
    # Candidate GRAPH_CSE_20261020: caller opt-in, default OFF. No fusion,
    # reassociation or reduction rewrite. Full-workload NVIDIA+AMD A/B and
    # three-vendor identity remain pending; no performance claim is made.
    # Matching is graph-structural, independent of shape or dataset names.
    var q = OpsProgram()
    q.n_in = p.n_in
    q.in_shapes = p.in_shapes.copy()
    q.in_kinds = p.in_kinds.copy()
    q.params = p.params.copy()
    q.param_shapes = p.param_shapes.copy()
    q.out_shapes = p.out_shapes.copy()
    var refs = List[Int]()
    for k in range(p.n_in):  # driver-loop: over input refs remain unchanged
        refs.append(k)
    for k in range(p.n_nodes()):  # driver-loop: over original graph nodes
        var args = List[Int]()
        for a in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over remapped input refs
            var r = p.args[a]
            args.append(r if r < 0 else refs[r])
        var found = -1
        if _cse_safe(p.code[k]):
            for j in range(q.n_nodes()):  # driver-loop: over earlier candidate nodes
                if _same_node(p, k, q, j, args):
                    found = j
                    break
        if found >= 0:
            refs.append(q.n_in + found)
            continue
        refs.append(q.n_values())
        q.code.append(p.code[k])
        for a in range(len(args)):  # driver-loop: over node reference words
            q.args.append(args[a])
        for a in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: over node integer words
            q.ints.append(p.ints[a])
        for a in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: over node float words
            q.fs.append(p.fs[a])
        q.end_node()
    for k in range(len(p.outs)):  # driver-loop: over output refs
        var r = p.outs[k]
        q.outs.append(r if r < 0 else refs[r])
    return q^


struct ProgramTrace(Movable):
    var enabled: Bool
    var nodes: List[Int]
    var codes: List[Int]
    var shapes: List[String]
    var enqueue_ns: List[Int]
    var wait_ns: List[Int]
    var output_bytes: List[Int]
    var sync_count: Int
    var sync_ns: Int
    var peak_tape_bytes: Int

    def __init__(out self):
        self.enabled = False
        self.nodes = List[Int]()
        self.codes = List[Int]()
        self.shapes = List[String]()
        self.enqueue_ns = List[Int]()
        self.wait_ns = List[Int]()
        self.output_bytes = List[Int]()
        self.sync_count = 0
        self.sync_ns = 0
        self.peak_tape_bytes = 0
