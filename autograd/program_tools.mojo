# SPDX-License-Identifier: Apache-2.0
# Graph metadata transformations and diagnostics; no tensor arithmetic here.
from std.memory import bitcast
from autograd.program import OpsProgram


def program_op_name(code: Int) -> String:
    var names = List[String](
        "invalid", "matmul", "add", "subtract", "multiply", "divide", "relu", "sigmoid",
        "tanh", "exp", "log", "sqrt", "scale", "transpose", "reshape", "sum", "mean",
        "softmax", "log_softmax", "linear", "layer_norm", "dropout", "gather_rows",
        "permutation", "mse", "mae", "huber", "bce_with_logits", "cross_entropy",
        "detach", "full", "copy", "embedding", "reshape_nd", "permute", "sum_nd",
        "mean_nd", "max", "matmul_nd", "concat", "narrow", "take", "softmax_nd",
        "log_softmax_nd", "conv2d", "max_pool2d", "avg_pool2d", "reserved", "erf", "gelu",
        "softplus", "silu", "leaky_relu", "elu", "hard_sigmoid", "hard_swish", "clip",
        "abs", "floor", "ceil", "round", "sin", "cos", "sign", "reciprocal", "pow",
        "minimum", "maximum", "where", "compare", "logical", "expand", "argmax",
        "pad", "any_all", "cumsum", "index",
    )
    if code > 0 and code < len(names):
        return names[code]
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
    for i in range(len(args)):  # driver-loop: node input references
        if args[i] != q.args[q.a_at[j] + i]:
            return False
    for i in range(p.i_at[k + 1] - p.i_at[k]):  # driver-loop: integer argument words
        if p.ints[p.i_at[k] + i] != q.ints[q.i_at[j] + i]:
            return False
    for i in range(p.f_at[k + 1] - p.f_at[k]):  # driver-loop: float argument words
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
    for k in range(p.n_in):  # driver-loop: input refs remain unchanged
        refs.append(k)
    for k in range(p.n_nodes()):  # driver-loop: original graph nodes
        var args = List[Int]()
        for a in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: remapped input refs
            var r = p.args[a]
            args.append(r if r < 0 else refs[r])
        var found = -1
        if _cse_safe(p.code[k]):
            for j in range(q.n_nodes()):  # driver-loop: earlier candidate nodes
                if _same_node(p, k, q, j, args):
                    found = j
                    break
        if found >= 0:
            refs.append(q.n_in + found)
            continue
        refs.append(q.n_values())
        q.code.append(p.code[k])
        for a in range(len(args)):  # driver-loop: node reference words
            q.args.append(args[a])
        for a in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: node integer words
            q.ints.append(p.ints[a])
        for a in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: node float words
            q.fs.append(p.fs[a])
        q.end_node()
    for k in range(len(p.outs)):  # driver-loop: output refs
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
