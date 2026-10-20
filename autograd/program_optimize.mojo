# SPDX-License-Identifier: Apache-2.0
# Opt-in graph metadata passes. No element arithmetic or storage ownership edits.
from autograd.program import OpsProgram
from autograd.program_tools import _same_node, _cse_safe


def _header(p: OpsProgram) -> OpsProgram:
    var q = OpsProgram()
    q.live = p.live
    q.n_in = p.n_in
    q.in_shapes = p.in_shapes.copy()
    q.in_kinds = p.in_kinds.copy()
    q.params = p.params.copy()
    q.param_shapes = p.param_shapes.copy()
    q.out_shapes = p.out_shapes.copy()
    return q^


def _args(p: OpsProgram, k: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over node reference metadata
        out.append(p.args[i])
    return out^


def _ints(p: OpsProgram, k: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: over node integer metadata
        out.append(p.ints[i])
    return out^


def _append(p: OpsProgram, k: Int, mut q: OpsProgram, code: Int,
            args: List[Int], ints: List[Int]):
    q.code.append(code)
    for i in range(len(args)):  # driver-loop: over node reference metadata
        q.args.append(args[i])
    for i in range(len(ints)):  # driver-loop: over node integer metadata
        q.ints.append(ints[i])
    for i in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: over unchanged float attributes
        q.fs.append(p.fs[i])
    q.end_node()


struct ProgramOptimization(Movable):
    var program: OpsProgram
    var source_nodes: List[Int]
    var pass_names: List[String]
    var before: List[Int]
    var after: List[Int]
    var rewrites: List[Int]
    var last_use: List[Int]
    var uses: List[Int]

    def __init__(out self, p: OpsProgram):
        self.program = p.copy()
        self.source_nodes = List[Int]()
        for k in range(p.n_nodes()):  # driver-loop: over immutable source node provenance
            self.source_nodes.append(k)
        self.pass_names = List[String]()
        self.before = List[Int]()
        self.after = List[Int]()
        self.rewrites = List[Int]()
        self.last_use = List[Int]()
        self.uses = List[Int]()

    def record(mut self, name: String, before: Int, rewrites: Int):
        self.pass_names.append(name)
        self.before.append(before)
        self.after.append(self.program.n_nodes())
        self.rewrites.append(rewrites)


def _reshape_count(p: OpsProgram, k: Int) -> Int:
    # Only fully explicit positive dimensions. -1, flatten, squeeze and
    # unsqueeze keep their original runtime validation and are not folded.
    var at = p.i_at[k]
    var end = p.i_at[k + 1]
    if p.code[k] == 14:
        if end - at != 2:
            return 0
    elif p.code[k] == 33:
        if end - at < 2 or end - at > 9 or p.ints[at] != 0:
            return 0
        at += 1
    else:
        return 0
    if p.a_at[k + 1] - p.a_at[k] != 1 or p.f_at[k + 1] != p.f_at[k]:
        return 0
    var count = 1
    for i in range(at, end):  # driver-loop: over explicit reshape dimension metadata
        var d = p.ints[i]
        if d <= 0 or count > 0x7fffffffffffffff // d:
            return 0
        count *= d
    return count


def _permutation(p: OpsProgram, k: Int) -> List[Int]:
    # A full explicit permutation carries its own exact-rank validator.
    # Normalize negative axes from its declared rank, never captured sizes.
    var out = List[Int]()
    var at = p.i_at[k]
    var rank = p.i_at[k + 1] - at - 1
    if p.code[k] != 34 or rank < 1 or rank > 8:
        return out^
    if p.ints[at] != 0 or p.a_at[k + 1] - p.a_at[k] != 1 or p.f_at[k + 1] != p.f_at[k]:
        return out^
    var seen = List[Bool](length=rank, fill=False)
    for i in range(rank):  # driver-loop: over declared permutation rank metadata
        var axis = p.ints[at + 1 + i]
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank or seen[axis]:
            out.clear()
            return out^
        seen[axis] = True
        out.append(axis)
    return out^


def _swap_axes(p: OpsProgram, k: Int) -> List[Int]:
    var out = List[Int]()
    if p.a_at[k + 1] - p.a_at[k] != 1 or p.f_at[k + 1] != p.f_at[k]:
        return out^
    var at = p.i_at[k]
    var n = p.i_at[k + 1] - at
    if p.code[k] == 13 and n == 0:
        out.append(-2)
        out.append(-1)
    elif p.code[k] == 34 and n == 3 and p.ints[at] == 1:
        # Larger magnitudes are invalid for every supported rank. Preserve
        # the original refusal instead of negating arbitrary signed words.
        if p.ints[at + 1] < -8 or p.ints[at + 1] > 7 or p.ints[at + 2] < -8 or p.ints[at + 2] > 7:
            return out^
        out.append(p.ints[at + 1])
        out.append(p.ints[at + 2])
    return out^


def _views(mut plan: ProgramOptimization, mut removable: List[Bool], reshape: Bool):
    var p = plan.program.copy()
    var q = _header(p)
    var changed = 0
    for k in range(p.n_nodes()):  # driver-loop: over topological graph metadata rewrite
        var args = _args(p, k)
        var ints = _ints(p, k)
        var code = p.code[k]
        if len(args) == 1 and args[0] >= p.n_in:
            var j = args[0] - p.n_in
            if reshape:
                var count = _reshape_count(p, k)
                if count > 0 and count == _reshape_count(q, j):
                    # Both accept exactly the same element count. Retain the
                    # final reshape and its output shape/count validator.
                    args[0] = q.args[q.a_at[j]]
                    removable[j] = True
                    changed += 1
            else:
                var outer = _permutation(p, k)
                var inner = _permutation(q, j)
                if len(outer) > 0 and len(outer) == len(inner):
                    ints.clear()
                    ints.append(0)
                    for axis in range(len(outer)):  # driver-loop: over explicit permutation axes
                        ints.append(inner[outer[axis]])
                    args[0] = q.args[q.a_at[j]]
                    removable[j] = True
                    changed += 1
                    # Even an identity keeps this permutation operation:
                    # variable-rank callers must still meet its exact rank.
                else:
                    var a = _swap_axes(p, k)
                    var b = _swap_axes(q, j)
                    if len(a) == 2 and len(b) == 2:
                        if (a[0] == b[0] and a[1] == b[1]) or (a[0] == b[1] and a[1] == b[0]):
                            # An involution becomes an identity swap with the
                            # stricter original axis. _swap still validates
                            # ndim >= 2 and both original axes' minimum rank.
                            var need_a = a[0] + 1 if a[0] >= 0 else -a[0]
                            var need_b = a[1] + 1 if a[1] >= 0 else -a[1]
                            var guard = a[0] if need_a >= need_b else a[1]
                            code = 34
                            ints.clear()
                            ints.append(1)
                            ints.append(guard)
                            ints.append(guard)
                            args[0] = q.args[q.a_at[j]]
                            removable[j] = True
                            changed += 1
        _append(p, k, q, code, args, ints)
    q.outs = p.outs.copy()
    plan.program = q^
    plan.record("explicit_reshape_chain" if reshape else "permutation_composition", p.n_nodes(), changed)


def _prune(mut plan: ProgramOptimization, removable: List[Bool]):
    var p = plan.program.copy()
    var needed = List[Bool](length=p.n_nodes(), fill=False)
    for k in range(p.n_nodes()):  # driver-loop: over preserve original execution/validation roots
        needed[k] = not removable[k]
    for i in range(len(p.outs)):  # driver-loop: over preserve every output, including aliases
        if p.outs[i] >= p.n_in:
            needed[p.outs[i] - p.n_in] = True
    var k = p.n_nodes() - 1
    while k >= 0:  # driver-loop: over reverse reachability through graph references
        if needed[k]:
            for i in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over predecessor references
                if p.args[i] >= p.n_in:
                    needed[p.args[i] - p.n_in] = True
        k -= 1
    var q = _header(p)
    var refs = List[Int]()
    var origins = List[Int]()
    for i in range(p.n_in):  # driver-loop: over input reference identity
        refs.append(i)
    for i in range(p.n_nodes()):  # driver-loop: over compact proven dead intermediates
        if not needed[i]:
            refs.append(-1)
            continue
        var args = _args(p, i)
        for j in range(len(args)):  # driver-loop: over remap graph references
            if args[j] >= 0:
                args[j] = refs[args[j]]
        refs.append(q.n_values())
        origins.append(plan.source_nodes[i])
        _append(p, i, q, p.code[i], args, _ints(p, i))
    for i in range(len(p.outs)):  # driver-loop: over remap outputs
        var r = p.outs[i]
        q.outs.append(r if r < 0 else refs[r])
    plan.program = q^
    plan.source_nodes = origins^
    plan.record("dead_intermediates", p.n_nodes(), p.n_nodes() - plan.program.n_nodes())


def _duplicates(mut plan: ProgramOptimization):
    var p = plan.program.copy()
    var q = _header(p)
    var refs = List[Int]()
    var origins = List[Int]()
    for k in range(p.n_in):  # driver-loop: over input reference identity
        refs.append(k)
    for k in range(p.n_nodes()):  # driver-loop: over exact deterministic node matching
        var args = _args(p, k)
        for i in range(len(args)):  # driver-loop: over canonical predecessor references
            if args[i] >= 0:
                args[i] = refs[args[i]]
        var found = -1
        if _cse_safe(p.code[k]):
            for j in range(q.n_nodes()):  # driver-loop: over earlier nodes, same attribute words
                if _same_node(p, k, q, j, args):
                    found = j
                    break
        if found >= 0:
            refs.append(q.n_in + found)
        else:
            refs.append(q.n_values())
            origins.append(plan.source_nodes[k])
            _append(p, k, q, p.code[k], args, _ints(p, k))
    for k in range(len(p.outs)):  # driver-loop: over preserve output reference order
        var r = p.outs[k]
        q.outs.append(r if r < 0 else refs[r])
    plan.program = q^
    plan.source_nodes = origins^
    plan.record("exact_duplicate_elimination", p.n_nodes(), p.n_nodes() - plan.program.n_nodes())


def program_optimize(p: OpsProgram, enabled: Bool = True) -> ProgramOptimization:
    # GRAPH_STRUCTURAL_20261020 is caller opt-in, default OFF at public entry.
    # No performance promotion before full-workload NVIDIA+AMD A/B and the
    # existing identity contract. Pass rules do not key on tensor dimensions.
    # Training always replays the original graph: merging gradient paths can
    # change reduction order. Views preserve word order, including NaN payloads
    # and signed zero; all rank/element-count validators remain represented.
    var plan = ProgramOptimization(p)
    if enabled:
        var removable = List[Bool](length=p.n_nodes(), fill=False)
        _views(plan, removable, True)
        _views(plan, removable, False)
        _prune(plan, removable)
        _duplicates(plan)
    var graph = plan.program.copy()
    plan.last_use = List[Int](length=graph.n_values(), fill=-1)
    plan.uses = List[Int](length=graph.n_values(), fill=0)
    for k in range(graph.n_nodes()):  # driver-loop: over logical reference liveness only
        for i in range(graph.a_at[k], graph.a_at[k + 1]):  # driver-loop: over node input uses
            var r = graph.args[i]
            if r >= 0:
                plan.last_use[r] = k
                plan.uses[r] += 1
    for i in range(len(graph.outs)):  # driver-loop: over outputs remain live through consumption
        var r = graph.outs[i]
        if r >= 0:
            plan.last_use[r] = graph.n_nodes()
            plan.uses[r] += 1
    return plan^
