# SPDX-License-Identifier: Apache-2.0
# Opt-in graph metadata passes. No element arithmetic or storage ownership edits.
from autograd.program import OpsProgram
from autograd.rankn import RN_MAX_COUNT, RN_MAX_RANK
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
    var relu_elements_before: Int
    var relu_elements_after: Int

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
        self.relu_elements_before = 0
        self.relu_elements_after = 0

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


def _swap_permutation(p: OpsProgram, k: Int, rank: Int) -> List[Int]:
    # Only an adjacent explicit permutation proves this exact rank. An
    # isolated swap remains rank-polymorphic and must keep its own validator.
    var axes = _swap_axes(p, k)
    var out = List[Int]()
    if len(axes) != 2 or rank < 2:
        return out^
    var a = axes[0] + rank if axes[0] < 0 else axes[0]
    var b = axes[1] + rank if axes[1] < 0 else axes[1]
    if a < 0 or b < 0 or a >= rank or b >= rank:
        return out^
    for j in range(rank):  # small-loop(rank: at most eight axes): proven permutation metadata
        out.append(b if j == a else (a if j == b else j))
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
                if len(outer) > 0 and len(inner) == 0:
                    inner = _swap_permutation(q, j, len(outer))
                elif len(inner) > 0 and len(outer) == 0:
                    outer = _swap_permutation(p, k, len(inner))
                if len(outer) > 0 and len(outer) == len(inner):
                    code = 34
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



def _fixed_shape(p: OpsProgram, value: Int) -> List[Int]:
    # Parameters retain their exact shape at every replay. An explicit
    # positive reshape validates its element count before the candidate runs.
    # Captured input shapes are not proof: run_dynamic can change them.
    var dims = List[Int]()
    if value < 0:
        var index = -value - 1
        if index < len(p.param_shapes):
            var shape = p.param_shapes[index]
            for k in range(shape.ndim):  # driver-loop: over fixed parameter shape metadata
                dims.append(shape.d[k])
    elif value >= p.n_in:
        var node = value - p.n_in
        if _reshape_count(p, node) > 0:
            var start = p.i_at[node] + (1 if p.code[node] == 33 else 0)
            for k in range(start, p.i_at[node + 1]):  # driver-loop: over explicit reshape metadata
                dims.append(p.ints[k])
    return dims^


def _expand_counts(p: OpsProgram, node: Int) -> Tuple[Int, Int]:
    if p.code[node] != 71 or p.a_at[node + 1] - p.a_at[node] != 1 or p.f_at[node + 1] != p.f_at[node]:
        return (0, 0)
    var source = _fixed_shape(p, p.args[p.a_at[node]])
    var target = _ints(p, node)
    if len(source) < 1 or len(source) > RN_MAX_RANK or len(target) < 1 or len(target) > RN_MAX_RANK:
        return (0, 0)
    var before = 1
    var after = 1
    var target_count = 1
    var rank = max(len(source), len(target))
    for axis in range(rank):  # driver-loop: over statically proven broadcast dimensions
        var a = source[axis - rank + len(source)] if axis >= rank - len(source) else 1
        var b = target[axis - rank + len(target)] if axis >= rank - len(target) else 1
        # Match the existing positive/addressable shape contract; never elide
        # a failing broadcast or move arithmetic ahead of its refusal.
        if a < 1 or b < 1 or (a != b and a != 1 and b != 1):
            return (0, 0)
        var size = max(a, b)
        if after > RN_MAX_COUNT // a or before > RN_MAX_COUNT // size or target_count > RN_MAX_COUNT // b:
            return (0, 0)
        after *= a
        before *= size
        target_count *= b
    return (before, after)


def _hoist_relu(mut plan: ProgramOptimization):
    # GRAPH_BROADCAST_RELU_20261020: opt-in only. ReLU is F(relu(F(x)))
    # and expand either aliases or copies through F (the same FTZ map).
    # F(F(x)) == F(x), so moving ReLU before expand preserves every word,
    # including signed zero, NaNs and subnormals. No floating arithmetic is
    # reassociated and no reduction is rewritten. Training stays original.
    # Adjacent, single-consumer, non-output expands only. Static shapes prove
    # that the retained expand cannot newly fail after the moved arithmetic.
    var p = plan.program.copy()
    var q = _header(p)
    var uses = List[Int](length=p.n_values(), fill=0)
    for k in range(len(p.args)):  # driver-loop: over captured graph references
        if p.args[k] >= 0:
            uses[p.args[k]] += 1
    for k in range(len(p.outs)):  # driver-loop: over externally consumed graph values
        if p.outs[k] >= 0:
            uses[p.outs[k]] += 1
    var origins = List[Int]()
    var changed = 0
    var k = 0
    while k < p.n_nodes():  # driver-loop: over adjacent graph metadata candidates
        var counts = (0, 0)
        if k + 1 < p.n_nodes() and uses[p.n_in + k] == 1:
            var following = k + 1
            if p.code[following] == 6 and p.a_at[following + 1] - p.a_at[following] == 1 and p.args[p.a_at[following]] == p.n_in + k and p.i_at[following + 1] == p.i_at[following] and p.f_at[following + 1] == p.f_at[following]:
                counts = _expand_counts(p, k)
        # A real broadcast has more elementwise work than its source. This
        # cost rule covers every positive shape, not a benchmark dimension.
        if counts[0] > counts[1] and counts[1] > 0:
            _append(p, k + 1, q, 6, [p.args[p.a_at[k]]], List[Int]())
            _append(p, k, q, 71, [p.n_in + k], _ints(p, k))
            origins.append(plan.source_nodes[k + 1])
            origins.append(plan.source_nodes[k])
            plan.relu_elements_before += counts[0]
            plan.relu_elements_after += counts[1]
            changed += 1
            k += 2
        else:
            _append(p, k, q, p.code[k], _args(p, k), _ints(p, k))
            origins.append(plan.source_nodes[k])
            k += 1
    q.outs = p.outs.copy()
    plan.program = q^
    plan.source_nodes = origins^
    plan.record("broadcast_relu_hoist", p.n_nodes(), changed)


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
        _hoist_relu(plan)
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
