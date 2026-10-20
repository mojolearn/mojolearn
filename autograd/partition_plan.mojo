# SPDX-License-Identifier: Apache-2.0
# Physical residency is separate from this immutable logical graph. Tensor
# splitting partitions independent output columns, never the contraction K.
from std.memory import memcpy
from autograd.program import OpsProgram
from autograd.rankn import ShapeN


struct GraphBytes(Copyable, Movable):
    var shape: ShapeN
    var words: List[Float32]
    var integer: Bool

    def __init__(out self):
        self.shape = ShapeN.matrix(1, 1)
        self.words = List[Float32]()
        self.integer = False

    def __init__(out self, shape: ShapeN, var words: List[Float32], integer: Bool = False):
        self.shape = shape
        self.words = words^
        self.integer = integer


struct PartitionPlan(Copyable, Movable):
    var program: OpsProgram
    var parameters: List[GraphBytes]
    var trainable: List[Bool]
    var stages: Int
    var tensor_shards: Int
    var placement: List[Int]
    var levels: List[Int]
    var owners: List[Int]
    var max_level: Int
    var partitioned_products: Int
    var origins: List[Int]
    var column_first: List[Int]
    var column_width: List[Int]

    def __init__(out self):
        self.program = OpsProgram()
        self.parameters = List[GraphBytes]()
        self.trainable = List[Bool]()
        self.stages = 1
        self.tensor_shards = 1
        self.placement = List[Int]()
        self.levels = List[Int]()
        self.owners = List[Int]()
        self.max_level = 0
        self.partitioned_products = 0
        self.origins = List[Int]()
        self.column_first = List[Int]()
        self.column_width = List[Int]()

    def schedule(mut self) raises:
        var p = self.program.copy()
        self.levels = List[Int](length=p.n_nodes(), fill=1)
        self.owners = List[Int](length=len(p.params), fill=-1)
        self.max_level = 0
        for k in range(p.n_nodes()):  # driver-loop: topological graph metadata
            var level = 1
            for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: node dependencies
                var rf = p.args[j]
                if rf >= p.n_in:
                    if rf - p.n_in >= k:
                        raise Error("partition plan is not topologically ordered")
                    level = max(level, self.levels[rf - p.n_in] + 1)
                elif rf < 0 and self.owners[-rf - 1] < 0:
                    self.owners[-rf - 1] = self.placement[k]
            self.levels[k] = level
            self.max_level = max(self.max_level, level)
        for j in range(len(self.owners)):  # driver-loop: parameters used only as outputs
            if self.owners[j] < 0:
                self.owners[j] = j % self.stages


def _emit_node(mut p: OpsProgram, code: Int, args: List[Int], ints: List[Int], fs: List[Float32]) -> Int:
    p.code.append(code)
    for v in args:  # driver-loop: node argument references
        p.args.append(v)
    for v in ints:  # driver-loop: node integer metadata
        p.ints.append(v)
    for v in fs:  # driver-loop: node floating metadata
        p.fs.append(v)
    p.end_node()
    return p.n_in + p.n_nodes() - 1


def column_bytes(a: GraphBytes, lo: Int, width: Int) raises -> GraphBytes:
    if a.shape.ndim != 2 or lo < 0 or width < 1 or lo + width > a.shape.cols():
        raise Error("tensor partition requires a valid rank-two column range")
    var data = List[Float32](length=a.shape.rows() * width, fill=Float32(0))
    # Byte packing, no host arithmetic on tensor values. The full unpartitioned
    # parameter is never uploaded to a model-stage executor.
    for row in range(a.shape.rows()):  # driver-loop: packed parameter row copies
        memcpy(dest=data.unsafe_ptr() + row * width,
               src=a.words.unsafe_ptr() + row * a.shape.cols() + lo, count=width)
    return GraphBytes(ShapeN.matrix(a.shape.rows(), width), data^, a.integer)


def partition_plan(p: OpsProgram, values: List[GraphBytes], trainable: List[Bool],
                   stages: Int, tensor_shards: Int) raises -> PartitionPlan:
    if stages < 1 or tensor_shards < 1 or tensor_shards > stages:
        raise Error("partition plan requires 1 <= tensor_shards <= logical stages")
    var plan = PartitionPlan()
    plan.stages = stages
    plan.tensor_shards = tensor_shards
    plan.program.n_in = p.n_in
    plan.program.in_shapes = p.in_shapes.copy()
    plan.program.in_kinds = p.in_kinds.copy()
    var uses = List[Int](length=len(p.params), fill=0)
    for j in range(len(p.args)):  # driver-loop: parameter use counts
        if p.args[j] < 0:
            uses[-p.args[j] - 1] += 1
    for j in range(len(p.outs)):  # driver-loop: output references
        if p.outs[j] < 0:
            uses[-p.outs[j] - 1] += 1
    var split_weight = List[Bool](length=len(p.params), fill=False)
    var split_node = List[Bool](length=p.n_nodes(), fill=False)
    for k in range(p.n_nodes()):  # driver-loop: tensor-partition admission
        var count = p.a_at[k + 1] - p.a_at[k]
        if tensor_shards < 2 or (p.code[k] != 1 and p.code[k] != 19) or count < 2:
            continue
        var wr = p.args[p.a_at[k] + 1]
        if wr >= 0:
            continue
        var j = -wr - 1
        var sh = p.param_shapes[j]
        # This split is based on independent output columns and single owner
        # use, not a benchmark shape. Shared weights retain one whole owner.
        if uses[j] != 1 or sh.ndim != 2 or sh.cols() < 2 or values[j].integer:
            continue
        if count == 3:
            var br = p.args[p.a_at[k] + 2]
            if br >= 0:
                continue
            var b = -br - 1
            if uses[b] != 1 or p.param_shapes[b].ndim != 2 or p.param_shapes[b].rows() != 1 or p.param_shapes[b].cols() != sh.cols():
                continue
            split_weight[b] = True
        split_weight[j] = True
        split_node[k] = True
    var param_map = List[Int](length=len(p.params), fill=0)
    for j in range(len(p.params)):  # driver-loop: retained whole parameters
        if split_weight[j]:
            continue
        param_map[j] = -len(plan.parameters) - 1
        plan.parameters.append(values[j].copy())
        plan.origins.append(j)
        plan.column_first.append(0)
        plan.column_width.append(0)
        plan.trainable.append(trainable[j])
        plan.program.params.append(0)
        plan.program.param_shapes.append(p.param_shapes[j])
    var refs = List[Int]()
    for j in range(p.n_in):  # driver-loop: input value mapping
        refs.append(j)
    for k in range(p.n_nodes()):  # driver-loop: graph lowering
        var hs = List[Int]()
        var ints = List[Int]()
        var fs = List[Float32]()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: remapped node refs
            var old = p.args[j]
            hs.append(param_map[-old - 1] if old < 0 else refs[old])
        for j in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: node metadata
            ints.append(p.ints[j])
        for j in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: node metadata
            fs.append(p.fs[j])
        var base_stage = min(stages - 1, k * stages // max(1, p.n_nodes()))
        if not split_node[k]:
            refs.append(_emit_node(plan.program, p.code[k], hs, ints, fs))
            plan.placement.append(base_stage)
            continue
        var old_weight = -p.args[p.a_at[k] + 1] - 1
        var columns = p.param_shapes[old_weight].cols()
        var pieces = min(tensor_shards, columns)
        var outputs = List[Int]()
        for part in range(pieces):  # driver-loop: logical tensor partitions
            var lo = columns * part // pieces
            var width = columns * (part + 1) // pieces - lo
            var args: List[Int] = [hs[0]]
            var weight = column_bytes(values[old_weight], lo, width)
            args.append(-len(plan.parameters) - 1)
            plan.program.params.append(0)
            plan.program.param_shapes.append(weight.shape)
            plan.parameters.append(weight^)
            plan.origins.append(old_weight)
            plan.column_first.append(lo)
            plan.column_width.append(width)
            plan.trainable.append(trainable[old_weight])
            if len(hs) == 3:
                var old_bias = -p.args[p.a_at[k] + 2] - 1
                var bias = column_bytes(values[old_bias], lo, width)
                args.append(-len(plan.parameters) - 1)
                plan.program.params.append(0)
                plan.program.param_shapes.append(bias.shape)
                plan.parameters.append(bias^)
                plan.origins.append(old_bias)
                plan.column_first.append(lo)
                plan.column_width.append(width)
                plan.trainable.append(trainable[old_bias])
            var plain = List[Int]()
            if p.code[k] == 19:
                plain.append(0)
            outputs.append(_emit_node(plan.program, p.code[k], args, plain, fs))
            plan.placement.append((base_stage + part) % stages)
        # Concatenate the independently computed output columns. Nonlinear
        # activation follows the whole tensor (softmax must see all columns).
        var joined = _emit_node(plan.program, 39, outputs, [-1], List[Float32]())
        plan.placement.append(base_stage)
        if p.code[k] == 19 and ints[0] != 0:
            var act = 6 if ints[0] == 1 else (7 if ints[0] == 2 else (8 if ints[0] == 3 else 17))
            joined = _emit_node(plan.program, act, [joined], List[Int](), List[Float32]())
            plan.placement.append(base_stage)
        refs.append(joined)
        plan.partitioned_products += 1
    for j in range(len(p.outs)):  # driver-loop: remapped output values
        var old = p.outs[j]
        plan.program.outs.append(param_map[-old - 1] if old < 0 else refs[old])
        plan.program.out_shapes.append(p.out_shapes[j])
    plan.schedule()
    return plan^
