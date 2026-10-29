# SPDX-License-Identifier: Apache-2.0
# Operation-local native metadata for repeated replay of an admitted graph.
from autograd.program import OpsProgram


struct PreparedReplayNode(Movable):
    var ints: List[Int]
    var fs: List[Float32]
    var original_last_int: Int

    def __init__(out self, p: OpsProgram, node: Int):
        self.ints = List[Int](capacity=p.i_at[node + 1] - p.i_at[node])
        self.fs = List[Float32](capacity=p.f_at[node + 1] - p.f_at[node])
        for j in range(p.i_at[node], p.i_at[node + 1]):  # driver-loop: over captured node integer metadata once per operation
            self.ints.append(p.ints[j])
        for j in range(p.f_at[node], p.f_at[node + 1]):  # driver-loop: over captured node float metadata once per operation
            self.fs.append(p.fs[j])
        # Do not validate dropout here: even malformed operation metadata must
        # retain its ordinary selected-node failure position during replay.
        self.original_last_int = self.ints[len(self.ints) - 1] if len(self.ints) != 0 else 0


struct PreparedProgramReplay(Movable):
    # This inventory belongs to one admitted program snapshot and one public
    # operation. Never cache it across program replacement or share it between
    # concurrent replays. Each conditional component gets its own inventory.
    var nodes: List[PreparedReplayNode]
    var handles: List[Int]
    var values: List[Int]

    def __init__(out self):
        self.nodes = List[PreparedReplayNode]()
        self.handles = List[Int]()
        self.values = List[Int]()

    def __init__(out self, p: OpsProgram):
        self.nodes = List[PreparedReplayNode](capacity=p.n_nodes())
        var max_arguments = 0
        for k in range(p.n_nodes()):  # driver-loop: over captured graph metadata once per operation
            self.nodes.append(PreparedReplayNode(p, k))
            max_arguments = max(max_arguments, p.a_at[k + 1] - p.a_at[k])
        self.handles = List[Int](capacity=max_arguments)
        self.values = List[Int](capacity=p.n_in + p.n_nodes())
