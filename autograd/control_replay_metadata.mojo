# SPDX-License-Identifier: Apache-2.0
# One native replay owner per component of a public control-flow operation.
from autograd.program import OpsProgram
from autograd.program_replay_metadata import PreparedProgramReplay


struct ControlReplay(Movable):
    var ready: Bool
    var graph: PreparedProgramReplay
    var parameters: List[Int]
    var no_grad: List[Bool]

    def __init__(out self):
        self.ready = False
        self.graph = PreparedProgramReplay()
        self.parameters = List[Int]()
        self.no_grad = List[Bool]()

    def prepare(mut self, p: OpsProgram):
        # Prepare only at the component's first actual execution. A false
        # initial predicate, zero-iteration limit or unselected branch must not
        # execute or validate its body's operation-specific metadata early.
        if not self.ready:
            self.graph = PreparedProgramReplay(p)
            self.ready = True
