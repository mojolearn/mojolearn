# SPDX-License-Identifier: Apache-2.0
# Native conditional-training metadata. Component graphs remain immutable;
# one shared parameter table gives each handle one optimizer update/state.
from autograd.program import OpsProgram, same_shape


def _signature(base: OpsProgram, other: OpsProgram) raises:
    if base.n_in != other.n_in:
        raise Error("ValueError: conditional training components need the same input count")
    for k in range(base.n_in):  # driver-loop: over component input contracts
        if base.in_kinds[k] != other.in_kinds[k] or base.in_shapes[k].ndim != other.in_shapes[k].ndim:
            raise Error("ValueError: conditional training components need the same input ranks and kinds")


def _merge(mut table: OpsProgram, component: OpsProgram) raises:
    for k in range(len(component.params)):  # driver-loop: over component parameter handles
        var found = -1
        for j in range(len(table.params)):  # driver-loop: over shared parameter identity
            if table.params[j] == component.params[k]:
                found = j
                break
        if found < 0:
            table.params.append(component.params[k])
            table.param_shapes.append(component.param_shapes[k])
        elif not same_shape(table.param_shapes[found], component.param_shapes[k]):
            raise Error("ValueError: shared conditional parameter shapes disagree")


def _bind(component: OpsProgram, table: OpsProgram) -> OpsProgram:
    var out = component.copy()
    var index = List[Int]()
    for k in range(len(component.params)):  # driver-loop: over component parameter references
        for j in range(len(table.params)):  # driver-loop: over shared parameter identity
            if component.params[k] == table.params[j]:
                index.append(j)
                break
    for k in range(len(out.args)):  # driver-loop: over graph argument references
        if out.args[k] < 0:
            out.args[k] = -index[-out.args[k] - 1] - 1
    for k in range(len(out.outs)):  # driver-loop: over graph output references
        if out.outs[k] < 0:
            out.outs[k] = -index[-out.outs[k] - 1] - 1
    out.params = table.params.copy()
    out.param_shapes = table.param_shapes.copy()
    return out^


def _used(component: OpsProgram, table: OpsProgram) -> List[Bool]:
    var flags = List[Bool](length=len(table.params), fill=False)
    for k in range(len(component.params)):  # driver-loop: over branch parameter membership
        for j in range(len(table.params)):  # driver-loop: over shared parameter identity
            if component.params[k] == table.params[j]:
                flags[j] = True
    return flags^


struct ProgramChoice(Movable):
    var body: OpsProgram
    var condition: OpsProgram
    var otherwise: OpsProgram
    var body_parameters: List[Bool]
    var other_parameters: List[Bool]

    def __init__(out self):
        self.body = OpsProgram()
        self.condition = OpsProgram()
        self.otherwise = OpsProgram()
        self.body_parameters = List[Bool]()
        self.other_parameters = List[Bool]()


def conditional_program(body: OpsProgram, condition: OpsProgram, otherwise: OpsProgram) raises -> ProgramChoice:
    _signature(body, condition)
    _signature(body, otherwise)
    if len(body.outs) != 1 or len(otherwise.outs) != 1 or len(condition.outs) != 1:
        raise Error("ValueError: conditional training needs one output per branch and one scalar predicate")
    var table = body.copy()
    _merge(table, otherwise)
    _merge(table, condition)
    var choice = ProgramChoice()
    choice.body = _bind(body, table)
    choice.condition = _bind(condition, table)
    choice.otherwise = _bind(otherwise, table)
    choice.body_parameters = _used(body, table)
    choice.other_parameters = _used(otherwise, table)
    return choice^
