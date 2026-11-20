# SPDX-License-Identifier: Apache-2.0
"""Native preprocessing program records, independent of Python admission.

One stage is a fixed 16-word ABI record. Appends preserve that record's order;
parallelism belongs to the existing executors of the program, not its 14 fields.
"""


def prep_program_stage(mut program: List[Int32], op: Int, total: Int, *params: Int):
    program.append(Int32(op))
    program.append(Int32(total))
    for i in range(len(params)):  # small-loop(params: at most fourteen native stage arguments)
        program.append(Int32(params[i]))
    for i in range(14 - len(params)):  # small-loop(params: pad the fixed fourteen-word stage ABI)
        program.append(0)
