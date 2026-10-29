# SPDX-License-Identifier: Apache-2.0
# Independent pure native plan compile root. Never execute this fixture.
from core.metrics_program_plan import MetricProgramPlan


@export
def prepare_native_plan(count: Int, declared: Bool) abi("C") -> Int:
    try:
        var plan = MetricProgramPlan()
        var input = plan.input(count)
        var scratch = plan.scratch(count)
        var out = plan.alloc(count)
        plan.stage(0, count, [input, scratch, out])
        if declared:
            _ = plan.want(out, count, -1, 1)
            _ = plan.want(out, count, input, 2)
        var known = False
        var ranges = plan.download(known)
        return len(ranges) + len(plan.stages)
    except:
        return -1
