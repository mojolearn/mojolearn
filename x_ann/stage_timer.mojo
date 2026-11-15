# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage timings for the x_ann device drivers (lane ann-apple, 2026-09-28).

Built with `-D MOJOLEARN_ANN_STAGES`, `mark` synchronizes the context and prints
`ANN-STAGE <driver> <stage> <ms>` to stdout, so one speed run shows where a
driver's wall time goes. Without the define (the default), `mark` compiles
to nothing: no sync, no print, no clock read, no bit or schedule change.
Before (lane envgate-classical-deviceplan, docs/plans/ENV_GATES_2026-11-15.md)
this was a `getenv` per timer construction, i.e. per driver call."""
from std.sys import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

comptime ANN_STAGES_ON = is_defined["MOJOLEARN_ANN_STAGES"]()


struct AnnStages(Movable):
    var on: Bool
    var name: String
    var t: Int

    def __init__(out self, name: String):
        self.on = ANN_STAGES_ON
        self.name = name
        comptime if ANN_STAGES_ON:
            self.t = perf_counter_ns()
        else:
            self.t = 0

    def mark(mut self, ctx: DeviceContext, stage: String) raises:
        comptime if ANN_STAGES_ON:
            ctx.synchronize()
            self.host(stage)

    def host(mut self, stage: String):
        comptime if ANN_STAGES_ON:
            var now = perf_counter_ns()
            print("ANN-STAGE", self.name, stage, Float64(now - self.t) / 1.0e6)
            self.t = now
