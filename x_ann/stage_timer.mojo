# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage timings for the x_ann device drivers (lane ann-apple, 2026-09-28).

With `MOJOLEARN_ANN_STAGES` set, `mark` synchronizes the context and prints
`ANN-STAGE <driver> <stage> <ms>` to stdout, so one speed run shows where a
driver's wall time goes. Unset (the default), `mark` does nothing: no sync,
no print, no bit or schedule change."""
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext


struct AnnStages(Movable):
    var on: Bool
    var name: String
    var t: Int

    def __init__(out self, name: String):
        self.on = String(getenv("MOJOLEARN_ANN_STAGES")) != ""
        self.name = name
        self.t = perf_counter_ns()

    def mark(mut self, ctx: DeviceContext, stage: String) raises:
        if not self.on:
            return
        ctx.synchronize()
        self.host(stage)

    def host(mut self, stage: String):
        if not self.on:
            return
        var now = perf_counter_ns()
        print("ANN-STAGE", self.name, stage, Float64(now - self.t) / 1.0e6)
        self.t = now
