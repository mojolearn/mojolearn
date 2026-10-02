#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Arm plumbing the benchmark board adds to bench/speed/forest_speed_arm.py:
PER-ARM MEMORY (`--mem`). Off unless asked for, so the driver's output is
unchanged without it.

OUR CPU IS NEVER RACED (Andrew, Oct 2 2026). The `ours-cpu` proxy arm that
lived here (a MOJOLEARN_VENDOR=cpu worker in the round-robin) is deleted. The
host column's same-bits digest comes from a separate `--host-digest` process
(forest_speed_arm.py, tools/aft_idcheck.sh), which prints hashes and no time.

PER-ARM MEMORY
--------------
Every arm here shares one process, so memory is read around each arm's fit
through the runner's fit_context (entry and exit are outside the timer):
the host peak is resettable (Linux VmHWM after clear_refs, macOS interval
max phys_footprint) and so is per arm and round; a process-level GPU figure
(nvidia-smi / rocm-smi per pid) is the WHOLE PROCESS, every arm's pools
included, and its method says so.
Lines: FSPEED-MEM lane arm round host_mb gpu_mb children_mb host_method
gpu_method.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TOOLS = os.path.join(_ROOT, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import bench_board_probe as probe      # noqa: E402

def _one_line(text, n=400):
    return " ".join(str(text).split())[:n]


# ---------------------------------------------------------------------------
# Memory around each arm's fit
# ---------------------------------------------------------------------------

class TreeMem(object):
    """fit_context for speed_gbdt_arm.run: one FSPEED-MEM line per arm and
    round (round 0 is the warm-up)."""

    def __init__(self, lane):
        self.lane = lane
        self.probes = {}

    def _probe(self, arm_name):
        if arm_name not in self.probes:
            dev = "cpu" if (arm_name.endswith("-cpu") or "-cpu-" in arm_name) else "gpu"
            lib = "mojolearn" if arm_name.startswith("ours") else arm_name.split("-")[0]
            self.probes[arm_name] = probe.MemProbe(dev, shared=True, library=lib)
        return self.probes[arm_name]

    def emit(self, arm_name, r, m):
        def f(v):
            return "-" if v is None else ("%.1f" % v)
        print("FSPEED-MEM lane=%s arm=%s round=%d host_mb=%s gpu_mb=%s children_mb=%s "
              "host_method=%s gpu_method=%s"
              % (self.lane, arm_name, r, f(m.get("host_mb")), f(m.get("gpu_mb")),
                 f(m.get("children_mb")), _one_line(m.get("host_method") or "-").replace("=", ":"),
                 _one_line(m.get("gpu_method") or "-").replace("=", ":")), flush=True)

    def context(self, arm_name, r):
        mem = self

        class _Ctx(object):
            def __enter__(self):
                mem._probe(arm_name).start()
                return self

            def __exit__(self, exc_type, exc, tb):
                if exc_type is not None:
                    return False
                m = mem._probe(arm_name).stop()
                mem.emit(arm_name, r, m)
                return False
        return _Ctx()
