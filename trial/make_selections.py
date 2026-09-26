"""Mojo 1.2 nightly trial: the 0.8.19 release selections, per backend.

cuda: the 0.8.19 NVIDIA selection as recorded (copied to trial/selection-cuda.json).
metal: the lanes of the recorded 0.8.19 Metal column.
cpu: the union of both, minus the lanes degenerate on cpu-host.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lane_applicability  # noqa: E402
import lane_select  # noqa: E402

c = json.load(open(os.path.join(ROOT, "trial/selection-cuda.json")))
m = json.load(open(os.path.expanduser("~/mojolearn-evidence/release-check/69a519c1522d/metal/column.json")))
ml = {k.split("/")[0] for k in m["cells"]}
print("metal lanes", len(ml), "cuda lanes", len(c["lanes"]), "same", ml == set(c["lanes"]))
order = lane_select.all_lanes()
metal = dict(c, backend="metal", column="apple-metal", lanes=[n for n in order if n in ml], dropped={},
             summary="0.8.19 Metal column lanes (Mojo 1.2 trial)")
json.dump(metal, open(os.path.join(ROOT, "trial/selection-metal.json"), "w"), indent=1)
skip = lane_applicability.degenerate("cpu-host")
u = ml | set(c["lanes"])
cpu = [n for n in order if n in u and n not in skip]
json.dump(dict(c, backend="cpu", column="cpu-host", lanes=cpu, dropped={n: skip[n] for n in u if n in skip},
               summary="0.8.19 Metal+NVIDIA lanes runnable on cpu-host (Mojo 1.2 trial)"),
          open(os.path.join(ROOT, "trial/selection-cpu.json"), "w"), indent=1)
print("cpu lanes", len(cpu))
