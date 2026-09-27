"""THE PREP LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `prep` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_prep",) once bindings/build_x_prep.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_prep", binding="_mojolearn_x_prep_host",
                        routes="_mojolearn_x_prep", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_prep",)
FAMILIES = (
    dict(
        family="x_prep",
        binding="_mojolearn_x_prep_host",
        routes="_mojolearn_x_prep",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("x-prep-dummy",),
        inference_lanes=(),
        forest_kinds=(),
        classes=("ExpansionDummyScaler",),
        display="the expansion proof dummy",
        host_modules=("x_prep/host/dummy_oracle.mojo",),
        exports=(
            "x_prep_host_numeric_mode", "x_prep_host_vendor", "x_prep_host_column", "x_prep_host_sabotage",
            "x_prep_l1_mean", "x_prep_scale", "x_prep_numeric_mode", "x_prep_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the proof dummy's CPU route (lane/algos-prep; removed before merge).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {"x-prep-dummy": "the expansion proof dummy"}
PUBLIC_PENDING_LANES = {"x-prep-dummy": "no reference"}
