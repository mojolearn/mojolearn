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
        training_lanes=(
            "x-prep-robust-scaler", "x-prep-maxabs-scaler",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "RobustScaler", "MaxAbsScaler",
        ),
        display="preprocessing additions, naive Bayes and discriminant analysis",
        host_modules=(
            "x_prep/host/program.mojo", "x_prep/common.mojo", "x_prep/prims.mojo", "x_prep/eigh.mojo",
            "x_prep/units.mojo",
        ),
        exports=(
            "x_prep_host_numeric_mode", "x_prep_host_vendor", "x_prep_host_column", "x_prep_host_sabotage",
            "x_prep_run", "x_prep_numeric_mode", "x_prep_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the prep lane's CPU route (preprocessing additions, naive Bayes, discriminant analysis).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-prep-robust-scaler": "RobustScaler",
    "x-prep-maxabs-scaler": "MaxAbsScaler",
}
PUBLIC_PENDING_LANES = {
    "x-prep-robust-scaler": "no reference",
    "x-prep-maxabs-scaler": "no reference",
}
