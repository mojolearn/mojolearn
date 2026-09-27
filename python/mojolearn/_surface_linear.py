"""THE LINEAR LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `linear` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_linear",) once bindings/build_x_linear.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_linear", binding="_mojolearn_x_linear_host",
                        routes="_mojolearn_x_linear", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_linear",)
FAMILIES = (
    dict(
        family="x_linear",
        binding="_mojolearn_x_linear_host",
        routes="_mojolearn_x_linear",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(
            "x-sgd-clf", "x-sgd-reg",
        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=(
            "SGDClassifier", "SGDRegressor",
        ),
        display="the linear expansion lane (SGD, GLMs, Huber, Bayesian, LARS, quantile, CV, isotonic)",
        host_modules=(
            "x_linear/ops.mojo", "x_linear/dispatch.mojo", "x_linear/sgd.mojo",
        ),
        exports=(
            "x_linear_host_numeric_mode", "x_linear_host_vendor", "x_linear_host_column", "x_linear_host_sabotage",
            "x_linear_fit", "x_linear_decision", "x_linear_numeric_mode", "x_linear_vendor",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        wheel_note="Ships: the linear expansion lane's CPU route (x_linear/, pass 1, PENDING).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-sgd-clf": "SGDClassifier",
    "x-sgd-reg": "SGDRegressor",
}
PUBLIC_PENDING_LANES = {
    "x-sgd-clf": "no reference",
    "x-sgd-reg": "no reference",
}
