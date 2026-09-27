"""THE CNN LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `cnn` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_cnn",) once bindings/build_x_cnn.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_cnn", binding="_mojolearn_x_cnn_host",
                        routes="_mojolearn_x_cnn", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_cnn",)
FAMILIES = (
    dict(
        family="x_cnn",
        binding="_mojolearn_x_cnn_host",
        routes="_mojolearn_x_cnn",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("x-cnn-conv2d", "x-cnn-conv1d", "x-cnn-pool"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("Conv2d", "Conv1d", "MaxPool2d", "AvgPool2d", "MaxPool1d", "AvgPool1d"),
        display="the CNN layers (conv, pooling, normalization) and the small CNN trainer",
        host_modules=("x_cnn/ops.mojo", "x_cnn/host/ops_host.mojo"),
        exports=(
            "x_cnn_host_numeric_mode", "x_cnn_host_vendor", "x_cnn_host_column", "x_cnn_host_sabotage",
            "x_cnn_gemm", "x_cnn_conv2d_forward", "x_cnn_conv2d_backward", "x_cnn_conv_shape",
            "x_cnn_pool_shape", "x_cnn_maxpool2d_forward", "x_cnn_maxpool2d_backward",
            "x_cnn_avgpool2d_forward", "x_cnn_avgpool2d_backward",
            "x_cnn_numeric_mode", "x_cnn_vendor",
        ),
        gate="tools/identity_break.py (tools/algos_lane_check.sh)",
        wheel_note="Ships: the CNN lane's CPU route (convolution, pooling and the CNN trainer).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-cnn-conv2d": "Conv2d forward and backward",
    "x-cnn-conv1d": "Conv1d forward and backward",
    "x-cnn-pool": "MaxPool and AvgPool (1d, 2d) forward and backward",
}
PUBLIC_PENDING_LANES = {
    "x-cnn-conv2d": "no reference",
    "x-cnn-conv1d": "no reference",
    "x-cnn-pool": "no reference",
}
