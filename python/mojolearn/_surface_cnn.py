"""THE CNN LANE'S HOST SURFACE FRAGMENT.

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
        training_lanes=("x-cnn-conv2d", "x-cnn-conv1d", "x-cnn-pool", "x-cnn-trainer", "x-cnn-batchnorm",
                        "x-cnn-dropout2d", "x-cnn-globalpool", "x-cnn-resnet-block",
                        "x-cnn-gcn", "x-cnn-sage", "x-cnn-conv-options",
                        "x-cnn-pool-options", "x-cnn-bn-options",
                        "x-cnn-gnn-options", "x-cnn-trainer-options"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("Conv2d", "Conv1d", "MaxPool2d", "AvgPool2d", "MaxPool1d", "AvgPool1d", "CNNClassifier", "BatchNorm2d", "BatchNorm1d", "Dropout2d",
                 "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "BasicBlock", "GCNConv", "SAGEConv"),
        display="the CNN layers (conv, pooling, normalization) and the small CNN trainer",
        host_modules=("x_cnn/ops.mojo", "x_cnn/host/ops_host.mojo", "x_cnn/host/gemm_host.mojo"),
        exports=(
            "x_cnn_host_numeric_mode", "x_cnn_host_vendor", "x_cnn_host_column", "x_cnn_host_sabotage",
            "x_cnn_gemm", "x_cnn_conv2d_forward", "x_cnn_conv2d_backward", "x_cnn_conv_shape",
            "x_cnn_conv_block_forward", "x_cnn_conv_block_backward",
            "x_cnn_pool_shape", "x_cnn_maxpool2d_forward", "x_cnn_maxpool2d_backward",
            "x_cnn_avgpool2d_forward", "x_cnn_avgpool2d_backward",
            "x_cnn_relu_forward", "x_cnn_relu_backward", "x_cnn_add", "x_cnn_linear_forward",
            "x_cnn_linear_backward", "x_cnn_softmax_xent", "x_cnn_sgd", "x_cnn_adam",
            "x_cnn_batchnorm_forward", "x_cnn_batchnorm_backward", "x_cnn_dropout2d", "x_cnn_mul",
            "x_cnn_spmm", "x_cnn_gcn_norm", "x_cnn_pad2d_forward", "x_cnn_pad2d_backward",
            "x_cnn_adaptive_pool", "x_cnn_graph_op",
            "x_cnn_numeric_mode", "x_cnn_vendor",
            "x_cnn_res_alloc", "x_cnn_res_free", "x_cnn_res_upload", "x_cnn_res_download", "x_cnn_res_argmax", "x_cnn_res_gather",
            "x_cnn_conv_block_forward_r", "x_cnn_conv_block_backward_r", "x_cnn_linear_forward_r",
            "x_cnn_linear_backward_r", "x_cnn_softmax_xent_r", "x_cnn_sgd_r", "x_cnn_adam_r", "x_cnn_fit_epoch_r",
            "x_cnn_idn2_flags", "x_cnn_epoch_rows", "x_cnn_adam_hyper_d", "x_cnn_fit_epoch_d",
        ),
        gate="tools/identity_break.py (tools/algos_lane_check.sh)",
        wheel_note="Ships: the CNN lane's CPU route (convolution, pooling, normalization, the CNN trainer, graph convolution).",
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {
    "x-cnn-conv2d": "Conv2d forward and backward",
    "x-cnn-conv1d": "Conv1d forward and backward",
    "x-cnn-pool": "MaxPool and AvgPool (1d, 2d) forward and backward",
    "x-cnn-trainer": "the small CNN trainer (CNNClassifier)",
    "x-cnn-batchnorm": "BatchNorm2d / BatchNorm1d forward and backward",
    "x-cnn-dropout2d": "Dropout2d (Philox channel mask)",
    "x-cnn-globalpool": "global and adaptive average / max pooling",
    "x-cnn-resnet-block": "the ResNet BasicBlock forward and backward",
    "x-cnn-gcn": "GCNConv (PyG) forward and backward",
    "x-cnn-sage": "SAGEConv (PyG) forward and backward",
    "x-cnn-conv-options": "Conv2d padding modes, same/valid padding and groups",
    "x-cnn-pool-options": "pooling ceil_mode and divisor_override",
    "x-cnn-bn-options": "BatchNorm momentum=None and track_running_stats=False",
    "x-cnn-gnn-options": "SAGEConv max aggregation, normalize and project",
    "x-cnn-trainer-options": "CNNClassifier Adam, AdamW, Nesterov and dampened SGD",
}
PUBLIC_PENDING_LANES = {
    }
