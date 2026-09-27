"""THE SEQUENCE LANE'S HOST SURFACE FRAGMENT (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `sequence` expansion lane and merged by host_surface.py at import
(`EXPANSION_LANES`). Literal data only: no imports, no calls but dict(...).
The names this lane may declare are its own:

  GPU_BINDINGS          ("_mojolearn_x_sequence",) once bindings/build_x_sequence.sh builds
                        it; every packaging list then carries it
                        (`host_surface.py --expansion-gpu-bindings`)
  FAMILIES              (dict(family="x_sequence", binding="_mojolearn_x_sequence_host",
                        routes="_mojolearn_x_sequence", ...),), the same keys as a
                        family in host_surface.FAMILIES
  TRAINING_LANE_NAMES   {lane: "the name the docs use"} for the family's training_lanes
  PUBLIC_PENDING_LANES  {lane: "no reference"} until a release record admits it
"""
GPU_BINDINGS = ("_mojolearn_x_sequence",)
FAMILIES = (
    dict(
        family="x_sequence",
        binding="_mojolearn_x_sequence_host",
        routes="_mojolearn_x_sequence",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("sequence-lstm", "sequence-gru", "sequence-rmsprop", "sequence-adagrad",
                        "sequence-autoarima", "sequence-stl", "sequence-var",
                        "sequence-mlp", "sequence-rnn", "sequence-lion",
                        "sequence-adafactor", "sequence-lamb",
                        "sequence-adamax", "sequence-nadam",
                        "sequence-lr-schedulers", "sequence-layernorm",
                        "sequence-theta", "sequence-croston",
                        "sequence-ets", "sequence-garch",
                        "sequence-prophet", "sequence-moe"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("LSTMRegressor", "LSTMClassifier", "GRURegressor", "GRUClassifier", "RMSprop", "Adagrad", "AutoARIMA", "STL", "VAR", "MLPClassifier",
                 "MLPRegressor", "RNNRegressor", "RNNClassifier", "Lion", "Adafactor", "LAMB", "Adamax", "NAdam", "StepLR",
                 "ExponentialLR", "OneCycleLR", "LayerNorm", "Theta",
                 "OptimizedTheta", "DynamicTheta", "DynamicOptimizedTheta", "AutoTheta",
                 "CrostonClassic", "CrostonOptimized", "CrostonSBA", "ETS", "DampedETS", "GARCH",
                 "ProphetForecaster", "MoEBlock"),
        display="the sequence lane's recurrent networks and optimizers",
        host_modules=("sequence/ops.mojo", "sequence/exec.mojo", "sequence/recurrent.mojo",
                      "sequence/pyapi.mojo", "sequence/stl.mojo", "sequence/dispatch.mojo", "sequence/vecar.mojo", "sequence/mlp.mojo",
                      "sequence/mlp_fit.mojo", "sequence/adafactor.mojo",
                      "sequence/layernorm.mojo", "sequence/nm.mojo", "sequence/theta.mojo",
                      "sequence/croston.mojo", "sequence/ets.mojo",
                      "sequence/garch.mojo", "sequence/prophet.mojo",
                      "sequence/moe.mojo"),
        exports=(
            "x_sequence_host_numeric_mode", "x_sequence_host_vendor", "x_sequence_host_column",
            "x_sequence_host_sabotage", "x_sequence_numeric_mode", "x_sequence_vendor",
            "rnn_fit", "rnn_predict", "rnn_n_params", "optimizer_step", "stl", "var_fit", "var_forecast", "mlp_fit", "mlp_predict", "adafactor_step", "lamb_step", "layer_norm", "theta", "croston", "ets", "garch", "prophet_fit", "prophet_predict", "moe_forward",
        ),
        gate="tools/algos_lane_check.sh (pass 1: CPU == GPU bitwise)",
        wheel_note=(
            "Ships: the recurrent fits and predictions and the optimizer step, the GPU "
            "binding's element bodies looped on the host."
        ),
        ships_in_wheel=True,
    ),
)
TRAINING_LANE_NAMES = {"sequence-lstm": "LSTMRegressor / LSTMClassifier",
                       "sequence-gru": "GRURegressor / GRUClassifier",
                       "sequence-rmsprop": "RMSprop",
                       "sequence-adagrad": "Adagrad",
                       "sequence-autoarima": "AutoARIMA",
                       "sequence-stl": "STL",
                       "sequence-var": "VAR",
                       "sequence-mlp": "MLPClassifier / MLPRegressor",
                       "sequence-rnn": "RNNRegressor / RNNClassifier",
                       "sequence-lion": "Lion",
                       "sequence-adafactor": "Adafactor",
                       "sequence-lamb": "LAMB",
                       "sequence-adamax": "Adamax",
                       "sequence-nadam": "NAdam",
                       "sequence-lr-schedulers": "StepLR / ExponentialLR / OneCycleLR",
                       "sequence-layernorm": "LayerNorm",
                       "sequence-theta": "Theta / AutoTheta",
                       "sequence-croston": "CrostonClassic / CrostonOptimized / CrostonSBA",
                       "sequence-ets": "ETS / DampedETS",
                       "sequence-garch": "GARCH",
                       "sequence-prophet": "ProphetForecaster",
                       "sequence-moe": "MoEBlock"}
PUBLIC_PENDING_LANES = {"sequence-lstm": "no reference", "sequence-gru": "no reference",
                        "sequence-rmsprop": "no reference",
                        "sequence-adagrad": "no reference",
                        "sequence-autoarima": "no reference",
                        "sequence-stl": "no reference",
                        "sequence-var": "no reference",
                        "sequence-mlp": "no reference",
                        "sequence-rnn": "no reference",
                        "sequence-lion": "no reference",
                        "sequence-adafactor": "no reference",
                        "sequence-lamb": "no reference",
                        "sequence-adamax": "no reference",
                        "sequence-nadam": "no reference",
                        "sequence-lr-schedulers": "no reference",
                        "sequence-layernorm": "no reference",
                        "sequence-theta": "no reference",
                        "sequence-croston": "no reference",
                        "sequence-ets": "no reference",
                        "sequence-garch": "no reference",
                        "sequence-prophet": "no reference",
                        "sequence-moe": "no reference"}
