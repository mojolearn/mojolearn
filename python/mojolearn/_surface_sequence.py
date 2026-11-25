"""THE SEQUENCE LANE'S HOST SURFACE FRAGMENT.

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
                        "sequence-prophet", "sequence-moe",
                        # lane tape-verify (2026-10-21): one seeded step through the autograd tape
                        "sequence-tape-sequential", "sequence-tape-program",
                        # lane seq-blocks-3 (2026-11-22): the Sequential sequence-block baselines, one seeded step
                        "sequence-blocks-cnn1d", "sequence-blocks-resnet1d", "sequence-blocks-inceptiontime",
                        "sequence-blocks-gru-lengths", "sequence-blocks-lstm-lengths",
                        # lane neural-blocks-3 (2026-11-22): models composed from the public layers
                        "sequence-composed-transformer", "sequence-composed-mamba1",
                        "sequence-composed-mamba2", "sequence-composed-mamba3",
                        "sequence-composed-recurrent",
                        "sequence-composed-conv1d",
                        # lane nn-losses (2026-11-23): one seeded step per added loss
                        "sequence-composed-losses",
                        # lane nn-losses-2 (2026-11-23): log_softmax with nll / kl_div, and ctc
                        "sequence-composed-log-softmax-ctc",
                        # nn-recurrent begin (tools/identity_lanes/nn_recurrent.py)
                        "sequence-blocks-bigru-lengths", "sequence-blocks-bilstm",
                        # nn-recurrent end
                        # lane rnn-parity-D (2026-11-25): the recurrent estimators' options (tools/identity_lanes/rnn_parity.py)
                        "sequence-rnn-weighted-steps", "sequence-rnn-bidi-options",
                        # lane rnn-step-bits (2026-11-25): step() equals the whole pass at every input width
                        "sequence-rnn-step-widths",
                        # nn-attention begin: classic attention models (tools/identity_lanes/nn_attention.py)
                        "sequence-attention-encoder", "sequence-attention-cross",
                        # nn-attention end
                        # nn-conv begin
                        "sequence-nnconv-transpose", "sequence-nnconv-pools", "sequence-nnconv-pads", "sequence-nnconv-upsample", "sequence-nnconv-shuffle",
                        # nn-conv end
                        # nn-activations begin
                        "sequence-composed-activations",
                        # nn-activations end
                        # nn-norms begin: group and instance norm models, one seeded step
                        "sequence-norms-resnet1d-gn", "sequence-norms-instance",
                        # nn-norms end
                        # lane nn-optim begin (2026-11-23): Sequential.compile's rmsprop, adagrad, lamb and schedules
                        "sequence-optim-rmsprop", "sequence-optim-adagrad", "sequence-optim-lamb",
                        # lane nn-optim end
                        # nn-conv3d begin
                        "sequence-composed-conv3d", "sequence-composed-pool3d",
                        # nn-conv3d end
                        # lane verify-coverage-audit (2026-10-21): MoEBlock.backward,
                        # Sequential export/load/predict, import_model
                        "sequence-moe-backward", "sequence-export-roundtrip",
                        "sequence-onnx-import", "sequence-program-save-load",
                        "sequence-device-tensor",
                        # Native device-parameter updates and consumed state;
                        # references remain pending until actual column runs.
                        "sequence-adafactor-device", "sequence-lamb-device",
                        # lane verify-coverage-owed (2026-10-21): state_dict and
                        # load_state_dict of the sequence optimizers
                        "sequence-optimizer-state",
                        # lane verify-gaps-20261022: the functional layer_norm_backward
                        "sequence-layernorm-backward",
                        # lane verify-coverage-20261024: DynamicTheta and OptimizedTheta
                        # (tools/identity_lanes/public_gap_methods.py)
                        "public-theta-variants",
                        # w3-fix-13-s2 (2026-11-25): lanes added since the shipped table was admitted;
                        # RECORD OWED: the release's one verification records them
                        "check-generic-sequence",
                        "fn-layer-norm-backward",
                        "fn-layer-norm-forward",
                        "generic-attrs-sequence",
                        "generic-digest-sequence",
                        "generic-params-sequence",
                        "generic-pickle-sequence",
                        "m-autoarima-predict",
                        "m-prophet-add-seasonality",
                        "nn-ep-layernorm",
                        "nn-ep-moe",
                        "nn-ep-rnn-state",
                        "nn-ep-sequential-conv",
                        "nn-ep-sequential-dense",
                        "nn-ep-sequential-lm",
                        "nn-pg-imported-program",
                        "nn-pg-program-decode",
                        "nn-pg-session-blocks",
                        # lane rnn-parity-B (2026-11-25): the per-step weighted model on ops.Session
                        "ops-per-step-weighted-embedding",
                        "nn-tr-schedules",
                        "nn-tr-seqopt-state",
                        # nn-blocks-scan begin: Session.scan over a captured cell (tools/identity_lanes/nn_scan.py)
                        "ops-scan-cell",
                        # nn-blocks-scan end
                        "public-theta-variants-forecast",
                        "sequence-croston-forecast",
                        "sequence-ets-forecast",
                        "sequence-theta-forecast",
                        # nn-blocks-pg begin: ops.Session optimizer parameter groups (tools/identity_lanes/nn_pg.py)
                        "ops-param-groups", "ops-param-groups-program-fit", "training-param-groups",
                        # nn-blocks-pg end
                        # nn-blocks-scatter begin: ops.Session scatter_add and segment reductions
                        "ops-scatter-segment",
                        # nn-blocks-scatter end
                        # nn-blocks-pgseq begin: Sequential.compile param_groups (tools/identity_lanes/nn_pgseq.py)
                        "sequential-param-groups", "training-param-groups-device",
                        # nn-blocks-pgseq end
                        # nn-blocks-composed begin: the composed tagger (tools/identity_lanes/nn_composed.py)
                        "ops-composed-tagger",
                        # nn-blocks-composed end
                        ),
        inference_lanes=(),
        forest_kinds=(),
        classes=("LSTMRegressor", "LSTMClassifier", "GRURegressor", "GRUClassifier", "RMSprop", "Adagrad", "AutoARIMA", "STL", "VAR", "MLPClassifier",
                 "MLPRegressor", "RNNRegressor", "RNNClassifier", "Lion", "Adafactor", "LAMB", "Adamax", "NAdam", "StepLR",
                 "ExponentialLR", "OneCycleLR", "LayerNorm", "Theta",
                 "OptimizedTheta", "DynamicTheta", "DynamicOptimizedTheta", "AutoTheta",
                 "CrostonClassic", "CrostonOptimized", "CrostonSBA", "ETS", "DampedETS", "GARCH",
                 "ProphetForecaster", "MoEBlock", "Sequential"),
        display="the sequence lane's recurrent networks and optimizers",
        host_modules=("sequence/ops.mojo", "sequence/exec.mojo", "sequence/recurrent.mojo",
                      "providers/python/sequence_pyapi_door.mojo", "sequence/stl.mojo", "sequence/dispatch.mojo", "sequence/vecar.mojo", "sequence/mlp.mojo",
                      "sequence/mlp_fit.mojo", "sequence/adafactor.mojo",
                      "sequence/layernorm.mojo", "sequence/nm.mojo", "sequence/theta.mojo",
                      "sequence/croston.mojo", "sequence/ets.mojo",
                      "sequence/garch.mojo", "sequence/prophet.mojo",
                      "sequence/moe.mojo", "autograd/model.mojo", "autograd/model_ops.mojo",
                      "providers/python/autograd_pyapi_door.mojo", "autograd/ops_api.mojo", "autograd/losses.mojo",
                      "autograd/metrics.mojo", "autograd/data.mojo",
                      "autograd/profile.mojo", "autograd/optim.mojo", "autograd/higher.mojo",
                      "autograd/model_checkpoint.mojo", "training/checkpoint.mojo",
                      "autograd/program.mojo", "autograd/weight_import.mojo",
                      # the backward bodies moved out of sequence/recurrent*.mojo and sequence/mlp.mojo
                      "autograd/kernels/recurrent.mojo", "autograd/kernels/recurrent_scan.mojo",
                      "autograd/kernels/mlp.mojo"),
        exports=(
            "x_sequence_host_numeric_mode", "x_sequence_host_vendor", "x_sequence_host_column",
            "x_sequence_host_sabotage", "x_sequence_numeric_mode", "x_sequence_vendor",
            "rnn_fit", "rnn_predict", "optimizer_step", "mlp_fit", "mlp_predict", "layer_norm", "moe_forward",
            # theta, croston, ets, garch, prophet_*, var_fit, var_forecast and stl left the Python surface
            # (lane calltable-xseq, 2026-11-11): the sequence owner reaches them
            # through the native table (core/xseq_entry_table.mojo).
            # the model-description trainer (autograd/model.mojo)
            "model_native_api_v1", "model_fit", "model_predict", "model_n_params", "model_labels", "model_gather_labels",
            "model_fit_labels", "model_predict_labels",
            "moe_backward",
            # the operation session (autograd/ops_api.mojo; mojolearn.ops)
            "ops_session_create", "session_public_api_v1", "ops_session_public_connect",
            "ops_shape",
            # checkpoints (autograd/model_checkpoint.mojo)
            "model_save_checkpoint", "model_checkpoint_info", "model_load_checkpoint",
            "ops_program_fit_native", "ops_session_call", "ops_session_initialize", "ops_program_public_call", "ops_program_execute",
            "ops_program_inspect", "ops_program_control",
            "owned_program_public_api_v1", "ops_owned_program_connect", "ops_owned_program_call",
            "ops_distributed_open", "ops_distributed_load", "ops_distributed_fit",
            "ops_distributed_run", "ops_distributed_state", "ops_distributed_residency",
            "ops_distributed_save", "ops_distributed_close", "ops_distributed_enter", "ops_distributed_exit",
            "ops_partitioned_open", "ops_partitioned_load", "ops_partitioned_fit",
            "ops_partitioned_run", "ops_partitioned_state", "ops_partitioned_residency",
            "ops_partitioned_save", "ops_partitioned_close", "ops_partitioned_enter", "ops_partitioned_exit", "ops_partitioned_profile",
            # weight import (autograd/weight_import.mojo; Sequential.load_state_dict, Session.load_weights)
            "model_import_params",
            # the ONNX model reader (autograd/import_reader.mojo, import_resolve.mojo; importers.onnx.read)
            "onnx_read", "onnx_buffers",
            # the lowering onto session operations (autograd/import_lower.mojo; importers.import_model)
            "onnx_import", "onnx_import_file", "ops_imported_run", "ops_import_model", "ops_torch_program_run",
            # model verification reports (autograd/model_verify.mojo; python -m mojolearn verify-model)
            "verify_digest", "verify_file_sha256", "verify_model_info", "verify_fill",
            # manifest sync (lane ne1-k-fail-b, 2026-11-07): registered in the binding source
            "onnx_native_api_v1", "ops_graph_record",
            # lane x2-small-mlp-on-executor (2026-11-09): the trainer's captured program
            "small_mlp_program",
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
                       "sequence-moe": "MoEBlock",
                       "sequence-tape-sequential": "Sequential forward, backward and optimizer step",
                       "sequence-tape-program": "ops.Session backward and step, and Program.fit",
                       "sequence-blocks-cnn1d": "Sequential 1-D CNN (conv1d, max_pool1d, adaptive_avg_pool1d)",
                       "sequence-blocks-resnet1d": "Sequential ResNet-1D with batch_norm1d and residual_block",
                       "sequence-blocks-inceptiontime": "Sequential InceptionTime module (parallel, batch_norm1d)",
                       "sequence-blocks-gru-lengths": "Sequential gru with sequence lengths",
                       "sequence-blocks-lstm-lengths": "Sequential lstm with sequence lengths",
                       "sequence-composed-transformer": "Sequential composed: rms_norm, attention",
                       "sequence-composed-mamba1": "Sequential composed: selective_scan",
                       "sequence-composed-mamba2": "Sequential composed: ssd_scan",
                       "sequence-composed-mamba3": "Sequential composed: trapezoid_scan",
                       "sequence-composed-recurrent": "Sequential composed: lstm, gru",
                       "sequence-composed-conv1d": "Sequential composed: conv1d, narrow",
                       "sequence-composed-losses": "Sequential losses: l1, smooth_l1, nll, kl_div, bce pos_weight",
                       "sequence-composed-log-softmax-ctc": "Sequential log_softmax (nll, kl_div) and ctc",  # nn-losses-2
                       # nn-recurrent begin
                       "sequence-blocks-bigru-lengths": "Sequential gru, two bidirectional layers, lengths",
                       "sequence-blocks-bilstm": "Sequential lstm, bidirectional",
                       # nn-recurrent end
                       # lane rnn-parity-D (2026-11-25)
                       "sequence-rnn-weighted-steps": "GRUClassifier: output='sequence', lengths, per-step sample_weight, class_weight, clip_grad_norm",
                       "sequence-rnn-bidi-options": "LSTMRegressor: bidirectional, dropout, bias=False, validation, partial_fit, h_0 / c_0, step refused",
                       "sequence-rnn-step-widths": "LSTMRegressor: step one step at a time equals predict, 8 / 64 / 65 / 300 inputs",
                       # nn-attention begin
                       "sequence-attention-encoder": "Sequential transformer_encoder_layer, positional_encoding",
                       "sequence-attention-cross": "Sequential transformer_decoder_layer, multihead_attention",
                       # nn-attention end
                       # nn-conv begin
                       "sequence-nnconv-transpose": "Sequential conv_transpose1d and conv_transpose2d",
                       "sequence-nnconv-pools": "Sequential adaptive pools and global_max_pool",
                       "sequence-nnconv-pads": "Sequential zero, constant, reflection and replication pads",
                       "sequence-nnconv-upsample": "Sequential upsample (nearest, linear, bilinear)",
                       "sequence-nnconv-shuffle": "Sequential pixel_shuffle and pixel_unshuffle",
                       # nn-conv end
                       # nn-activations begin
                       "sequence-composed-activations": "Sequential composed: the activation and identity layers",
                       # nn-activations end
                       # nn-norms begin
                       "sequence-norms-resnet1d-gn": "Sequential ResNet-1D with group_norm",
                       "sequence-norms-instance": "Sequential 1-D CNN with instance_norm1d",
                       # nn-norms end
                       # nn-conv3d begin
                       "sequence-composed-conv3d": "Sequential composed: conv3d, max_pool3d, adaptive_avg_pool3d",
                       "sequence-composed-pool3d": "Sequential composed: grouped conv3d, avg_pool3d, max_pool3d",
                       # nn-conv3d end
                       "sequence-moe-backward": "MoEBlock backward",
                       "sequence-export-roundtrip": "Sequential export, load and predict",
                       "sequence-onnx-import": "import_model on an ONNX file",
                       "sequence-program-save-load": "ops.Program save and Session.load_program",
                       "sequence-device-tensor": "resident sequence tensors through an optimizer",
                       "sequence-adafactor-device": "Adafactor resident parameters, gradients and state",
                       "sequence-lamb-device": "LAMB resident parameters, gradients and state",
                       "sequence-optimizer-state": "NAdam and RMSprop state_dict and load_state_dict",
                       "sequence-layernorm-backward": "layer_norm_backward",
                       "public-theta-variants": "DynamicTheta / OptimizedTheta",
                       # lane nn-optim begin
                       "sequence-optim-rmsprop": "Sequential rmsprop with cosine_warm_restarts and multi_step",
                       "sequence-optim-adagrad": "Sequential adagrad with reduce_on_plateau and cosine",
                       "sequence-optim-lamb": "Sequential lamb with one_cycle and linear_warmup",
                       # lane nn-optim end
    # w3-fix-13-s2 (2026-11-25): lanes added since the shipped table (RECORD OWED)
    "check-generic-sequence": "estimator.check across devices for every public sequence and neural class",
    "fn-layer-norm-backward": "layer_norm_backward on host and device-resident inputs",
    "fn-layer-norm-forward": "layer_norm_forward on host and device-resident inputs",
    "generic-attrs-sequence": "the fitted attributes and properties of every public sequence and neural class",
    "generic-digest-sequence": "digest_ and training_digest_ of every public sequence and neural class",
    "generic-params-sequence": "get_params, set_params and clone round trips of every public sequence and neural class",
    "generic-pickle-sequence": "pickle, deepcopy and clone-refit round trips of every public sequence and neural class",
    "m-autoarima-predict": "AutoARIMA: predict",
    "m-prophet-add-seasonality": "ProphetForecaster: seasonality",
    "nn-ep-layernorm": "LayerNorm: layernorm",
    "nn-ep-moe": "MoEBlock: moe",
    "nn-ep-rnn-state": "GRUClassifier, LSTMClassifier, RNNClassifier: state",
    "nn-ep-sequential-conv": "Sequential convolution layers: forward, export and reload",
    "nn-ep-sequential-dense": "Sequential dense layers: forward, export and reload",
    "nn-ep-sequential-lm": "Sequential: lm",
    "nn-pg-imported-program": "Session: program",
    "nn-pg-program-decode": "Session: decode",
    "nn-pg-session-blocks": "Mamba1Block, Session: blocks",
    "ops-per-step-weighted-embedding": "Session: per-step weighted GRU with an embedding, h_0, reverse, streaming",
    "nn-tr-schedules": "StepLR, ExponentialLR, OneCycleLR: schedules",
    "nn-tr-seqopt-state": "Adagrad, Adamax, LAMB: state",
    # nn-blocks-scan begin
    "ops-scan-cell": "Session: scan over a captured cell (GRU cell, linear recurrence, reverse, lengths, captured)",
    # nn-blocks-scan end
    "public-theta-variants-forecast": "DynamicTheta, OptimizedTheta: forecast",
    "sequence-croston-forecast": "CrostonClassic, CrostonOptimized, CrostonSBA: forecast",
    "sequence-ets-forecast": "DampedETS, ETS: forecast",
    "sequence-theta-forecast": "AutoTheta, DynamicOptimizedTheta, Theta: forecast",
    # nn-blocks-pg begin
    "ops-param-groups": "Session: optimizer parameter groups, global clip, state dict",
    "ops-param-groups-program-fit": "Program.fit with an optimizer that has parameter groups",
    "training-param-groups": "SGD, AdamW: parameter groups, one clip over every group, state dict",
    # nn-blocks-pg end
    # nn-blocks-pgseq begin
    "sequential-param-groups": "Sequential: compile param_groups and no_decay, checkpoint resume",
    "training-param-groups-device": "SGD, AdamW: parameter groups stepped from device gradients",
    # nn-blocks-pgseq end
    # nn-blocks-scatter begin
    "ops-scatter-segment": "Session: scatter_add, segment_sum, segment_mean, segment_max",
    # nn-blocks-scatter end
    # nn-blocks-composed begin
    "ops-composed-tagger": "Session: a masked tagger (embedding, scan cell with dropout, parameter groups), resumed",
    # nn-blocks-composed end
                       }
PUBLIC_PENDING_LANES = {
    # lane w3f2-main (2026-11-24): "no reference", not "unwatched": the shipped table
    # (verify_reference/table.json) carries no cell for these lanes, so every part reads OWED
    # (gating, never a pass) until the release record admits them (test_host_surface).
    "sequence-tape-sequential": "unwatched",
    "sequence-tape-program": "unwatched",
    "sequence-blocks-cnn1d": "no reference",
    "sequence-blocks-resnet1d": "no reference",
    "sequence-blocks-inceptiontime": "no reference",
    "sequence-blocks-gru-lengths": "no reference",
    "sequence-blocks-lstm-lengths": "no reference",
    "sequence-composed-transformer": "no reference",
    "sequence-composed-mamba1": "no reference",
    "sequence-composed-mamba2": "no reference",
    "sequence-composed-mamba3": "no reference",
    "sequence-composed-recurrent": "no reference",
    "sequence-composed-conv1d": "no reference",
    "sequence-composed-losses": "no reference",  # lane nn-losses
    "sequence-composed-log-softmax-ctc": "no reference",  # lane nn-losses-2
    # nn-recurrent begin
    "sequence-blocks-bigru-lengths": "no reference",
    "sequence-blocks-bilstm": "no reference",
    # nn-recurrent end
    "sequence-rnn-weighted-steps": "no reference",  # lane rnn-parity-D
    "sequence-rnn-bidi-options": "no reference",  # lane rnn-parity-D
    "sequence-rnn-step-widths": "no reference",  # lane rnn-step-bits
    # nn-attention begin
    "sequence-attention-encoder": "no reference",
    "sequence-attention-cross": "no reference",
    # nn-attention end
    # nn-conv begin
    "sequence-nnconv-transpose": "no reference",
    "sequence-nnconv-pools": "no reference",
    "sequence-nnconv-pads": "no reference",
    "sequence-nnconv-upsample": "no reference",
    "sequence-nnconv-shuffle": "no reference",
    # nn-conv end
    # nn-activations begin
    "sequence-composed-activations": "no reference",
    # nn-activations end
    # nn-norms begin
    "sequence-norms-resnet1d-gn": "no reference",
    "sequence-norms-instance": "no reference",
    # nn-norms end
    # nn-conv3d begin
    "sequence-composed-conv3d": "no reference",
    "sequence-composed-pool3d": "no reference",
    # nn-conv3d end
    "sequence-moe-backward": "unwatched",
    "sequence-export-roundtrip": "unwatched",
    "sequence-onnx-import": "unwatched",
    "sequence-program-save-load": "unwatched",
    "sequence-device-tensor": "unwatched",
    "sequence-adafactor-device": "unwatched",
    "sequence-lamb-device": "unwatched",
    "sequence-optimizer-state": "unwatched",
    "sequence-layernorm-backward": "unwatched",
    "public-theta-variants": "unwatched",
    # lane nn-optim begin
    "sequence-optim-rmsprop": "no reference",
    "sequence-optim-adagrad": "no reference",
    "sequence-optim-lamb": "no reference",
    # lane nn-optim end
    # nn-blocks-scatter begin
    "ops-scatter-segment": "no reference",
    # nn-blocks-scatter end
}
