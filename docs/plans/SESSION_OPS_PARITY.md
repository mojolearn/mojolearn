# Session ops parity with Sequential (lane session-ops, 2026-11-25)

Andrew (2026-11-25): a model author must be able to use every layer Mojolearn already has inside a custom graph (a
second input, an auxiliary loss, a user-defined scan cell, a custom backward rule). `ops.Session` exposes the
EXISTING kernels Sequential uses; no new kernel is written where one exists. This file is the inventory and the
design; the sub-lanes update their rows ("added" with the Session name) as they land.

## How a layer becomes a Session op (no existing op pays)

- **Same nodes, same bits.** A Session op calls the Sequential layer's own forward function on the tape
  (`nn_conv_transpose2d`, `rankn_conv3d`, `graph_group_norm`, `nn_mha_forward`, ...). It records the same nodes
  and tags, so its forward and backward are Sequential's bits by construction; a native check compares the two.
- **Dispatch.** Session codes 130..179 (`autograd/session_layer_rules.mojo`): conv 130..139, pool 140..149,
  resize/shuffle/pad 150..159, the rest 160..179. `OpsSession._op_run_tensors` tests the range once on the chain's
  no-match path (after scan and custom op), which no older operation reaches; `_layer_op` picks the family
  method. A layer op is a plain operation, so a scan cell or a custom operation's program may use it.
- **Options that an existing op lacks** (conv2d dilation, a pool's padding or ceil_mode) never add a word to the
  existing op: the door chooses a NEW code at UNPACK when the call uses them, so `conv2d(x, w)` still records
  OPS_CONV2D with the instructions it ran before.
- **Backward.** Sequential differentiates its custom tags with ModelRules; a session uses StepOpRules /
  ImportRules. `OpsSession.layer_nodes` is set when a layer op records a node; every backward entry (Session
  scalar and seeded backward, the custom-op paths, Program executor, control VJP, Program.fit) tests it once per
  call and then wraps its usual rules in `LayerRules[...]` (NnConvRules, NNAttentionRules, NnCtcRules first, then
  the inner rules unchanged). A session that never used a layer op runs exactly its old rules. Program.fit tests
  the Bool once per batch beside a whole backward (no lookup, no allocation).
- **Program capture.** `PROGRAM_MAX_CODE` 129 -> 179; capture, save, load and replay go through `_op_run` like
  every op. Frontend codes `_SOP_*` are 1000 + the session code.
- **Higher order.** `vjp`/`jvp`/`hvp` and `create_graph=True` differentiate with NeuralRules / recorded
  backward and refuse a layer node by name ("custom backward rule not registered"), as they refuse the
  custom op.

## Inventory

Sequential layer codes are `python/mojolearn/_x_sequence_model.py` `_LAYER`. "exists" = a Session method today.

| Layer (Sequential code) | Sequential kernel / forward | Session op | Backward rule | Owner |
|---|---|---|---|---|
| linear (1) | tape linear | `linear` exists | exists | - |
| relu, sigmoid, tanh, softmax (2..5) | tape elementwise / softmax | exist | exist | - |
| layer_norm (6), rms_norm (35) | layer_norm_core, graph rms | exist | exist | - |
| dropout (7) | tape dropout | exists | exists | - |
| residual, branch, concat, multiply, narrow (8, 19, 20, 37, 38) | containers / rankn | `add`, `concat`, `mul`, `split`/`index` exist | exist | - |
| reshape, permute, flatten (9..11) | rankn layout | exist | exist | - |
| conv2d (12) | rankn_conv2d | `conv2d` exists (no dilation, square padding) | exists | conv (dilation / per-axis padding as a new code) |
| max_pool2d, avg_pool2d (13, 14) | rankn pools | `max_pool2d`, `avg_pool2d` exist; PyTorch's options (per-axis tuples, padding, dilation, ceil_mode, count_include_pad, divisor_override) added as codes 142 / 143, chosen by the door only when used | exist | pool: added |
| conv1d (15) | rankn_conv2d over (N, C, 1, L) | missing | exists (RN conv tags) | conv |
| max_pool1d, avg_pool1d (16, 17) | rankn pools over (N, C, 1, L) | `max_pool1d` (140), `avg_pool1d` (141) added (dilation, ceil_mode, count_include_pad on the same node) | exists | pool: added |
| adaptive_avg_pool1d (18) | tape mean / rankn avg pool (Sequential 18), nn_adaptive_avg_pool1d for other lengths | `adaptive_avg_pool1d` (147, rank word 1) added, any output size | exists / NnConvRules | pool: added |
| silu, gelu, softplus (32..34) | import unary | exist | exist | - |
| rotary (36) | graph rotary | exists | exists | - |
| embedding (39) | tape embedding | exists | exists | - |
| attention (40) | attention block | `attention_block`, `attention` exist | exist | - |
| rnn / lstm / gru (41) | recurrent scan | `rnn`, `lstm`, `gru` exist | RecurrentRules | - |
| selective_scan, ssd_scan, trapezoid_scan (42..44) | ssm primitives | `mamba_block` exists; raw scans missing | SsmRules | rest (expose if a tensor-level forward exists, else list) |
| lm_head (45) | lm_head_v2 | `lm_head_loss` exists | exists | - |
| batch_norm1d / 2d (46) | rankn_batch_norm_core | missing | RN tags | rest |
| identity (47) | none | not needed (the tensor itself) | - | - |
| leaky_relu, elu, hardswish, hardsigmoid (48, 49, 52, 53) | import unary | exist | exist | - |
| prelu (50), mish (51) | imp_binary PRELU, imp_unary MISH | missing | import tags | rest |
| group_norm, instance_norm (54, 55) | graph_group_norm, graph_instance_norm | missing | RN tags | rest |
| conv_transpose1d / 2d (56, 57) | nn_conv_transpose1d / 2d | missing | NnConvRules | conv |
| adaptive_avg_pool2d (58), adaptive_max_pool2d / 1d (59, 60), global_max_pool (61) | nn_adaptive_* | `adaptive_avg_pool2d` (147, rank 2), `adaptive_max_pool1d` / `2d` (148) added; PyTorch has no global max pool name: it is `adaptive_max_pool2d(x, 1)` (shape (N, C, 1, 1); Sequential 61's rankn_max returns (N, C)) | NnConvRules | pool: added |
| constant / zero pad (62) | imp_pad | `pad` exists | exists | - |
| reflection / replication pad (63, 64) | nn_pad | missing (`pad(mode=)`) | NnConvRules | resize |
| upsample (65) | nn_upsample (nearest, bilinear) | missing (`interpolate`) | NnConvRules | resize |
| pixel_shuffle / unshuffle (66) | nn_pixel_shuffle | missing | NnConvRules | resize |
| multihead_attention (67) | nn_mha_forward | missing | NNAttentionRules | rest |
| positional_encoding (72) | nn_pos_enc_forward | missing | NNAttentionRules | rest |
| conv3d (73) | rankn_conv3d | missing | RN conv tags | conv |
| max_pool3d, avg_pool3d, adaptive_avg_pool3d (74..76) | rankn_*_pool3d | `max_pool3d` (144), `avg_pool3d` (145), `adaptive_avg_pool3d` (146) added | RN pool tags | pool: added |
| log_softmax (77) | nn_log_softmax | `log_softmax` exists | exists | - |
| losses mse, l1, huber, bce_with_logits, cross_entropy | tape losses | exist | exist | - |
| losses smooth_l1, nll, kl_div, bce pos_weight, ctc | nn_losses bodies, nn_ctc | missing | loss kinds / NnCtcRules | rest |

Expected not exposable without a new kernel (each sub-lane confirms or exposes): conv_transpose3d (no 3-D col2im
path), adaptive_max_pool3d, interpolate modes trilinear / bicubic / area, padding modes circular and 3-D
reflection / replication.

session-ops-pool (confirmed): `adaptive_max_pool3d` (no 3-D adaptive max body: x_cnn's adaptive max is 2-D);
`adaptive_avg_pool3d` with an output size that neither is 1 nor divides the input (rankn_adaptive_avg_pool3d takes
only the global mean or equal tiles; x_cnn's uneven adaptive window bodies are 2-D); `return_indices=True` on every
max pool (the bodies hold the winners in PyTorch's flat order, but a Session operation returns one tensor handle;
refused by name). Unbatched inputs ((C, L), (C, H, W), (C, D, H, W)) are refused by rank: add the batch axis.

## Verify design (Andrew, 2026-11-25)

1. **One light verify lane per op**, both sides through `device=` (`docs/plans/VERIFY_THROUGH_DEVICE.md`), smallest
   shapes that exercise padding / stride / dilation edges, a sabotage arm per side that the lane catches.
2. **One generic composed-model lane driven by data cases**, never a lane per composition. A case is a data file
   under the verify fixtures: a captured Program graph plus tiny inputs, and optional train / save / resume steps.
   The runner runs every case on both sides and digests the outputs and updates. Adding a case = adding a data
   file, no code. One sabotage arm per side on the runner (it drops one node's gradient contribution).
   nn-blocks' `ops-composed-tagger` moves over as the first case (its arms kept as case-level faults where they
   fit, else on the runner). The minimum further cases, each crossing one piece of glue once: tape / backward
   accumulation (an input on two paths, a non-unit seed); Program capture + replay; gradient accumulation with a
   masked sum and denominator; param groups + resume; scan with dropout over varying lengths.
3. Verify keeps working from the built wheel and fails on a fault (AGENTS.md "The verify command is protected").

## Sub-lanes

| Sub-lane | Codes | Kernel range | Rows |
|---|---|---|---|
| session-ops-conv | 130..139 | 6900..6919 | conv1d, conv3d, conv_transpose1d / 2d, conv2d with dilation / per-axis padding |
| session-ops-pool | 140..149 | 6920..6939 | 1-D / 3-D pools, adaptive pools, global max, pool padding |
| session-ops-resize | 150..159 | 6940..6959 | interpolate, pixel_shuffle / unshuffle, reflection / replication pad |
| session-ops-rest | 160..179 | 6960..6979 | batch / group / instance norm, prelu, mish, multihead attention, positional encoding, losses, raw SSM scans |
| session-ops-verify | - | - | the generic composed-model lane and its data cases |
