# Session ops parity with Sequential (lane session-ops, 2026-11-25)

Andrew (2026-11-25): a model author must be able to use every layer Mojolearn already has inside a custom graph (a
second input, an auxiliary loss, a user-defined scan cell, a custom backward rule). `ops.Session` exposes the
EXISTING kernels Sequential uses; no new kernel is written where one exists. This file is the inventory and the
design; the sub-lanes update their rows ("added" with the Session name) as they land.

## How a layer becomes a Session op (no existing op pays)

- **Same nodes, same bits.** A Session op calls the Sequential layer's own forward function on the tape
  (`nn_conv_transpose2d`, `rankn_conv3d`, `graph_group_norm`, `nn_mha_forward`, ...). It records the same nodes
  and tags, so its forward and backward are Sequential's bits by construction; a native check compares the two.
- **Dispatch.** Session codes 130..189 (`autograd/session_layer_rules.mojo`): conv 130..139, pool 140..149,
  resize/shuffle/pad 150..159, the rest 160..179, the round's additions 180..189. `OpsSession._op_run_tensors` tests the range once on the chain's
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
- **Program capture.** `PROGRAM_MAX_CODE` 129 -> 189; capture, save, load and replay go through `_op_run` like
  every op. Frontend codes `_SOP_*` are 1000 + the session code.
- **Higher order.** `vjp`/`jvp`/`hvp` and `create_graph=True` differentiate with NeuralRules / recorded
  backward and refuse a layer node by name ("custom backward rule not registered"), as they refuse the
  custom op.
- **Scan and custom-backward cells.** `Session.scan` accepts a cell built from ordinary ops, layer ops included
  (what the composed tagger case exercises). A cell containing a custom-backward op (`custom_op`), or a scan
  inside a custom op's program, is REFUSED at bind by name; nesting is not implemented in this wave.
- **Digests.** Session and Program expose no `digest_` / `training_digest_`. Verify hashes their selected outputs
  and state directly; that is the contract, and no digest property is added to them.
- **Fixtures.** As small as possible while the intended fault still changes the result; the case that matters
  (repeated ids, unequal lengths, a second optimizer group) over a bigger tensor. Related cases that exercise the
  same implementation share one lane / data file, with separate assertions where the defects differ. No wall
  time is claimed; the integrated suite is timed only when it runs.

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
| conv2d (12) | rankn_conv2d | `conv2d` exists; dilation, per-axis stride / padding and padding 'valid' / 'same' **added** (code 131, chosen by the door only when used; the default call keeps code 44) | exists | conv |
| max_pool2d, avg_pool2d (13, 14) | rankn pools | `max_pool2d`, `avg_pool2d` exist; PyTorch's options (per-axis tuples, padding, dilation, ceil_mode, count_include_pad, divisor_override) added as codes 142 / 143, chosen by the door only when used | exist | pool: added |
| conv1d (15) | rankn_conv1d (rankn_conv2d over (N, C, 1, L)) | `conv1d` **added** (code 130; PyTorch's dilation through rankn_conv1d's new `dilation` argument, default 1, Sequential's call unchanged) | exists (RN conv tags) | conv |
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
| conv_transpose1d / 2d (56, 57) | nn_conv_transpose1d / 2d | `conv_transpose1d`, `conv_transpose2d` **added** (codes 133, 134; per-axis words on 2-D) | NnConvRules | conv |
| adaptive_avg_pool2d (58), adaptive_max_pool2d / 1d (59, 60), global_max_pool (61) | nn_adaptive_* | `adaptive_avg_pool2d` (147, rank 2), `adaptive_max_pool1d` / `2d` (148) added; PyTorch has no global max pool name: it is `adaptive_max_pool2d(x, 1)` (shape (N, C, 1, 1); Sequential 61's rankn_max returns (N, C)) | NnConvRules | pool: added |
| constant / zero pad (62) | imp_pad | `pad` exists | exists | - |
| reflection / replication pad (63, 64) | nn_pad | added: `pad(x, pads, value, mode='reflect' \| 'replicate')` (code 153; `mode` None / 'constant' keeps code 73; ONNX pad order kept, `value` before `mode`) | NnConvRules | resize |
| upsample (65) | nn_upsample (nearest, bilinear) | added: `interpolate` (code 150; nearest, linear (N, C, L), bilinear; per-axis size / scale_factor, align_corners, recompute_scale_factor) | NnConvRules | resize |
| pixel_shuffle / unshuffle (66) | nn_pixel_shuffle | added: `pixel_shuffle`, `pixel_unshuffle` (codes 151, 152; (N, C, H, W) only) | rank-N view / permute tags | resize |
| multihead_attention (67) | nn_mha_forward | missing | NNAttentionRules | rest |
| positional_encoding (72) | nn_pos_enc_forward | missing | NNAttentionRules | rest |
| conv3d (73) | rankn_conv3d | `conv3d` **added** (code 132) | RN conv tags | conv |
| max_pool3d, avg_pool3d, adaptive_avg_pool3d (74..76) | rankn_*_pool3d | `max_pool3d` (144), `avg_pool3d` (145), `adaptive_avg_pool3d` (146) added | RN pool tags | pool: added |
| log_softmax (77) | nn_log_softmax | `log_softmax` exists | exists | - |
| losses mse, l1, huber, bce_with_logits, cross_entropy | tape losses | exist | exist | - |
| losses smooth_l1, nll, kl_div, bce pos_weight, ctc | nn_losses bodies, nn_ctc | missing | loss kinds / NnCtcRules | rest |

### The round's additions (Andrew, 2026-11-25; not Sequential layers)

| Op | Reuses | Session op | Backward | Owner |
|---|---|---|---|---|
| `sequence_mask(lengths, max_length)` | scan's length mask construction where the semantics agree | missing | none (mask) | extra |
| `topk(x, k, axis)` (lowest index wins a tie, every vendor) | existing sort / argmax bodies where they fit | missing | gather of the upstream to the selected positions | extra |
| `gather(x, axis, index)` (torch.gather / take_along_dim) | rankn take / embedding runs | missing | scatter_add (repeated indices accumulate) | extra |
| `logsumexp(x, axis)` | log-softmax's numerics | missing | softmax of the row times the upstream | extra |
| soft-target `cross_entropy` (a target distribution) | the cross-entropy / log-softmax bodies; the hard-label path keeps its code | missing | softmax minus target | extra |

Expected not exposable without a new kernel (each sub-lane confirms or exposes): conv_transpose3d (session-ops-conv
confirms: col2im, im2col and the row copies carry the depth words, but nn_conv_transpose2d's host path is rank-4 only
and its output body op_nnc_ct_out indexes the bias channel over H W, not D H W; exposing it needs that body made
depth-aware, a change to Sequential's conv_transpose node, so it is listed, not written), adaptive_max_pool3d, interpolate modes trilinear / bicubic / area, padding modes circular and 3-D
reflection / replication.

Not exposable without a new kernel (session-ops-resize, confirmed): interpolate modes trilinear, bicubic, area,
nearest-exact, `antialias=True`, and any 5-D input (nn_upsample is 2-D only); pad mode circular and 5-D reflection /
replication (nn_pad covers (N, C, L) and (N, C, H, W)); pixel_shuffle / unshuffle on inputs other than 4-D (torch's
leading batch dims) is refused, not emulated.

session-ops-pool (confirmed): `adaptive_max_pool3d` (no 3-D adaptive max body: x_cnn's adaptive max is 2-D);
`adaptive_avg_pool3d` with an output size that neither is 1 nor divides the input (rankn_adaptive_avg_pool3d takes
only the global mean or equal tiles; x_cnn's uneven adaptive window bodies are 2-D); `return_indices=True` on every
max pool (the bodies hold the winners in PyTorch's flat order, but a Session operation returns one tensor handle;
refused by name). Unbatched inputs ((C, L), (C, H, W), (C, D, H, W)) are refused by rank: add the batch axis.

## Verify design (Andrew, 2026-11-25)

1. **One light verify lane per op**, both sides through `device=` (`docs/plans/VERIFY_THROUGH_DEVICE.md`), smallest
   shapes that exercise padding / stride / dilation edges, a sabotage arm per side that the lane catches.
2. **One generic composed-model lane driven by data cases**, never a lane per composition: `ops-composed-cases`
   (`tools/identity_lanes/composed_cases.py`, lane session-ops-verify). A case is a declarative spec in a data file,
   `python/mojolearn/verify_reference/composed-*.json` (shipped by every packer through the existing
   `verify_reference/*.json` glob; nothing to generate, no binary). Schema `mojolearn.composed-cases.v1`: a file
   holds related cases; a case has tiny `data` arrays (literal values or the harness's hashed stream) and a list of
   statements that call the public `ops.Session` / `Program` methods BY NAME (`{"op": "call", "fn": "s.matmul",
   "args": ["@x", "@w"], "as": "y"}`), so any op, including ops added later (sequence_mask, topk, gather, ...),
   is usable in a case with no runner change. Verbs: session, call, map, let, read, array, np (host oracles only),
   repeat, def/use (macros), path, roundtrip (an .npz checkpoint), state_save/state_load (optimizer_state_dict),
   find/pick (Program.params), and the checks digest, same (bytes), differ, zero, close, oracle (weighted
   cross-entropy in float64), fd_grad (central differences of a captured Program). The runner runs every case on
   the harness's side (both sides through `device=`), REFUSES on a failed check, and hashes the named parts
   `<case>:<part>` (Session and Program expose no `digest_`). Adding a case = adding a data file. References OWED.
   Cases: `composed-tagger.json` tagger (nn-blocks' `ops-composed-tagger`, moved over; its lane file deleted, its
   native check `checks/native_tests/nn_composed.mojo` and its four arms kept as the case's faults);
   `composed-glue.json` tape-two-paths (an input on two paths, a 0.37 loss scale; gradients against central
   differences), program-replay (capture, replay to the bit, save/load), grad-accum-masked (two microbatches with a
   masked sum over a shared denominator against the whole batch), groups-resume (Adam, two groups, save, fresh-session
   resume bit equal), scan-dropout-lengths (lengths 1 and 3, keys 0 and 1, zeros past each length).
   Runner arms (`autograd/session_ops_verify.mojo`, test-build defines, off in the product): DROP_ACCUMULATE /
   _HOST, the tape skips every gradient contribution after a tensor's first on one side (`autograd/tape.mojo`,
   comptime, so the product build runs the plain accumulate); tape-two-paths REFUSES on that side.
3. Verify keeps working from the built wheel and fails on a fault (AGENTS.md "The verify command is protected").

## Sub-lanes

| Sub-lane | Codes | Kernel range | Rows |
|---|---|---|---|
| session-ops-conv | 130..139 | 6900..6919 | conv1d, conv3d, conv_transpose1d / 2d, conv2d with dilation / per-axis padding |
| session-ops-pool | 140..149 | 6920..6939 | 1-D / 3-D pools, adaptive pools, global max, pool padding |
| session-ops-resize | 150..159 | 6940..6959 | interpolate, pixel_shuffle / unshuffle, reflection / replication pad |
| session-ops-rest | 160..179 | 6960..6979 | batch / group / instance norm, prelu, mish, multihead attention, positional encoding, losses, raw SSM scans |
| session-ops-extra | 180..189 | 6980..6999 | sequence_mask, topk, gather along an axis, logsumexp, soft-target cross-entropy |
| session-ops-verify | - | - | the generic composed-model lane and its data cases |
