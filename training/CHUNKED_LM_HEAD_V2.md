# Chunked LM-head / cross-entropy v2 foundation

Status: normative CPU oracle plus opt-in device inference and training stages,
exported as `chunked_lm_head_v2_loss` and `chunked_lm_head_v2_train`. The
training selector returns loss, `dHidden`, and `dWeight`; V1 remains the
default everywhere and no existing trainer is silently rerouted.

Profile name: `mojolearn.identical.lm_head.chunked.fp32.v2`.

## Fixed arithmetic contract

- Vocabulary chunks are storage only. The CPU oracle walks 256-wide chunks
  and the device GEMM slice in `training/chunked_lm_head_v2.mojo` is 1024
  wide (`LM_HEAD_V2_CHUNK`); neither width reaches a reduction, because
  every fold below visits vocabulary ids or rows in one serial order.
- An LM-head logit is one `identical_mul_add` chain over hidden features in
  ascending feature order, followed by `ftz`.
- For each row, the global maximum visits vocabulary ids `0..V-1`. Chunk
  boundaries never reset it. `identical_fmax` supplies its total-order rule.
- For each row, the exponential denominator visits vocabulary ids `0..V-1`
  in one serial `ftz(acc + ftz(exp))` chain. Summing per-chunk totals, using a
  tree, or normalizing each chunk is not this profile.
- Row losses fold rows ascending, then divide once by `Float32(rows)`.
- `dHidden[row, feature]` folds vocabulary ids ascending. It may not combine
  independently reduced chunk gradients.
- A `dWeight[token, feature]` cell folds rows ascending. Vocabulary chunks own
  disjoint output rows and may be scheduled independently because no two
  chunks reduce into the same cell.
- V2 recomputes logits deterministically between passes. A production kernel
  may retain one `[rows, 256]` logits chunk, plus `[rows]` max and denominator
  arrays. It must not retain `[rows, vocab]` logits, exponentials, or dlogits.
- Validation precedes output mutation: positive rows/width, vocabulary at
  least two, exact buffer shapes, first non-finite hidden/weight value, then
  targets ascending.

## Python

`mojolearn.training.chunked_lm_head_loss(hidden, weight, targets,
return_grad=False)` is the Python door. It returns the mean loss as a float,
or `(loss, d_hidden, d_weight)` with `return_grad=True`. Both training
bindings register the two entries it calls: the GPU binding over the device
kernels, and the training host binding (`_mojolearn_training_host`) over
`chunked_lm_head_v2_oracle_forward` and `chunked_lm_head_v2_oracle`, so a
CPU-only install runs the normative oracle. The host binding's negative
control, `-D MOJOLEARN_HOST_SABOTAGE=1`, folds the row losses and every
`dWeight` cell over rows descending; measured, it moves `dWeight` and left the
loss bits unchanged at the two shapes tried. No identity lane carries the door yet;
`tools/lane_accounting.py::DECLARED_LANELESS` names what it owes.

## Integration inventory

The current byte-LM trainer allocates full `[M,V]` logits and CE intermediates
in `training/byte_lm.mojo::TrainBuffers`, then calls the v1 GEMM, CE forward,
CE backward, and LM-head GEMM backward. The pooled-head, offload, and model-
pool variants use the same full tensors. A production v2 should first be added
beside those calls under an explicit profile selector; it must not silently
replace v1 or the forward-only public logits API, whose output necessarily is
`[M,V]`.

The smallest executable stage is
`training/checks/chunked_lm_head_oracle.mojo`. Its checker uses a 513-token
fixture crossing two full chunks and a one-token tail. It proves repeated
loss/dHidden/dWeight bits, compares loss and gradients against the independent
v1 composition within `2e-5`, and rejects a deliberately wrong target-chunk-
local normalization.

For a realistic `B=8`, `L=2048`, `V=50,257`, one full logits tensor is
823,410,688 floats (3.07 GiB). The v2 scratch bound is 4,227,072 floats
(16.13 MiB), more than 194x smaller. Persistent parameters and returned model
gradients are excluded from both figures.

Build and run:

```sh
pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL \
  training/checks/chunked_lm_head_check.mojo -o /tmp/chunked-lm-head-check
/tmp/chunked-lm-head-check
```

The success sentinel is `CHUNKED_LM_HEAD_V2_OK`.

The device/CPU bit gate is:

```sh
pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL \
  training/checks/chunked_lm_head_device_check.mojo \
  -o /tmp/chunked-lm-head-device-check
/tmp/chunked-lm-head-device-check
```

It crosses two complete vocabulary chunks and a one-token tail, checks the
loss, row maxima, denominators, `dHidden`, and `dWeight` bit-for-bit against
the CPU oracle, and checks repeated device loss and gradient cells. The stage allocates only three
`[rows]` scratch arrays and one scalar beyond its inputs: `(3*rows+1)*4`
bytes. It never allocates full logits or dlogits.

## ByteTrainer selection

`ByteConfig(..., chunked_lm_head_v2=True)` selects the device-resident V2
forward and backward inside `ByteTrainer`. The selector is included in the
checkpoint/profile string. The default is `False`.

When selected, full logits become one reusable `[rows,256]` chunk. CE
exponential/dlogit storage and CE ones/workspace become one-cell placeholders;
the LM-head workspace is sized only for that chunk. Row maximum, denominator,
and row-loss arrays remain. The integration never invokes a V1 kernel with
those placeholders.

The V2 route is a memory/maximum-shape option, not a speed default. Local Metal
evidence at B=1, L=64, DM=64, V=8192 retained 16,389 rather than 1,323,009
head-path cells. Chunk GEMM cut scalar V2 from 232--242 ms to 122--157 ms,
while V1 remained faster. See
`bench/evidence/2026-09-20_byte_lm_chunked_head_v2.md`.

## NI34 source-only model integration, 2026-10-06

`MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2=1` is a default-OFF IDENTICAL selector,
suppressed by `MOJOLEARN_IDN_ALL_OFF`. It supplies the default for ByteConfig's
existing option, including both native binding constructors. ByteTrainer's
existing GEMM-logit GPU forward/backward now has a native Mojo counterpart in
`training/chunked_lm_head_host.mojo`, reached by `byte_host_loss` and
`byte_host_gradient`. Canonical GEMM cells and vocabulary/gradient fold order
are independent of the temporary vocabulary chunk width. The pooled Byte head
is outside this integration.

Samba loss-and-gradient calls use the existing standalone serial-logit V2 head
only for no-ignore means whose divisor equals the local row count. Ignored
rows and custom/microbatch divisors retain the existing full native head. These
are two explicitly different head graphs: Byte uses canonical GEMM logits;
Samba's standalone profile uses its existing serial feature dot. Each graph
requires within-version host/NVIDIA/AMD/Apple identity, not equality to the
other graph. Full-logits inference remains the existing API.

Byte checkpoint profiles distinguish this head and non128 GEMM leaves. Samba
checkpoints carry native `training_experiment_profile` metadata (chunked head,
CE token-total tree, GEMM leaf), and mismatches are refused. Missing older
metadata means baseline. Source only: no compilation, execution, verification,
quality assessment or timing has been performed for these edits.

## GEMM-panel arm (the default profile since 2026-11-19; the only profile since 2026-11-20, when the scalar serial profile was deleted)

Promoted on the A/B of 2026-11-19 (NVIDIA L40S, job nvc-ab-20261119T125554Z-6f7afc, base and arm in one job):
train-chunked-lm-head-loss 33,016.9 -> 284.5 ms, the three FP64 differences smaller under the panels; numbers beside
`LMHEAD_FAST_GEMM_PANEL` in `training/chunked_lm_head_panel_math.mojo`. "The arm" below is this profile; "the default
profile" in the text below names the scalar serial profile it replaced.

An experiment arm of the public door (`chunked_lm_head_v2_loss`,
`chunked_lm_head_v2_train`) and of the executor op `lm_head_v2`
(`OPS_LM_HEAD_V2`, device and host arm: `autograd/lm_head_v2_op.mojo`,
`autograd/lm_head_v2_host.mojo`), which calls the door's forward and backward
over the executor's storage. One define switches the device door
(`training/chunked_lm_head_v2.mojo::_gp_door`), the host column
(`training/chunked_lm_head_gemm_panel_host.mojo`) and the oracle
(`chunked_lm_head_v2_oracle_forward`, `chunked_lm_head_v2_oracle`) together:
one build, one set of words on every vendor and on the CPU. The words differ
from the default profile's (logit contraction and every fold below), so the
arm's hashes are compared across columns of the same build, never with a
default build. The device kernels and the host loops call the same per-unit
bodies (`training/chunked_lm_head_panel_math.mojo`). The ByteLM head keeps its
own folds. The op adds only what is its own: float class codes, a code that
names no vocabulary entry (shifted logit +0.0, so the row loss is
`log(denominator)`, and no one-hot in its dlogit) and the upstream factor
`ftz(g * upstream)` per gradient cell.

Fold contract under the arm, with `P = LM_HEAD_V2_CHUNK = 1024` vocabulary
entries per panel and `LMHEAD_GP_LEAF = 64` per denominator leaf. Under the
arm both constants are arithmetic, not storage: changing either is a
different profile.

1. Operands: hidden and weight are read only through the GEMM, whose leaf
   flushes both operands (`ftz`); no separate flush pass exists.
2. Logits: panel `p` holds `H . W[pP : pP+n]^T` by the identical FP32 GEMM
   contract (`OP_NT`, `m = rows`, `n = n_p`, `k = width`), the contraction the
   ByteLM head uses. It is recomputed in each of the three passes (maximum,
   denominator, gradients); no `[rows, vocab]` owner exists.
3. Row maximum: `identical_fmax` over the row's logits. A total-order
   selection, so the visiting order does not reach the result.
4. Denominator: with `shifted = ftz(ftz(logit) - ftz(max))`, leaf `j` folds
   `ftz(exp(shifted))` over vocabulary ids `64j .. 64j+63` ascending from
   `+0.0`, every add flushed. The row's leaves, in ascending order, reduce
   level by level: node `q` of a level is `ftz(ftz(a) + ftz(b))` of nodes
   `2q` and `2q+1` of the level below, an odd tail carried bit for bit. The
   root is the denominator. A panel holds whole leaves, so panel boundaries
   do not reach this fold.
5. Row loss and the mean: unchanged (`-(shifted_target - log(denominator))`,
   then the 256-row leaves and balanced tree of the default profile).
6. dlogit: `ftz((ftz(ftz(exp(shifted)) / ftz(denominator)) - onehot) / rows)`
   per cell, as in the default profile.
7. dWeight: rows `pP .. pP+n-1` are `dlogit_p^T . H` by the GEMM contract
   (`OP_TN`, `m = n_p`, `n = width`, `k = rows`). Final: nothing folds across
   panels.
8. dHidden: panel partial `D_p = dlogit_p . W[pP : pP+n]` by the GEMM
   contract (`OP_NN`, `m = rows`, `n = width`, `k = n_p`); the partials, in
   ascending panel order, reduce by the same level-by-level pair rule as the
   denominator leaves. The implementation is a binary counter holding
   `floor(log2(panels)) + 1` partials.

Execution: every device kernel runs one thread per output unit (operand
element, (row, panel) for the maximum, (row, leaf), (row, tree node), panel
cell, dHidden cell); none walks the vocabulary or the rows. On the host every pass splits its units across
all cores and the GEMM splits inside; the loop over panels is the only serial
driver. Under the arm the door's input validation is one parallel scan per
operand with the default's refusals, order and words, and spans beyond signed
32-bit kernel indexing are refused by name.

Scratch beyond the operands and returned gradients, in floats:
`rows * P` (panel) `+ 1.5 * rows * ceil(vocab / 64)` (leaves and their tree)
`+ (floor(log2(panels)) + 1) * rows * width` (dHidden partials) plus the GEMM
workspace. At 8,192 x 32,768 x 1,024: 8.4M + 6.3M + 6 x 8.4M = 65M floats
(260 MB) against 268M for one full logits tensor.

Quality: the same mathematics in a different summation order (pairwise in
place of serial chains). The checks that judge this function by tolerance are
unchanged and apply to the arm as written: loss within rtol 2e-5 of the
Float64 reference and gradients within rtol 1e-4, atol 1e-6
(`checks/native_tests/training_chunked_lm_head_door.mojo`), and 2e-5 absolute
against the v1 composition (`training/checks/chunked_lm_head_check.mojo`).
The device check compares the arm's door with the arm's oracle word for word
at three panels plus a one-token tail. The host negative control under the
arm (`-D MOJOLEARN_HOST_SABOTAGE=1`) folds every denominator leaf descending.
Source only: nothing here has been compiled, run or timed.
