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
`dWeight` cell over rows descending. No identity lane carries the door yet;
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
