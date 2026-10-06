# Graph neural and channel dropout source handoff

Branch `ideas/neural-identical-20261006-v2`, based on `main` at `fd6cf8045`.
Changes remain uncommitted. Compilation, testing, lint, verification and timing
were not run, as requested. No binary or result evidence exists.

NI55 adds four-feature CSR message-passing tasks sharing graph metadata; NI56
reuses the existing graph-normalization workspace route and its reference switch;
NI58 shares CSR traversal across four GraphSAGE maxima and exact tie counts.
All preserve the original scalar statements per output as their intended
same-bits contract. NI57 adds a distinct numerical version with 64-edge leaves,
adjacent balanced merges and odd tails carried without padding, through the
common host/device `spmm_at` element function. Its initial leaf implementation
is serial: it establishes an experimental arithmetic path, not parallel speed.

NI59 computes each channel mask once inside a block and applies it in the same
launch. NI60 keeps the mask table but reuses each loaded value for four spatial
outputs. Both retain the complete dense output mask. They are alternative
schedules and must not be silently combined; the selector records their conflict.

Every new flag is IDENTICAL-only, default OFF and disabled by
`MOJOLEARN_IDN_ALL_OFF`. The host retains its original same-bits path for schedule
experiments and uses the shared NI57 contract when that numerical flag is set.
No claim of cross-vendor identity or quality is made from source alone.

Source and controls are retained in [neural_aux.json](neural_aux.json),
[`neural_aux_contract.mojo`](../../x_cnn/neural_aux_contract.mojo),
[`neural_aux_ops.mojo`](../../x_cnn/neural_aux_ops.mojo), and the common graph
element/device dispatch sources. Remaining work includes all compilation and
same-version identity/quality evidence, full GCN/GraphSAGE/dropout consumer
recipes and full-operation A/B timing. The default GraphSAGE mean recipe does
not qualify its max aggregation; synthetic component timings do not qualify
full graph-neural training.
