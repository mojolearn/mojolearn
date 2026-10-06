# NN59 explicit residual-dropout layer source handoff

Branch `ideas/neural-identical-ab-20261006-r3`, forked from main
`fd6cf80453a6f18eb02e81566c824e7da106ccf0`. Root owns commit/push. No compile,
test, static checker, identity, quality or timing execution was performed.

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output.

The chosen workload is a complete, explicit residual-dropout layer, exposed as
`mojolearn.training.residual_dropout(values, residual, *, p, seed, stream,
offset)` and its paired-gradient `residual_dropout_backward(gradient, ...)`.
It does not alter CNN, transformer or other existing model behavior. The user
supplies the same RNG coordinates and probability to forward and backward.

`training/residual_dropout_contract.mojo` is the pure shared host/device graph
and Philox4x32-10 coordinate mapping. It has no GPU imports. The FP32 scale is
computed natively; dropout rounds and flushes before the separately rounded
residual addition. dValues uses the same keep/scale mask, while dResidual
preserves the incoming gradient words. The seed is uint64, stream uint32 and
offset a nonnegative admitted Int64 element coordinate.

A selects `MOJOLEARN_NN59_DROPOUT_RESIDUAL` in IDENTICAL mode: one fused
forward kernel and one paired-gradient kernel. B adds `MOJOLEARN_NN59_CONTROL`:
dropout plus residual kernels and separate backward producers. OFF/ALL_OFF
also select the baseline schedule. CPU forward/backward uses the same pure
scalar graph, with separate baseline passes. Public shape, finite-value,
probability, span, RNG-coordinate and alias admission occurs before outputs
are published. Empty arrays admit configuration but touch no pointer.

GPU and CPU boundary modules are `bindings/residual_dropout_boundary.mojo`
and `bindings/residual_dropout_boundary_host.mojo`, exporting
`residual_dropout_binding` and `residual_dropout_backward_binding`. Root has authored both training-builder registrations and public
`training.py` imports/exports. No selected-arm programming remains for this
explicit layer; all execution evidence remains unrun. The Python API module
`python/mojolearn/_residual_dropout_impl.py` only checks metadata, chooses the
binding and passes buffers; it does no arithmetic on tensor data.

Whole-operation timing must include admission, preparation/upload, allocation,
all forward/backward kernels, finite scans, required synchronization and output
download/consumption. Source integration is complete for this chosen explicit
layer; four-column identity, model quality and performance remain unproven.
Further placements inside particular models are additional independently
scoped experiments. Exact paths and arm definitions are in
[residual.json](residual.json).
