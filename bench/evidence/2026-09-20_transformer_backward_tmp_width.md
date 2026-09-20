# Transformer backward fan-in scratch right-sizing

`LlamaBackwardStages.tmp0/tmp1/tmp2` were each allocated as
`M * max(d_model, intermediate)`.  A complete caller audit shows that they
only receive dA results from gate/up or q/k/v projections, all shaped
`[M, d_model]`, before same-shaped fan-in kernels.  They are unrecorded scratch;
no trace or public buffer owns their excess tail.

The change allocates each as `M * d_model`.  It changes no kernel, launch,
index, arithmetic, output, or synchronization.  For GPT-3-small
B1/L2048/d_model768/intermediate3072, retained scratch falls by
`3 * 2048 * (3072 - 768) * 4 = 56,623,104` bytes (54 MiB).  The saving scales
linearly with batch and sequence length.

Apple M4 validation:

- the 17-case clause-(a) gate passed all 37 stages, 412,172 cells bitwise;
- the full clauses B/C/D/E/F run exited zero, including batch/length fixtures,
  refusal behavior, carry theorem, and exact routing checks;
- the wide-intermediate fixture (`d_model=32`, `intermediate=300`) passed,
  directly exercising the corrected allocation regime;
- `git diff --check` passed; no cloud resource was used.

This is an allocation-only exact win.  Runtime should be neutral aside from
lower allocation/initial-fill pressure; no speed claim is made.
