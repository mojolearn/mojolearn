# Missing GPU fixture evidence, Apple M4, 2026-09-20

This complete full-part column covers 56 lanes and all nine fixtures (504
cells), one repetition per the release task's policy. Every cell is STABLE;
1,557 numeric parts compare exactly with the currently recorded CPU results.
No numeric disagreement or refusal was observed.

The harness/Python source witness is the clean committed checkout
`ea12ec7dae18204d7272aa8f2adce13db8b561fa`. Native Metal bindings were copied
from the fresh `release089-macos` build of
`a97676ba18a09fb577ef9faae45ab19a01eec848`; binary digests are in the column.
Between these commits, the only Mojo change is the CPU-only HDBSCAN row-norm
lifetime fix. Neither the GPU arithmetic nor `tools/identity_break.py`
changed. This is a source/binary developer column, not an installed-wheel
qualification receipt.

The lane list is the nonparallel CPU-only numeric evidence gaps from the
reference audit, except the 19 neural lanes measured independently on AMD
in `2026-09-20_amd-neural-full-parts`. Full properties were retained; no
`--partial-column` option was used. The intentional broad investigation was
explicitly enabled with `MOJOLEARN_APPLE_FULL_DIAGNOSTIC=1`.

The nine `gbdt-categorical-ctr-tables/model` parts retain their existing
CPU role: the CPU loads a GPU-saved model and does not write one. Their GPU
model bytes are evidence; they are not a fabricated CPU model-writer check.
