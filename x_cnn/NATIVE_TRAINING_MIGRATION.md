# Native CNN training migration

This is an uncompiled source change. It does not establish product compliance
with the no-Python rule or cross-vendor identity. Existing frozen measurements
and binaries do not cover these new entries.

`training_device.fit_epochs` and `host.training_host.fit_epochs` accept the
typed `TrainingSpec` contract. They own the epoch and minibatch loops, row-order
generation, forward/backward/optimizer kernel sequence, optimizer scratch,
synchronization, loss readback, and ordered epoch-loss reduction. No Python
objects, callbacks or imports occur in either entry or the local source import
closure inspected for this change. The caller still owns model/data/tape
resident handles.

The epoch seed remains `epoch_key(seed, first_epoch + epoch)`. Adam starts each
epoch at the absolute completed-step count plus one. SGD uses two fixed native
scalar rows, preserving its first-ever-step flag. Full and final-short-batch
plans are unchanged. The trainer executes the requested number of epochs;
there is no new early stopping condition. The epoch mean follows the same
ordered Neumaier sum and division as the former Python wrapper.

The transitional Python model now calls one `x_cnn_fit_epochs_d` entry. It has
no Python epoch/minibatch fallback and no tiled SGD hyperparameter table. A
binding without the new entry is rejected, never silently routed through the
old Python training loop. Legacy single-epoch exports remain available to
existing tests; they are not a production-compliance exemption.

Two unrelated Python-facing CSR/GCN adapters formerly inside `ops_host.mojo`
were moved unchanged into the existing binding adapter. This prevents the new
typed host trainer from importing Python through its kernel module. Native CSR
and GCN implementations are unchanged.

Remaining required migration work is explicit:

- Compiled public model construction, argument/label/input handling and returns.
- Native model/data/tape allocation, ownership and parameter readback replacing
  the remaining Python setup/resource loops.
- Native prediction minibatch/layer orchestration.
- Removal of transitional Python-facing product interfaces after a supported
  compiled interface preserves their behavior.

Only Python AST parsing, source import/loop inspection, whitespace checks and
sabotage-patch applicability were checked. No build, model execution, numerical
verification or timing was performed. The sabotage patch now covers the new
host training output seam as well as legacy entries. A future authorized freeze
must compile all required targets and evaluate identity/quality and full-workload
A/B coverage before any performance/default claim.
