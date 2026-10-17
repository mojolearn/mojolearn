# Host loops on the GPU path (Andrew, 2026-10-17)

The CPU is the driver of the GPU path, as it is in PyTorch: it walks the model
structure, decides branches and launches kernels. A `for` loop in host code is
not a defect by itself. A loop that touches the data is.

## The rule

A host loop inside a GPU fit, transform, predict or training step is allowed
only when it is a **driver loop**. Everything else is CPU work on the GPU path
and is a defect to fix.

1. **Driver loop (allowed).** Iterates over model or launch structure:
   features, columns, words, classes, layers, leaves, tree depth, iterations or
   epochs, config entries, argument spans, buffer handles, launch sequences.
   Its trip count and its body cost are independent of the number of rows,
   samples, elements, tokens or non-zeros. The body reads metadata (sizes,
   offsets, counts, shape flags) and enqueues launches; it never reads a data
   value. Every driver loop carries a tag on the `for` line or the line above:
   `# driver-loop: over <what>`. The hook accepts a tagged loop only when its
   range is not a data dimension (`n`, `rows`, `n_samples`, `n_rows`,
   `elements`, `tokens`, `nnz`, `len(x)`, `len(y)`, `count`, a downloaded
   buffer length, and the like; the hook keeps the denylist) and its body
   indexes no downloaded buffer. A false tag is refused, not trusted.
2. **Readback reduction (defect).** The host downloads a device buffer longer
   than a few scalars and sums, scans, compares or otherwise computes over it
   (a convergence flag per block, a log-likelihood from per-column terms, a
   finiteness scan). Fix: a device reduce kernel with the same fold order on
   every vendor and in the host column, then one scalar readback. A
   convergence check reads one flag, or runs every K iterations.
3. **Data loop (defect).** Any host loop over rows, samples, elements, tokens
   or non-zeros, or over a per-element buffer (packing, scatter, label
   mapping, bounds checks over values). Fix: a kernel.

A fixed-size scalar readback (at most 8 values: a flag, a loss, a norm) is
allowed; each one is a device sync, so at most one per outer iteration.

## Enforcement

`tools/hooks/no_host_routes.py` keeps `mojo-host-loop` as the owed rule for
classes 2 and 3. Tagged driver loops are a reported count, never debt. The
baseline only shrinks. The triage of the 98 sites found on 2026-10-17 is in
`docs/plans/HOST_LOOPS_TRIAGE.md`.
