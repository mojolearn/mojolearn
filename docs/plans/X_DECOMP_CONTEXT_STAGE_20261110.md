# Decomposition context publication and download stage ownership

The native PCA/TSVD descriptor drivers release the GIL around execution. The
former `xd_ctx` Optional test/store could open competing device contexts, and
concurrent Apple downloads could overwrite the shared pinned stage before its
host copy completed.

`x_decomp/device.mojo` now reuses `process_ctx` with an explicit single-device
plan and a new slot name. The original default device root remains unchanged.
The helper publishes the process-lifetime context under its native lock and
uses acquire/release publication for subsequent reads.

The pinned stage has one native lease covering lazy creation, every existing
DMA chunk, synchronization and its final host copy. The lease releases on
errors. Its new layout has a new slot name/version. Creation publishes the
buffer only after synchronization succeeds. Raw downloads and uploads do not
acquire this stage lease. Transfer thresholds, chunk sizes, routes, copy counts,
all numerical bodies and defaults remain unchanged. No generator/template for
these definitions was found in the tracked source.

Removed: the duplicate Optional-only context owner. The pinned stage remains
because it is the measured transfer path; only its ownership is synchronized.

Source inspection and `git diff --check` only. Compilation and native concurrent
first-use/stage fixtures are queued with the orchestrator; no runtime, speed or
cross-device identity claim. Compile all affected GPU x_decomp provider images
and transitive consumers in the bundled integration build. No A/B define or
numerical/bit change is intended. Adjacent resident-pool metadata work belongs
to the finite-state lane and is a separate dependency.
