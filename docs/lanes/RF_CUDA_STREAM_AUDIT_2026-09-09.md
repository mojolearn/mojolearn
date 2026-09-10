# RF CUDA stream audit, 2026-09-09

Status: no production stream scheduler enabled. The installed API supports real
stream objects, but this Apple M4 machine cannot establish how a CUDA-created
stream becomes a selectable context view. No remote training pod was used.

## Evidence and the historical 1.7x claim

The handoff's HIGGS 1M H100 numbers, 5762 ms / 3314 ms, give 1.7387x. They are
historical end-to-end measurements, not a controlled attribution experiment.
The accompanying explanation (“per-node-per-feature launches with global
atomics vs cuML's per-level shared-memory histograms plus four streams”) is too
strong. Current MojoLearn already dispatches shared versus global histogram
kernels from `SharedMemoryConfig` in
`ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo:2397`;
the grid batches work items and columns. Kernel shape and missing concurrent
streams are candidates to measure separately, not demonstrated causes of the
entire ratio.

cuML's actual independent-tree scheduling is confirmed in
[the pinned v26.08 source](https://github.com/rapidsai/cuml/blob/265b9da6a0e75dbef071a3168398b993a5ff6f0e/cpp/src/randomforest/randomforest.cuh#L336):
OpenMP selects one stream per worker, samples rows on it, and passes it to tree
fit, then joins the stream pool. Local checkouts examined:
`upstream/cuml-v26.08.00` at `265b9da6a0e75dbef071a3168398b993a5ff6f0e` and
`upstream/cuml` at `00094f7e4e4b5da3a968d193a4da6085fa38f11b`.

MojoLearn's `ensemble/randomforest.mojo:2515` currently has K host slots but all
sampling, builders, split downloads, and synchronizations receive the same
DeviceContext. This reduces host synchronization overhead; it does not create
concurrent GPU streams. GBDT's integer stream IDs also do not select a backend
queue: `gbdt/gpu_lib/gpu_single_worker.mojo:438` explicitly launches on `self.ctx`.
Changing the device vendor alone cannot make those integer IDs real streams.

## Runtime capability probe

[DeviceContext API](https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext/)
and [its public source](https://github.com/modular/modular/blob/main/max/mojo/max/gpu/host/device_context.mojo)
expose `num_streams`, `select_stream`, `create_stream`, and `enqueue_wait_for`.
The first two are views over a runtime-owned pool. `create_stream` returns a
DeviceStream; the public interface has no DeviceStream-to-DeviceContext
conversion. DeviceStream can enqueue kernels, but the existing RF launcher
chain requires DeviceContext for launches, allocations, and copies.

The isolated `checks/probes/rf_stream_api.mojo` compiles with installed Mojo
1.0.0 (`ed45d567`). On Apple M4 it reports `api metal initial_streams 1` and an
explicit unsupported result. Exploratory direct calls additionally established:

- `select_stream(1)`: `invalid stream id: 1`.
- `create_stream()`: `createStream is not supported on this device`.

Run on an authorized CUDA machine with:

```sh
tools/with_build_lock.sh pixi run mojo run checks/probes/rf_stream_api.mojo
```

It checks whether `create_stream()` grows the pool and whether the new index
can be selected and synchronized. A pass is API reachability only, not proof of
kernel concurrency or forest identity. A failure identifies the exact missing
bridge. Separate DeviceContexts per slot are not a verified substitute: shared
buffer allocation ownership, copy routing, and destruction must be checked
before using existing raw-pointer builder inputs across independent contexts.

## Bounded implementation after that probe passes

1. Keep one physical context/pool, select one actual context view per RF slot,
   allocate its Builder on that view, and keep every launch in its tree on it.
   Join shared quantiles, bins, X, y, and sampler setup once before priming slots.
2. Do not adopt shared `SplitStaging`: its prefix download in
   `builder.mojo:897` assumes all writes precede one queue's copy. Retain each
   builder's existing private split buffers/downloads instead, then synchronize
   the relevant slot before consuming host splits.
3. Row IDs already have a device buffer per slot (`randomforest.mojo:1832`).
   Per-tree RNG is a pure seed/tree-id hash (`rng_seed_for`), so scheduling must
   never replace tree IDs with completion order. Store finished trees at their
   original indices, preserving prediction reduction order.
4. `RowSampler.h_rows` is shared host staging, but its upload paths synchronize
   before reuse. Keep those barriers initially; private staging is a later
   optimization. Optional sorted-row scratch is one shared set: prohibit
   ROWS_SORTED_SAMPLE with multistream until it is private per slot.
5. OOB masks use disjoint tree regions; join all slots before OOB reads and
   before freeing shared inputs. Keep context views and slot buffers alive
   until every kernel/copy finishes, including error cleanup.
6. Explicitly inspect copy routing: runtime DeviceBuffer copies can insert
   cross-stream synchronization. Allocate slot scratch on the slot context;
   do not assume passing another context moves buffer-associated transfers.
7. Initially opt in only on CUDA. Compare serial and concurrent full model
   fingerprints in FAST, DETERMINISTIC, IDENTICAL for bootstrap on/off,
   weights, classification/regression, OOB, several stream counts, tree counts
   not divisible by slots, and trees finishing in the prime phase. Include a
   repeat-run check and memory/race sanitizer where supported.
8. Capture an Nsight Systems trace proving distinct concurrent GPU queues,
   then run matched ABBA full-fit timings with histogram changes separately
   toggled. Report overlap and timing independently; host threads alone are
   not evidence of GPU overlap.

No production scheduler rewrite is retained because pool creation/binding has
not been verified on NVIDIA. The probe and ownership plan make that next step
concrete without claiming an untested speedup.
