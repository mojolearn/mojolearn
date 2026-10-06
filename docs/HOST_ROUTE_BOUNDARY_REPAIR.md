# Host-route boundary repair draft (2026-10-06)

**Uncompiled, untested, identity unverified, and unmeasured.** This private
source draft is a future freeze. It does not modify any measured binaries,
accepted receipts, default switches, checker, or baseline.

The observed pre-push failure contained 49 findings. Fifteen strict findings
remain independently actionable after remote ancestry is refreshed. This
source work addresses two actual device scheduling defects and separates
CPU-only implementations that were imported transitively by GPU bindings.
It does not claim that the gate now passes.

## Device scheduling repairs

- Adam status folding: replace four serial walks across partial status rows
  with 128-thread tile minima followed by the existing integer `Atomic.min`.
  Initialize the same four caller-owned status fields to the existing `n`
  sentinel. Integer minimum preserves first-failure indices independently of
  order; that source argument still needs compilation and execution evidence.
- C42 active triangle: replace the single-thread row walk with live flags,
  the existing exclusive integer scan, and rank-based scatter. Ascending
  active-row order is preserved by construction. Allocate scratch only when
  the existing opt-in C42 path is selected; retain it through the existing
  synchronization. No benchmark dimension or new dispatch threshold is added.

## CPU/GPU module boundaries

`xtrees.api` previously imported `shap_host` for the compile-time CPU cache
branch. The shared import graph therefore reached CPU task helpers from the
GPU trees binding. Move the four cache ABI functions to separate
`shap_cache_device_api` and `shap_cache_host_api` modules, registered by their
respective binding entry points. Names, argument checks, opt-in guard, and
registration order are preserved. The GPU module calls only the existing
device cache; the CPU module calls only the existing host cache.

`core.host_storage` now owns uninitialized float-list allocation and typed
address adapters. It does not traverse model/data values, copy buffers, choose
thread counts, or run CPU kernels. `core.host_tasks` owns the existing CPU
copy and row scheduling functions, moved verbatim from `host_lanes`.
`host_predict_threads` retains its CPU task-count policy and re-exports the
storage adapters for existing callers. Pointer-only callers import storage
directly. CPU consumers of copy/row scheduling import `host_tasks` explicitly.
No thread cap, work threshold, arithmetic, or CPU parallel execution is changed.

Two unused `host_f32_uninit` imports in the device NN20 summary modules were
removed. Their CPU summary implementation remains unchanged.

## Source audit and acceptance still required

A literal import inventory covered all 80 original direct imports of the
three affected modules and checked alternate import spellings. Before the
changes, four GPU-named bindings reached CPU lane/task policy modules:
byte-LM, training, transformer (through the unused summary imports), and
x_trees (through shared cache registration). After the changes, the inventory
finds no such path to `host_lanes`, `host_predict_threads`, `host_tasks`, or
`shap_host`. This is source dependency evidence, not Mojo symbol resolution,
compiler acceptance, or a substitute for the repository policy gate.

The five moved allocation/address/copy/row functions have byte-identical
function text. CPU TreeSHAP and its parallel unit implementation are not
replaced with serial execution or marked as a GPU exception. No `cpu-route`
label, baseline growth, hook bypass, toolchain patch, or unsupported Modular
feature is introduced.

Before integration: review the source, authorize and run the repaired
host-route gate, then admit a new frozen build and identity evidence on the
affected columns. Full end-to-end A/B evidence is required for affected Adam
and C42 estimator workloads, including neighboring shapes and a non-board
case where required by repository policy. Current no-compilation instructions
remain in force; none of those numerical acceptance steps has been run.

External source-only inventories, patches and commit logs are retained under
`mojolearn-evidence/six-lane-full-ab-20261006/quality-review/publication-hook-diagnosis`.
