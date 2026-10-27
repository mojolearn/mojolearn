# Generation metadata admission, 2026-10-27

This source repair captures stable local CausalLM metadata once per public
forward/generate operation or stream read. The stream's next read admits current
metadata again. It follows the earlier tree and neural admission repairs in
`experiments/native_metadata_once_20261027` and
`experiments/neural_metadata_once_20261027`.

## Boundary and ownership

- `providers/python/generation_metadata_pyapi.mojo` is the integration edge. It
  reads model/block fields and retains their Python buffer owners. Exact internal
  Arrays contribute a checked C-order address once. Foreign buffer owners retain
  the existing per-use copy path; freezing their bytes could change aliasing with
  mutable state, so this change deliberately does not do that.
- `core/generation_metadata.mojo` contains only native dimensions, option words,
  scalar limits and address-vector assembly. It cannot import providers, bindings
  or `std.python`, directly or transitively; the dependency checker includes this
  new root with no debt baseline.
- `bindings/causal_lm_public.mojo` remains mixed orchestration. Prepared execution
  uses the captured fields while passing current token length, batch, cache
  capacity, cache counts, positions and state/input/output addresses separately.
  Holding a Python owner is lifetime management, not compiled Python code.

Local snapshots are built lazily after state validation and parallel/resident
routing. A stream builds its snapshot inside its existing error cleanup scope.
No snapshot is stored on the model, stream handle, global map or object-id cache.
All captured owners remain live through their final compiled kernel call.
Concurrent mutation during an active call gets no new thread-safety guarantee.
Replacement/refit/checkpoint changes between public operations are admitted anew.

Transformer dimensions, extended option words, optional weights, int15 planes and
arithmetic-profile identity are captured; live profile compatibility checks and
cached-token progression remain. Mamba1 captures inner width; Mamba2 captures head
and convolution widths and dt limits. Both preserve live continuation state.
Embedding and output head capture their dimensions, normalization options and
owners as part of the same operation. Standalone block/embedding/head entry points
keep their signatures and capture only their own needed metadata.

## What this does not establish

The local token/layer loop no longer reconstructs stable layer dimensions and
weight addresses from model dictionaries each step. Whole generation is **not**
claimed Python-free or free of CPython object work. Remaining work includes:

- State ownership/usability validation, live mutable buffer validation and
  copy/writeback; these must not become stale snapshots.
- Resident route eligibility and resident session setup/key/state transfer.
  Existing resident sessions already own their compiled execution state. These
  routes are intentionally not made to construct unused local snapshots.
- Per-step dispatch registry/native-session ownership lookups and environment
  routing. Those are separate from weight metadata and some refer to mutable
  session state. Compiled-extension dispatch is not itself Python bytecode.
- Foreign-buffer copying/protocol metadata, output allocation/boxing and stream
  tokenization. These need their own supported boundary work before a broader
  migration claim; a folder relocation cannot remove their runtime dependencies.
- Distributed workers retain their existing boundary; local snapshots are not
  built for a parallel operation.

The shared source applies to Apple, NVIDIA and AMD. This lane performs no builds,
GPU execution or timings. Root owns Mac-only compilation and public runtime
validation. Other vendor validation and performance remain unproven; existing
measurement campaigns are untouched. There is no numerical kernel change or
speedup claim.

## Validation and integration commands

Source checks: `tools/test_generation_metadata_boundary.py`, existing neural/tree
metadata and runtime-fallback scope guards, plus
`tools/check_neural_native_boundary.py`. The combined suite passed 34 tests and 14
subtests; 80 dependency-closure modules passed. Python probe syntax passed.
These checks are source evidence only. `validation.json` records status.

After a frozen candidate build, use the candidate package/extension environment:

```sh
MOJOLEARN_GENERATION_DEVICE=gpu MOJOLEARN_NEURAL_NATIVE_DISPATCH=1 python -m pytest -q python/mojolearn/tests/test_generation_metadata_snapshot.py
MOJOLEARN_GENERATION_DEVICE=cpu python -m pytest -q python/mojolearn/tests/test_generation_metadata_snapshot.py
```

The 11 authored candidate cases exercise three model families, new input/batch
shapes and live state positions, multiple-token vs chunked reads, checkpoint
resume, exact Array and foreign head-owner replacement/in-place writes, layer
weight replacement, owner lifetime and failed snapshot admission cleanup. They
force the local forward route. Existing `test_generation_native.py` and relevant
fixed15/resident/distributed tests remain the route-preservation checks; no claim
that those checks ran is made here. Integration should retain test exits, actual
compiled source closure and candidate binary hashes. Any failure needs a concrete
source repair before qualifying the affected route.

Full logs and source manifest:
`~/mojolearn-evidence/generation-metadata-once-20261027/`.
