# Captured graph optimization source round — 2026-10-20

Internal implementation and qualification record. This is not a public capability
or performance claim. The candidate is opt-in through existing `optimize=True`
execution/inspection paths; the default remains false.

`autograd/program_optimize.mojo` now applies an ordered native pass pipeline:

1. **Explicit reshape chains.** Two consecutive reshapes with positive, explicit
   dimensions and equal element counts become the final reshape of the original
   input. The final count check remains. Inferred dimensions, squeeze, unsqueeze
   and flatten are not folded, so their validation stays intact.
2. **Permutation composition.** Two valid full permutations of the same rank
   compose their axis metadata. Even an identity retains a permutation operation
   to enforce the original exact-rank check on variable-rank inputs. Two identical
   axis swaps become one identity swap retaining the stricter original axis;
   `_swap` still enforces rank at least two and the original axis bounds.
3. **Dead intermediates.** Remove only intermediates made unnecessary by those
   proven rewrites. Reachability includes every output and every other original
   operation as a root. Thus an unrelated original operation, validation, copy,
   detach or random operation does not disappear merely because its result is
   unused. An intermediate that remains an output or has another consumer stays.
4. **Exact duplicate elimination.** Preserve the existing bitwise attribute
   comparison and random/copy/detach exclusions. No floating point algebra,
   reassociation, fusion or reduction-order transformation is introduced.

The captured program is immutable. Training and VJPs replay that original graph,
preserving gradient accumulation and optimizer order. Optimized inference has
independent numerical/finite-difference comparisons against those derivatives.
Parameter references and all input validation remain part of the program; no
parameter tables or checkpoint formats are rewritten by these passes.

Inspection and profile results expose an ordered pass report: node counts before
and after, rewrite counts, original source-node indices, use counts and logical
last uses. This is logical reference liveness, not a storage-recycling policy.
The tape retains responsibility for pending views and buffer aliases. No memory
reclamation workaround is introduced around the confirmed Mojo PythonObject
ownership blocker. Host profiling intervals are not GPU compute timings.

The verifier gains three required cases covering NaN payloads, signed zero,
infinities/subnormals, retained intermediate/repeated outputs, independent output
buffers, variable ranks, invalid reshape/rank refusals and recovery, copy/detach
boundaries, lazy control flow, bounded loops, serialization and derivatives.
The existing post-optimization training oracle now includes transpose chains.
The source enumerates 20 cases and 88 numerical digests. Cheap controller tests
validate the reporting and audit machinery; they do not qualify native execution.

Remaining: the parent integrates and compiles changed sequence source once per
freeze, runs native positive/refusal controls, and qualifies installed packages
and required physical vendors. The known standard-library ownership leak remains
a separate blocker; no unsupported compensation or global completion claim is
permitted. Full-dataset end-to-end NVIDIA+AMD A/B, with preserved identity and
quality, is required before any performance default promotion. No such timing
evidence is claimed for this source round.
