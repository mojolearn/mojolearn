# Mutex handoff repair, 2026-09-16

Branch: `fix/amd-merge-ordering`, based on `9257fd6de`. The RF investigation
branch and its reserved source files were read only. The release branch and
reference table were not changed. `AMD_MUTEX_ORDERING_INTEGRATION.patch`
is an unapplied integration patch for the reserved production files. The
new helper alone does **not** change the shipping forest.

## Diagnosis

The pre-claim acquire-load / relaxed-CAS protocol is not a valid general
mutex under the C++ memory model. No stale atomic load is needed to expose
the defect:

1. A acquire-loads the free lock.
2. B claims it, updates the payload, and release-stores the free value.
3. A's relaxed CAS succeeds, reading B's release, then A reads the payload.

A's acquire predates B's write, while its CAS is relaxed. There is no
synchronizes-with edge from B's release to A's payload read. Mutual exclusion
in the lock word is insufficient to order the ordinary payload accesses.
The claim that only the holder writes the unlock value does not exclude
this interleaving.

An acquire CAS paired with the existing release unlock closes **this** hole,
not just its timing window: a successful RMW reads the immediately preceding
modification, which must be the unlock it claims. A weak CAS is fine; a
spurious failure must retry without touching the payload.

## Portable implementation

`core/device_mutex.mojo` implements relaxed test / weak relaxed CAS /
**post-success acquire-load**. It uses operations the pinned Apple toolchain
can compile. The post-claim load is required even though its result is unused.

The successful CAS is sequenced before the load. Write-read coherence forbids
that load from reading a modification earlier than this CAS. While the holder
owns the lock, other participants cannot modify it. Thus the load reads the
holder's own CAS. That RMW extends the previous release's release sequence;
an acquire reading it synchronizes with that release. Initialization is
ordered separately by completion before the kernel launch. This argument
assumes conforming atomics at device scope, and does not establish GPU
scheduler progress or correctness of the compiler's lowering.

This preserves both the forest's `0 -> 1` protocol and k-NN's `-2 -> -1`
consumer claim. Block-wide barriers and release stores remain at their
existing sites. A capability-selected acquire-CAS is also a sound, potentially
cheaper solution where measured; keeping the old pre-acquire protocol as an
Apple fallback would leave its formal defect intact.

## Alternatives

- Making only the payload store atomic-release is insufficient. Existing
  ordinary reads would still be unpaired; the multiword `Split` is not a
  scalar supported by the current Atomic API. Per-field atomics do not give
  a consistent snapshot. Even a hypothetical whole-struct release store and
  acquire load need a protocol forcing the reader to observe the preceding
  update, or an atomic compare/update loop. Atomicity alone is not that rule.
- Relaxed CAS followed by a device acquire fence is another standard
  atomic-to-fence pattern where the backend supports it. The repository's
  Metal fence limitation makes the post-claim load worth testing.
- A two-pass reduction can remove the mutex: publish one candidate per
  (node, sampled column), then reduce distinct-column candidates with the
  existing gain/column ordering. It needs scratch capacity, initialization,
  a second launch, and separate handling of purity/metadata. Within-column
  range merging must retain its existing pinned order. This is a larger
  change than repairing the handoff and is not implemented here.

## How much the observations establish

The four traced divergences support a lost-candidate interpretation,
conditional on tracing faithfully capturing every candidate and comparison.
They do not independently identify a particular cause: stale reads, competing
writes, an upstream omitted candidate, or a compiler defect could produce
similar symptoms. A provably invalid lock is worth fixing regardless.

The other lane's commit `85588252491dd002dbbbec1abd1e9f2a883f1253` records
gfx942 acquire-CAS compilation and a 300-per-arm A/B: stock 7 moved, acquire
0 moved, with different binary digests. Those are **inherited observations**,
not runs made on this branch. Its `(1 - 7/300)^300` calculation treats the
estimated control rate as known. A two-sided Fisher exact comparison is
approximately **0.01508**. Zero failures in 300 gives a one-sided 95% upper
failure-rate bound of **0.9936%**, not proof of zero failures. The formal
argument repairs the protocol; hardware tests assess its implementation.

## Other spellings

The source census at `9257fd6de` found four production compare-exchange sites:

| File | Claim | Assessment |
|---|---|---|
| `ensemble/decisiontree/batched_levelalgo/split.mojo` | 0 -> 1 | General mutex; gap applies |
| `extratrees/impl/decisiontree/batched_levelalgo/split.mojo` | 0 -> 1 | General mutex; gap applies |
| `neighbors/impl/detail/fused_l2_knn.mojo` | 0 -> 1 | Multiple producers can miss an intervening exchange cycle |
| same k-NN file | -2 -> -1 | Single consumer restricts interleavings; matching spelling, not independent proof of this same failure |

`neighbors/mutex_probe_main.mojo` contains two more copies and an overbroad
justification. Its runtime success does not validate general mutual exclusion
ordering. The GBDT single-pass reorder kernel instead acquire-loads published
status/payload words without a subsequent CAS; it is not another spelling of
the split load/claim gap. Comments mentioning CUDA CAS are not extra live
claim sites. This census does not claim to certify every GPU synchronization
protocol in the repository.

## Validation

- `python3 tools/check_mutex_handoff_model.py`: PASS. In its bounded fresh-load
  schedules, stock lacks payload ordering in 4/6 valid schedules; acquire-CAS
  and post-claim acquire-load each have 0/6. It is a small counterexample
  enumerator, not a full C++ model checker. `--protocol stock` was run and
  **exited 1**; the negative control was observed to fail.
- `mojo build -j 1 -I . core/device_mutex_check.mojo`: PASS using the existing
  activated pixi environment. Direct invocation without activation first
  failed to locate `std`; that was an invocation error, corrected.
- Apple M4, exclusive `mac_slot.sh metal`, one CPU thread: PASS. Sixteen
  contention launches (eight per state pair), two blocks and 128 claims each,
  all final counts 256 and no checksum handoff errors. The skip-write sabotage
  produced count 0 and was rejected by the same check. This smoke does not
  exercise the full k-NN producer/consumer protocol or the forest.
- `git apply --check docs/lanes/AMD_MUTEX_ORDERING_INTEGRATION.patch`: PASS
  against this branch's base. The patch was not applied because the user
  reserved those files for the other active lane; an exception was requested.
- gfx942 helper and end-to-end RF validation: pending. The guarded
  `tools/device_mutex_leg.sh` runs only the helper/model check; it does not
  claim to discharge the forest A/B or cross-vendor identity gates.

Primary memory-model references:
[C++ atomic ordering](https://eel.is/c++draft/atomics.order),
[C++ coherence and release sequences](https://eel.is/c++draft/intro.races),
[C++ fences](https://eel.is/c++draft/atomics.fences), and
[LLVM atomic semantics](https://llvm.org/docs/Atomics.html).
