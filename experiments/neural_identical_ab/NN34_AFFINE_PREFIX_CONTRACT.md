# NN34 affine-prefix component contract, draft v2

Source: `mamba/impl/ops/neural_scan_profile.mojo`. This is an isolated, uncompiled component. No public Mamba/Samba caller imports it. Compilation, verification, vendor identity, quality and timing have not been run.

The A arm is selected by `MOJOLEARN_NN34_AFFINE_PREFIX` in IDENTICAL mode, unless `MOJOLEARN_IDN_ALL_OFF` is defined. The B arm omits the define and evaluates a sequential prepared-factor recurrence. This A/B is component scope; it is not full-model performance evidence. New profile bits may differ from the old version. Host, NVIDIA, AMD and Apple must agree within the candidate profile.

## Prepared factors and rounding

Inputs are Float32 arrays `a[t, chain]` and `b[t, chain]`, with mathematical recurrence `h[t] = a[t] * h[t-1] + b[t]`. Preparing those factors from raw Mamba inputs, projecting outputs and all derivatives are outside this component. Its component baseline is not asserted to reproduce an entire incumbent selective-scan implementation.

Each input leaf is explicitly flushed with `ftz`. For an older left affine summary `(LA, LB)` and newer right summary `(RA, RB)`, composition is exactly:

```
A = ftz(identical_mul(ftz(RA), ftz(LA)))
B = ftz(identical_mul_add(ftz(RA), ftz(LB), ftz(RB)))
h = ftz(identical_mul_add(ftz(A), ftz(chunk_boundary), ftz(B)))
```

One leaf is carried as its two words; no synthetic identity multiplication or zero addition is inserted. Adjacent equal-sized subtrees merge first. An unmatched right subtree carries unchanged into the next level. Every prefix has its own canonical tree; the binary-counter slots implement the same adjacent-pair tree independently of request length, vendor, launch geometry and scheduling. All implementations call the shared compose, prefix and evaluate helpers.

The component changes floating-point association. It does not establish quality, stability or error bounds by mathematical associativity. Long contexts, severe decay, cancellation, extremes and multi-step training remain required future work.

## Absolute chunks and prefix behavior

Chunks contain absolute positions `[32k, 32k+32)`. The constant bounds prefix work and retained state; it is not chosen using benchmark dimensions. Absolute position starts at zero and counts tokens already consumed. New request length never changes chunk partitioning.

Within a chunk, every output evaluates its entire affine prefix against the hidden state at the preceding completed chunk boundary. Intermediate outputs do not replace that boundary. At the 32nd token the final output becomes the next boundary and all slots reset to positive zero.

Prefill computes three ordered phases:

1. Every `(token, chain)` task constructs its prefix summary from the absolute chunk start, or from restored prefix slots for a partially consumed first chunk. It reads no future token. The final token also writes the next checkpoint's slots into separate buffers.
2. One task per chain propagates boundary values through completed chunks in ascending order and retains the preceding boundary for any unfinished last chunk.
3. Every output task evaluates its prefix against its recorded boundary. The final token writes the next checkpoint's `last` word.

Decode pushes one leaf into the same binary-counter slots, drains the same canonical prefix and applies the same boundary evaluation. Thus split prefill requests and repeated one-token decode have the same specified arithmetic graph as one full request. This is a design contract, not executed identity evidence.

The initial parallel prefill repeats some subtree construction per prefix. Eliminating this repeated work is a later scheduling optimization; do not silently substitute a different scan tree.

## State and buffer ownership

A checkpoint consists of all of:

- Profile id: 2 for this affine-prefix graph, 1 for the component sequential baseline.
- Next absolute token position and chain count.
- `boundary[chains]`: hidden state at the last completed chunk boundary.
- `last[chains]`: most recent output, or the initial hidden state at position zero.
- `slots_a[chains, 6]` and `slots_b[chains, 6]`: affine summaries for the occupied bits of `absolute_position % 32`.

Slot occupancy is implied by the position; consumed and inactive slots are explicitly positive zero. The level-5 slot exists while forming a complete 32-token prefix and is cleared before checkpoint completion. Restoring only `last` at a nonzero position is invalid. State from different profile ids cannot be exchanged. Model serialization, file formats, validation of checkpoint contents, transactional restore and failure recovery are not implemented.

Initialize position-zero chains with `nn34_init_chain` on the host or `nn34_init_kernel` on the device. Their initial hidden values are flushed and installed in both `boundary` and `last`; all slots are cleared. The scalar position is owner metadata and is advanced once per successful operation, never independently by each chain task.

Prefill requires disjoint input and output state buffers. Arrays `a`, `b`, `output`, `prefix_a` and `prefix_b` each contain `tokens * chains` floats. Each boundary/last array contains `chains` floats, each slot array `chains * 6`, and `chunk_boundaries` contains `ceil((absolute_start % 32 + tokens) / 32) * chains`. Tensor inputs, output and scratch may not overlap in ways that create concurrent reads and writes. Buffer allocation and extent validation belong to the future owning caller.

Device prefill queues the three phases in one ordered device context. The owner holds all allocations until synchronization and output consumption, then commits the output checkpoint and new position. Decode updates in place; a failed operation requires discarding the state. Empty requests do not invoke the component and retain the existing state. Ragged streams with separate absolute positions are not implemented by this shared-position API.

## Pending integration and acceptance

Still required: raw Mamba factor preparation and emission order; every selective-scan backward gradient and optimizer consequence; a full public prefill/decode owner; checkpoint serialization and restore; ragged streams; model refusal/exception semantics; and complete generated-host/model caller migration.

No compilation, static checks, generation, tests or model execution were performed. Future acceptance must establish same-version host/NVIDIA/AMD/Apple words, prefix/decode/resume equivalence, gradient correctness, long-context quality and full-dataset end-to-end NVIDIA+AMD A/B performance. Component timings alone cannot promote this profile.
