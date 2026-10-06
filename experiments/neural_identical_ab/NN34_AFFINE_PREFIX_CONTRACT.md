# NN34 Mamba-1 affine-prefix profile, source draft v2

Sources: `mamba/impl/ops/neural_scan_profile.mojo`, `neural_mamba_scan.mojo`, `modeling_mamba.mojo`, `selective_scan_backward.mojo` and their generated host counterparts. Mamba-1 public prefill, decode, resident sessions, zero-state backward and host inference now select this graph. All source is uncompiled/unverified. Vendor identity, quality and timing have not been run.

The A arm is selected by `MOJOLEARN_NN34_AFFINE_PREFIX` in IDENTICAL mode, unless `MOJOLEARN_IDN_ALL_OFF` is defined. The B arm omits the define and retains the incumbent model recurrence. The isolated prepared-factor component also remains available. Model integration is source-only and is not full-model performance evidence. New profile bits may differ from the old version. Host, NVIDIA, AMD and Apple must agree within the candidate profile.

## Prepared factors and rounding

Inputs are Float32 arrays `a[t, chain]` and `b[t, chain]`, with mathematical recurrence `h[t] = a[t] * h[t-1] + b[t]`. `neural_mamba_scan.mojo` prepares Mamba-1 factors with the existing `delta*A -> exp`, `(delta*B)*u` pinned seams. It emits the ascending C-state dot and separately rounded u*D skip. Remaining block projections/gating/residuals retain their model call sites. Its component baseline alone is not asserted to reproduce the entire model.

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

Slot occupancy is implied by the position; consumed and inactive slots are explicitly positive zero. The level-5 slot exists while forming a complete 32-token prefix and is cleared before checkpoint completion. Restoring only `last` at a nonzero position is invalid. State from different profile ids cannot be exchanged. Public `Mamba1State` carries `profile_id`, `absolute_position`, `affine_boundary`, `affine_a` and `affine_b` alongside conv_window/h. Buffer shapes and profile/position metadata are checked by the API shell; native model refusal covers finite values of all added arrays. Session open/export/load copies all state. Python state serialization retains these arrays and scalars; no incompatible partial h-only migration is accepted. Raw native callers must supply the documented full checkpoint. A failed call makes the Python NN34 state or resident session unusable; restore a complete saved checkpoint into a fresh state/session.

Initialize position-zero chains with `nn34_init_chain` on the host or `nn34_init_kernel` on the device. Their initial hidden values are flushed and installed in both `boundary` and `last`; all slots are cleared. The scalar position is owner metadata and is advanced once per successful operation, never independently by each chain task.

Prefill requires disjoint input and output state buffers. Arrays `a`, `b`, `output`, `prefix_a` and `prefix_b` each contain `tokens * chains` floats. Each boundary/last array contains `chains` floats, each slot array `chains * 6`, and `chunk_boundaries` contains `ceil((absolute_start % 32 + tokens) / 32) * chains`. Tensor inputs, output and scratch may not overlap in ways that create concurrent reads and writes. `nn34_mamba_forward` owns scratch and disjoint next-state buffers through synchronization, then copies the checkpoint into its owner. Binding-side shapes come from fixed Mamba dimensions and scalar metadata. At absolute position zero, h supplies the initial boundary.

Device prefill queues the three phases in one ordered device context. The owner holds all allocations until synchronization and output consumption, then commits the output checkpoint and new position. The public model uses the disjoint prefill path even at one token; the explicit low-level decode component remains in-place and requires discarding state after failure. Empty requests do not invoke the component and retain the existing state. Ragged streams with separate absolute positions are not implemented by this shared-position API.

## Backward and public caller scope

The existing public Mamba-1 backward is zero-state prefill only. NN34 keeps that API and supplies its actual tree VJP. For each state chain, it visits absolute chunks in descending order and each chunk's output-prefix losses newest first. It rebuilds each canonical prefix tree, walks nodes in reverse creation order, and propagates both affine-A and affine-B adjoints. Each node pins separately rounded products/additions; right-A receives the A-product contribution before the B-product contribution. The next chunk's boundary adjoint attaches only to the completed chunk's final prefix.

Prepared-factor B adjoints occupy the first half of the gradient buffer, A adjoints the second. Existing channel contractions and discretization derivatives consume these values; the exp path uses factor-A adjoint times its rounded exp, rather than the old sequential `dh*h_previous`. All downstream x, normalization, projection, convolution, A_log, D and bias gradients remain connected to the public backward/optimizer consumers. No new state-cotangent or arbitrary carried-state training API is claimed.

The generated host forward/backward, alternate neural-host Mamba1BlockInference oracle, and GPU model now share the selected graph. Public state carries one absolute position for each uniform-length request. Existing fresh ragged inference invokes independent row calls; an independently positioned ragged carried-state API is outside the selected arm.

## Remaining acceptance

All chosen model wiring is source-authored. No compilation, static verification, tests, identity, model execution, gradient check, quality evaluation or timing was performed. Host files were authored with `tools/mamba_host_gen.py --source-write-only`, which omits post-generation verification and stale comparisons; this is not accepted generated-source/compile evidence.

Future acceptance must establish same-version host/NVIDIA/AMD/Apple words, prefill/decode/resume equivalence, gradient behavior, long-context and multi-step training quality, and full-dataset end-to-end NVIDIA+AMD A/B performance. The per-prefix tree VJP deliberately recomputes nodes and may be expensive; source completeness is not a speed claim. Component timings alone cannot promote this profile.
