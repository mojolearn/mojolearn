# Pooled/offloaded LM-head CE buffer aliasing

`BytePooledHead`, used by model-pool and offloaded-replay training, retained
five full `[tokens, vocab]` buffers even though the standard byte trainer had
already qualified the exact two-buffer lifetime schedule.

The pooled head now uses that same schedule: CE shift aliases logits, while CE
weights and dlogits alias exponent storage.  Each transition loads one cell
before overwriting that cell on the same in-order context.  These pooled seams
have no `IdentityTrace` owner, and call ordering matches the standard trainer.
No kernel, arithmetic, reduction, launch, or synchronization changed.

At GPT-3-small B1/L2048/V50257 this removes
`3 * 2048 * 50257 * 4 = 1,235,116,032` bytes (about 1.15 GiB).  Savings scale
linearly with batch, sequence length, and vocabulary size.

Apple M4 local qualification ran the single-device portions of
`byte_lm_offload_check` under IDENTICAL mode.  All three cases passed exact
loss, gradient, parameters, first/second moments, optimizer flags, invalid-token
atomicity, rollback, and replay-after-rollback comparisons against
`ByteParallelTrainer`:

- one layer, one logical shard;
- two layers, three logical shards (run twice by the upstream matrix).

The temporary local-only matrix restriction was reverted; the checked-in test
remains unchanged.  Multi-device cases remain covered by their existing cloud
gate.  No cloud resource was created for this allocation-only change.
