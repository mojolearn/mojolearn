# GPT-3-small input/step orchestration audit

Base: `21fbc561f`. This audit found no production source change worth carrying.

The current pretokenized path already has bounded, ordered asynchronous
prefetch (`TokenBatches.prefetch`, commit `ad94232dc`). It reads only the
verified mmap, retains exact logical step order, bounds live batches by queue
depth, and rethrows a producer failure at its exact step.

On an Apple M4, the existing benchmark was repeated at sequence length 2048:

| batch | bytes/batch | modeled consumer | sync median | prefetch median | ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 8,196 | 1.0 ms | 0.256380 s / 200 | 0.253193 s / 200 | 1.01259x |
| 64 | 524,544 | 1.5 ms | 0.094155 s / 48 | 0.091499 s / 48 | 1.02903x |

The ordered byte digests matched (`3f8f1e2e...` and `ead5924f...`). At the
production GPT-3-small shape B1/L2048, loading and copying 8 KiB is already
hidden by a native step hundreds of milliseconds long; another async layer
cannot materially improve the end-to-end step.

`_array` admission/copy was measured separately over 1,400 calls per shape.
Median per-call time was 21.05 us at B1, 22.62 us at B8, 34.99 us at B64,
and 239.46 us at B256. Even four B1 logical shards cost roughly 84 us before
the native call. Avoiding this copy is unsafe: the public trainer promises it
does not alias mutable caller memory, and the copied arrays must remain live
through the native call.

The binding then copies each admitted int32 buffer into a Mojo `List[Int32]`.
Replacing this with borrowed host pointers would require changing the native
trainer interfaces and their lifetime/error contract for only 8 KiB per B1
shard. Removing either Python or native range validation would also change
which public error wins. Both ideas were rejected without source changes.

Resident training already avoids per-step model serialization. Its timed
`train_step` publishes only scalar loss/step metadata in lean mode; full state
and gradient serialization occurs only when explicitly exporting a witness or
checkpoint. Consequently there is no hidden model-sized Python serialization
inside the normal production step to eliminate.
