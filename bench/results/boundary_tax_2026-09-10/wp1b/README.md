# WP1b actual transpose staging probe

Apple M4, Mojo 1.0.0 (ed45d567), FAST/default compilation. Synthetic finite
Float64 C-order input; asymmetric per-cell values and signed zeros. Uses the
production `_mojolearn._tiled_transpose_to_f32` and `hostptr.copy_f32` directly.
Allocations and exact per-cell Float32-bit oracle are outside measured regions.
No DMA, training, Python conversion or device kernel is timed.

```sh
tools/with_build_lock.sh nice -n 19 pixi run mojo build -I . -I bindings bench/boundary/float64_pinned_transpose.mojo -o /tmp/wp1b-transpose
tools/with_build_lock.sh nice -n 19 /tmp/wp1b-transpose
```

Build and run exit zero. Every output cell matches the independent transpose
address/cast oracle in both arms, one warmup plus five alternating pairs.
At 1M×28, direct pinned staging minimum is 12.358ms versus 17.208ms for malloc
transpose plus SIMD pinned copy (28.2% lower staging time); baseline max/min
spread is 11.5%. This supports investigating fusion, not a whole-fit speed claim.

The 2M×20 window is conservatively **void**: baseline full spread is 29.3%,
although its endpoint drift is only +14.2%. Retain raw results without using
that shape to justify implementation/default decisions. No repeated timing
attempt was made to seek a favorable window. NVIDIA/AMD measurements are owed.
