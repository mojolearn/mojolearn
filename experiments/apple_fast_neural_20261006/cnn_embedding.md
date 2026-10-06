# CNN and embedding implementation notes

Branch `ideas/apple-fast-neural-20261006`, forked from local main `fd6cf8045`.
**Not tested: no compilation, verification, test, or measurement was run.**
Controls and A/B arm definitions are in `cnn_embedding.json`. Every new
mechanism is disabled unless its explicit define and Apple GPU FAST guard hold.
Historical defaults and IDENTICAL controls are retained.

E01 and E06 reuse existing disabled runtime mechanisms, now with explicit
source-only experiment recipes and inline qualification notes. E02–E05 and
E07–E10 add executable source paths, not just descriptions or unused switches.

Convolution tile controls derive the spatial rows, output channels, reduction
depth, channels per thread and shared arrays together. Their values follow
power-of-two work decomposition and storage tradeoffs, not benchmark shapes.
All loops retain complete reduction work and spatial/channel bounds. E05
assigns the saved im2col matrix to output-channel tile zero; every spatial
tile still saves its own complete patch. Other channel tiles still stage the
same operands for their own output. The full training operation must include
these saved columns and the existing backward route.

The embedding gather adds a float8 kernel for divisible widths and retains
the existing float4/scalar fallback. Launch geometry is independently selected
for all helpers and their grids. E09 reuses the existing immutable-table token
API; no Python data computation was added. The context was already shared,
so no redundant context experiment is claimed. FAST residency works both
with and without the atomic-backward arm. E10 extends existing exact-size
scratch pools and also integrates them into the atomic route, whose earlier
return would otherwise bypass pooling. Fresh gradients are fully seeded,
accumulated gradients are uploaded from the caller, and pooled outputs and
IDs are overwritten. Buffers return to the pool after output completion.
An exception does not publish a newly refused table as a valid cache entry.

Future build targets are `bindings/build_x_cnn.sh` and
`bindings/build_embedding.sh`. Those scripts have not been invoked. The
standalone Embedding controls are not a claim about the separate embedded
training binding or byte-LM embedding implementation. Qualify the actual
public callers that reach these files.

Full CNN/embedding-consumer workload mapping is pending. The saved
`tools/apple_speed_cnn/fastq.py` synthetic quality suite is a useful recipe
reference but is not proof of full-dataset qualification. Later work must
retain dataset hashes, complete dimensions/steps and output consumption, then
evaluate CNN accuracy/loss, convolution and embedding gradients, repeated IDs,
padding rows, accumulation, table replacement, zero-token uploads, resize,
empty inputs and refusal. Neither bit equality across versions nor a faster
single kernel can replace task quality. Cold and repeated use are distinct.

Remaining work is intentionally unperformed: all build, correctness/quality,
coverage, numerical identity where relevant, performance, and interaction
qualification. No speed or quality result exists for this delivery.
