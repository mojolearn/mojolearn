# afn-attn A/B requests (transformer-forward, binding transformer, shape full)

Arm A is FAST with no afn define (main's kernels), arm B is FAST with the one
define. Every define is FAST + Apple only; IDENTICAL is unchanged. All touch
only the transformer-forward board lane (and, through the shared
`llama_decoder_layer_forward` entry, the byte LM forward on a build that
passes the define to that binding; this lane does not request those).
Profile and mechanisms: docs/apple-fast/notes/neural-attn.md.

**afn-attn-norm-sg (`MOJOLEARN_AFN_ATTN_NORM_SG`).** RMSNorm with one
simdgroup per row (256 blocks of 256 threads instead of 16 blocks with one
thread per row folding 384 values serially), also for residual1+norm2.
Expected: norm1 (1.24 ms on the M3) and norm2 drop to tens of microseconds.
Risk: low; f32, fold order only.

**afn-attn-rope-cache (`MOJOLEARN_AFN_ATTN_ROPE_CACHE`).** Both RoPEs, the
cache append and the two cache copies in one launch, and no synchronize.
Expected: rope_and_cache (1.19 ms) to one small launch, one host wait fewer.
Risk: low; elementwise, same arithmetic. Fresh full-causal prefill only.

**afn-attn-flash (`MOJOLEARN_AFN_ATTN_FLASH`).** One online-softmax launch on
the simdgroup matrix unit, no score stash (100 MB per call at this shape), no
scan, no flag readback (about three host round trips fewer). Expected: the
largest single gain (attention 4.6 ms). Risk: medium; online-softmax
rescaling changes rounding more than the others (still f32); watch the
quality judge's max error.

**afn-attn-gqa-tile (`MOJOLEARN_AFN_ATTN_GQA_TILE`).** FLASH with the heads
of one KV group sharing a K/V tile. At the board shape (n_kv == n_heads) it
is FLASH at GROUP 1, so it should read the same as afn-attn-flash; it is
requested to prove it is exercised and correct there.

**afn-attn-fuse-pre (`MOJOLEARN_AFN_ATTN_FUSE_PRE`).** norm1 folded into one
q+k+v projection launch with RoPE and the cache append in its epilogue; norm2
folded into the gate/up projections. Expected: about 6 launches and one wait
fewer, no norm1_out/norm2_out round trip. Risk: medium; this lane's own
64x64x32 MMA GEMM replaces main's GEMM for those projections, so its speed
against main's GEMM decides the sign.

**afn-attn-fuse-mlp (`MOJOLEARN_AFN_ATTN_FUSE_MLP`).** One gate+up launch with
the SwiGLU epilogue, residual adds in the o_proj and down_proj epilogues.
Expected: 3 launches and two elementwise passes fewer. Risk: as fuse-pre
(own GEMM).

**afn-attn-arena (`MOJOLEARN_AFN_ATTN_ARENA`).** The workspace's ~35 buffers
as views of one arena and no wait between the uploads and the forward.
Expected: lower per-launch host cost (fewer live Metal allocations) and one
wait fewer. Risk: low; storage location only, same bytes.

**afn-attn-all (`MOJOLEARN_AFN_ATTN_ALL`).** Every candidate at once.
