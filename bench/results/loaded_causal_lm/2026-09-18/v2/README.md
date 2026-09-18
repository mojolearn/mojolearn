# All supported checkpoint families, profile v2

Capture source `c9c3b96b7`; native paired neural bindings are the clean/fault
builds recorded in `../native-build-witness.json` (native source unchanged).
All other host bindings and Metal binaries are retained builds, with loaded
binding hashes embedded in each capture. This remains source-tree evidence,
not rebuilt installed-wheel certification.

24 cases: Llama tied+untied, Mistral(window3), Qwen2(nonzero QKV bias),
Qwen3(nontrivial Q/K norm), Phi3(fused QKV/gate-up mapping), Mamba1 and Mamba2,
each in FP32/BF16/int8. Every fixture has two nontrivial layers and a sharded
checkpoint, batch2, five input tokens, prefill3 and two carried-state steps.
Window3 crosses its cache ring. Mamba2 exercises open-chunk recurrent state;
it does not exercise its 256-token chunk boundary.

All checks pass on clean CPU and Metal: exact full versus prefill/decode logits,
batch invariance, reload, reset, repeated greedy, mapping and composition fault.
All144 captured part hashes match across CPU and Metal. Each case reloads an
independent model and verifies exact output equality. Compiler fault readback
is False/True for clean/fault. The native descending-GEMM-leaf control changes
all120 floating parts (logits, prefill, decode1, decode2, state), while all24
greedy IDs remain unchanged. `comparison.json` contains the per-case result.

Serial shared-slot runs: clean CPU13.28s, Metal6.25s, CPU native fault13.68s,
90-second per-run bound, nice19 and supported math-library thread limits1.
The temporary Metal binary symlink was removed after captures.

Still owed: NVIDIA/AMD, installed-wheel replay, physical multi-GPU execution,
capacity/latency measurements, real external checkpoint smoke, ragged prompts,
long-context/chunk-boundary fixtures. Profile v1 captures remain unchanged.
