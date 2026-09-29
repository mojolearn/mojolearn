# Matched FP32/fixed15 comparison

User requested actual comparison on NVIDIA and AMD, 2026-09-29.
Isolated branch experiment/profile-fair-compare; no public default changes
are merged by this experiment. Based on the FP32-default restoration branch.

First comparison: latest existing implementations, equal resident weight
strategy, explicit profiles, same SmolLM2 checkpoint and prompts, seven
alternating rounds. Prompt lengths 32/512; output lengths 1/32/128. Session
setup included in both intervals, initial load/packing separate. Hashes
outside timings. Refuse a missing resident path; never compare it silently
against a per-layer fallback. Full resident-versus-reference gate before
timing; repeat tokens and last-logit hashes checked during timing.

AMD's resident path is excluded from shipping dispatch. This experiment
adds a private keyword permitting it on HIP only for gate/benchmark calls.
No backend identity is spoofed. A gate failure or hang prevents timing.
This is qualification, not production admission; source sabotage and wider
coverage are still required before promoting the AMD resident path.

This does NOT prove both numerical profiles have reached their performance
limits. FP32 already has resident generation and fused attention, whereas
fixed15 has integer-specific decode/heads kernels. Those are not interchangeable
arithmetic. Follow-up transferable-optimization ablations must retain FP32
bits and compare before/after within that profile before attributing any
remaining difference to precision. No Mojo arithmetic edits in this stage.

Resources: AMD do-amd responds and its last hung-model retry passed; root
cause of the earlier hang remains unestablished. Shared NVIDIA nvc1/nvc2
have been retired and nvc3 refuses SSH. No rentals/extensions authorized.
Submit AMD now; NVIDIA remains pending an existing assigned live queue.
