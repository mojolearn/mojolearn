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

## Submitted runs

- AMD: request `1790684366433-speed-profile-fair-compare-94ddca292b`,
  source 94ddca292, command capped at 1200 seconds. Output directory
  `/root/mojolearn-wt/steward-do-amd/bench/results/profile-compare-rqH6JXto`.
  Resident gate GREEN: full prefill and decoded logits equal the per-layer
  reference, generated tokens and final logits agree, both profiles. Timing
  ongoing at this checkpoint; interim 512+32 ratio 1.466 (fixed15 slower).
- NVIDIA: user explicitly authorized provisioning after existing shared
  pods were found gone. Created ONE RTX 4090, pod g9y473zwd912ir, nvc1,
  $0.74/hour. Queue handles automatic retirement after 30 idle minutes.
  Job `nvc1-0001`, 45-minute cap, source c5a2c6dfb. Running, building.
  Output directory `/root/mojolearn-profile-fair-compare/bench/results/profile-compare-x4f5ILGM`.
  Model staged from R2, all six pinned files verified. No new model download
  from Hugging Face. Do not resync its source tree while the job runs.

Source differences after AMD submission: NVIDIA provenance-marker support,
one test expectation and documentation. Benchmark and native arithmetic are
the same. A patch-synced NVIDIA tree's git HEAD is its merge base: its
`.profile_compare_commit` marker records the actual submitted source.

Code audit: FP32 already offers fused attention; fixed15 forces its own eager
path to honor quantized QK arithmetic. The heads-at-once and phase-fence
changes optimize that fixed15 path, not a shared FP32 kernel left deliberately
slow. Both runs use the current dispatch decisions of their own profile.
Any further FP32 optimization needs a separate bit-preserving ablation.

These fixed token prompts are synthetic performance probes, not quality
evaluations. Generation can diverge between profiles; cross-vendor identity
compares each profile with itself, never FP32 with fixed15. Compare model,
prompt and output hashes across the finished reports before drawing a
cross-vendor conclusion. Retain all samples, and report regression rows.

FP32 default restoration remains on fix/fixed15-opt-in (00b0057ea), NOT main.
26 CPU selector tests passed on the remote Mac; the current experiment
inherits that change. No wheel or production dispatch has been published.
