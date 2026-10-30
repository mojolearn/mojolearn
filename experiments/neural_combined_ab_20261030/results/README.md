# Frozen NVIDIA round f048701e

One L40S, 13 allocated CPU cores. One excluded warmup and one scored whole
stream/read operation per cell. Both saved checkpoints, batch one, prompt 128,
128 generated tokens; model loading is recorded separately. NVIDIA only, no
performance default promotion, no cross-vendor ledger admission.

All 32 full-workload requests executed: 24 measured with original quality
success; four completed generation and recovered quality from exact retained
matching model/prompt/output evaluations; four SmolLM2 forced-engine requests
were refused because they conflicted with resident-forward admission. Original
failures remain in the raw records. The smoke has the same coverage breakdown.
There are no pending or budget-skipped cells in this round.

Scored seconds (one sample; no noise estimate):

| Model | Public read | Baseline s0 k0 e0 | Reuse s0 k1 e0 | Sampler s1 k0 e0 | All flags s1 k1 e1 |
|---|---:|---:|---:|---:|---:|
| mojolearn-162m | 128 | 7.241 | 8.361 | 6.706 | 5.576 |
| mojolearn-162m | 1 | 8.909 | 8.979 | 8.386 | 7.599 |
| SmolLM2-135M | 128 | 4.138 | 3.315 | 15.399 | 15.401 |
| SmolLM2-135M | 1 | 4.927 | 4.809 | 17.767 | 17.727 |

`s` is sampling policy, `k` requests legacy KV-upload reuse, and `e` requests
the native generation engine. Resident-forward is requested in every cell.
These settings are not independent executed optimizations:

- The 162m model has no final norm. No-final-norm residency is disabled by a
  separate compile-time experiment in these frozen artifacts. Its engine-off
  cells executed the legacy path and its engine-on cells the native engine;
  KV-upload reuse had no resident loan to reuse.
- SmolLM2 sampler-on cells executed resident sampling before engine selection.
  Reuse and engine flags were no-ops for that generation route. Its all-flags
  read128 timing is about 3.72 times the baseline; this remains scoped to that
  route, not to the follow-up explicit-route combinations.
- KV reuse applies only within a compatible multi-token public read. Read1
  cannot skip intermediate uploads within that read.

Every supported cell has a completed native route witness and retained output
bytes. Token-output hashes agree across executed routes and read sizes within
each sampling policy for these fixtures. This is neither cross-vendor identity
nor a claim about every intermediate tensor. Quality is a finite, same-model
teacher-forced logits/NLL check, not independent statistical qualification.
The next explicit-route matrix is a separate frozen round.

`f048701e-nvidia.json` preserves the original statuses, timings, routes, quality
recovery links, exact input identities, compiled artifact hashes, hardware and
source provenance. `f048701e-nvidia-retention.json` is the supported
`bench_board_evidence` content-addressed PUT/GET receipt for those records.
`f048701e-mac-retention.json` verifies the complete 3,408,373,854-byte run bundle
and 176,933,814-byte recovery bundle on the Mac against their R2-retained bytes.

### Repaired modular route round: 99c25a6d, NVIDIA L40S

`99c25a6d-nvidia.json` records all 44 requests separately. Both smoke and full
workloads completed 40 supported cells, with four explicitly unsupported 162m
resident cells and no failed or pending cells. The repaired install passed 43
focused tests and four actual overlapping-call probes. Independent streams
preserved serial outputs and state; overlapping reads of one stream were refused.
This demonstrates public-call overlap and ownership, not simultaneous GPU kernels.

For SmolLM2 read128, explicit legacy FP32 plus KV reuse took 2.854 s versus
4.040 s for legacy FP64 without reuse (ratio 0.706). The complete auto arm took
15.392 s (3.809), selecting resident sampling; explicit native-engine FP32 with
reuse requested took 7.701 s (1.906), with reuse inapplicable there. For 162m
read128, the complete auto arm selected native-engine and took 5.584 s versus
6.948 s (0.804). Read1 results and every losing configuration are retained too.
These are single scored observations after one excluded warmup, not noise-adjusted
speed claims. Model loading is recorded separately from generation.

All executed routes and both read sizes produced identical output bytes within
each model/sampling-policy group on this NVIDIA install. All 40 full cells have
finite same-model teacher-forced quality scores. This is neither cross-vendor
identity nor independent quality qualification. Defaults remain unchanged.
The original f048 sampler-only slowdown and aea ownership failures remain intact.

The full 5.23 GB archive and all required source, binary, model-state and output
bytes are retained in R2 and on the Mac. The old pod's automatic lease teardown
preceded aggregate Mac recovery after a session interruption; prerequisites were
already on both stores, and the completed per-cell/final archives survived in R2.
The recovered aggregate and per-cell receipts are recorded without claiming that
its post-run final receipt was embedded in its own archive.
