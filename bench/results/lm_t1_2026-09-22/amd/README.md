# T1, AMD arm: cross-vendor at the target shape on an MI325X, and the endurance

Evidence only. `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md` section 10, T1(b) and E7,
2026-09-22 on a DigitalOcean `gpu-mi325x1-256gb` droplet (gfx942), rented by
`tools/do_extra_leg.sh amd --segment-lease 150 --dollar-cap 12` (priced live:
150 minutes at $3.80 an hour is at most $9.50; `leg.txt`), body
`tools/lm_t1_body.sh` arm `amd`, commit 424b8fab4 built on the box. RunPod's
MI300X had no capacity (`../amd-runpod-refused/create_response.json`).

**Cross-vendor, the 162M shape, K = 64.** The NVIDIA arm's seed checkpoint
(step 0) and checkpoint 1 were fetched from R2 and verified by sha256
(`from-nvidia-verified.log`), then:

| arm | from | steps | s per optimizer step | hash s | verdict |
|---|---|---|---|---|---|
| xvendor | NVIDIA checkpoint 0 | 1..3 | 141.9 (first), 138.6, 138.5 | 3.5 | PASS: every state hash, gradient hash, loss and lr equals the NVIDIA chain |
| arrival | NVIDIA checkpoint 1 | 2..3 | 142.3 (first), 138.9 | 3.6 | PASS: lands on the NVIDIA boundary hash |

So the same three optimizer steps of the GPT-3 Small shape, on an H100 and
on an MI325X, are the same bits, and the arrival replay across vendors
works at this size. The MI325X step is 3.47 times the H100's 39.9 s.

**Endurance, E7.** `tools/lm_ce_alias_probe.py` at batch 4 for 2,000 steps
under a 4,200 s budget: the budget ended the run at step 1,916 (2,000 steps
need 4,350 s at this speed), so `lm_attention_endurance_check.py` found no
result file; `endurance_summary.json` is computed from the per-step log
instead. Median 2.1758 s a step over steps 2 to 50 and 2.1756 over the last
50, maximum 2.235; every layer `FUSED_RAN` forward and backward on every
one of the 1,916 steps (22,992 of 22,992); eager storage 432 bytes on every
step; device memory 38,661 MB on every sample; loss 10.98 to 1.66, none
non-finite. The repaired attention path holds on AMD as it did on NVIDIA.
