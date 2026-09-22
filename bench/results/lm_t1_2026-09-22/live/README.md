# T1c: the live NVIDIA plus AMD segment at the target shape

Evidence only. `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md` section 10, T1(c),
2026-09-22. A RunPod H100 80GB HBM3 pod (coordinator, shards 0 to 43) and a
DigitalOcean MI325X droplet (worker, shards 44 to 63) trained the same three
optimizer steps of the 162M shape (K = 64, batch 4, the enwik8 id stream, the
T1 recipe) from the NVIDIA T1 seed checkpoint, through
`tools/lm_segment.py --live-role coordinator|worker` over the chained fold,
the AMD worker reaching the coordinator through an ssh tunnel opened on the
AMD box (`link.log`: TUNNEL_UP). Bodies rendered by `tools/lm_segment_leg.py`;
the boxes were joined by `tools/lm_live_link.sh` after the orchestrator's
first two attempts (a recipe and stream mismatch, refused on both boxes by
name; then a DigitalOcean bundle unpack failure).

**Result: PASS.** Both workers reported the same full-state sha256 after each
step, and every state hash and gradient hash equals the one-box H100 chain
of `../nvidia/one` (d333b045..., 094b4e5a..., 6a408d48...); checkpoint 3
written by the coordinator's replica has the same sha256 as T1's checkpoint 3
(5dcbb3c4...). So a step split live between an H100 and an MI325X in
different data centers is the same bits as the step on one H100.

**Timing, per step, from the coordinator's record (`nvidia/coordinator.jsonl`):**

| phase | seconds | what it is |
|---|---:|---|
| gradients in | 97 | both workers computed their shards (44 on the H100, 20 on the MI325X) and replied |
| fold, NVIDIA worker | 184 | the host fold of its 44 held gradients into the prefix (NumPy on the pod's host) |
| fold, AMD worker | 50 | the 649 MB prefix through the tunnel, the host fold of 20 gradients, the 649 MB total back |
| apply | 8 | the total to the NVIDIA worker and both AdamW updates |
| **step** | **335 to 397** | against 39.9 s for the same step on one H100 |

The cost is the HOST FOLD, not the wide-area link: the two 649 MB crossings
are inside the 50 s of the AMD fold, so the tunnel carried at least 90 MB/s
(commit-to-commit 336 and 342 s on the worker side, `amd/chain.jsonl`; the
first "step" of 5,462 s is the coordinator's wait for the worker to be
rented). Two fixes, neither changing a bit: the NumPy fold in place instead of
through copies (a few times), and the fold on the device with the kernel
`train_step` already uses (the whole 184 s and most of the 97 s). Owed
before T3's segment 3.

**Cost.** H100 $3.49 an hour for 120 minutes (mostly waiting for the AMD box
across two rentals), MI325X about 30 minutes; about $9 in all.
