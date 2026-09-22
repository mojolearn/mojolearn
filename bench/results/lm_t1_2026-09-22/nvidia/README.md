# T1, NVIDIA arm: the segment runner at the target shape on 2x H100

Evidence only. `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md` section 10, T1(a) and
T1(d), 2026-09-22 on a RunPod pod with two H100 80GB HBM3 (driver 580.126.09),
commit 424b8fab4's source built on the box (`build_base` 63 s, `build_byte_lm`
141 s), body `tools/lm_t1_body.sh`, arm `nvidia`. Shape 4x2048 d768 12 layers
12 heads ff2048 v50257, 162,147,840 parameters, K = 64 shards per optimizer
step (524,288 tokens), the enwik8 id stream under the 50,257-id vocabulary
staged from R2, learning-rate table 6e-4 peak with a 2-step warmup.

| arm | devices | steps | s per optimizer step | hash s per step | verdict |
|---|---|---|---|---|---|
| one | 0 | 1..3 | 50.9 (first), 39.9, 39.9 | 8.0, 8.0, 9.1 | PASS |
| two | 0,1 | 1..3 | 30.7 (first), 20.2, 20.2 | 7.6, 7.8, 7.6 | PASS, every state hash equals `one` |
| replay | 0 | 2..3 from checkpoint 1 | 48.4 (first), 39.9 | 7.9, 8.0 | PASS, every state hash equals `one` |

- The seed checkpoint (step 0) was drawn on the box with NumPy 2.5.2 and is
  1,945,780,159 bytes, sha256 5871ef0f...; checkpoints 1 and 3 are the same
  size, sha256 21c46ad0... and 5dcbb3c4... (`checkpoints.sha256`). Save 4.4
  to 4.9 s each; upload to R2 54 to 56 s each (about 35 MB/s), keys
  `runs/t1/2026-09-22/nvidia/`.
- The hash cost is the state (1.95 GB) and the summed gradient (649 MB) read
  back and sha256'd on the host: about 8 s, 20 percent of a one-device step
  and 40 percent of a two-device step. Worth reducing before the run (a
  device-side digest, or hashing every step only inside record windows).
- Two devices divide the step by 1.98 and change no bit.

Files: `status.txt` (the body's timeline), `one/`, `two/`, `replay/` (chain,
segment.json, manifest), the logs, `recipe.json`, `gpu.txt`, `binding.txt`.
No checkpoint is in the repository; the three are in R2 under the keys above.
