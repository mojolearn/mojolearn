# T2: the rehearsal of the GPT-3 Small run, 2026-09-23

Route A only, three segments of 10 optimizer steps at the target shape
(162,147,840 parameters, batch 4, length 2048, vocabulary 50,257, K=64 shards
of 8,192 tokens a step) over the pinned FineWeb-Edu id stream, driven by
`tools/lm_run_driver.py` from `spec.json` with `recipe.json` (hash scheme
`sha256.v1`, the first). Every segment landed; every handoff was replayed by
the receiving box from the sender's checkpoint two steps back and agreed;
the live segment's two boxes wrote the same chain line at every step.

| segment | box | steps | s per step (median) | hash s | arrival replay | checkpoints |
|---|---|---|---|---|---|---|
| A/1 | NVIDIA H100 80GB (RunPod) | 0 to 10 | 40.0 | 7.9 | none (the seed) | 8, 10 |
| A/2 | AMD MI325X (DigitalOcean) | 10 to 20 | 138.9 | 3.6 | PASS (steps 9, 10 from ckpt 8) | 18, 20 |
| A/3 | H100 coordinator + MI325X worker, chained over an ssh tunnel | 20 to 30 | 101.1 (H100), 103.4 (MI325X) | 5.0 | PASS (steps 19, 20 from ckpt 18, on the H100) | 28, 30 |

Mean loss over the 64 shards: 10.951 at step 1, 8.221 at step 30. State
digest at step 30: `b34773af42833906...` on both boxes of the live pair;
the H100's and the MI325X's chain lines agree on state, gradient, the 64
losses and the learning-rate bits at every step 21 to 30
(`A-3-nvidia/segment/chain.jsonl`, `A-3-amd/segment/chain.jsonl`).

## Segment 3 took five attempts; none failed on the arithmetic

1. Orchestration ordering (the worker gave up before the coordinator listened).
2. A stalled 12 GB staging transfer held the pod two hours.
3. A hash-scheme mismatch: `sliced-sha256-8.v2` had landed on main between
   segment 2 and the replay, so the H100 hashed the same bits under the new
   scheme and read DISAGREE on the digests while all 64 losses agreed.
   Fixed: the recipe names the scheme (c0bac0f02), a line without the field
   is the first scheme (6e0a5b22c, test 0a89e39cb).
4. Started before that fix landed; superseded.
5. This one. The driver then recorded arrival FAIL with no checkpoints from
   a previous attempt's directory; fixed in dc6c300f9 and the record
   rewritten with `lm_run_driver.py reland --segment A/3` (`ledger.json`;
   `driver.log` carries both landings).

## Files

`spec.json`, `recipe.json`, `ledger.json`, `driver.log`, `A-3-live.log`
(the orchestrator), and per box `status.txt`, `leg.txt`, `gpu.txt`,
`binding.txt`, `segment/{segment.json,chain.jsonl,manifest.tsv,log.txt}`,
`arrival/{segment.json,chain.jsonl,log.txt}` where a replay ran, and the
coordinator's `segment/coordinator.jsonl`. Checkpoints are in R2 under
`runs/t2/2026-09-22/`. The full leg directories (bundles, console logs)
stay in `~/mojolearn-evidence/gpt3-run/t2/`.
