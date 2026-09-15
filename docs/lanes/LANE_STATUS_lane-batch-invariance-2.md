# LANE STATUS: lane/batch-invariance-2 (agent "batch2")

Goal: close the three gaps the `batch` part of tools/identity_break.py declared
it does not test: the backward pass (part `batchgrad`, `--batch-grad`),
serving-scale B (part `batchscale`, `--batch-scale`), and ragged, padded
sequence batches (the new `lengths=` argument and part `ragged`, `--ragged`).

## Done (on the branch)
- python/mojolearn/_ragged.py and `lengths=` on TransformerBlock, Mamba1/2/3Block,
  SambaStack forward, the byte LM trainer and LanguageModelInference logits and
  next_bytes. Contract text: transformer contract 7.5, a section in each Mamba
  contract. Tests: python/mojolearn/tests/test_ragged_lengths.py (20 pass on
  Metal, 14 pass and 6 skip by name on the CPU-only install; two sabotages of
  the copies each fail it).
- identity_break.py: batchgrad, batchscale, ragged parts, opt-in, their own
  JSON keys and `summary (<part>):` lines; three commits, one per part.
- M4 smoke (base fixture) green for all three parts; sabotage arms seen to fail.

## Running / owed
- the full M4 record (every fixture, two repeats) for the lanes the parts
  declare, the CPU-only host column for the byte LM host lanes
- one H100 leg (RunPod) and one AMD leg (DigitalOcean MI325X or Hot Aisle
  single GPU) at the branch head, the three-column diff under
  bench/results/identity_break/2026-09-15_batch2/
- the CPU identity gate on the branch, then merge to main

## Next commands
    W=<worktree>; cd $W
    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
      --lanes <declared lanes> --batch-grad --batch-scale --ragged --vendor apple-m4 --json apple-m4.json
    python3 tools/identity_break.py --diff apple-m4.json nvidia-h100.json amd-mi325x.json
