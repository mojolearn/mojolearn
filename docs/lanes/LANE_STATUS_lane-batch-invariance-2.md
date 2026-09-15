# LANE STATUS: lane/batch-invariance-2 (agent "batch2")

Goal: close the three gaps the `batch` part of tools/identity_break.py declared
it does not test: the backward pass (part `batchgrad`, `--batch-grad`),
serving-scale B (part `batchscale`, `--batch-scale`), and ragged, padded
sequence batches (the new `lengths=` argument and part `ragged`, `--ragged`).

## Done (on the branch)
- python/mojolearn/_ragged.py and `lengths=` on TransformerBlock, Mamba1/2/3Block,
  SambaStack forward, the byte LM trainer and LanguageModelInference logits and
  next_bytes. Contract text: transformer contract 7.5, a section in each Mamba
  contract. Tests: python/mojolearn/tests/test_ragged_lengths.py.
- identity_break.py: batchgrad, batchscale, ragged parts, opt-in, their own
  JSON keys and `summary (<part>):` lines.
- Record at 1cc7a2f47: bench/results/identity_break/2026-09-15_batch2/ (Apple M4,
  one H100, one MI325X; both boxes rented before the 10:30 ET rule and verified
  deleted). Reviewed on restart; see its README, section "Review findings".
- Restart (2026-09-15 about 11:15 ET), after merging origin/main, M4, one core:
  six lanes on base read IDENTICAL x4 against the record, the Apple sabotage arm
  moves every hashed cell of all four parts, test_ragged_lengths 20 passed (with the
  agent's CPU host byte LM binding copied in; 19 passed and 1 skipped by name
  without it).

## Owed
- The CPU identity gate on the merged head, then merge to main.
- The CPU-only column for byte-lm-host-infer and -threaded.
- AMD and NVIDIA columns at a later commit: the next release record only.
