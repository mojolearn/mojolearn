# T0 on the M4: the six-segment tooling at a small shape

Evidence only. `mojolearn verify` does not run this and no release depends on
it. This is the local, no-rental step of `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md`
section 10 for the five Python-side items of section 8, taken 2026-09-22 on
the local Apple M4 (Metal, one job at a time under `tools/mac_slot.py`) at
shapes far below the target. It proves the protocol and the tooling and says
nothing about the 162M model; T1 owes every number at the target shape.

Every command is in `tests/`, every output in `logs/`, and the scratchpad
path is written `<scratchpad>`.

## E2: a per-step learning rate (`logs/set_lr.log`)

`ParallelByteLanguageModelTrainer.set_lr` over the new native
`byte_lm_parallel_set_lr`, on the `tools/par_lm_xvendor.py` recipe (2 blocks,
d32, vocabulary 512, K = 4): one trainer that runs step 1 at one rate, calls
`set_lr`, and runs step 2 lands on the same state hash as a trainer opened
fresh at the second rate for step 2; leaving the rate unchanged lands
elsewhere; `export_raw` hashes as `state_dict` does; a rate of 0 is refused.

## E1: the segment runner (`logs/segment.log`, `segments/`)

`tools/lm_segment.py` at shape 2x32 d32 2 layers over the 50,257-id enwik8
token stream (3,235,008 parameters), K = 4, 12 optimizer steps, checkpoints
every 4, boundaries at 4, 8 and 12:

- route A ran segments 1, 2 and 3 from the seed checkpoint, each from the
  previous segment's boundary checkpoint (`segments/A1..A3`);
- route B ran segment 1 from the seed and segment 2 from route A's
  checkpoint 4, each held to route A's chain step for step: PASS
  (`segments/B1`, `segments/B2`);
- an arrival replay from the boundary-minus-two checkpoint (step 2) for two
  steps against route A's chain: PASS (`segments/ARR`);
- the zero-moments control from the same checkpoint: DISAGREE at its first
  step on the state hash, exit 1 (`segments/CTRL`);
- `compare` over A1, B1 and ARR: 16 fields compared, 0 disagreements;
  `manifests` over A1 and B1: 2 checkpoints compared, 0 disagreements;
- a recipe with K = 3 against a K = 4 checkpoint: REFUSED before any device
  work.

## E5: the chained fold, live (`logs/live_chained.log`, `live/`)

`tools/live_xvendor.py local` on one device under a lock, the recipe above,
6 steps: the gathered group of two workers (shards 0,2 and 1,3), the chained
group of two (blocks 0,1 and 2,3) and the chained group of three with unequal
blocks (0; 1; 2,3) all agree with the recorded one-process M4 column
(`bench/results/par_lm_xvendor/2026-09-21/apple-m4-1gpu.json`, 6 steps
compared, 0 differ) and with each other on every state hash and every summed
gradient hash. Host tests in `python/mojolearn/tests/test_cross_vendor.py`
hold the chained group equal to the gathered one with fake trainers, refuse a
non-contiguous block, and hold the NumPy and pure-Python folds equal.

## E3: the token stream (`logs/tokens.log`, `tokens/`)

`tools/fineweb_tokens.py --text` over the first 20 MB of FineWeb-Edu shard
000's text with the pinned vocabulary `vocab/mojolearn-bpe-fineweb-edu-50257-v1`
(sha256 bfe4a401...): 92,012 documents, 4,158,497 ids, 4.79 bytes per id,
max id 50,255, the last 50 documents held out as the validation range,
**17.7 MB/s** of text encoded on the M4 (so 13 GB of text is about 12
minutes on one core). `TokenBatches` reads the stream, `lm_segment.py recipe`
accepts it, the first document round-trips through the tokenizer and its ids
are the head of the stream.

## E4: the segment lease

`tools/gemm_remote_leg.sh` and `tools/do_extra_leg.sh` refuse, by name,
`--segment-lease` without `--dollar-cap`, a segment lease of 60 minutes or
less or over 2,880, `--minutes` together with `--segment-lease`, a cap that
is not a dollar figure, and still refuse `--minutes 90` as before. The price
check itself (RunPod `costPerHr` after the create, DigitalOcean
`price_hourly` before it) runs only on a real rental and is owed to T1.

## Not shown here

Anything at the target shape: the per-step hash cost at 1.95 GB of state,
checkpoint save and upload times, the chained fold's bytes on a wide-area
link, and the multi-device step. T1 in the plan.
