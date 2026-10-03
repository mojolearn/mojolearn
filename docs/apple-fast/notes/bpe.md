# Lane af-bpe notes: byte-level BPE training and encode, Apple FAST

## WHERE BPE TRAINING AND ENCODING RUN TODAY (main 8897404da, read 2026-10-03)

BOTH RUN ON THE HOST CPU, IN EVERY TIER. There is no device kernel anywhere in `tokenizer/`.

- The only tokenizer binding is `bindings/_mojolearn_tokenizer_host.mojo`, built by
  `bindings/build_tokenizer_host.sh` -> `bindings/build_host_family.sh`, which REFUSES any
  `MOJOLEARN_NUMERIC_MODE` other than identical and compiles no accelerator target. The FAST and IDENTICAL
  board arms therefore load the SAME host `.so` (`python/mojolearn/tokenizer.py` `_binding()` /
  `_native_trainer()` -> `_backend.load_host_module("_mojolearn_tokenizer_host")`), which is why the M3
  board shows FAST == IDENTICAL on bpe-train (47.8 / 47.9 ms) and the encode gap (52.2 / 58.6) is noise
  between two runs of one binary.
- `bpe-train` = `BpeVocabularyTrainer.train` -> host `bpe_train` -> `tokenizer/train/bpe_train.mojo::train_bpe`:
  ONE host thread. Pre-tokenize every document (`tokenizer/impl/pretokenize.mojo`), dedup the pre-tokens
  into groups (`PieceGroups`, FNV open addressing), count adjacent pairs once into `PairTable`, then the
  merge loop: a binary heap picks the max (count, smallest key), only the groups in `where[pair]` are
  rewritten and their pair counts adjusted. 0 kernels, 0 device buffers.
- `bpe-encode` = `BpeTokenizer.encode_batch` -> host `bpe_encode_batch`: documents split over host
  threads (`host_parallelize`, `host_predict_task_count`), each `BpeTokenizer.encode_bytes`:
  pre-tokenize, then `tokenizer/impl/bpe.mojo::bpe_append` per pre-token (min-rank merge loop over a
  byte-keyed FNV rank table). 0 kernels.
- README lists BPE training among the host training lanes; the trainer's own docstring states "there is
  no GPU path here".

## Profile of the host path per call (board shapes: train 1 MiB enwik8, vocab 4096, min_frequency 2;
## encode 2,048 documents of 2,048 chars, 4 MiB)

| step | train | encode |
|---|---|---|
| Python -> binding crossings | 1 train + 1 sizes + 1 copy | 1 |
| host copies | every document into a List (1 MiB) | every document into a List (4 MiB) |
| pre-tokenize | 1 thread, whole corpus | per doc, host threads |
| core loop | 3,840 merges, heap + touched groups | per pre-token min-rank loop, host threads |
| kernels / syncs / device buffers | 0 / 0 / 0 | 0 / 0 / 0 |

## What this lane adds (FAST + Apple only; everything behind its own define, default OFF)

A NEW FAST-only binding `bindings/_mojolearn_tokenizer_fast.mojo` (`bindings/build_tokenizer_fast.sh`,
refuses any tier but fast; output `python/mojolearn/_mojolearn_tokenizer_fast.so`) and the device module
`tokenizer/fast/bpe_device.mojo`. Every device line sits under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` AND its define. The host binding,
`tokenizer/train/bpe_train.mojo`, `tokenizer/impl/*` and `tokenizer/encoding.mojo` are not edited, so
IDENTICAL (and the host binding every tier loads today) compiles main's code unchanged.
`python/mojolearn/tokenizer.py` routes to the fast binding only when `MOJOLEARN_NUMERIC_MODE=fast`, the
fast `.so` exists, and the binding reports the device path compiled in (`bpe_fast_flags`); otherwise it
runs main's host path, so a fast build with no define is main.

Device train profile (per pass; one pass = one merge, or up to 7 with MERGE_BATCH):
3 launches (top-K partials over the live pair entries, one-threadgroup top-K + selection, apply over
groups), no wait. One 32-byte state readback + wait per 64 passes. Setup: 3 uploads, 2 init launches.
Teardown: one download of the merges (2 x n_merges int32). Live buffers: ~12 (1 u8 + 3 i64 + 8 i32
pools under LIVEBUF: 3).

Device encode profile per call: 6 uploads (text, pre-token bounds, doc offsets, doc pre-token index,
rank table arena + 3 index arrays unless LIVEBUF keeps the table), 2 launches (pre-token merge,
per-document compaction), 2 downloads (ids, counts), 1 wait.

REMAINING HOST STEP (not removed by this lane, stated so the manager does not read the device path as
host-free): pre-tokenization (the GPT-2 pattern, leftmost-first with possessive runs and a lookahead) and
the training corpus's pre-token dedup still run on the host before the first upload. A device
pre-tokenizer is the next lane: per-byte class codes, then boundary rules (contractions, ` ?\p{L}+`
etc., `\s+(?!\S)`), a segmented scan for run ends, then a device hash dedup.

## Exactness argument for MERGE_BATCH (same merges, same order, same tie count)

Selection list L = the top K=8 live pairs by (count desc, key asc), key = left * V + right (V =
vocab_size). q0 = L[0] is the host's winner. A later L[j] is merged in the same pass iff, walking j = 1,
2, ... and stopping at the first failure: (1) q0 is not a self pair (a != b) and no accepted pair is;
(2) L[j] shares no token with any accepted pair; (3) count(L[j]) > count(L[j+1]) strictly (so it is
above every list entry after it and every pair outside the list, all <= count(L[K-1])); (4) count >=
min_frequency and the vocabulary has room. Merging a pair (a, b) changes only pairs touching a or b
(they lose occurrences) and creates pairs with the new token, each of whose occurrences maps injectively
to an occurrence of an old pair (x, a) or (b, x) touching the merged tokens (except for a self pair,
where the bound is the merged pair itself; hence rule 1). Every such old pair other than accepted ones
is ranked after L[j] with a strictly smaller count, so after the earlier merges of the pass L[j] is still
the unique maximum: the host would pick it next, with no tie (the host's n_ties_broken increments only
on q0, as L[1] == L[0] in count). Token-disjoint pairs cannot overlap in the symbol stream, so applying
them in one pass equals applying them in order.

## Exactness argument for the device apply pass

Each group is rewritten by one thread, left to right, non-overlapping (the host's rewrite). Count deltas
are exact integer atomics: each old pair touching a merged symbol is subtracted once (responsibility: the
merge covering its right symbol, else the merge covering its left), each new pair touching a new token is
added once (responsibility: the new token on its right, else the one on its left). Integer adds commute,
so the table after the pass equals a full recount whatever order the threads ran in; the selection is a
total order on (count, key), so the reduction order cannot reach the result.
