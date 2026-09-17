# Can this trainer be fed 10 billion tokens?

Lane `lane/lm-training-shakedown`, question 4. Source reading plus what leg 1
staged; no separate GPU arm was needed to answer it.

**Short answer: no, and the gap is bigger than a loader.** The corpus is in
R2, the tokenizer algorithm is in the repository, and nothing joins them to
the trainer. Three separate pieces are missing and one of them is a decision,
not code.

## What is actually there

FineWeb-Edu 10BT IS in the R2 store, exactly as the brief says, and
`tools/dataset_store.sh:53` describes it as "the corpus for a real LM run":

    corpus/fineweb-edu-10BT/NNN_00000.parquet   NNN = 000 .. 013
    14 shards, 28,518,193,415 bytes, pinned by size and sha256 in
    bench/results/dataset_store/manifest.tsv, staged to
    training/corpus/fineweb-edu-10BT/

It is a GROUP key: `push`, `verify` and `presign-put` take the group name and
expand to one key per shard. enwik8 (100,000,000 bytes) and pile_github
(97,124,565 bytes) are single keys and are what every neural leg has ever
used. Leg 1 staged enwik8 in 7 seconds, so staging is not the problem.

## Gap 1: the loader reads the whole corpus into host RAM

`tools/lm_step_memory_probe.py:229`, `CorpusBatches`:

    raw = self.path.read_bytes()
    self.data = np.frombuffer(raw, dtype=np.uint8)
    self.modulus = len(raw) - length - 1

One file, read entire, resident. `tools/byte_lm_real_text_capture.py` is worse
for this purpose: it materializes ALL 128 batches up front
(`full_schedule = b''.join(... for step in range(128))`) before step 1.

The only chunked reads anywhere in the LM stack are sha256 hashers and the
Linux sealed-memfd checkpoint reader. There is no `mmap`, no `np.memmap`, no
generator, no iterator on the token path, and no seam to hook one into:
`train_step(ids)` takes a materialized `int32[batch, length + 1]` and nothing
else -- no path, no dataset object, no cursor.

This is the SMALLEST of the three gaps. A `np.memmap` over a pre-tokenized
uint16 file is a contained change to a driver, not to the library.

## Gap 2: there is no parquet reader and no tokenizer in the training path

Nothing in the repository reads parquet into the trainer. More importantly,
`CorpusBatches` does not tokenize at all -- it casts RAW BYTES to int32 and
calls them token ids. Its own docstring says so: "the byte LM's next-byte
schedule at this shape, no tokenizer, no normalization."

The tokenizer itself EXISTS and is good: `python/mojolearn/tokenizer.py`
exposes `GPT2Tokenizer` over the host binding `_mojolearn_tokenizer_host`,
implementing the GPT-2 pre-tokenizer pattern and byte-level BPE by merge rank.
It is simply not wired to the trainer, in either direction.

## Gap 3: 99.5% of the two largest tensors would train on nothing

This is the one that is easy to miss and is not a loader problem at all.

At the target shape the vocabulary is 50,257 and the embedding and lm_head are
`(50257, 768)` each: 38,597,376 parameters apiece, **77,194,752 of the
162,147,840 total, 47.6% of the model**. Feed it raw bytes and every id is
below 256, so rows 256 to 50,256 of both tensors receive a gradient of exactly
zero forever. Nearly half the model would be untrained, and the loss curve in
MEASURED.md section 6 -- which IS falling, genuinely -- is the curve of a
byte-level model wearing a 50,257-row coat.

So a real 50,257-vocab run needs real BPE ids, which needs a vocabulary.

## The vocabulary is a decision, not a task

mojolearn deliberately **ships no vocabulary**
(`python/mojolearn/tokenizer.py`: "MOJOLEARN SHIPS NO VOCABULARY
(2026-09-15)"), and the GPT-2 table was removed under the no-third-party-data
rule. Two ways forward and they are not equivalent:

  1. **Train our own.** `python/mojolearn/_bpe_trainer.py` is a
     bitwise-deterministic byte-level BPE vocabulary trainer, held byte for
     byte against `tokenizer/train/bpe_train.mojo` by
     `pixi run check-bpe-trainer`. It is deterministic by construction, which
     is the right property for a model trained under the IDENTICAL contract:
     the vocabulary is part of what the model IS, and a model whose
     vocabulary cannot be rebuilt from source we control is only half
     reproducible. Cost: a training pass over a corpus sample, unmeasured.
  2. **Let the user supply one** and never distribute it. Fine for a library,
     awkward for a published model.

Nobody has decided this, and it gates everything downstream.

## What the work actually is, scoped

In dependency order. None of it is exotic; the point is that it is four things
and not one.

  1. **Decide the vocabulary** (above). Then train it and pin it in R2 with a
     size and sha256 like every other artifact. UNSCOPED until decided.
  2. **A pre-tokenization pass**: 14 parquet shards -> text -> `GPT2Tokenizer`
     -> a flat uint16 id stream (50,257 fits in uint16), written as shards and
     pushed to R2 under a new key with the same pinning. 10B tokens is 20 GB
     as uint16 against the 28.5 GB of parquet. One-time, parallel, CPU-only,
     and a RunPod CPU pod is the right box for it
     (`docs/RUNPOD_CPU_LEG.md`). The determinism requirement is that the
     same shards always produce the same ids, which the tokenizer already
     gives.
  3. **A memory-mapped batch reader** to replace `CorpusBatches`: `np.memmap`
     over the uint16 shards, the same `step k row b` schedule, the same
     manifest checking. Small, and it needs its own corpus manifest schema
     since `mojolearn.byte-lm.corpus.v1` describes a single raw-byte file with
     `vocabulary: 256`.
  4. **A held-out split that is actually held out.** enwik8's manifest already
     declares `train_range` [0, 90000000], `validation_range` and
     `test_range`, and `CorpusBatches` ignores all three -- it wraps modulo
     the WHOLE file. At 2,000 steps that does not matter (step 1,999 reads
     byte 4,093,952, well inside the train range) but a run of 610,352 steps
     at batch 8 consumes 10B tokens and would read the test set unless the
     reader honours the split. A 200-hour run that quietly trains on its own
     test set is worth more care than it currently gets.

## The part that is genuinely fine

Staging. `tools/stage_from_r2.sh` presigns short-lived GETs on the Mac, pipes
a fetch-and-verify script to the box over ssh stdin, checks every file against
`bench/results/dataset_store/manifest.tsv` by size and sha256, and hard-links
corpora into the box's source tree. The credential never leaves the Mac. Leg 1
staged 100 MB in 7 seconds; 20 GB of pre-tokenized ids would be minutes, once,
per box. **Nothing about the data pipeline blocker is about moving bytes.**
