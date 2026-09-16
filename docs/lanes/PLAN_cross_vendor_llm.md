# Draft plan: a cross-vendor bitwise-identical language model

Status: DRAFT FOR DISCUSSION. Nothing here is authorized or started. Written 2026-09-16.

## The claim worth making

Nobody publishes a language model whose training is bitwise reproducible across three GPU vendors. If we train a GPT-3 Small shaped model and show that Apple, AMD and NVIDIA produce **the same bits**, and anyone can download the artifacts and check the hashes, that is a claim a reader can falsify rather than take on faith. That is the whole thesis of the library, demonstrated at a scale people recognize.

## Correction 1: the handoff design does not prove what we want

Training a slice on Apple, handing the checkpoint to AMD, then finishing on NVIDIA produces **one model, not three**. Each segment ran on exactly one vendor, so nothing was ever computed twice and no cross-vendor comparison exists. It proves checkpoints are portable, which is worth something, but it is not a bitwise identity claim.

The claim needs the same work done twice or more, on different silicon, compared bit for bit. So the design should be **replication, not relay**.

**Proposed design.** Train the full run once on the cheapest capable vendor, saving a checkpoint every N steps. Then, for each checkpoint boundary, take checkpoint `i` and run the next N steps **again on a different vendor**, and verify it reproduces checkpoint `i+1` exactly. Rotate which vendor replicates which segment. The headline becomes: every segment of this training run was reproduced on at least one other vendor, bit for bit, and here are the hashes.

If budget allows, add one **complete independent second run** on another vendor. That gives the strongest sentence available: two full training runs, different vendors, identical final weights.

## Correction 2: the overtraining factor is about 120x, not 12x

GPT-3 Small is 125M parameters and GPT-3 trained every model on 300B tokens, about 2,400 tokens per parameter. Compute-optimal scaling puts the right figure near 20 tokens per parameter, so about **2.5B tokens**, not 25B. That is roughly **1/120th** of GPT-3's token budget, not 1/12th.

This is good news and it is the single biggest cost lever in the plan.

Rough compute: `C ≈ 6ND` = 6 × 125e6 × 2.5e9 ≈ **1.9e18 FLOPs** for one full run.

## Correction 3: Apple cannot carry a third of this

The M4 is a laptop GPU. For a training workload of this shape it is one to two orders of magnitude off an H100, and our identity mode is slower than an unconstrained implementation. An even three-way split would put Apple's third in the weeks.

Give Apple a **proving segment**, not a third: one or two checkpoint intervals, enough to show the Apple column reproduces the same bits, with the bulk of the FLOPs on rented silicon. The claim does not weaken. "Apple reproduced segments 7 and 12 exactly" is the same evidence as "Apple did a third of the work".

## DECIDED: the tokenizer, and why we build no vocabulary trainer

Settled Sep 16 2026. Three things were being confused and they are separate:

- **Tokenizer**: code that turns text into numbers. **We have one**, written in Mojo, with its own identity lane.
- **Vocabulary**: a data file listing the pieces and their numbers. We have none; we deleted OpenAI's GPT-2 table deliberately.
- **Vocabulary trainer**: a tool that builds that file from a corpus. We have none, and **we are not writing one**.

**What the vocabulary is actually for.** Not for verifying the tokenizer, which already uses a synthetic vocabulary mojolearn generates, so our tests and identity lane need no real one. It is for **training cost**: byte-level tokenization needs roughly 4x more tokens for the same text, so a 32k to 64k vocabulary makes the GPT-3 Small run about four times cheaper, and makes the model comparable to others of that shape.

**The decision: train our own vocabulary using Hugging Face `tokenizers` (Apache-2.0), and write no trainer of our own.** Measured Sep 16: their BPE output is bitwise reproducible on every axis testable at one core, and the reason is structural rather than incidental, since the serialization is id-ordered so run-to-run variation has nowhere to leak in. SentencePiece BPE also reproduces the vocabulary, though corpus order changes the model file's bytes while leaving the vocabulary and the tokenization identical; and its sampling path does **not** reproduce and cannot be pinned, as 0.2.2 rejects `random_seed` as an unknown field.

**SETTLED Sep 16 2026, measured, $0.25 on one rented CPU pod.** Hugging Face `tokenizers` BPE came back IDENTICAL on every axis that could be moved: five repeated runs, vocabulary 1k / 8k / 32k, corpus order forward / reversed / rotated, **threads 1 / 2 / 4 / 8 / 16**, **arm64 M4 against x86_64 Linux**, and library versions 0.20.3 / 0.22.1 / 0.23.2. The thread axis, which was the one most likely to break it, did not.

**So: pin Hugging Face `tokenizers` BPE. Build nothing.** There is nothing left to fix. The stability is structural rather than lucky, verified rather than assumed: the vocabulary serializes in strict id order (checked to be id order and not merely alphabetical) and selection runs over integer counts, so there is no float accumulation to vary.

**Two hard exclusions, both measured:**
- **Never unigram.** HF unigram produces the same token set but 1,953 of 8,000 scores differ (median delta 1.78e-15, max 5.4e-3), and across 54,417 held-out lines 2 retokenize to different pieces and 281 more get different ids. Float-accumulation noise, which is fatal to a bitwise claim however small.
- **Never SentencePiece's sampled mode** (`input_sentence_size` with shuffle): 946 pieces unique per run, 135 of 400 lines splitting differently, and it **cannot be pinned**, since 0.2.2 rejects `random_seed` as an unknown field.

**SentencePiece BPE on a whole corpus** is the "files differ, behavior identical" case: `sp.vocab` and the piece/score/id structure are identical on every axis including threads, while `sp.model` bytes move because the proto embeds `model_prefix`, the input list order and the NFKC table. Usable, but HF is the cleaner claim.

**What we would state publicly, and it is short:** BPE and never unigram, fixed corpus bytes and vocabulary size, version pinned as hygiene rather than necessity. Thread count, file order and architecture need no promise at all.

**Evidence limits, recorded rather than glossed:** one 16 MB English corpus, nothing at real training scale, and three library versions is evidence rather than a guarantee about future versions.

**What ships where.** The wheel carries tokenizer code and **no vocabulary and no third-party data**. The vocabulary file travels with the model on Hugging Face, alongside the weights, which is what every model does. PyTorch ships no tokenizer, no vocabulary and no trainer, so it is not a precedent either way.

**The vocabulary is built once and frozen.** Changing it invalidates the model, since every number would mean something else.

## Correction 4: the tokenizer changes the token budget

We removed the GPT-2 vocabulary table deliberately, and our models are byte-level. A byte-level model needs roughly 4x more tokens to see the same text, so "2.5B tokens" has to be stated in the units we actually train in. Either:
- train byte-level and size the budget in bytes, stating it plainly; or
- train with a user-supplied vocabulary, which we do not ship, and document which one the run used so a reader can reproduce it.

Decide this before sizing anything, because it moves the compute estimate by about 4x.

## Correction 5: we are not ready today, and the reason is informative

Two things say wait:

1. **We have a live reproducibility defect on AMD.** `rf-score-weighted` produces different bits between two runs on the same MI300X, confined to the regression parts, found during the 0.8.6 record. It is a forest lane, not a neural one, so it does not directly implicate this path, but it is proof that our AMD code can be nondeterministic in ways we have not characterized. Starting a multi-day cross-vendor training claim while a known same-machine nondeterminism is open would be building on sand.

2. **Our neural identity evidence is tiny-fixture only.** The `byte-lm` and `mamba2` lanes prove identity on small inputs. Long training runs introduce shapes that small fixtures never reach: different grid and tile selections at larger batch sizes, reduction trees that vary with occupancy, accumulation over thousands of steps where a single last-bit difference compounds. Nothing we have recorded tests that.

## Correction 6: the multi-GPU path should be part of the claim

Andrew's suggestion, and it is a good one. A single-GPU reproducibility result is a weaker claim than it looks, because the hard part of deterministic training is exactly what multi-device introduces: cross-device reductions, gradient folds whose order must be pinned, and copy traffic between contexts. That copy traffic is the same hazard class as the MI300X peer-copy stale read we found in September, where a kernel read a destination before it was written.

So train the model on **more than one GPU per vendor** and make the cross-device fold part of what is being proven. The claim becomes: this model was trained on multiple devices, and the multi-device result is bitwise reproducible across vendors. That is a much harder thing to do and a much better sentence.

Two consequences for the plan:

- **Box sizing changes.** Segments now need two-device boxes rather than one, which raises the per-hour cost and narrows which providers work. RunPod supports two-GPU NVIDIA and two-GPU AMD; Hot Aisle pins one GPU per body, so a two-device AMD run goes to RunPod.
- **The Apple leg becomes single-device by necessity**, since there is one GPU in the machine. State that plainly rather than papering over it: Apple reproduces the single-device segments, and the multi-device claim covers NVIDIA and AMD.

**This depends on a prerequisite we have not measured.** We do not currently know which algorithms have a multi-GPU path at all, let alone which are bitwise reproducible across devices. That audit is running now. The neural training path does have one (the `par-*` lanes cover it), but our evidence for it is tiny-fixture only, exactly as for the single-device case. The calibration run below should therefore be **multi-device from the start**, since adding devices later would invalidate everything it established.

## The pipeline is a CHAIN, and it is only as strong as its weakest link

"End to end reproducible" means: give someone the corpus, the seed and the config, and they get our model back **byte for byte**, on different hardware. That is six links, and a single weak one makes the whole claim worthless. Status as of Sep 16 2026:

| # | Link | Where it runs | Status |
|---|---|---|---|
| 1 | Corpus to vocabulary | **CPU only** (no GPU path exists in any implementation) | Being built. Claim is cross-ARCHITECTURE, not cross-vendor |
| 2 | Text to tokens | Host integers | **Done**, has an identity lane |
| 3 | **Data ordering** (shuffle, sharding, batching) | Host | **UNVERIFIED — check whether a lane exists** |
| 4 | Training kernels | GPU | Proven on TINY fixtures; **unproven over thousands of steps**. The real gap |
| 5 | Checkpoint save and reload | Both | Partially proven: model cells hash saved bytes, reload equality checked per lane, **never across a long run** |
| 6 | Inference | Both | Largely proven; 182 real batch parts, 17 named n/a, 0 undeclared |

**Link 3 is the one nobody has looked at and it is as load-bearing as any kernel.** If the example order, the shuffle seed handling or the shard assignment varies between runs or between machines, the model differs no matter how perfect the arithmetic is. Before the LLM run, establish whether a deterministic data-ordering lane exists and build one if it does not.

**Link 5 matters more under the replication design** than it would otherwise, because that design restarts from checkpoints repeatedly. A checkpoint that does not round-trip bit-exactly makes every replicated segment meaningless, and it would look exactly like a vendor disagreement.

## The calibration run, which should come first

Before committing real money, do a **small version of the whole thing**: a 10M to 20M parameter model, one to two hours of training, the full replication design across all three vendors.

It answers every open question cheaply:
- Do our neural training kernels stay bitwise identical across vendors over thousands of steps, or does something drift?
- What is our actual tokens-per-second on each vendor in identity mode, which is the only honest input to a cost estimate?
- Does the checkpoint format round-trip bit-exactly across vendors?
- Does a segment restarted from a checkpoint reproduce the original continuation exactly, which is the property the whole design rests on?

If the calibration run holds, the full run is a scaling exercise. If it does not, we have found the bug for a few dollars instead of a few hundred.

## Cost, with the uncertainty stated

Our identity-mode throughput on this workload is **unmeasured**, so these are ranges, and the calibration run replaces them with numbers.

| Item | Estimate |
| --- | --- |
| Calibration run, all three vendors, 10 to 20M params | $10 to $30 |
| Full run, one vendor, 125M params, ~2.5B tokens equivalent | $30 to $80 |
| Segment replication on a second vendor, ~20% of the FLOPs | $10 to $30 |
| Apple proving segments | free, machine time only |
| Optional second complete run for the strongest claim | $30 to $80 |
| Failed runs, restarts, contingency | 1.5x to 2x the above |
| **Total, replication design** | **$100 to $250** |
| **Total, with a second complete run** | **$200 to $450** |

Wall clock is realistically **two to four days**, not one, once setup, failures and reruns are counted. A single clean training run may well be under a day; the surrounding work is not.

## Artifacts, if it works

- The model weights at every checkpoint, with hashes.
- The per-vendor reproduction record: which vendor reproduced which segment, and the matching hashes.
- The exact commit, wheel hash, dataset and seed, so a reader can rerun it.
- A verification command that checks a downloaded checkpoint against our published hash.

Publishing to Hugging Face is the right home for the weights. Two cautions: the training corpus must be openly licensed, since we tell people where the data came from, and we ship no third-party data ourselves; and the model card should state exactly what the bitwise claim covers and what it does not.

## Recommendation

Yes, do it, and it is a genuinely big deal. But do it in this order:

1. Close the AMD `rf-score-weighted` nondeterminism, or characterize it well enough to know it does not touch this path.
2. Run the calibration model end to end across three vendors.
3. Only then size and commit the 125M run.

The calibration run is the decision point, and it costs about as much as one of tonight's wasted AMD legs.
