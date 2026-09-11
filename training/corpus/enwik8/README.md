# Pinned English corpus for the byte language model (enwik8)

The English kind of ENGINEERING_RULES section 9 for every neural timing and
quality claim, from 2026-09-11 night (the code kind is
`training/corpus/pile_github`). Andrew asked for two corpora that are the
norm, generalize and carry published benchmarks: "is shakespeare a good
file? what generalizes? what is the norm? what are the benchmarks? we
should take 2 corpora that generalize and that have benchmarks".
`training/corpus/tinyshakespeare` (1.1 MB, a nanoGPT quick-start demo with
no leaderboard) is retired as a timing corpus; it stays for the byte LM
validation runs that already pin it.

`input.txt` is the first 10^8 bytes of the English Wikipedia XML dump of
2006-03-03: the Hutter Prize file and the most cited byte-level language
modeling benchmark. The source is `http://mattmahoney.net/dc/enwik8.zip`
(36,445,475 bytes, sha256
`547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc`), whose
single member is 100,000,000 bytes with the published md5
`a1fa5ffddb56f4953e226637dabbb36a` and sha256
`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`
(computed on the Mac 2026-09-11). No normalization, no tokenizer.

The benchmark split is bytes [0, 90M) train, [90M, 95M) validation and
[95M, 100M) test, metric bits per character. Published: Transformer-XL 1.06
(12 layers) and 0.99 (24 layers), arXiv 1901.02860; T12 1.11, arXiv
1808.04444; SHA-RNN 1.076, arXiv 1911.11423. Those numbers come from full
training runs; a timing leg matches the metric, not the budget.

The probe's schedule starts at byte 0, so a three-step timing run reads the
dump's `<siteinfo>` header and the first articles. Step time does not
depend on which bytes a step reads beyond that.

The bytes are NOT committed. `manifest.json` pins the source, the hashes and
the length, and `tools/fetch_corpus_enwik8.sh` rebuilds `input.txt` and
refuses any byte that does not match.

Consumers: `tools/lm_step_memory_probe.py --corpus training/corpus/enwik8/input.txt`,
`tools/torch_lm_step_opponent.py --corpus enwik8`, `tools/attention_step_leg.sh`,
`tools/torch_lm_step_opponent_leg.sh`.

License: Wikipedia text, CC BY-SA and GFDL. This repository redistributes
none of it, only its hash.
