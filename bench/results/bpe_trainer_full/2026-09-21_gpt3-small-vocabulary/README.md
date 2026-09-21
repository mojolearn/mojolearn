# 2026-09-21 the GPT-3 Small run's vocabulary

ONE vocabulary for the GPT-3 Small shape run, trained by our own trainer on the corpus the run
reads. R2 only (derived from third-party text): `vocab/mojolearn-bpe-fineweb-edu-50257-v1/`.

| | |
|---|---|
| corpus | FineWeb-Edu sample-10BT shard `000_00000.parquet` (sha256 b1ba7b2c...), ALL 726 row groups, 726,000 documents |
| text | `tools/fineweb_text.py SHARD OUT_DIR`: each document's UTF-8 text plus one 0x0A, one file per row group, 3,470,451,588 bytes, sha256 of the concatenation 2fdce5685e3665dd67ac45d6eb54689edcf5d12405932859e79f908086510394 |
| trainer | `tokenizer/train/train_main.mojo OUT 50256 2 rg*.txt` at main d2b1f49f7: each file one document, 50,256 ranks, min_frequency 2 |
| result | 50,256 tokens, 50,000 merges, 3,137,688 pre-token groups, 32,143 ties broken; 77.9 s, 2.5 GB peak, Apple M4, one core |
| ranks.tsv | bfe4a401789e734c139ee6d872848833d0058b1b274757334c955236a57d9b24 |
| tokenizer.json | e5004bb72cb1ce67bc1517b7a20f513efce7e3b6b9e081abe974db805546bc12 |

## Why this sample size

Three samples of the same shard, each scored on 5,000 documents of shard 013 (23,590,061 bytes)
that none of them saw. FineWeb-Edu's `token_count` column is the GPT-2 tokenizer's count for the
same documents: 5,146,266.

| sample | bytes | tokens on the held-out text | against GPT-2's table | count of the last merge in its sample |
|---|---:|---:|---:|---:|
| row groups 0 to 19 | 95,788,129 | 4,976,106 | 3.31% fewer | 24 |
| row groups 0 to 104 | 504,893,432 | 4,962,428 | 3.57% fewer | 124 |
| all 726 row groups | 3,470,451,588 | 4,958,917 | 3.64% fewer | not counted |

Compression has flattened by 505 MB. The larger sample is kept for the tail: a token merged on 24
occurrences is an accident of the sample and its embedding row is barely trained.

## The builder on three machines (96 MB sample, 0.8.11 wheel, before the speedup)

`body.template.sh` and `door.py`: `pip install mojolearn==0.8.11` in a fresh venv, the sample staged
from R2 (`corpus/fineweb-edu-10BT/vocab-sample/000-rg0-19.txt`, sha256 30f55413... checked on the
box), `BpeVocabularyTrainer(vocab_size=50256, min_frequency=2, backend="mojo")`. The trainer is host
code; the rented GPU boxes contribute their CPUs.

| machine | wall | ranks.tsv sha256 | tokenizer.json sha256 |
|---|---:|---|---|
| Apple M4 (repo checkout, 0.8.11 binding) | 730.7 s | 06a8b2733537f408531b8b4deb959c5d84e78b85ef6004548af558e27137d812 | c89330d6c3ed18212c3aece3c7550f16624a8e3cb39f5a014838833a7e09b21b |
| RunPod MI300X pod, AMD EPYC 9474F | 1,554.6 s | 06a8b273...d812 | c89330d6...b21b |
| RunPod RTX 4090 pod | see `nvidia/` when it lands | | |

The faster builder (d2b1f49f7, `../2026-09-21_incremental-counts/`) reproduces 06a8b273... in 2.4 s.

NOT DONE: the whole-shard vocabulary (bfe4a401...) has been built on the M4 only.
