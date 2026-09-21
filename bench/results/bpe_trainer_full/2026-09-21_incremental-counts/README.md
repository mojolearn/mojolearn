# 2026-09-21 the vocabulary trainer keeps its pair counts across merges

`tokenizer/train/bpe_train.mojo` recounted every adjacent pair in every pre-token group on each
of the 50,000 merges. It now builds the counts once and, per merge, touches only the groups that
hold the merged pair; the winner comes off a heap ordered by the same `(count, key)` rule. Counts
are integer sums, so the table is the one a full recount builds and the files are the same bytes.

Apple M4, one core, `tokenizer/train/train_main.mojo` built with `mojo build -j 1`, 50,256 ranks,
min_frequency 2. "before" is the 0.8.11 binding through `BpeVocabularyTrainer(backend="mojo")`
(the September 18 row is that day's EPYC 9575F CLI run).

| corpus | bytes | groups | before | after | ranks.tsv sha256, before = after | ties |
|---|---:|---:|---:|---:|---|---:|
| FineWeb-Edu 10BT shard 000, first 5,000,000 bytes | 5,000,000 | 60,186 | 66.4 s | 0.35 s | d8fc4eb92b1bbfb69e789ca1b1184a117434ce188899e68c9e5e983211425a4b | 46,659 |
| the September 18 recipe (10 MB enwik8 + 10 MB pile_github) | 20,000,000 | 220,165 | 1,116.6 s | 1.2 s | 3d547b17821cf46502f275a441dd6ded9682a4ddcacde993a1ff836f39c4122d | 47,825 |
| FineWeb-Edu 10BT shard 000, row groups 0 to 19 | 95,788,129 | 332,278 | 730.7 s | 2.4 s | 06a8b2733537f408531b8b4deb959c5d84e78b85ef6004548af558e27137d812 | 45,762 |
| FineWeb-Edu 10BT shard 000, row groups 0 to 104 | 504,893,432 | 918,308 | stopped | 10.9 s | 42dd91a48dd5da147b4f5e15945984a2183f636dfca031424d8d5497cef8e71f | 41,526 |

`tokenizer.json` matched in every compared row too (5927384e..., 7ae8b893..., c89330d6...). The
tie counts are equal before and after. The 505 MB row has no "before": that run was stopped after 24 minutes, unfinished.

Gates on this tree: `check-bpe-trainer` PASS (the Python reference, byte for byte, 284 tie-broken
selections), `check-bpe-trainer-sabotage` SEEN TO FAIL (6 failures).

The merge loop raises if the merged pair's count is not exactly zero after its merge, which is
what a group missing from the pair's holder list would cause.
