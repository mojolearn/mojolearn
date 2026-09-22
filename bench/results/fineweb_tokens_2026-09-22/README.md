# The GPT-3 Small run's token stream, produced three times on three CPUs

Evidence only. E3 of `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md`, 2026-09-22.
FineWeb-Edu sample-10BT shards 000, 001, 002, 003 (train) and 013 (held out,
every one of its rows), staged from R2 and pinned, one document per parquet
row, through the pinned vocabulary `vocab/mojolearn-bpe-fineweb-edu-50257-v1`
(ranks sha256 bfe4a401..., identity in every manifest), by
`tools/fineweb_tokens.py` at commit 83dc038d5 with the published wheel
mojolearn 0.8.14's tokenizer binding (`pip_freeze.txt`), on three RunPod
CPU pods rented by `tools/runpod_cpu_leg.sh` (bodies in `bodies/`, presigned
URLs redacted).

| run | pod flavor | CPU | encode MB/s | seconds | tokens.i32 sha256 |
|---|---|---|---|---|---|
| 1 | cpu3c | AMD EPYC 7702P (host load 160) | 20.8 | 1,082 | 4cf7181ba71058e7... |
| 2 | cpu5c | AMD EPYC 4564P | 74.8 | 344 | 4cf7181ba71058e7... |
| 3 | cpu3c | AMD EPYC 7713 | about 30 | 742 | 4cf7181ba71058e7... |

**All three streams are byte-identical**: 12,442,225,788 bytes,
3,110,556,447 ids from 3,098,101 documents, 4.75 bytes per id, max id
50,255, sha256 `4cf7181ba71058e70d7329dbae32de53c5750b49ae95331cebde4676f922209d`.
The train range is ids [0, 2,926,502,182) and the validation range, shard
013, is the rest. The three manifests differ only in timings and the tool
record (`run1/manifest.json` against `run2/` and `run3/`).

**What is in R2.** Run 3's stream, split into seven parts of 2,000,000,000
bytes (the last 442,225,788) plus the manifest, under
`corpus/fineweb-edu-10BT/tokens/mojolearn-bpe-fineweb-edu-50257-v1/`, pinned
by size and sha256 in `bench/results/dataset_store/manifest.tsv`
(`run3/parts.sha256`); the joined parts hash to the stream's sha256
(`run3/parts_joined.sha256`). A box stages the group and runs
`cat tokens.i32.part0? > tokens.i32`; `TokenBatches` then verifies the whole
against the manifest. Runs 1 and 2 uploaded nothing: a single PUT to R2 is
capped at 5 GB and their 12.4 GB PUT came back HTTP 400 (`upload_tokens.log`),
which is why run 3 exists and uploads parts. Run 2's manifest alone is at
`.../witness-2/manifest.json`.

**Cost.** $0.105, $0.061 and $0.118; about 13 to 20 minutes each including
staging 9.1 GB of parquet.
