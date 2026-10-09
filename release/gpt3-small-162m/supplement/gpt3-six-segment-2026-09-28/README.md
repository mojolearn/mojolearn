# Six-segment GPT-3 Small-shaped training evidence

This supplement supports the route figure in Section 6, the scale result in
Section 7.2, and the scale evidence in the verification appendix. It is separate
from the release 0.8.10 fixture record.

Both routes compute 5,000 optimizer steps from the same initialization, with
64 logical shards of four 2,048-token sequences per step: 2,621,440,000 tokens
per route. The recipe records the model dimensions, AdamW settings, seed,
token-stream identity, and exact FP32 learning-rate table.

| Segment | Steps | Route A | Route B |
|---|---|---|---|
| 1 | 1–1,000 | 2 × NVIDIA H100 80 GB HBM3 | 1 × AMD MI325X |
| 2 | 1,001–2,000 | 2 × NVIDIA H100 80 GB HBM3 | 2 × AMD MI300X |
| 3 | 2,001–2,400 | 1 × H100 + 1 × MI325X, jointly | 1 × H100 |
| 4 | 2,401–3,900 | 2 × AMD MI300X | 2 × H100 |
| 5 | 3,901–4,900 | 2 × H100 | 2 × AMD MI300X |
| 6 | 4,901–5,000 | 1 × Apple M3 Ultra | 1 × NVIDIA L40S |

The table uses completed segment receipts, not the earlier plan. In particular,
B/4 used two H100s, and B/6 used an L40S. Route B's checkpoint-source receipts
show continuation from its own checkpoints after the shared initialization.
B/5's retained launch receipt is for its restart from its own step-4,100 file;
the initial launch's extracted checkpoint references also retain its incoming
step-3,900 checkpoint from B/4.

## What the checker checks

From the paper repository root:

```sh
python3 tools/check_gpt3_evidence.py
```

This is an **evidence-file checker**, not CPU training or CPU replay. It reads
the saved records and checks:

- Manifest hashes and the preceding-line hash links in each original chain.
- Complete, nonduplicated coverage of the 5,000 logical steps in each route.
- Equality at every paired step of state and summed-gradient digests, all 64
  loss bit patterns, learning-rate bits, hash scheme, and shard indices.
- Agreement of every learning-rate entry with the recipe and identical recipe
  and token-stream identities across segment receipts.
- Checkpoint continuity, matching digests for all 56 common post-initialization
  checkpoints, and the route-B checkpoint source paths.
- Matching overlap records after restarts in A/4 and B/5, counted only once.
- All 400 paired coordinator/worker records in the joint A/3 segment.
- Eight retained two-step arrival replay chains against their sender records.
- Positive controls that match and deliberate changes detected at step 101.

It writes `results/gpt3-six-segment-2026-09-28.json`. A successful check means
the saved evidence is internally consistent and supports the reported
comparisons. It does not independently establish that the logged arithmetic
was executed correctly. That requires recomputing training or selected steps.
Checkpoint equality here compares recorded full-file SHA-256 values; the
checker does not download and hash the approximately 1.95 GB checkpoint files.

## Contents and provenance

`index.json` identifies the files for each segment and control. The original
JSONL chains are gzip-compressed without modifying their contents.
`manifest.json` gives each source path relative to the local `gpt3-run`
evidence root, its original SHA-256, the stored file SHA-256, and whether it
is a projection. Receipt projections retain numerical results, checkpoint
digests and environment information while omitting hostnames, transfer URLs,
and infrastructure bookkeeping. Environment text files retain GPU models,
installed binding versions and wheel hashes. Apple's wheel directory contained
both 0.8.19 and 0.8.24; its binding receipt identifies 0.8.24 as the installed
version for the completed segment.

A/4 and B/5 contain separate original attempt and resumed chains. Each chain
has valid internal links; overlaps are compared by numerical fields, rather
than treating a reconstructed union as one uninterrupted hash chain.
The completed ledger also records arrival PASS for A/4 and B/5, but the checker
only counts the eight arrivals whose original chains are included here.

The `sliced-sha256-8.v2` scheme covers all array bytes. Each array is divided
into eight contiguous ranges; the eight SHA-256 hex digests are concatenated
and hashed. State combines the array digests in parameters, first moment,
second moment, flags order. It is not sampling eight array elements.

The final checkpoint SHA-256 recorded by both routes is:

```text
4129921e99ed404db1ca4b7b94ebc43c0bee18e7711deb326b31ec89e0b402c8
```

The original chains and checkpoint payloads are also stored under
`runs/t3/2026-09-22/` in the project's R2 artifact store. This supplement does
not provide public access to that store and does not bundle those checkpoint
payloads. No claim of CPU replay or CPU training at this model shape is made.

To regenerate this supplement from the original local evidence root:

```sh
python3 tools/check_gpt3_evidence.py --import-root /path/to/gpt3-run
```
