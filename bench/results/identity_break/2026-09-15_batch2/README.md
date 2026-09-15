# The batchgrad, batchscale and ragged parts: first record (2026-09-15)

Lane `lane/batch-invariance-2`. The three opt-in parts of
`tools/identity_break.py` (`--batch-grad`, `--batch-scale`, `--ragged`) on 26
lanes, nine fixtures, two repeats, every other part (train, infer, model,
batch) run as usual, at commit 1cc7a2f47. The batch2 agent committed this
directory unreviewed at Andrew's stop (d4a3cb79b). It was reviewed on restart,
2026-09-15 about 11:15 ET; the review is the last section.

| column | box | how | cells |
|---|---|---|---|
| `apple-m4.json` | Apple M4, Metal, this Mac, one core, shared machine | six chunks merged with `--merge`, one binding set (23 sha256 values, `merged_separate_builds: false`), `commit_source: git rev-parse HEAD` | 234 |
| `nvidia-h100-sm_90a.json` | RunPod H100 80GB HBM3, one GPU, pod xbhbu11dsetza0 | 24 bindings built on the box from 1cc7a2f47; lanes split into two harness processes on the one GPU (a, b), merged | 234 |
| `amd-mi325x-gfx942.json` | DigitalOcean MI325X, one GPU, droplet 600680353 | as the H100, `MOJOLEARN_GPU_ARCHS=gfx942` | 234 |

Both boxes were rented at about 09:46 ET, before Andrew's 10:30 ET rule
limiting GPU boxes to PyPI release records. Both were deleted and verified
gone (pod `DELETE -> 204`, then `HTTP 404`; droplet `DELETE -> 204`, then
`HTTP 404`). The leg directories, logs and body are in
`~/mojolearn-evidence/batch2/` (`2026-09-15_134552-nvidia-h100-batch2`,
`2026-09-15_134606-amd-mi325x-batch2`, `leg_body.sh`).

## Verdicts

`diff.three-columns.txt`:

    summary: IDENTICAL=234
    summary (infer/model): IDENTICAL=351, N/A=117
    summary (batch): IDENTICAL=234
    summary (batchgrad): IDENTICAL=108, N/A=126
    summary (batchscale): IDENTICAL=189, N/A=45
    summary (ragged): IDENTICAL=108, N/A=126

No MOVED, BATCH_MOVED, DIVERGENT or REFUSED cell. Every N/A names its reason:
`no-backward` (99), `mean-reduction-fixed-batch` (18, the byte LM trainers),
`mean-reduction` (9, SmallMLPTrainer), `not-in-the-serving-set` (45),
`no-sequence-axis` (126), `no-save` (108), `function` (9).

`diff.vs-166-lanes-record.txt` puts the three columns beside the 166-lane
record's three on the same 26 lanes: train IDENTICAL=234 and batch
IDENTICAL=234 across all six, so the new parts did not move any existing hash.
The new parts read NOT-COMPARED there because the old record has no such keys.

## The sabotage, seen to fail

`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 MOJOLEARN_IDENTITY_BATCHGRAD_SABOTAGE=1`,
base fixture, one repeat, on transformer, mamba2, samba, embedding, byte-lm,
byte-lm-host-infer and pca. batchscale and ragged have no switch of their own:
they read `MOJOLEARN_IDENTITY_BATCH_SABOTAGE` (`PART_SPECS` in
identity_break.py). Diffed against each box's production column:

| column | batch | batchgrad | batchscale | ragged |
|---|---|---|---|---|
| `nvidia-h100-sm_90a.sabotage.json` | BATCH_MOVED=7 | BATCH_MOVED=4 | BATCH_MOVED=6 | BATCH_MOVED=5 |
| `amd-mi325x-gfx942.sabotage.json` | BATCH_MOVED=7 | BATCH_MOVED=4 | BATCH_MOVED=6 | BATCH_MOVED=5 |
| `apple-m4.post-merge-smoke.sabotage.json` (restart, 6 lanes) | BATCH_MOVED=6 | BATCH_MOVED=4 | BATCH_MOVED=5 | BATCH_MOVED=4 |

Every hashed cell of every part moved; the rest are N/A by name (pca and the
byte LM lanes have no batchgrad, pca and embedding no ragged, embedding no
batchscale). Train and infer cells stay IDENTICAL, as they must.

## Restart check on the merged tree

After merging origin/main (450423c95), on the M4, one core, shared machine,
with a Metal set whose 23 sha256 values equal `apple-m4.json`'s (copied, not
rebuilt): transformer, mamba2, samba, embedding, byte-lm and pca, base
fixture, one repeat, all parts. `diff.post-merge-smoke.four-columns.txt`
against the three committed columns: every base cell IDENTICAL x4 (summary
counts include the other fixtures' three-column rows). The sabotage arm is the
table row above. `python/mojolearn/tests/test_ragged_lengths.py`: 19 passed and
1 skipped by name (LanguageModelInference, no host byte LM binding in the
worktree), then 20 passed once the agent's CPU host byte LM binding was copied
into `python/mojolearn/host/`.

## Review findings

1. There was no README. This file is it.
2. Each column's provenance checks out. The two GPU JSONs equal the merge of
   the box halves `identity_break.<vendor>.a.json` and `.b.json` in the leg
   archives cell for cell, and the sabotage JSONs equal the boxes' copies. All
   three columns carry commit 1cc7a2f47. The Apple chunks share one binding set.
3. The hashes in `diff.three-columns.txt` come from the right columns. It
   regenerates byte for byte from the three JSONs, apart from its
   `require-columns` line, which came from a `--require-columns` flag. The
   vs-166 diff regenerates byte for byte against
   `2026-09-14_166-lanes/{apple-m4,nvidia-h100-sm_90a,amd-mi325x-gfx942}.json`.
4. The Apple column had NO sabotage arm at 1cc7a2f47. The agent's M4 sabotage
   runs (its scratchpad `sab1`, `sab2`) predate the final harness. The restart
   arm above closes that gap.
5. The Apple Metal set was built at df617c699, not 1cc7a2f47. Between the two,
   only `checks/hardware_matrix_check.mojo` and `checks/kernel_matrix.mojo`
   changed (the hardware matrix admission tables). `lengths=` is Python only.
6. The GPU legs ran two harness processes on one GPU at once. On the MI325X
   that setup once gave a MOVED cell and a refusal (record2, STOP_STATE). Here
   every cell is STABLE and matches the other columns, so nothing was hidden,
   but a release record should run one process per GPU.
7. The byte-lm-host-infer and -threaded cells are host CPU bindings on every
   column. They show that the host path agrees across three machines, not that
   the GPUs agree.
8. Owed: the CPU-only column for the byte LM host lanes (the agent's
   `cpucol` run never produced output), and the AMD and NVIDIA columns at a
   later commit, with the next release record only.
