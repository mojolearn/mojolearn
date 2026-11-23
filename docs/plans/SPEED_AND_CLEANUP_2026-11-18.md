# Speed and cleanup plan, 2026-11-18

Andrew's order of 2026-11-18 (the comparison with an old release's board was dropped on 2026-11-23; measurement is
ad hoc, only when Andrew asks): fix what is known, then clean up (boundary files, test code, duplicates), and release only when everything is right. No haste. Bits
only have to match between GPU vendors within one version. Opponents are set aside. The rules are in `AGENTS.md` part 8.

Every item names its evidence file under `~/mojolearn-evidence/briefs-20261118/` (`<name>-RESULT.md`). The live list of
owners and leftovers is `FOLLOWUPS.md` in the same directory. Nothing below has been timed unless it says so.

## 1. This round: fixes, then one full timing of our latest

Merged into `preinteg/wave2-20261113` (head 722ad41eed, parse-level compile and tool tests green on GitHub):

| Row or defect | Cause | Evidence |
|---|---|---|
| MLP predict 2.5x on a wide input | serial byte-walk finite scan, seven callers | `fx-mlp-infer` |
| pls-canonical transform 2.0x (and CCA, PLS, ALS, MinCovDet, dictionary learning) | resident check never matched | `fx-rows-rest` |
| train-test-split | one pageable download per output; harness copies inside the clock | `fx-rows-rest`, `fx-harness` |
| pools, batchnorm, gcn, graphsage 1.2-1.9x | per-call host work; a 70 MB device clear per forward | `fx-cnn-layers`, `fx-rows` |
| kernel-shap, isotonic predict | float64 round trip; knots joined per call | `fx-rows-rest` |
| host buffers freed during an asynchronous copy (product and 587 check sites) | no keep-alive after the synchronize; the gate could not see it | `fx-lifetime`, `fx-tool-tests` |
| A/B verdicts: 12 arms deleted, 2 promoted, 3 undecided | | `fx-verdicts`, `fx-verdicts2` |

Left on purpose: radius-neighbors fit (a 1 ms safety check on a call that does no work); knn-imputer and
iterative-imputer (host noise by their own records).

Then: the GitHub build of that head, main fast-forwarded to it, and a short start-up job on the pod. No board is
timed unless Andrew asks for it.

## 2. Speed arms, default off, timed in the same pod session

Each is built as a B arm from its lane branch against the build above and merges only on a win.

| Arm | Rows | Estimate (not measured) | Bits |
|---|---|---|---|
| `MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA` PROMOTED 2026-11-19 (mamba3-decode 0.80, mamba2-decode 0.84, control 0.92; `_OFF` restores) | mamba2/3-decode, samba-decode | -200 to -600 ms a row | same |
| `MOJOLEARN_LMHEAD_FAST_GEMM_PANEL` (door, op and oracle) PROMOTED 2026-11-19 (33,017 -> 284.5 ms, L40S; `_OFF` restores) | chunked LM head, 33 s | to about 1-2 s | change, all columns |
| `MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK` PROMOTED 2026-11-19 (four decode rows 0.91-0.94, about 0.94-0.98 net of drift; `_OFF` restores; refusal checks RUN OWED) | transformer-decode, causal-lm-generate | -60 to -230 ms a row | same |
| `MOJOLEARN_LLAMA_FAST_KV_STRIDE_CAP` (measured 2026-11-19: noise; deleted, docs/TOMBSTONES.md) | transformer decode, grows with length | -100 to -170 ms | may change |
| Mamba-3 backward S17 tail as a parallel fold (lane open) | samba-train-step 45.5 ms | -5 to -13 ms | change, all columns |
| Samba step scratch held, registry resident (lane open) | samba-train-step | -3 to -10 ms | same |
| `MOJOLEARN_XCNN_FAST_TYPED_DOOR` (boundary pilot) | average pools | 10-20 us a call | same |
| 15 arms from converted environment switches; 12 arms of batch ab4; 3 undecided | | | |

Owed before sizing more neural work: a call profile of the decode, generate and train-step rows, and one diagnostic
build with the phase timers (`ideas-neural-speed`, `ideas-training-speed`: no profile exists for these rows).

Needs Andrew's word: Mamba decode reports on request (changes what `step` returns); a constant-time Mamba-2/3 decode
recurrence (decode bits would differ from forward); the recurrent scan on the NVIDIA column only (-525 ms over ten GRU
and LSTM rows, a per-vendor route).

## 3. Duplicates: the same job done in several places (`ideas-functional-duplicates`)

Now, mechanical, bits unchanged (lanes open; merged after the full timing, then the touched rows are timed again):

- 55 private serial host finite scans onto `core/host_finite_scan.mojo` (lane `fx-host-scans`).
- 27 upload/download helpers with an element loop and five identical copies onto one transfer module, resample's
  download included (lane `fx-transfer-helpers`).

Next, small: `metrics/impl/binary_ranking.mojo` onto the eight-bit radix sort (12 launches instead of 128); 15 private
zero-fill kernels and 121 zero-filled lists that are then fully overwritten; a per-family check that a second identical
call uploads nothing.

Later, each with an A/B: device finite scans (with the shared input admission); the 22 "already on device" stores and
their six key schemes (after the boundary cleanup); the five LM heads (after the panel arm's verdict); the three
squared-distance and softmax bodies; the three device Adam bodies (bits change); the GBDT one-bit sort sites as an arm.

Leave alone: hand forwards and their tape twins, GBDT histogram kernels, the forest builders, the GEMM routes, top-k
selections, RNG streams, Apple `fast` (`afn_*`, paused), oracles, pools and arenas, L-BFGS, tombstones.

## 4. Composed models (`ideas-compose-kernels`)

A model composed with `mojolearn.ops.Session` or `Sequential` reaches the sequence GEMM, the fused linear layer, the
norms, the losses and a resident optimizer. It does not reach the Mamba steps, the attention block with its cache,
Samba, the chunked LM head, the fused CNN block or the recurrent scans: those are session op codes 81-93 with no public
name. Identity is inherited by construction; its coverage has holes.

1. Name the block ops on `ops.Session` (about 6 files, no bits change). First settle whether a recording session can
   train through a step op.
2. One identity lane for the forward and backward of every public op, at shapes that cross the kernel route rules.
3. `Session.step` as one launch (an arm); the five optimizer kinds and gradient clipping exposed.
4. A check that runs `OPS_LM_HEAD_V2` on both columns against the oracle.
5. Docs: the contract sentence says "Mojolearn"; the `state` docstring matches what exists.

## 5. Boundary files (`ideas-boundary`, `fx-typed-pilot`)

Not a file merge (only 68 near-duplicate files; merging buys no build time and breaks 2,900 path references). The work
is typed parameter and context tables in place of about 7,470 by-name reads and 2,112 operation-name string tests. The
pilot on the average pools decides it by A/B. If it wins: one core lane, then 20-30 family lanes in three or four
waves, after the A/B verdicts have landed on the switch sites.

## 6. Test code (`ideas-test-harness`)

The Mojo checks are about 454,000 lines in 1,214 files and grew 70,000 lines in two weeks; literal repetition is about
7 percent. The value of a shared kit is one fix point for buffer lifetimes, not size.

1. `checks/native_tests/kit.mojo` (about 250 lines: upload and download that drain before returning, compare helpers)
   and a rule that new checks use it.
2. After the release work: the 139 lifetime-prone check files onto the kit; registration in the registry table
   (about 5,400 lines and 96 wrapper files removed).
3. Needs Andrew's word: 235 standalone checks (60,000 lines) that no task, workflow or tool runs: register or retire.

## 7. Other owed cleanup (`ideas-refactor`, `fx-env-dead`, `fx-dead-small`, `fx-tools-prune`)

- Shared input admission (`admit_x`) in the up to 206 doors that still carry their own check.
- The 26 `xn_*` ops and their Python door, kept alive by six legacy oracle tests.
- 58 environment reads kept; four more that Python tests flip within one run; 303 undecided experiment defines.
- Mamba-3 owned-weights methods that only raise; `tools/amd_central.sh` and the Hot Aisle leftovers.
- Tooling: the persistent race worker; host units compiled once across build legs; 47 tool scripts kept as unsure.
- Latent defect, lane open: the K/V cache replay assumes an append-only cache (`review-kv-shadow`).

## 8. Release

Only when Andrew says, after everything above that is not explicitly deferred: one cross-vendor verification of every
lane on every device from the built wheels, then publish, then measure the published wheel.
