# HANDOFF 2026-09-18 evening: where we are, what remains

Written for a session with NO context, at the point Claude Code ran out of
weekly capacity. Everything below is merged to `main`; nothing is in flight.

## State at handoff (verified, not assumed)

- Every lane opened this session is MERGED; each branch is 0 commits ahead of main.
- No worktree has uncommitted work. No `mojo`, `train_main` or `mac_slot` process is running.
- RunPod API pod list: `[]`. All Hot Aisle and DigitalOcean boxes rented this
  session were terminated and 404-verified by the lanes.
- Andrew's standing orders, still in force:
  - NO NEW LANES without asking.
  - Every subagent commits and pushes EVERY step, as WIP, in its own worktree
    under `~/mojolearn-wt/`.
  - Local compute: one core, nice 19. All Metal work goes through `mac_slot.sh metal`.
  - Apple must be BITWISE IDENTICAL, with NO documented exceptions.

## What landed today (read the lane status doc before touching any of it)

| work | merged at | status doc |
|---|---|---|
| Tokenizer renamed `GPT2Tokenizer` -> `BpeTokenizer` (old name is a deprecated alias); tokenize-once built into training (`mojolearn.lm_corpus`, `tools/lm_train.py`; default trains OUR vocab, `--vocab` for the user's, `--bytes` for byte mode); vocab identity carried in checkpoints; verifier lanes `bpe-vocabulary`, `tokenized-corpus` | 5bde47f20, 0e4715f7c | `LANE_STATUS_tokenized-corpus.md` |
| Mojo BPE trainer is the default (`BpeVocabularyTrainer(backend="auto")`); dense VxV table replaced by `PairCounts`; full 50,256-rank build 34.8 GB -> 94 MB peak; Mojo == Python byte for byte; ranks sha `3d547b17...` reproduced | 0cccafd97 | `LANE_STATUS_bpe-builder-native.md` |
| Exact masked-tail attention replay + bounded eager storage ON for AMD and Apple (NVIDIA already on). AMD 1.27x geomean late step, all 700-step witnesses == NVIDIA. Apple correct but INERT in default training (see R3) | 1c05865f5 | `LANE_STATUS_attention-replay-vendors.md` |
| Apple GEMM seam repaired to round-then-flush (`lib_zero_fma_repair_for`, Apple only). Seam probe `62a6b5621e27c707` on all three vendors; no existing Apple cell moved; NVIDIA/AMD device code byte-identical. Cost about +16 to 23% of the Apple LM step | 6d2783f8f | `LANE_STATUS_apple-seam-repair.md` |
| Attention corner predicate: verified already fixed on main (signed-zero laundering; dQ 3449 real corners, dk/dv 248 conservative) | lane/attention-corner-predicate | `LANE_STATUS_attention-corner-predicate.md` |

The vocabulary (50,257 ids) and the tokenized enwik8 are in R2, pinned in
`bench/results/dataset_store/manifest.tsv`. NO vocabulary ships in the repo or
wheel, and none should (Andrew, Sep 18).

## What remains: Apple is NOT yet bitwise identical everywhere

Only GEMM (plus kNN from 2026-09-09) is repaired. The figures are from the
apple-seam lane's STATIC audit and are approximate. See
`LANE_STATUS_apple-seam-repair.md` for the file list.

- **R1. About 390 device sites that round then flush, outside GEMM and kNN.**
  - By family: Mamba 104, training 48 (optimizer 38), ARIMA 29, transformer 26,
    Holt-Winters 20, GP 19, decomposition 19, resample 16, and smaller ones.
  - Apple differs at the 315-triple boundary.
  - Same repair as GEMM: `lib_zero_fma_repair_for`-style, with block admission
    where the kernel is hot.
- **R2. About 20 device sites with NO flush after the FMA**:
  - `core/gram_splitk`, `column_stats`, GLM dot/softmax, GBDT cursor and
    add-model-value, RF/ExtraTrees split gain, spectral Nystrom predict, UMAP
    optimizer.
  - Apple flushes the subnormals that NVIDIA and AMD keep.
  - **DECISION OWED BY ANDREW:**
    - (a) flush on all three vendors, which moves NVIDIA/AMD bits at those
      sites and needs a reference regen by the table.json owner. This is the
      recommended option.
    - (b) make Apple keep subnormals, by software emulation, which is slow.
- **R3. Apple attention replay is inert.** Apple's default schedule never runs the
  estash kernels that contain the replay. Metal on NVIDIA's schedule already
  matches NVIDIA's hashes. Switching Apple's default needs its own qualification,
  and the speed gain on Apple is unmeasured. This is a candidate and needs
  Andrew's go-ahead.
- **R4. About 230 plain `ftz(x*y)` / `ftz(x/y)` sites.** These are exposed only if
  Metal's plain multiply and divide also flush before rounding. That is UNTESTED,
  and a seconds-long probe settles it. Do this before R1 is sized.
- **R5. NaN payloads differ per vendor on an overflow fixture** (Apple, NVIDIA and
  AMD are all different). This is a bitwise difference. Candidate.
- **R6. GEMM repair cost.** The vocab-head backward GEMMs take the slow exact path
  on real data, about 3x slower. A finer-grained admission check is a candidate.
  Which values trip it is not instrumented.

## Other open items (candidates, NOT opened)

- BPE trainer: incremental pair-count updates. The full recount is now nearly
  all the runtime, and the output would be identical. `train_main` should read
  bytes, not refuse invalid UTF-8.
- AMD zdot kernel has no replay: a zdot corner would still refuse. Zero were
  seen in training.
- `release_qualified` stays `False`. It is a LABEL, not a gate. It belongs to the
  once-per-release four-column record including Apple (the Apple column is
  about 10.77 h on the one Mac). Andrew may prefer to delete the field.
- Forest-deadlock lead: `LANE_STATUS_lane-forest-deadlock.md:469` counts about
  127 more sites of the DEVIATION 2520 shape (buffer releases with no
  `synchronize()` before `DeviceContext` destruction). It is a lead, not a
  census, and it can hang any program loading a second model.
- Unmerged branches from OTHER sessions, left untouched:
  - `lane/rf-score-weighted-nondeterminism` (17 commits)
  - `lane/neighbors-rest`
  - `lane/gbdt-rest`
  - `lane/identity-record-next`
  - `lane/rf-mutex-mechanism`
  - `lane/forest-train-speed`
  - `lane/cpu-training-gate-speed`
  - `lane/136-lane-record`
- The broader picture from this morning is in `docs/lanes/STATUS_2026-09-18.md`.

## Suggested order when capacity returns (every item needs Andrew's OK first)

1. R4 probe (seconds, Apple slot). It decides whether R1 grows by about 230 sites.
2. Andrew's decision on R2, then R1 + R2 as repair lanes, family by family:
   training/optimizer and transformer first, because they gate the tri-vendor
   training plan.
3. R3 if the tri-vendor training plan is going ahead.
4. R5 and R6.

## Traps that cost time today

- A killed session's subagents leave work UNTRACKED. Today one lane's only file was
  untracked in its worktree, and a vocab binary was in a reaped `/private/tmp`
  scratchpad. Commit-every-step exists for this.
- The old Mojo BPE trainer peaked at 34.8 GB on a 16 GB Mac. It survived only by
  memory compression. Fixed, but never run an unbounded job locally.
- RunPod AMD and Hot Aisle MI300X were both out of capacity at times. The AMD
  evidence is split between Hot Aisle MI300X (the 700-step pairs) and
  DigitalOcean MI325X (same gfx942).
- A finish script printing an EMPTY summary line reads like a clean pass (the
  kNN runner). Read the evidence, not the console.
