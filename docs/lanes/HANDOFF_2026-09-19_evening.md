# HANDOFF 2026-09-19 evening

Written for a session with NO context. main = `3b80c4fe5`, branch
`lane/lm-attention-fallback` at the same commit, version still **0.8.8**,
**zero pods live**, working tree clean, `2408 passed / 153 skipped`.

`docs/lanes` was deliberately emptied today (165 files) along with
`archive/` (34). Every `LANE_STATUS_`, `LANE_BODY_`, `PLAN_`, `HANDOFF_`
and `BRIEF_` file is gone by Andrew's instruction. **43 source files still
cite those paths in "WHY THIS FILE EXISTS" comments and are now dangling.**
Those references were removed after a reachability audit confirmed the source
files themselves are active APIs, kernels, checks or guarded cloud runners.
This file is the one deliberate exception to the purge.

---

## 1. THE MAMBA2 HOST REPAIR LANDED

The host Mamba2 padding repair landed on main as `ad5c37272`. No agent owns
the files now. It removes padded oracle work while preserving the recorded
base train, infer, batch and batchscale hashes.

Its brief, so you can judge the result:

- PROFILE first, do not guess.
- **The speedup must not move a single bit.** No FP reassociation, no
  accumulation-order change, no precision change, no reduction reordering.
  Legal: hoisting invariant work, removing recomputation, killing an
  accidental O(n^2) walk -- the SAME arithmetic in the SAME order.
- If the only way to make it fast changes bits, STOP and report that. That
  is a finding, not a failure.
- Prove it with `identity_break` before/after on
  `mamba1,mamba2,mamba3,mamba2-dtlimit,mamba2-bf16w,mamba2-int8w`,
  `--repeats 2`, ZERO parts moved, plus before/after timing.

A completed agent earlier today did the GPU columns; its work is merged.

---

## 2. WHY THAT AGENT EXISTS: the 58% nobody had ever summed

Per-cell `timing.stages` has been in every record for weeks and nobody
totalled it. Summed over
`bench/results/identity_break/2026-09-17_reference-regen/*.json`:

```
whole column                20437 s
  batchscale (ALL lanes)    11906 s   58.3%     <- ONE optional part
  mamba2* batchscale         9108 s   44.6%     <- four lanes of it
```

Per cell: `mamba2` **362.5 s**, `mamba3` **2.1 s**, `mamba1` **0.2 s** --
and mamba1/2/3 run the SAME shape (`dm,di,nh = 32,64,1`,
`_seq(X,2,16,dm)`). mamba2 is **176x mamba3 at identical shape**.

`mamba2`'s own `train` stage is 13.7 s. The fit is fine. `batchscale`
(`_batchscale_block`, tools/identity_break.py ~8860) forwards **L=1024
nineteen times** -- a 1024-row call at sub-batches {1,17,64,256} plus B=2 at
L=1024 against 15 prefixes -- while the lane's fixture is L=16.

So this is a REAL DEFECT users hit when calling `Mamba2Block.forward` on a
long sequence, not verifier overhead. **Do not "fix" it by deleting the lane
or the check** -- that deletes the only evidence of it. Its batchscale is
live (18 distinct values across columns), not inert.

Generated launch structure is NOT the cause: mamba2 declares "9 kernels, 12
launches", mamba3 "8 kernels, 9 launches".

---

## 3. GAPS: every matrix count is 0

```
No GPU column at all ................. 0
GPU column on fewer than 3 classes ... 0
No CPU verifier declared ............. 0
Sabotage not seen to move a build .... 0
Batch undeclared ..................... 0
```

`verification_matrix.py --check` passes: "no gap list names unreachable
work". What is actually left:

1. **3 `par-*` lanes with no two-device column**: `par-forecast-arima`,
   `par-forecast-holtwinters`, `par-ivf`. (52 of 55 now carry one.)
   `par-ivf` and the gbdt drivers REFUSE on a CPU column by design -- no CPU
   implementation of the cooperative multi-GPU driver -- so they need a real
   2-GPU box. `par-forecast-arima` and `-holtwinters` DO run on CPU.
2. **6 kernel/nystroem NVIDIA `classical_host` saved-model recordings**
   (`SAVED_MODEL_INFERENCE_OWED`). A different artifact from an identity
   cell; today's promotion did not pay it. Only Apple recordings exist
   (`bench/results/classical_host/2026-09-18-apple-kernel-variants`).
   RECORDING REQUIRES A GPU BOX -- the CPU binding cannot manufacture its
   own expected answer. All six are confirmed selectable routes, so this is
   one command on a rented NVIDIA box, not a research task:

   ```sh
   python3 tools/classical_host_gate.py record \
     bench/results/classical_host/2026-09-19-nvidia-kernel-variants \
     --lanes kernel-ridge-poly,kernel-ridge-sigmoid,kernel-ridge-laplacian,\
nystroem-poly,nystroem-sigmoid,nystroem-laplacian
   ```

   then `classical_host_gate.py check` against it on the CPU box. Note
   `record` had been raising AttributeError on main for a while before
   2026-09-16 -- ahead of every other refusal -- so if it dies instantly,
   suspect the tool, not the lane.
3. **The three-column release record.** See section 5.
4. **0.8.9** -- not cut. 98 commits and 4253 insertions of shipping kernel
   source since v0.8.8, including new arithmetic (ordered boosting 789
   lines, gbdt_oracle_ordered 582, binarization 405, sample_quantile 220,
   the public qr/eigh/svdvals doors 534). That is a feature release, not a
   patch.

---

## 4. TRAPS THAT COST TIME TODAY -- read before you trust a green result

- **`--emit-reference` needs no binding and no GPU.** It regenerates from
  committed records. Scoped + additive via `--lanes` and `--reference-table`
  is how 18 lanes were promoted for $0. It REFUSED correctly when run
  without `--batch-checks` rather than drop gbdt-query-rmse's batch cells.
- **Cell hash values are LISTS** (one per repeat), not strings. A comparison
  written as `isinstance(v, str)` silently skips every part and reports
  "nothing moved". This produced a false "five sabotage arms proved nothing"
  before it was caught.
- **`par_diff` exit 0 over a truncated column means nothing.** A green diff
  over one cell is a check that cannot fail.
- **`complete: false` is refused by `admit()`** -- "incomplete
  identity_break checkpoint" -- with AND without `par_axis`. A truncated
  column's hashes can be read and agree perfectly and still credit nothing.
  A pair was committed as "landed" on exactly that mistake and corrected at
  `3b80c4fe5`.
- **Two legs with the same `MOJOLEARN_GAP_SLUG` collide silently and the
  later commit wins.** One agent overwrote another's three committed files
  four minutes after they landed, then restored them byte-for-byte. No guard
  exists. Pick a unique slug.
- **The per-phase `timeout cap` scales with REMAINING lease.** A leg that
  spends 50 minutes building starts its real phase with 60 seconds and exits
  124. Warm the cache or rent longer.
- **`denormal_ftz` is NOT a redundant twin of `denormal`.** Its signal is
  ACROSS columns: vendor A's `denormal` matching vendor B's `denormal_ftz`
  means B flushes subnormals and A does not. A within-column redundancy
  measurement of it is measuring the wrong axis, and nearly justified
  deleting the one detector for the divergence this project exists to catch.

---

## 5. THE EVIDENCE STORY, and its honest weakness

A column = one box running lanes x 9 fixtures -> `cells["lane/fixture"]
["part"] = hash`. Three columns at ONE commit, diffed cell for cell, is the
claim. Sabotage arms keep it from being vacuous.

`bench/results/identity_break/2026-09-14_166-lanes/` is the model artifact:
Apple M4 + H100 + MI325X, all at `1eea14f80`, with a published
`diff.three-columns.txt`. **6685 of 6686 part comparisons identical**, one
known exception (`kmeans-sqrt/wide`).

**The weakness to state plainly in any paper:** that record covers 166 lanes
and we now have 256, and the SHIPPED table
(`python/mojolearn/verify_reference/table.json`, 2187 cells, 314 records) is
an ACCRETION -- **7587 of its parts (47%) rest on a single device class**,
and only 40 of 243 lanes have every part on three or more. Every lane has
been SEEN on three vendors; they have not all been seen in ONE run. Closing
that is one fresh 3-column record at today's commit: NVIDIA + AMD are cheap
rentals, Apple is the constraint.

**Save `lane_seconds` on that Apple run.** The 8.3-hour figure in the old
notes was extrapolated from a single lane and has never been measured.

---

## 6. THE BINCACHE FIX -- why runs were slow

`MOJOLEARN_GAP_LANES`/`_SLUG`/`_COMMIT_DIR`/`_BUDGET` were IN the cache key,
so a leg could only hit if it re-ran the identical lane list into the
identical directory. Fixed at `900038b24` (`MOJOLEARN_GAP_` added to
`NON_BUILD_PREFIXES`). Same box, same image, different lane list:

```
build phase              1738 s -> 160 s
elapsed_after_families   1725 s ->  59 s
outcomes             56 miss    -> 55 hit
```

`bincache/v1/` now holds its first GPU partitions (`sm_89/` 110 objects,
`gfx942/` 55). Budget the next par leg in minutes, not an hour.
