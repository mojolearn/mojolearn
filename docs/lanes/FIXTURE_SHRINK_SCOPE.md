# FIXTURE SHRINK SCOPE (lane/identity-fixtures-light, 2026-09-16)

**THE SHRINK HAS LANDED.** Thirteen lanes now hash a smaller input. Any cell
recorded for one of them at the old size is superseded: `LANE_REVISIONS` in
`tools/identity_break.py` carries an entry for every one, so an old column
reads **OWED to the next record**, not DIVERGENT against different bytes.

If you are recording, re-record these thirteen. Everything else is untouched.

## The gate every one of them passed

A smaller fixture that can no longer fail is worse than the slow one it
replaced, so the rule was mechanical and per lane: shrink, run the family's
sabotage host build at the NEW size, and require the cell to read DIVERGENT.
An arm that went inert meant the size was stepped back up until it fired
again. **No lane ships at a size whose sabotage arm did not fire.** The arms
were also confirmed live at the ORIGINAL sizes first, so a firing arm is not
an artifact of the change.

Runs: CPU host route, one core, `nice -n 19`, `MOJOLEARN_NUMERIC_MODE`
set explicitly so a run cannot silently refuse before the first fit and
produce nothing. Production set and sabotage twin built from this tree for all
thirteen families.

## A. CHANGED (13 lanes)

`rows` means the lane fitted the full 20,000 x 16 fixture and now slices it.
`steps` means training steps. No arithmetic knob moved: tree counts, depths,
losses, optimizer settings, the seasonal period and the model shapes are all
untouched. What came down is INPUT SIZE and STEP COUNT.

| lane | before | after | lever |
|---|---|---|---|
| `gbdt-parametric-losses` | 20000 rows | **1500** | rows (ten fits of 8 trees) |
| `gbdt-nan-modes` | 20000 rows | **1500** | rows |
| `gbdt-lossguide-newtoncosine` | 20000 rows | **1500** | rows |
| `gbdt-pair-logit` | 20000 rows | **1500** | rows; 178 query groups remain, and the second fit walks the first 40 |
| `hdbscan` | 6000 rows | **4000** | rows; floor set by the Boruvka round count |
| `hdbscan-leaf` | 6000 rows | **2000** | rows; same measurement, leaf arm holds lower |
| `spectral` | 2000 rows | **512** | rows (O(n^2) eigen work) |
| `holtwinters` | 512 obs | **128** | observations; still many periods of the seasonal 12 |
| `byte-lm` | 3 steps | **1** | AdamW steps |
| `byte-lm-resident` | 3 steps | **1** | AdamW steps, with its stateless replay cut to match; its SHAPE came down later the same day, section A2 |
| `samba` | 6 windows, 3 steps | **2 windows, 1 step** | steps |
| `samba-untied-dropout-accum` | 96 rows, 3 steps | **32 rows, 1 step** | steps; 32 rows KEPT because `accumulation_is_aligned(256, 4)` is False, so fewer rows would delete the A=4 claim |
| `mamba2-dtlimit` | `(2, 16, 32)` slab | **`(2, 8, 32)`** | sequence length |

Two sizes were set by measurement rather than taste, and are floors:

- **hdbscan / hdbscan-leaf.** The Boruvka round count is one of the integers
  these lanes HASH. Measured across 6000/4000/3000/2000/1500/1000/750, it
  holds at 5 down to 4000 (excess-of-mass) and down to 2000 (leaf), and falls
  to 4 below. Each lane sits at the smallest size still reaching the fifth
  merge round.
- **samba-untied-dropout-accum.** `accumulation_is_aligned` admits A=4 only at
  32 rows (512 tokens); at 16 rows it is refused. The rows are the claim, so
  only the step count came down.

## A2. MODEL SHAPE, NOT INPUT SIZE (lane/neural-shape-shrink, 2026-09-16)

The shrink above moved INPUT SIZE and STEP COUNT, and on Metal it bought the
neural lanes nothing (`byte-lm` 231.8 s to 242.0 s, `samba` worse). The reason
is in the launch count. One byte LM training step issues
`(64 + 23*G)*L + (32 + 5*G)` kernel launches for `L` blocks, and **no launch on
the shipped path sits inside a loop over sequence length or d_model**. Steps
and sequence length remove arithmetic and not one launch. DEPTH removes
launches.

| lane | before | after | lever |
|---|---|---|---|
| `byte-lm-resident` | 2 blocks, d_model 32, ff 64 | **1 block, d_model 16, ff 32** | depth and width; Metal 173.95 s to **77.26 s**, 2.25x; 211 launches per step to 124 |

`LANE_REVISIONS` carries it at `shape-l1-d16-ff32-1`. The arm was confirmed
DIVERGENT at the current size first and again at the new one.

**Everything else in these families is a floor**, with the mechanism in
`docs/lanes/LANE_STATUS_lane-neural-shape-shrink.md`:

- `byte-lm` is the published profile's ONLY device column. `training/byte_lm.mojo`
  pins it as a comptime and the gradient oracle is defined against two blocks.
- `mamba1/2/3`, `mamba2-dtlimit`, `transformer`, `transformer-window` are each a
  SINGLE block, so there is no depth axis. The Mamba d_model 32 floor was seen
  to refuse at 16; `transformer` admits narrower shapes and was measured not to
  pay (1.121 s to 1.090 s, and `n_heads=1` is worse).
- `samba` and `samba-untied-dropout-accum` are a Mamba-3 layer plus an attention
  layer, and the heterogeneous stack is the claim. d_model 32 was seen to refuse
  at 16, and the head count and MLP width are flat.

## B. UNCHANGED

Every other lane of the harness. Record against them freely.

**The old bucket B is cancelled.** It protected the GBDT family on the
grounds that their cost was a Metal command-queue leak rather than fixture
size. That attribution was wrong twice over: the queues were never ours (they
belonged to `DockHelper`, an Apple XPC service), queue pressure does not
degrade a run (4663 queues, zero failures, normal progress), and the GBDT
lanes were in fact fitting the full 20,000-row fixture. All four are in
bucket A above.

## C. UNDECIDED

Empty.

## D. NOT CHANGED: the global batch knob

`BATCH_ALONE` stays at 16. It changes no recorded hash, so it looked like free
money, but measured on Metal it is worth about 12% (byte-lm full 231.8 s
against `--no-batch` 204.8 s) and the shipped negative control cannot judge
the change: `MOJOLEARN_IDENTITY_BATCH_SABOTAGE` perturbs the first element of
the whole-batch answer, which row 0 alone already catches, so it fails
identically at `alone=16` and `alone=4`. Lowering it on that evidence would be
a verification that cannot fail. A discriminating probe (perturb only row
k > alone) is owed before anyone touches it.

## E. `par-*` is out of the release record's scope

Not a fixture change: no `par-*` lane body is edited and no `par-*` cell
moves. All 39 are excluded from what a FULL-COLUMN record runs, declared in
`tools/identity_break.py` as `RECORD_EXCLUDED_PREFIXES` and enforced in
`run()`, which prints `# OUT OF RECORD SCOPE (39 lanes)`. `--lanes` is never
filtered, so the two-device `par` legs and the CPU identity gate are
unaffected. A record now runs 160 lanes.

Known consequence: all 11 CPU-covered `par-*` lanes are in
`host_surface.record_covered_lanes()`. `TRAINING_GPU_COLUMNS` still points at
the 166-lane record, which carries them, so nothing moves today. The day those
columns are repointed at a record taken under the new scope, those 11 lose
their GPU columns unless dropped from the covered set or admitted as OWED.

## F. The Apple column runs once per PyPI release

Also enforced in code, not prose: `refuse_routine_apple_column()` refuses a
full-column Apple run of more than `APPLE_COLUMN_LANE_LIMIT` lanes unless
`MOJOLEARN_APPLE_RELEASE_RECORD` names the release. Routine verification goes
on a rented CPU pod, bitwise equal to Metal, in parallel, about $0.24/hour.
See ENGINEERING_RULES.md section 12, docs/RELEASE_CHECKLIST.md section 5b and
docs/VERIFY.md.

## G. The thirteen references are stale by design, and that is now guarded

`verify_reference/table.json` still carries references taken at the OLD sizes
for all thirteen lanes. It is **not** regenerated here on purpose: a
regeneration today also carries 72 changed reference values across ten lanes
this branch never touched (newer committed records winning over older ones),
which is not this lane's change to make, and the release regenerates the table
as a normal step.

What makes that safe is a check rather than a memo. `build_table` now records
`lane_revisions`, and `_verify_reference.stale_reference_lanes()` returns any
lane whose fixture has moved past the reference the table carries. A table
generated before that key existed records no revision, which counts as stale.
`_verify_all` drops those lanes from the comparison, names them, and refuses
outright if nothing comparable is left. It was watched failing first: against
today's table it names exactly these thirteen, it goes quiet against a table
carrying current revisions, and it catches a single corrupted revision.

So nothing compares against the stale references in the meantime, and the next
person who shrinks a fixture and forgets to regenerate gets a refusal instead
of a DIVERGENT that looks like our identity claim being false.
