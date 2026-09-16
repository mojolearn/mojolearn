# LANE STATUS: three arms that could not fail (2026-09-16)

Branch `lane/dead-arms`, cut from `main` at `16b29b9cf`. No box rented. Every
job took a slot through `mac_slot.sh` (the Metal jobs took the Metal lock),
one at a time, one core, `nice -n 19`, `MOJOLEARN_NUMERIC_MODE=identical`.
The 0.8.6 macOS wheel in a private venv under this session's scratchpad
supplied the bindings and this worktree supplied the harness, so nothing was
built and no result here can be confounded by a build. The sha256 of every
mojolearn binding each process had MAPPED was printed from inside the process
next to its results, for example `_mojolearn_gbdt.so 78690951dd2af1ba`,
`_mojolearn_mamba.so 207408a0ba597c13`, `_mojolearn_transformer.so
61d1049d0e353ed1`.

`lane/shrink-blindness-audit` (merged as `16b29b9cf`) named three of these in
its section 5 and gave a mechanism for each. **Every mechanism was reproduced
here before anything was changed**, and one of the three turned out to reach a
lane the audit did not name.

## THE HEADLINE

| arm | the state on main this morning | after |
|---|---|---|
| `gbdt-nan-modes` | `nan_mode` Min and Max hashed the SAME bytes at 1500, 6000 and 20000 rows | DISTINCT on all nine fixtures |
| `mamba3`, `transformer`, **`transformer-window`** | two same-shape RMSNorm weights were the SAME TENSOR, so exchanging them was the identity function | the exchange is DETECTED on all nine fixtures, in all four parts |
| `mamba2-dtlimit` | `dt_limit=(0.1, 0.1)`, a clamp returning a CONSTANT, read IDENTICAL to production | a constant clamp is DETECTED on all nine fixtures, and `dt_bias` moves the cell in BOTH directions |

**`transformer-window` carries the same defect as `transformer` and was not in
the audit's list.** It is the same weights dict built by the same helper.

## 1. `gbdt-nan-modes`: the NaN reached no split

### The mechanism, reproduced

`_with_nan` wrote NaN into columns 5, 6 and 7 of every eighth row. `labels_for`
builds `y_clf` from columns 3 and 4 only (`s01 = X[:,3] + 0.5*X[:,4]`), so the
fitted ensemble splits on columns 3 and 4 only and no NaN ever reaches a split
decision. Printed from the fit itself, not inferred:

```
  columns holding NaN: [5, 6, 7]
  labels_for reads columns 3 and 4 (s01 = X[:,3] + 0.5*X[:,4])
  NaN columns intersect the label columns? False
  Min-arm splits on feature columns: [3, 4]      (parsed from the saved model text)
  NaN columns 5,6,7 among them? []

  shipped _with_nan (cols 5,6,7) n=1500:  min=a21e31ec9cc7597d max=a21e31ec9cc7597d  IDENTICAL (arm INERT)
  shipped _with_nan (cols 5,6,7) n=6000:  min=58abad67c1742b5c max=58abad67c1742b5c  IDENTICAL (arm INERT)
  shipped _with_nan (cols 5,6,7) n=20000: min=7f15e34e477a4eae max=7f15e34e477a4eae  IDENTICAL (arm INERT)
```

### The fix

`_with_nan` keeps the NaN it had on columns 5, 6 and 7 and ALSO places one on
column 3 (every eighth row) and on column 4 (four rows later), staggered so no
row loses both label columns at once. Keeping 5, 6 and 7 is deliberate: the
fixture now places a NaN both on columns the trees split on and on columns they
do not. `labels_for` runs on the CLEAN fixture, so **no label moves**, and
columns 3 and 4 are the two columns no fixture perturbs, so the same NaN lands
on all nine.

### It fires, with values

```
  fixed _with_nan n=1500:  min=da54f1f980861f61 max=d8e2a3e5305a859b  DISTINCT
  fixed _with_nan n=6000:  min=09a481c0d2b4a17f max=46ac0f77158d0c14  DISTINCT
  fixed _with_nan n=20000: min=c363c109d9c09cf5 max=1a761d036a18be15  DISTINCT

  row     0: Min -0.090179309   Max +0.006575228     (a NaN row)
  row     1: Min +2.154427767   Max +2.633731365
  row     2: Min +0.387351811   Max -0.071271203
  row     3: Min -2.452134848   Max -2.163101912
  max |Min - Max| over the fit: 1.986947417
```

and with the shipped fixture, on every fixture (the staggered version):

```
  base         min=5d7ce7cc59e261b0 max=e57525a0fb5169d2 DISTINCT
  ties         min=13cc3194def6ba8a max=ebe6c62a716f1e48 DISTINCT
  hashed       min=f73018929c188417 max=5c690fb375214219 DISTINCT
  wide         min=06b5d515d833b61b max=6d19cf352d1593e7 DISTINCT
  denormal     min=5d7ce7cc59e261b0 max=e57525a0fb5169d2 DISTINCT
  denormal_ftz min=5d7ce7cc59e261b0 max=e57525a0fb5169d2 DISTINCT
  dupes        min=ca1a067b4a87c5fe max=e57525a0fb5169d2 DISTINCT
  odd          min=9dfd010df838667f max=970e4742a0b39b76 DISTINCT
  negative     min=85b31578f0f57530 max=68ee75afe6295f7b DISTINCT
```

`ties` matters here. An arm that fires on `base` and is inert on the integer
fixture is the defect all over again (`docs/lanes/SABOTAGE_AUDIT_2026-09-16.md`
section 5), and this one fires on `ties` as well.

### Production is unmoved as a behaviour

`tools/identity_break.py --lanes gbdt-nan-modes --repeats 2`, all nine
fixtures, this Mac, one core:

```
cells=9 stable=9 moved=0 refused=0
infer: stable=9   model: stable=9   batch: stable=9
```

## 2. `mamba3`, `transformer` and `transformer-window`: the norms were one tensor

### The mechanism, reproduced, and why the byte-lm remedy does not apply

`_block_weights(..., ones=(...))` set several norm weights to a vector of ones.
Where two of them have the SAME SHAPE the block reads two bitwise-equal
tensors, so exchanging them on the way in is the identity function.

```
  mamba1               equal same-shape pairs: none
  mamba2               equal same-shape pairs: none
  mamba3               equal same-shape pairs: [('B_norm.weight', 'C_norm.weight')]
  transformer          equal same-shape pairs: [('input_layernorm.weight', 'post_attention_layernorm.weight')]
  transformer-window   equal same-shape pairs: [('input_layernorm.weight', 'post_attention_layernorm.weight')]
  mamba2-dtlimit       equal same-shape pairs: none

  mamba3               SWAP B_norm.weight <-> C_norm.weight: 63de4bf6b9f8262a -> 63de4bf6b9f8262a  BLIND
  transformer          SWAP the two layernorms:              295d4e62d4c78b14 -> 295d4e62d4c78b14  BLIND
  transformer-window   SWAP the two layernorms:              49ffb2316f238e6d -> 49ffb2316f238e6d  BLIND
```

**This is a permutation the lane cannot see, not a tensor it never reads.** A
`+0.25` on either member alone moves every one of the six cells:

```
  mamba3              B_norm.weight  + 0.25 -> d55b01c2faa3cc94 DETECTED
  mamba3              C_norm.weight  + 0.25 -> 0e2f175e15a67edc DETECTED
  transformer         input_layernorm.weight + 0.25 -> c48dcc721b94b33b DETECTED
  transformer         post_attention_layernorm.weight + 0.25 -> 8d2d77d4a52c8998 DETECTED
  transformer-window  input_layernorm.weight + 0.25 -> cdfc82433d6beae8 DETECTED
  transformer-window  post_attention_layernorm.weight + 0.25 -> 003e98dd9120a1fe DETECTED
```

**The byte-lm remedy does not apply and was not copied.** `byte-lm` is curable
by a second AdamW step because its two norms separate by 2.000e-03 after one.
These three lanes have no optimizer at all: `_block_fit` runs a forward, a
prefill, one decode step and a backward on the weights it is handed, and
nothing writes them back. There is no step in which the two could separate, at
any size. So the remedy has to be at initialisation.

### The fix

`_block_weights` gains `near_one=(...)`: `1 + ` hashed uniform on `[-1/8, 1/8)`,
seeded per lane and per tensor. The two members of each pair move from `ones`
to `near_one`; every other `ones` name is untouched, so `mamba1`, `mamba2` and
`mamba2-dtlimit` keep theirs.

A per-element vector rather than a second constant, because it also separates
a norm that scales per channel from one that scales by a single value. Within
an eighth of unity, so no activation is crushed and no denormal is
manufactured:

```
  mamba3       B_norm.weight  first 4 [1.0229710340499878, 1.1091084480285645, 1.0968594551086426, 1.10806143283844]
               C_norm.weight  first 4 [0.8783536553382874, 1.0379772186279297, 1.0437241792678833, 0.9010683298110962]
               bitwise equal? False   both within an eighth of 1? True
  transformer  input_layernorm.weight          first 4 [0.9321880340576172, 1.076145052909851, 1.0535906553268433, 1.017050862312317]
               post_attention_layernorm.weight first 4 [0.9353950619697571, 0.9485383629798889, 0.9784401059150696, 1.0296516418457031]
               bitwise equal? False   both within an eighth of 1? True
```

### It fires, on every fixture, in every part

All 27 cells (three lanes x nine fixtures) DETECT the swap, and the parts that
move are `['backward', 'forward', 'prefill', 'step']` in every one of them, so
this is not one path noticing. On `base`:

```
  mamba3              base production b282d7e1efb22e6b  swapped bb0656e9729d4dae  DETECTED
  transformer         base production f2cb3899e9bfe21c  swapped 7518e31091b70c89  DETECTED
  transformer-window  base production 998e7f404405725f  swapped 52bba120f4accd91  DETECTED
```

and the forward output element by element, `base`, mamba3, 1024 of 1024
elements differ:

```
  y[0, 0, 0]: production +1.444779932e-01   swapped +1.443656534e-01
  y[0, 0, 1]: production +1.915919781e-02   swapped +1.825287938e-02
  y[0, 0, 2]: production +7.389476895e-01   swapped +7.383573055e-01
  y[0, 0, 3]: production +4.146799743e-01   swapped +4.128238857e-01
  max |production - swapped| = 1.080845594e-01
```

transformer and transformer-window likewise, 1024 of 1024, max 4.805791378e-02.

### A wider consequence, reported and NOT acted on

A norm weight of ones also means the norm-weight MULTIPLY is unobservable: a
kernel that dropped `* w` entirely would produce the same bits as one that
multiplies by ones. That is true of `mamba1`, `mamba2` and `mamba2-dtlimit`'s
remaining `ones` norms as well, and it is not what this lane was asked to fix.
Those three keep their all-ones norms here, because moving them would take two
more PUBLIC lanes out of the shipped reference table for a defect nobody has
asked for. **It is a live hole and it is written down here rather than fixed.**

## 3. `mamba2-dtlimit`: the clamp was a constant

### What "one-sidedly" costs, measured

The dt this lane makes was read off the LIBRARY, not off a model of it: with
one bound taken out of the way, each bound was bisected until the cell moved,
which brackets the dt the cell actually hashes.

```
  base         measured dt range: [0.282595, 1.110428]
  ties         measured dt range: [0.644507, 1.069363]
  hashed       measured dt range: [0.470005, 1.270123]
  wide         measured dt range: [0.314445, 0.983253]
  denormal     measured dt range: [0.352379, 1.144791]
  denormal_ftz measured dt range: [0.352379, 1.144791]
  dupes        measured dt range: [0.407879, 1.038445]
  odd          measured dt range: [0.282595, 1.110428]
  negative     measured dt range: [0.457942, 0.588512]
```

The lane's clamp was `(0.01, 0.1)`. **Every dt on every fixture is above the
upper bound**, so S9 (`dt = clamp(softplus(dt_raw + dt_bias), lo, hi)`) returned
`hi` for every value, and the lane read one branch of a three-branch clamp.
Three consequences, each measured rather than argued:

```
  (1) dt_bias, L=8 and L=16, production cells 3c1d9aaeaa765468 and b9e7928a2a4d30cf
      +0.01 +0.10 +0.25 +1.00 +4.00 +16.00 and -1.00 : BLIND (unmoved), every one
      -4.00 -> 9e90117fe64a1952   -16.00 -> 7a8533f16c11e83e : DETECTED

  (2) the LOWER bound is inert. With hi fixed at 0.1, lo walked to 0, 0.001,
      0.005, 0.02, 0.05, 0.09 and 0.0999 moved NOTHING, at L=8 and at L=16,
      and on all nine fixtures.

  (3) dt_limit=(0.1, 0.1), a clamp that returns a CONSTANT for every input,
      read 3c1d9aaeaa765468 at L=8 and b9e7928a2a4d30cf at L=16: IDENTICAL to
      production. The lane exists for the clamp and could not tell it from a
      constant.
```

(3) is the whole cost in one line. The `MOJOLEARN_MAMBA2_SABOTAGE_CLAMP_BEFORE_SOFTPLUS`
arm in `mamba/impl/modules/mamba2.mojo` says in its own comment that it is
"witnessed only by an ACTIVE dt_limit fixture"; the fixture was active in name
only.

**This is not the shrink.** Every line above holds at L=16, the pre-shrink
length, as well as at L=8.

### The fix

`dt_limit` becomes `(0.5, 0.9)`, inside the dt the lane makes, so some values
clamp low, some clamp high and some pass through. Measured over all nine
fixtures with the library:

```
  base         lo live=True   hi live=True   constant-clamp DETECTED=True  dt_bias +/-0.01 and +/-0.25 all DETECTED
  ties         lo live=False  hi live=True   constant-clamp DETECTED=True  all DETECTED
  hashed       lo live=True   hi live=True   constant-clamp DETECTED=True  all DETECTED
  wide         lo live=True   hi live=True   constant-clamp DETECTED=True  all DETECTED
  denormal     lo live=True   hi live=True   constant-clamp DETECTED=True  all DETECTED
  denormal_ftz lo live=True   hi live=True   constant-clamp DETECTED=True  all DETECTED
  dupes        lo live=True   hi live=True   constant-clamp DETECTED=True  all DETECTED
  odd          lo live=True   hi live=True   constant-clamp DETECTED=True  all DETECTED
  negative     lo live=True   hi live=False  constant-clamp DETECTED=True  all DETECTED
```

against the shipped `(0.01, 0.1)`, which reads `lo live=False`,
`constant-clamp DETECTED=False` and every `dt_bias` probe BLIND on all nine.

**No pair can do better than eight and eight.** `ties` starts at 0.645 and
`negative` stops at 0.589, so a lower bound that bites on `ties` is above
everything `negative` has, and an upper bound that bites on `negative` is below
everything `ties` has. `(0.5, 0.9)` is the round pair inside the overlap.

## 4. Production is unmoved as a behaviour, and the controls

`tools/identity_break.py --lanes mamba1,mamba2,mamba3,transformer,transformer-window,mamba2-dtlimit,gbdt-nan-modes --repeats 2`,
nine fixtures, this Mac, one core, under the Metal lock. See
`~/mojolearn-evidence/dead-arms-2026-09-16/out_production_after.txt`.

```
cells=63 stable=63 moved=0 refused=0
infer: stable=63      model: stable=9 n/a=54      batch: stable=63      rlpair: stable=54
```

Nothing moved between repeats, nothing refused, on any of the seven lanes.

**The controls are `mamba1` and `mamba2`**, which share `_block_weights` with
the three changed lanes and were NOT changed. Their cells must be exactly what
the shipped reference table already holds, and they are. The four changed
lanes' new `train/base` cells match the standalone probes of sections 1 to 3
exactly, which is worth stating because the probes drive the lanes through a
patched `_block_weights` while this run drives them through the shipped one:

| lane | shipped table `train/base` | this run | |
|---|---|---|---|
| `mamba1` | `1609902abaf80a04` | `1609902abaf80a04` | control, UNMOVED |
| `mamba2` | `5b05a3ecbd70248e` | `5b05a3ecbd70248e` | control, UNMOVED |
| `mamba3` | `63de4bf6b9f8262a` | `b282d7e1efb22e6b` | moved once, as intended |
| `transformer` | `295d4e62d4c78b14` | `f2cb3899e9bfe21c` | moved once |
| `transformer-window` | `49ffb2316f238e6d` | `998e7f404405725f` | moved once |
| `mamba2-dtlimit` | `3c1d9aaeaa765468` | `dbe3d4e560695b2e` | moved once |
| `gbdt-nan-modes` | (no cell in the table) | `78578cb9693251c7` | moved once |

## 5. THE REFERENCE CONSEQUENCE, which is real

Measured against `origin/main` as merged here, not against the state this
branch was cut from. Main regenerated `python/mojolearn/verify_reference/table.json`
this morning (`lane/expose-stepfull`), and the regenerated table's own
`train/base` refs are **exactly the five pre-fix cells measured above**, which
is an independent confirmation of every "before" number in sections 1 to 3:

```
  mamba2-dtlimit      3c1d9aaeaa765468        mamba3              63de4bf6b9f8262a
  transformer         295d4e62d4c78b14        transformer-window  49ffb2316f238e6d
  mamba1              1609902abaf80a04        mamba2              5b05a3ecbd70248e   (the two controls)
```

**Five lanes' recorded cells move. One of the five costs nothing.**

| lane | revision was | is now | what it costs |
|---|---|---|---|
| `gbdt-nan-modes` | `rows-1500-1` | `rows-1500-nan-in-split-columns-2` | nothing: the regenerated table carries NO cell for it, so it is `no reference` before and after |
| `mamba2-dtlimit` | `seqlen-8-1` | `seqlen-8-dtlimit-straddle-2` | its reason moves from `unwatched` to `stale reference`; the table's cell now describes different bytes |
| `mamba3` | (none, PUBLIC) | `norms-near-one-1` | **held back from the public set until the next record** |
| `transformer` | (none, PUBLIC) | `norms-near-one-1` | **held back** |
| `transformer-window` | (none, PUBLIC) | `norms-near-one-1` | **held back** |

`release/0.8.7` is live and has not recorded yet (its build gate lifted today,
`06afb4075`), so this lands at the cheapest moment there is: the 0.8.7 record
captures the new revisions and regenerates the table in one pass. The three
that were public carried cells describing the all-ones bytes, so leaving them
public would have had a user read DIVERGENT for something that is not their
machine.

**The three new revisions are NOT sizes**, so they carry no `@floor` and are
declared in `NON_SIZE_REVISIONS` with what moved instead, which is what
`tools/fixture_floors.py` rule 1 requires. `gbdt-nan-modes` and
`mamba2-dtlimit` keep their size keys (`rows-1500-...`, `seqlen-8-...`) and
their existing floors: **neither fix moved a size.** The two `@floor` reasons
are updated, because both of them described the dead arm as dead.

### The guard was watched to FAIL before it was trusted

With the three `PUBLIC_PENDING_LANES` entries removed,
`test_host_surface.py::test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`
fails and names them:

```
E   AssertionError: these lanes' fixtures moved past the shipped reference and they are
    still public: ['mamba3', 'transformer', 'transformer-window']. Add them to
    PUBLIC_PENDING_LANES as 'stale reference' until the release regenerates the table
1 failed, 154 deselected
```

With them restored, the file passes. Re-run AFTER the merge, against main's
own copy of the test, both guards fire and each names its own lanes.

Restoring main's reasons exactly (`mamba2-dtlimit` back to `unwatched`, the
three norm lanes back to public):

```
E   AssertionError: these lanes are held back for a reason the shipped table no longer supports:
E       mamba2-dtlimit: held back as 'unwatched', but its fixture has moved past the shipped
E       reference ('seqlen-8-dtlimit-straddle-2'), so its reason is 'stale reference' and a run
E       would prove nothing
```

and with only the three norm lanes left public:

```
E   AssertionError: these lanes moved past the shipped reference and they are still public:
E       ['mamba3', 'transformer', 'transformer-window']. Add them to PUBLIC_PENDING_LANES as
E       'stale reference' until the release regenerates the table
```

Restored, `157 passed`. `tools/fixture_floors.py` reads
`ok: 15 fixture floor(s), none violated`, and its own `--self-test` refuses
every mutation it makes, including an untraceable reason on this lane's
`mamba2-dtlimit` floor.

## 6. THE APPLE RULE CHANGED MID-LANE, and what this lane did about it

`~/mojolearn-evidence/tools/LANE_RULES_2026-09-16.md` gained a rule on the
afternoon of 2026-09-16: **no lane takes an Apple or Metal identity cell or
column at all**, because the CPU host route produces the same bits at about
1/600th of the cost and Metal is one job at a time on one machine, so every
lane's Apple cell queues behind every other lane's. It names five lanes that
took Apple cells that day and together made Apple the serial bottleneck.

This lane's one Apple column (section 4) was started at 14:49, before that
rule existed, and was two thirds finished when it landed. It was allowed to
run to the end rather than cancelled, because cancelling would have spent the
43 minutes of Apple already used and handed nothing to anybody, and the
standing rule is that a new rule applies to the NEXT process. **No further
Apple or Metal work was taken on this branch.** Every other measurement here
is a data or parameter perturbation through the shipped bindings, and the
post-merge guard runs in section 5 touch no device at all.

## 7. What this does NOT claim

- One repeat per probe on the base fixture for the mechanism sections; the
  fire-or-not verdicts in sections 1, 2 and 3 are on all nine.
- No sabotage binary was built. Every result is a data or parameter
  perturbation through the shipped 0.8.6 bindings. None of them is a statement
  about a specific sabotage define, including
  `MOJOLEARN_MAMBA2_SABOTAGE_CLAMP_BEFORE_SOFTPLUS`, which is named above only
  for what its own comment says.
- The Apple column is the only device column touched, and only for this lane's
  own cells. No sweep, no re-record, no GPU column.
- The all-ones norm-weight multiply hole in `mamba1`, `mamba2` and
  `mamba2-dtlimit` (section 2) is reported, not fixed.

## Resume

    SP=<scratchpad>
    python3 -m venv $SP/venv086
    $SP/venv086/bin/pip install ~/mojolearn-evidence/release-0.8.6/macos-wheel/*.whl numpy
    WT=/Users/andrewhendel/mojolearn-wt/dead-arms
    MAC_SLOTS=4 bash ~/mojolearn-evidence/tools/mac_slot.sh metal nice -n 19 \
      env MOJOLEARN_NUMERIC_MODE=identical WT=$WT $SP/venv086/bin/python \
      ~/mojolearn-evidence/dead-arms-2026-09-16/<probe>.py

The probes and every run log are in
`~/mojolearn-evidence/dead-arms-2026-09-16/`: `arm1_nan.py` and
`arm1_variants.py` and `arm1_after.py` (the NaN arm), `arm2_norms.py` and
`arm2_after.py` (the norm swap), `arm3_dtlimit.py`, `arm3_pick.py`,
`arm3_grid.py`, `arm3_range.py` and `arm3_confirm.py` (the clamp).

## Pods

None rented on this branch.
