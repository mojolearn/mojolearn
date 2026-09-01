# gaussian_process differs Apple against AMD in ONE block, and that block is a SABOTAGE ARM

Found 2026-09-01 by diffing cards that no leg had ever compared. **Read the
CORRECTION below before quoting anything from this file**: the first reading
of this diff, on the same day, called the lane RED across vendors, and that
reading was wrong.

    Apple  bench/results/e1/2026-08-28_162228-MacBook-Air-1-terrabyte/lanes/gp.identical.card
    AMD    bench/results/e1/2026-08-28_203552-mojolearn-e2-amd/lanes/gp.identical.card

Both 3,494 lines. md5 `6e638e82a73e...` against `6bdeb28d6c81...`.

## What diverges

EIGHT lines, all inside ONE fit-and-predict block, which begins at card line
2119. `diff` over the two files returns these and nothing else:

    line  stage                    Apple             AMD
    2123  gp.kernel                58092f5ccc47e1a1  ddce183b7f4ba51d
    2124  gp.ridged                58092f5ccc47e1a1  ddce183b7f4ba51d
    2127  chol.jittered            58092f5ccc47e1a1  ddce183b7f4ba51d
    2128  chol.panel000.factored   db27960ef291b3ad  48c869baaa279711
    2129  chol.factor              274f147608c4c675  8c04837da5217bb1
    2133  gp.factor                274f147608c4c675  8c04837da5217bb1
    2145  gp.kcross                58092f5ccc47e1a1  ddce183b7f4ba51d
    2147  gp.v                     dcd3ffbe87a95146  67b9c35ec334f9f6

**IT ORIGINATES AT `gp.kernel`**, the RBF Gram and the first COMPUTED stage
of the block. Everything after it inherits. That is the identity card doing
exactly what it was built for: localizing a cross-vendor difference to one
stage rather than to a lane.

The other 3,486 lines are byte-identical, including every `chol.*` stage of
the other twenty-nine fits.

## CORRECTION 2026-09-01: the block is `GP_SAB_STD_EXP` on the planted fixture

The block is not one of a kind. Grouping every fit block by byte-identical
header AND identical `gp.x_train` input hash:

    Apple:  29 blocks -> gp.kernel 6687945852cf2969,  1 block -> 58092f5ccc47e1a1
    AMD:    29 blocks -> gp.kernel 6687945852cf2969,  1 block -> ddce183b7f4ba51d

Twenty-nine siblings with the same configuration and the same input agree
with each other AND across vendors. The odd block sits at the SAME card
position on both boxes. So this is not non-determinism: that block
deterministically takes a path the other twenty-nine do not, and that path
answers differently on Metal than on HIP.

**The path is `std.math.exp`, and it is reached because the check asked for
it.** The block is the sabotaged half of one clean-then-sabotaged pair driven
by `check_gp_sabotages`, and the arm is `GP_SAB_STD_EXP`, whose whole
statement is that a device `exp` is a VENDOR CHOICE in its last bit
(`gaussian_process/checks/gp_sabotage.mojo`, IDENTITY_PATHS row 12). The
identification is a source-and-record read, no re-run:

1. `gp_check.mojo::check_gp_sabotages` fits each fixture TWICE per arm, clean
   (`GP_SAB_NONE`) then sabotaged, and stops at the first fixture that moves
   a bit. Both fits write a block into the same card. That is the pair
   structure the card shows from line 1959 onward, 32 lines per block.
2. The arm order is the literal list in that function:
   DIST_DESCENDING, STD_EXP, EXPANDED_RBF, NO_FTZ_KERNEL,
   ALGEBRA_REASSOCIATE, VDOTV_PAIRWISE, MEAN_DESCENDING, LOGDET_RECOMPUTED,
   YALPHA_DESCENDING. The fixture order is planted, duplicate, handworked,
   ard, signed_zero.
3. Each leg's own log fixes how many fixtures each arm consumed before it
   moved, and the two logs are arm-for-arm identical, which is why the odd
   block lands at the same card position on both boxes. Arm 1 spent two
   fixtures (planted inert, duplicate moved), so arm 2 begins at the third
   pair: clean at 2087, sabotaged at 2119.
4. `gp.identical.log` on both boxes says, independently,
   `MUST FAIL  STD_EXP on planted: kernel[1] moved (0 earlier fixtures
   INERT to it)`. That is the same claim as the card: the sabotaged planted
   fit's kernel moved away from its clean twin.

So the two vendors disagree about `std.math.exp` on a planted RBF Gram, in a
fit that was constructed to call `std.math.exp` instead of `identical_exp`,
by a check whose printed verdict is MUST FAIL. **The shipped path is not
involved, and every stage the shipped path produced agrees byte for byte
across Metal and HIP.**

## Why nobody knew, and this part still stands

**Both legs printed `ALL PASSED [IDENTICAL]`.** Neither leg compares itself
against the other. A per-vendor run can only check a card against its own
oracle, and cross-vendor identity is by construction a claim about TWO cards,
so a green leg is not evidence for it. That is `[[reached-but-inert]]`
applied to an entire leg rather than to an arm, and it is why the diff had
never been taken. It remains true that nothing in either leg would have
noticed a real divergence.

The correction does not retire that lesson. It adds a second one: **a card
that mixes sabotaged runs with shipped runs must SAY WHICH IS WHICH, or the
first person to diff two of them reads a deliberate divergence as a defect.**
That is what happened here, in this file, on the day the diff was taken.

## Three consequences, restated

1. `gaussian_process`'s card carries the only cross-vendor evidence that
   exists for `cholesky`, because the AMD cholesky leg emitted no card of its
   own. That evidence is now POSITIVE: `chol.jittered`, `chol.panelNNN.*`,
   `chol.factor`, `chol.nb`, `chol.diag`, `chol.logdet` and `chol.solve.*`
   agree byte for byte between the M4 and the MI325X in every one of the
   twenty-nine unsabotaged fits, at n=2, 4, 8, 12 and 16. The three `chol.*`
   lines that differ are downstream of a Gram matrix that was deliberately
   computed with the wrong `exp`, so they say the factorization faithfully
   propagated a different input and say nothing against the lane. A card
   emitted by the cholesky lane itself is still owed.
2. The reason `gaussian_process` was withheld from the Python surface has
   DISSOLVED. It was pulled because its card was believed to diverge; the
   card agrees everywhere the shipped path reaches. Whether to expose it is
   the orchestrator's call and this file does not make it. What it can say is
   that the stated blocker no longer exists, and that the sentences quoting
   it in `SUPPORT_MATRIX.md`, `HANDOFF_2026-09-01.md`,
   `docs/PYPI_SURFACE_PLAN.md` and `cholesky/README.md` are now false in the
   unflattering direction and need the same correction this file just took.
3. THE CARD HEADER COULD NOT SAY WHAT MADE THAT BLOCK DIFFERENT from its
   twenty-nine siblings. **FIXED 2026-09-01.** It carried `n_train`, `d`, a
   kernel name that printed only the NUMBER of length scales, and
   `alpha_bits`, and all four are identical across all thirty. The field that
   separated them, `sabotage`, was not there at all. That was a
   CARD_GAPS-class defect in its own right and it is what cost this file its
   first reading.

## What the header now carries

`gaussian_process/estimator.mojo` writes, for a fit:

    gaussian_process fit: profile=mojolearn.identical.gp.fp32.v1 n_train=16
      d=1 kernel=RBF(ls1=0x3f800000) alpha_bits=0x00000000 sabotage=NONE

and for a prediction the same two additions, `kernel=` (which the predict
header omitted entirely) and `sabotage=`, beside the `n_star` and
`return_std` it already had.

Two fields changed:

- **`sabotage=<ARM NAME>`**, from `gp_sabotage_name`. It is the ONLY argument
  of `gpr_fit_host` that changes the arithmetic without changing an input, so
  it is precisely a decision the ALGORITHM makes and `CARD_GAPS.md`'s rule
  admits it. It is what names the block at line 2119.
- **the kernel's length scales, as bits**, inside `gp_kernel_name`. That
  function rendered every RBF node as `RBF(ls1)` whatever its length scale
  was, so `RBF([1.0])` and `RBF([2.0])` printed the same name. It does not
  separate these particular thirty blocks, which all fit
  `gp_fixture_kernel(GP_FIX_PLANTED)` and therefore all use `[1.0]`, and this
  file does not claim otherwise. It closes the same class of defect one step
  before it bites: the length scale is the only free parameter an RBF has and
  every cell of the Gram matrix is divided by it, so a header that omits it
  cannot say what produced the matrix underneath it.

Two candidates were considered and deliberately left out:

- **`elem_tpb` and `solve_tpb`.** Threads per block is SCHEDULING.
  `check_launch_invariance` exists to prove that no bit reads it, and
  `CARD_GAPS.md`'s rule is explicit that recording launch geometry in an
  identity trace would break the exact property that check establishes. Seven
  of the thirty blocks are that check's arms and they are byte-identical to
  each other by design; naming the arm would make them differ on paper for a
  reason the lane spends a whole check denying.
- **`optimizer`, `n_restarts_optimizer` and `normalize_y`.** They cannot
  vary. `gp_validate_optimizer` accepts only `"none"`, `0` and `False` and
  refuses everything else BY NAME (DEVIATIONS 1761 and 1764), so a header
  field for them would be a constant.

The build MODE is still absent, and that is `CARD_GAPS.md`'s cross-lane item
rather than this lane's: the fix there is a recorded integer at seq 0, not a
header comment, because both readers drop comments. It would not have
separated these thirty in any case, since one card is one build.

The header is a `#` comment line (`core/identity_trace.mojo:231-239`) that
both readers skip, so these additions move no record, change no stage list
and do NOT make a v2 of `mojolearn.identical.gp.fp32.v1`.

## What is owed

**RUN OWED, and nothing below has been compiled or run.** The identification
above is a read of committed source, cards and logs.

1. Rebuild and re-run the lane's checks under IDENTICAL with the card
   enabled, on ONE box, and read the thirty planted headers back. The fix is
   confirmed when they are no longer all identical: twenty-nine carry
   `sabotage=NONE` and the block whose `gp.kernel` is not
   `6687945852cf2969` carries `sabotage=STD_EXP`. Confirmed further if the
   pairs read clean-then-sabotaged down the whole sweep, arm by arm, in the
   order the check lists them.
2. Re-run the AMD leg the same way and diff the two cards again. The expected
   result is the same eight lines, now inside a block whose header says
   `sabotage=STD_EXP`, and the diff becomes self-explaining rather than
   alarming.
3. A cholesky card from the cholesky lane itself on a second vendor. The
   positive evidence in consequence 1 is real but it is second-hand, it only
   covers n <= 16, and it only covers the shapes a GP fit happens to ask for.
4. Diagnosing the `std.math.exp` difference itself is NOT owed and is not a
   defect. It is the measurement the arm exists to take, and the number to
   record is that it was WITNESSED across two vendors on 2026-08-28 rather
   than merely asserted from IDENTITY_PATHS row 12.
