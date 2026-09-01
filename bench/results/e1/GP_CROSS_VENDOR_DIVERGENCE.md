# gaussian_process is RED across vendors, and two green legs said otherwise

Found 2026-09-01 by diffing cards that no leg had ever compared.

    Apple  bench/results/e1/2026-08-28_162228-MacBook-Air-1-terrabyte/lanes/gp.identical.card
    AMD    bench/results/e1/2026-08-28_203552-mojolearn-e2-amd/lanes/gp.identical.card

Both 3,494 lines. md5 `6e638e82a73e...` against `6bdeb28d6c81...`.

## What diverges

EIGHT stages, all inside ONE fit block, which begins at card line 2119:

    gaussian_process fit: profile=mojolearn.identical.gp.fp32.v1
    n_train=16 d=1 kernel=RBF(ls1) alpha_bits=0x00000000

    line  stage                    Apple             AMD
    2123  gp.kernel                58092f5ccc47e1a1  ddce183b7f4ba51d
    2124  gp.ridged                58092f5ccc47e1a1  ddce183b7f4ba51d
    2127  chol.jittered            58092f5ccc47e1a1  ddce183b7f4ba51d
    2128  chol.panel000.factored   db27960ef291b3ad  48c869baaa279711
    2129  chol.factor              274f147608c4c675  8c04837da5217bb1
    2133  gp.factor                274f147608c4c675  8c04837da5217bb1
    2145  gp.kcross                58092f5ccc47e1a1  ddce183b7f4ba51d
          gp.v                     (inherits)        (inherits)

**IT ORIGINATES AT `gp.kernel`**, the RBF Gram and the first COMPUTED stage
of the block. Everything after it inherits. This is the identity card doing
exactly what it was built for: localizing a cross-vendor difference to one
stage rather than to a lane.

## Why it is not flaky, and not the whole lane

The block is not one of a kind. Grouping every fit block by byte-identical
header AND identical `gp.x_train` input hash:

    Apple:  29 blocks -> gp.kernel 6687945852cf2969,  1 block -> 58092f5ccc47e1a1
    AMD:    29 blocks -> gp.kernel 6687945852cf2969,  1 block -> ddce183b7f4ba51d

Twenty-nine siblings with the same configuration and the same input agree
with each other AND across vendors. The odd block sits at the SAME card
position on both boxes. So this is not non-determinism; that block
deterministically takes a path the other twenty-nine do not, and that path
answers differently on Metal than on HIP.

## Why nobody knew

**Both legs printed `ALL PASSED [IDENTICAL]`.** Neither leg compares itself
against the other. A per-vendor run can only check a card against its own
oracle, and cross-vendor identity is by construction a claim about TWO cards,
so a green leg is not evidence for it. That is `[[reached-but-inert]]` applied
to an entire leg rather than to an arm.

## Three consequences

1. This is the ONLY cross-vendor evidence that exists for `cholesky`. The AMD
   cholesky leg emitted no card at all, so the `chol.*` divergence above is
   visible only from inside gp's card.
2. `gaussian_process` must NOT be exposed through the Python surface until
   this is understood. It was on the shortlist for exactly that, chosen off a
   status block that was wrong in the flattering direction.
3. THE CARD HEADER CANNOT SAY WHAT MAKES THAT BLOCK DIFFERENT from its
   twenty-nine siblings. The header carries n_train, d, kernel and
   alpha_bits, and those are identical across all thirty. Whatever selects
   the divergent path is not in the header, which is a CARD_GAPS-class defect
   in its own right and is free to fix.

## What is owed

Diagnosing it is numerics and is not done here. The next step is the header
field, because without it the block cannot even be named. Then the RBF Gram's
own arithmetic on the two columns, since that is where it starts.
