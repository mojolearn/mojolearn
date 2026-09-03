# Apple cards, 2026-09-03, archived because they were only in /tmp

Two Apple IDENTICAL cards produced on 2026-09-03 lived only at their default
`/tmp` trace paths. Both were used as one side of a cross-vendor diff, and a
comparison whose Apple half is a temp file is not evidence anyone else can
check. They are archived here unmodified.

| card | records | diffed against | result |
|---|---|---|---|
| `mamba3.identical.card` | 28 | MI325X `2026-09-03_072159` and H100 `2026-09-03_073012` / `082634` | byte-identical |
| `transformer-backward.identical.card` | 37 | H100 `2026-09-03_091511` | byte-identical |

## Read the commit spread before quoting these

The mamba3 columns did NOT run at one commit: Apple at `c6e86966`, AMD at
`9d4aabbe`, NVIDIA at `a2b5f656`. The diff across that span touches
`checks/numerics.mojo` by NINE LINES, ALL OF THEM COMMENT -- verified with
`git diff c6e86966 a2b5f656 -- checks/numerics.mojo`, whose every changed line
begins `#:` -- plus two markdown files. So nothing mamba3 compiles moved and
the comparison holds, but the honest sentence is THREE VENDORS AT THREE
COMMITS, the same caveat mamba2 already carries, and not three at one commit.

The transformer-backward pair is Apple at the working tree of `5c0f7abd` and
NVIDIA at `eca21b78`; the diff between them is tools, packaging and docs only.

## What produced them

    tools/with_identical_mode.sh pixi run check-mamba3-block
    MOJOLEARN_TFB_CHECK_CLAUSE_B=1 ... _F=1 \
      tools/with_identical_mode.sh pixi run check-transformer-backward

The second needs all five clause flags: the default run skips (b) through (f),
and clause (f) is the only one that asks whether the gradient is the gradient.
