# CPU column for par-samba and par-samba-clip (2026-09-16)

The fourth column for the two lanes `lane/cpu-verifier-par-samba` covered:
the Apple M4's CPU column, one core, against the three GPU columns of
`bench/results/identity_break/2026-09-14_166-lanes`. Nothing was rented and
no Metal job ran (the 0.8.6 Apple column was recording on the same machine).

Host bindings: core, training, mamba and transformer, built IDENTICAL for
the CPU column, and a second set built with `-D MOJOLEARN_HOST_SABOTAGE=1`.
Fixtures: all nine. Repeats: 2.

| file | what it is |
|---|---|
| `cpu-apple-m4.json` | the production CPU column |
| `cpu-apple-m4.sabotage.json` | the same lanes on the MOJOLEARN_HOST_SABOTAGE build |
| `cpu-apple-m4.dropped-shard.json` | the fold patched to ignore every shard after the first |
| `cpu-apple-m4.reversed-fold.json` | the fold patched to take the shards in reverse order |
| `cpu-apple-m4.par-sweep.json` | every par-* lane whose family is in this build set, one fixture |
| `diff.four-columns*.txt` | each column diffed against the three GPU columns, `--require-columns 4` |
| `test_under_dropped_shard.txt` | the shard probe test under the dropped-shard patch |
| `par_sweep_column_check.txt` | `tools/cpu_identity_gate_check.py column` over the sweep |

## PRODUCTION (filled from diff.four-columns.txt)

The CPU column reproduces the recorded GPU bytes on every cell of both
lanes, over all nine fixtures:

    summary: IDENTICAL=18                  (train)
    summary (infer/model): IDENTICAL=36
    summary (batch): IDENTICAL=18
    require-columns 4 over ['par-samba', 'par-samba-clip']: OK

**72 cells IDENTICAL x4, zero DIVERGENT, zero short.** Sample hashes, equal
in all four columns: par-samba/base `359834bee4b2789a`, par-samba/denormal
`c04ab1019588d438`, par-samba-clip/base `387a63aafbcce06d`,
par-samba-clip/denormal `8c2ae3ff8e254f9b`.

This column was taken at commit 78fa7b7c0, the commit that carries the
change, so the evidence and the code it verifies are the same tree.

## The negative controls

- **Host sabotage.** The four families rebuilt with
  `-D MOJOLEARN_HOST_SABOTAGE=1`: DIVERGENT on all 72 cells (train 18,
  infer/model 36, batch 18). `diff.four-columns.sabotage.txt`.
- **A dropped shard.** `ordered_sum_gradients` patched to fold only the
  first shard (`parts[1:]` -> `parts[1:1]`): DIVERGENT on all 72 cells, and
  the shard probe test FAILS with "gradient 0 is not shard 0 + shard 1".
  `diff.four-columns.dropped-shard.txt`, `test_under_dropped_shard.txt`.
- **A reversed fold, which is NOT a control here and is recorded so nobody
  repeats it.** Wave 3 asked for the shards to be folded in reverse order.
  At the two logical shards these lanes use that CANNOT move a bit: IEEE
  addition is commutative, so `g0 + g1` and `g1 + g0` are the same bytes,
  and only associativity fails, which needs three or more parts. Measured
  both ways: the reversed fold left all 72 cells IDENTICAL x4
  (`diff.four-columns.reversed-fold.txt`), while a three-part reassociation
  differs in 63,705 of 200,000 float32 values. The shard's CONTRIBUTION, not
  its position, is what the dropped-shard arm holds these lanes to.

## The other par lanes

`cpu-apple-m4.par-sweep.json` ran every par-* lane whose host family is in
this four-family build set, on the base fixture. Every uncovered par lane
reads REFUSED by name. Of the seven covered lanes in the set, six are
STABLE, including both new ones (par-samba `359834bee4b2789a`,
par-samba-clip `387a63aafbcce06d`).

par-mlp is the seventh and reads REFUSED here, for a reason that predates
this branch: `SmallMLPTrainer` reaches `_mojolearn_linalg`, whose host
binding is not in this build set, and the lane fails with "no CPU
implementation of _mojolearn_linalg.linalg_numeric_mode". Reproduced with
this branch's pool admission present AND with those two lines
scratch-reverted: the same refusal both ways, so it is a build-set gap, not
a regression. The CPU identity gate builds every declared family and covers
par-mlp there.

## Reproducing

`docs/lanes/LANE_STATUS_lane-cpu-verifier-par-samba.md` carries the
commands. The one that costs a whole run if forgotten:
`MOJOLEARN_NUMERIC_MODE=identical` must be exported, because
`identity_break._run_reference` defaults the WANTED mode to `fast` and
refuses before the first fit.
