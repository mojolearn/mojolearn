# 2026-09-17, the reference regeneration record (lane/reference-regen)

WHY THIS RECORD EXISTS. `python/mojolearn/verify_reference/table.json` is built
from the committed columns under `bench/results/identity_break/`, and five
things were wrong with the shipped one at once:

1. five lanes read `stale reference` and were dropped from every comparison;
2. the table carried no `admission_policy`, so `verify --coverage` reported the
   `legacy` fallback;
3. thirty of the 246 appendix entries had no reference hash of any kind;
4. the four optional properties shipped ZERO hashes;
5. `stepfull` was declared by sixteen lanes and carried by nine.

Two of those were a regeneration that had never been run (the optional parts
are only emitted under `--emit-reference --batch-checks`, and the policy key is
written by the builder). The other three needed columns, which is this record.

## The columns

PLACEHOLDER

## What was checked before the table was rebuilt

Every new column was compared cell by cell against the hashes the SHIPPED
table already carried, because a newest-wins rebuild would otherwise replace a
reference several device classes agree on with this record's single value and
say nothing. PLACEHOLDER

The checker was watched to FAIL first: one STABLE cell of one column was
rewritten to `deadbeefdeadbeef` and it reported the cell by name together with
the four device classes standing behind the shipped value.
