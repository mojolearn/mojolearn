# group_id in the GradientBoosting pool (lane/gbdt-learning-to-rank, stage 1)

`GradientBoosting.fit(..., group_id=...)` checks CatBoost's Pool grouping in Python
(string or integer ids compared by their byte spelling, `_catboost.pyx:2171-2196`; each
group's rows consecutive, `libs/data/objects.cpp:60-87`), sends the run lengths to the
binding as a five-slot optional tail after the three optional float slots, and both GBDT
bindings refuse the grouping by name because no loss this implementation trains reads one.
The question these columns answer is whether anything moved for a fit WITHOUT `group_id`.

## Machine and builds

Apple M4, one core, shared machine (`nice -n 19`, `OMP_NUM_THREADS=1`,
`MOJOLEARN_CPU_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `MODULAR_THREAD_BUSY_WAIT_US=0`,
`mojo build -j 1`), `MOJOLEARN_NUMERIC_MODE=identical`. Base is origin/main at 0affb7c75;
stage 1 is that tree plus this change. sha256 of the bindings:

| binding | base | stage 1 |
|---|---|---|
| `_mojolearn_gbdt.so` (Metal, identical) | 855a48477fef7344 | 1e790db16ea89e54 |
| `_mojolearn_gbdt_host.so` (CPU column) | 285ad94b1532e8fc | 7da0f2b158c93029 |

The CPU columns load `_mojolearn_core_host.so` from the shared checkout beside the GBDT
host binding (`MOJOLEARN_HOST_DIR`), with the worktree's identical set moved aside so the
package takes its CPU-only route.

## Commands

    python tools/identity_break.py --lanes <the 16 gbdt-* lanes> --fixtures base,ties,odd \
        --repeats 1 --vendor apple-m4 --json apple-m4.<arm>.json
    MOJOLEARN_HOST_DIR=<dir> python tools/identity_break.py --lanes <same> --fixtures base,ties,odd \
        --repeats 1 --vendor cpu-apple-m4 --json cpu-apple-m4.<arm>.json
    python tools/identity_break.py --diff <a> <b>

## Verdicts (files beside this README)

| diff | train | infer/model | batch |
|---|---|---|---|
| `diff-metal.txt`: Metal base vs Metal stage 1 | IDENTICAL=48 | IDENTICAL=90, N/A=6 | IDENTICAL=48 |
| `diff-cpu.txt`: CPU base vs CPU stage 1 | IDENTICAL=36, REFUSED=12 | IDENTICAL=66, N/A=6, NOT-COMPARED=24 | IDENTICAL=36, NOT-COMPARED=12 |
| `diff-metal-cpu.txt`: Metal stage 1 vs CPU stage 1 | IDENTICAL=36, ONE-COLUMN=12 | IDENTICAL=66, N/A=6, ONE-COLUMN=24 | IDENTICAL=36, ONE-COLUMN=12 |

`diff-records-metal.txt` and `diff-records-cpu.txt` add the stage 1 column to the three
training GPU columns of `bench/results/identity_break/2026-09-14_166-lanes`. Their summary
lines count all nine fixtures of the records, so read the rows for base, ties and odd: every
Metal row is IDENTICAL x4 or N/A (186 IDENTICAL, 6 N/A), and every CPU row is IDENTICAL x4,
N/A, or IDENTICAL x3 with the CPU cell REFUSED (186 IDENTICAL, 6 N/A, 48 REFUSED). No
DIVERGENT or MOVED verdict appears on those fixtures in any of the five diffs.

The 12 refused CPU train cells are gbdt-ordered-rmse, gbdt-feature-freq,
gbdt-pointwise-l2-bayesian-eval and gbdt-categorical-ctr, which the GBDT host binding refuses
by name at base as well (lane/cpu-training-gbdt-ordered covers them, not merged at 0affb7c75).
The six N/A cells are the two adapters' model cells (`n/a:save-not-implemented`).

## Tests

`python/mojolearn/tests/test_gbdt_group_id.py`: 15 passed on the Metal route (126 with
test_gbdt_search_option_guards and test_gbdt_input_safety) and 15 passed on the CPU route.
Seen to fail first: against the BASE Metal binding the two refusal cases fail with
`gbdt_fit: params must hold 35 + n_class_weights, ... values (35) values, got 40`, the old
binding refusing the longer tail, which is what shows the group sizes reach the binding and
that the new binding is the one refusing.

## At the merge of origin/main (1d1e2e917)

Main's changes to the GBDT paths since 0affb7c75 were documentation wording, and its
identity_break.py grew new parts; both bindings were rebuilt at the merge (sha256 `_mojolearn_gbdt.so`
d02b8c4eaa124a47, `_mojolearn_gbdt_host.so` f4fef0067e2efa2c) and the same 16 lanes rerun:

| diff | train | infer/model | batch |
|---|---|---|---|
| `diff-metal-stage1-vs-merge.txt` | IDENTICAL=48 | IDENTICAL=90, N/A=6 | IDENTICAL=48 |
| `diff-cpu-stage1-vs-merge.txt` | IDENTICAL=36, REFUSED=12 | IDENTICAL=66, N/A=6, NOT-COMPARED=24 | IDENTICAL=36, NOT-COMPARED=12 |
| `diff-metal-cpu-merge.txt` | IDENTICAL=36, ONE-COLUMN=12 | IDENTICAL=66, N/A=6, ONE-COLUMN=24 | IDENTICAL=36, ONE-COLUMN=12 |

test_gbdt_group_id, test_gbdt_search_option_guards, test_gbdt_input_safety and
test_host_surface: 228 passed on the Metal route; test_gbdt_group_id 15 passed on the CPU
route. docs_facts --check and wheel_ci pins pass.

Not run here: NVIDIA and AMD columns (owed to the next release record, Andrew's Sep 15
rule), and fixtures other than base, ties and odd.
