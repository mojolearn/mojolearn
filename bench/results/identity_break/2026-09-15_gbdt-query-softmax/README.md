# gbdt-query-softmax (lane/gbdt-rest, 2026-09-15): CPU column recorded, Metal column OWED

QuerySoftMax, the softmax ranking loss, and the ranking losses' CatBoost loss
parameters. **This record is INCOMPLETE ON PURPOSE.** The Apple M4 Metal GPU
was degraded by a command-queue leak that evening (AGXCommandQueue at 6754
against a limit of 512, about 6400 with no live creator process), every GBDT
Metal fit ran roughly 20x slow, and a machine restart was the clearing action.
The coordinator stopped Metal work before the restart, so what is here is the
CPU column and the CPU-route tests; the Metal column and everything that
compares against it are owed. **No identity claim is made by this file.**

## What IS measured (Apple M4, one core, identical tier)

CPU column, `gbdt-query-softmax`, fixtures base, ties and odd, two repeats,
through the GBDT host binding (`cpu-apple-m4.json` here):

    cells=3 stable=3 moved=0 refused=0
    infer: stable=3   model: stable=3   batch: stable=3

    | lane                     | base             | ties             | odd              |
    | gbdt-query-softmax       | 25fc6e5d5a805a61 | b6e211efa3cf6954 | b1a5b4a15894d926 |
    | gbdt-query-softmax infer | ea2d05bd3086f534 | fe4f90f054fe3f2a | aceb8f3c4308c9c4 |
    | gbdt-query-softmax model | 15a9571d93ce33ae | aa34715b4256cf52 | 984fcb4eb1a55571 |
    | gbdt-query-softmax batch | aea366262b2cf1a6 | 94c7576f1927e226 | 4545e3bd60cb1b29 |

Tests on the CPU route: `test_gbdt_query_softmax`, `test_gbdt_yeti_rank` and
`test_cpu_training_gbdt_losses`, 21 passed.

Both bindings built from this tree, one core: the Metal identical GBDT binding
(rc=0, 352 s) and the GBDT host binding (rc=0, 125 s).

Light checks on this tree: `docs_facts --check` OK (13 facts, 12 marked spans),
`wheel_ci pins .` OK (56 build scripts), `wheel_ci inventory python/mojolearn`
OK (84 modules).

One finding, measured rather than assumed, and now carried by a test
(`test_lambda_is_read_where_the_second_derivative_is_read`): QuerySoftMax's
`lambda` enters the SECOND derivative only (`query_softmax.cu:182`), and this
loss's own defaults are Gradient leaves (whose Hessian is the leaf weight sum)
under a Cosine score (whose weight plane is the row weight), so `lambda` moves
nothing at the defaults and moves the model under Newton leaves. The lane's
parameter fit therefore runs Newton so `lambda` is actually hashed.

## What is OWED (all of it needs a healthy Metal GPU)

1. The Metal column for `gbdt-query-softmax` (3 fixtures, 2 repeats).
2. The Metal vs CPU diff on the new lane, `--require-columns 2`.
3. The Metal batch sabotage run (must read `batch_moved`).
4. The host sabotage build and its CPU column (must read DIVERGENT on every
   new cell against the Metal column).
5. The base-fixture spot check of the ten existing GBDT lanes, Metal and CPU,
   and against the committed stage 4 Metal column, to show this lane moved no
   existing hash.
6. The test route on Metal.
7. The CatBoost 1.2.10 CPU QuerySoftMax comparison (loss curve against their
   QuerySoftMax metric, plus final NDCG and DCG), a quality reference only.

The exact commands for all seven are in
`docs/lanes/LANE_STATUS_lane-gbdt-rest.md` under "Owed, with the exact
commands". The drivers and the raw logs of this attempt are outside the repo
at `~/mojolearn-evidence/gbdt-rest-2026-09-15/` (the /private/tmp scratchpad
does not survive a restart).

The earlier attempt's Metal cells in that directory read REFUSED, and the
reason is not a defect in this lane: the lane worktree's `python/mojolearn/
identical/` held only the GBDT binding, with no base `_mojolearn` extension,
so `as_f32_colmajor` could not find `transpose_f32`. A resumed run must place
a complete identical binding set in the worktree first; see the status doc.
