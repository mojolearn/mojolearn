# Isolation Forest single output allocation (2026-09-20)

`IsolationForest._run` formerly allocated both an `n_query` Float32 score
array and an `n_query` Int32 label array for every public scoring call. The
native ABI already guarantees that `want=score_samples/decision_function`
touches only the score pointer and `want=predict` touches only the label
pointer; its binding documentation explicitly permits zero for the unused
address. The Python boundary now allocates only the selected result and passes
zero for the other pointer. No model, kernel, arithmetic, or dispatch changes.

Apple M4 Metal, IDENTICAL build, 512 training rows, eight trees, fixed model
and query bits. The old arm adds the exact removed unused `empty(n, i4/f4)`
allocation around the otherwise identical new public call. Alternating-order
three-run medians:

| seed | features | query rows | method | exact | old s | new s | speedup | digest prefix |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 11 | 8 | 1,000,000 | predict | yes | .101608 | .099777 | 1.018x | `c61d1eff1bf6c9b7` |
| 29 | 16 | 1,000,000 | score_samples | yes | .139470 | .139166 | 1.002x | `754057ba6d700952` |
| 47 | 32 | 1,200,000 | decision_function | yes | .240090 | .233400 | 1.029x | `efbb90f8f45664e5` |

No case regressed. The deterministic allocation saving is 4 bytes per query
row (4.0 MB at one million rows, 4.8 MB at 1.2 million). Separate one-million
row `predict` processes measured `/usr/bin/time -l` peak RSS of 252,837,888
bytes with the old allocation and 247,480,320 bytes without it, a 5,357,568
byte reduction. Output length, sum, and SHA matched.

The focused pointer-boundary gate covers all three methods and sabotages any
attempt to pass a nonzero unused address.
