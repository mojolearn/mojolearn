# FAST forest device argmax qualification (2026-09-20)

Scope: FAST `parallel_groves` classifier `predict` only. The existing forest
kernel still writes its exact `rows * classes` Float32 vote workspace. A
one-thread-per-row kernel applies the established strict-`>` first-max rule
on device and returns `rows` Int32 codes. `predict_proba`, regressors, and
IDENTICAL/DETERMINISTIC dispatch are unchanged.

Apple M4 Metal, 1,000,000 rows, 16 Float32 features, 64 trees, depth 8,
fixed seeds. Each timing is a complete Python public prediction after warmup.
The comparison arm is the former public implementation (`predict_proba` or
`_vote`, host `argmax_rows`, then `decode_labels`).

Eight classes, seven repetitions (seconds):

| estimator | former raw | candidate raw | median | speedup |
|---|---|---|---|---|
| RF | .077157,.072573,.073839,.077798,.081190,.081180,.090757 | .069215,.063295,.063652,.066286,.068023,.071295,.078961 | .077798 -> .068023 | 1.144x |
| ET | .154163,.126856,.123046,.153357,.125216,.100977,.104917 | .075980,.066885,.063694,.071295,.070152,.073666,.072069 | .125216 -> .071295 | 1.756x |

Every one-million-row label array matched exactly. SHA-256:

- RF: `0f3a8ea24a250982fd45a5e284d1aafbad10d08ac348bae57461c42b65ccf1fd`
- ET: `b8fb24fae6c9d0c9a9d3577ef5a76721ec13d55c16d82608a3afbedcdd430dbf`

Binary RF is effectively flat (five repetitions: former median .039703 s,
candidate .038694 s; identical digest
`937c14670fe292b1806580a6935b76faefe6c3fc162c6cc23dfd9f0278d1d0da`).
This is not advertised as a binary-class speed win; the dispatch is retained
because it does not regress and avoids materializing the public probability
object.

`/usr/bin/time -l` process maxima were inconclusive because allocator/device
workspace high-water dominates (ET former 287,621,120 bytes, candidate
306,446,336 bytes). No RSS claim is made. The durable allocation reduction at
the output boundary is exact: `4 * rows * classes` host vote bytes become
`4 * rows` code bytes (32 MB -> 4 MB for this eight-class case); the device
vote workspace is intentionally retained so forest arithmetic is untouched.

Focused routing/protocol gate:

```text
36 passed in 1.19s
```

Both modified FAST bindings compiled and loaded on Metal. No cloud resource
was provisioned.

## Generalization matrix

Before promotion, a second qualification crossed both estimators with three
seeds, 2/3/8 classes, 8/16/64 features, depth 4/8/12, balanced and 90%-majority
labels, and deliberately tied leaf votes. Cases used 500,000 or 1,000,000
rows. `old` is the prior parallel-groves public composition and `new` is the
device-argmax path. Times are alternating-order three-run medians in seconds.
Accuracy is identical between arms because every label array matches; the
listed value is the common value. Tie cases intentionally force class 0/1 to
equal votes to exercise strict-`>` first-max, so their accuracy is not a
model-quality claim.

| estimator | case (seed) | C/F/D | rows | exact | accuracy | old | new | speedup |
|---|---|---:|---:|---|---:|---:|---:|---:|
| RF | balanced (11) | 2/8/4 | 500k | yes | .940640 | .009188 | .007768 | 1.183x |
| RF | imbalanced (29) | 2/16/8 | 1m | yes | .952125 | .033935 | .031355 | 1.082x |
| RF | tie (47) | 2/64/12 | 500k | yes | .500920 | .026477 | .023932 | 1.106x |
| RF | balanced (11) | 3/16/12 | 1m | yes | .999375 | .041459 | .038767 | 1.069x |
| RF | imbalanced (29) | 3/64/4 | 500k | yes | .932416 | .022644 | .020160 | 1.123x |
| RF | tie (47) | 3/8/8 | 1m | yes | .331500 | .031141 | .025568 | 1.218x |
| RF | balanced (11) | 8/64/8 | 500k | yes | .882494 | .044240 | .035897 | 1.232x |
| RF | imbalanced (29) | 8/8/12 | 1m | yes | .923375 | .094215 | .061173 | 1.540x |
| RF | tie (47) | 8/16/4 | 500k | yes | .118616 | .044730 | .034371 | 1.301x |
| ET | balanced (11) | 2/8/4 | 500k | yes | .928872 | .014575 | .013764 | 1.059x |
| ET | imbalanced (29) | 2/16/8 | 1m | yes | .951500 | .042213 | .036720 | 1.150x |
| ET | tie (47) | 2/64/12 | 500k | yes | .500920 | .034533 | .032192 | 1.073x |
| ET | balanced (11) | 3/16/12 | 1m | yes | .982125 | .039000 | .035010 | 1.114x |
| ET | imbalanced (29) | 3/64/4 | 500k | yes | .932416 | .022177 | .019919 | 1.113x |
| ET | tie (47) | 3/8/8 | 1m | yes | .331500 | .040091 | .034227 | 1.171x |
| ET | balanced (11) | 8/64/8 | 500k | yes | .865604 | .042724 | .031375 | 1.362x |
| ET | imbalanced (29) | 8/8/12 | 1m | yes | .913750 | .109062 | .055706 | 1.958x |
| ET | tie (47) | 8/16/4 | 500k | yes | .118616 | .039126 | .028201 | 1.387x |

All 18 repeated `predict_proba` byte hashes were stable. There were no timing
regressions; four binary/small-class cases fall below 10%, but the operation
still removes the materialized host vote matrix and remains faster. Dispatch
uses only the compiled numeric mode, configured inference engine, binding
availability, and shape metadata; it never inspects feature or label values.
Focused sabotage tests prove explicit `sequential` and both reproducibility
tiers cannot enter this entry point, even if the native symbol is present.
