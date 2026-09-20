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
