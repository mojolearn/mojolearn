# Native binary LogisticRegression prediction (2026-09-20)

Binary `LogisticRegression.predict` formerly exported every Float32 decision
score, expanded it into a Python-float list, built a Python-int code list, and
then decoded that list.  The new `qn_predict_binary` entry retains the same
decision function and strict `score > 0` threshold in native code and returns
Int64 codes directly.  Multiclass prediction and `decision_function` are
unchanged; old binaries retain the old fallback.

Apple M4 Metal, IDENTICAL build, 1,000,000 fixed query rows, balanced-order
three-run medians:

| seed | features | old s | new s | speedup | exact SHA-256 |
|---:|---:|---:|---:|---:|---|
| 3 | 8 | .082439 | .014404 | 5.72x | yes |
| 11 | 32 | .094168 | .027440 | 3.43x | yes |
| 29 | 64 | .114801 | .045237 | 2.54x | yes |

The respective output digests were `76c62a255412f887b`,
`ccd060410cd01292`, and `9bf8ed2172f79683`.  Isolated 8-feature processes
reduced peak RSS from 238,600,192 to 205,750,272 bytes (-32,849,920), with
the complete digest unchanged.  The focused gate includes exact zero and
negative-zero threshold cases, numeric-label native gathering, object labels,
and sabotages any attempt by the new path to call public `decision_function`.
Both Metal and CPU-host estimator bindings compile with two jobs.
