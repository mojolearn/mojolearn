# TargetEncoder device scratch default promotion


Promotion branch `lane/apple-fast-target-default`, based on current main
`12fdd6697`. Measured source `4d1ea20b2210660acef563b529f72bc57aae9ee4`.
Raw `AFC_DEF_ARM` records (not mixed-arm summary medians):

| Dataset | Main A ms | Candidate B ms | Gain | Digest, both arms |
|---|---:|---:|---:|---|
| taxi | 270.94762498745695 | 203.74366699252278 | 24.8% | e662ed62eea3df89 |
| istella | 250.42391597526148 | 179.81891601812094 | 28.2% | 6b3d8967d553dbb0 |

Quality receipt source/binary hashes match those measured arms. The checker
passed 108 exact dtype/shape/byte arrays across continuous, binary and
multiclass targets; fixed and automatic smoothing; inferred and supplied
categories; held-out/unknown categories; cross-fit and separate-fit outputs;
fitted encodings and target means. Fixed smoothing independently matches
the closed-form oracle at rtol/atol3e-5. Output_shape in timing alone is not
the quality evidence.

Source review: changes between measured integration and current main outside
this candidate are ARIMA files, so the baseline target path remains relevant.
The binding exposes scratch support only for FAST+Apple; `_OFF` removes that
entry and existing Python fallback restores previous alloc/count_neg/download
behavior. IDENTICAL and other vendors remain unchanged. Buckets retain explicit
zero initialization of unused row tails; numerical kernels/arithmetic do not
change. The old `MOJOLEARN_TARGET_SCRATCH` opt-in becomes harmless.

Isolation evidence: taxi harness09:25:14–09:25:17 UTC, istella09:25:18–09:25:21.
Manager reports long du ended09:13:09, serial backup hold began09:29:36 and
ended09:41:08. These identified activities do not overlap these harnesses.
Manager additionally reviewed cleanup receipts09:01–09:09:59, with next
cleanup09:41:09, and corroborated no known heavy-work overlap with these
recorded target windows. Single pairs do not estimate variance; preserve raw
tags and do not repeat timings. This is evidence of a material improvement on
both measured datasets, without claiming an empirical noise estimate.
Local bounded evidence: `~/mojolearn-evidence/apple-fast/sync/target-current-evidence-0959.txt`.
Remote raw race logs: `~/mq/out/race-gap26-target-current-{taxi,istella}/race.log`;
quality: `~/mq/out/gap26-target-current-quality-quality/{PASS.json,compare.log}`.

Manager's required compile verification (no scoring):

```sh
bash ~/mojolearn-evidence/apple-fast/sync/compile_arms_m2.sh lane/apple-fast-target-default x_prep MOJOLEARN_TARGET_SCRATCH_OFF
```

A now means default/new; B means OFF/old. `target_scratch_pair.py` understands
both original opt-in and promotion OFF manifests and verifies actual enabled
state per arm. If manager requests a new untimed default/OFF quality pair,
use the new exact source and a fresh quality tag; old quality source receipts
cannot attest newly compiled binaries. No repeats of measured taxi/istella
are needed. Main merge requires default/OFF builds. The manager reviewed
recorded isolation evidence; no canonical board or main was modified by this lane.
