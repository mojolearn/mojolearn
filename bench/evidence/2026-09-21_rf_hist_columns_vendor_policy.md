# RF four-column histogram tile vendor policy

The candidate computes four adjacent sampled-feature histograms per block and
shares each row-ID and sampled-label load across those columns. Each column
keeps its original integer/fixed-point accumulator and write location, so the
arithmetic order within a histogram does not change.

The NVIDIA H100 trial used exactly the canonical Taxi and Istella-S R2 data,
one excluded warmup plus five retained complete fits, and three alternating
fresh processes per arm. All 12 arm/dataset/process records matched the full
five-array model, fitted classes and metadata, complete predictions and
probabilities, accuracy, and log loss. Every within-process spread was at most
1.035. Selection logs showed the original `histogram_binned` route for the
baseline and `histogram_binned_columns4_*` for the candidate; a falsely
labelled baseline negative control was rejected.

| vendor | dataset | baseline median (ms) | tile4 median (ms) | tile4 / baseline | decision |
| --- | --- | ---: | ---: | ---: | --- |
| NVIDIA H100 | Taxi | 812.815 | 795.713 | 0.97896 | default on |
| NVIDIA H100 | Istella-S | 1290.605 | 1138.535 | 0.88217 | default on |
| Apple Metal | Taxi | 8134.5 | 7356.0 | 0.90429 | default on |
| Apple Metal | Istella-S | 21217.3 | 17685.9 | 0.83356 | default on |

The Apple runs used 1,000,000 rows and matched complete model, prediction,
probability, and quality outputs. Their observed spreads were 1.063/1.052 on
Taxi and 1.162/1.092 on Istella for baseline/candidate; stability is recorded
but is not a promotion gate under the current median policy.

NVIDIA and Apple therefore select tile4 by default.
`MOJOLEARN_RF_HIST_COLUMNS4_OFF=1` restores the original one-column route.
HIP remains default-off pending its MI300X trial, while
`MOJOLEARN_RF_HIST_COLUMNS4=1` remains the explicit AMD experiment switch.

A post-promotion H100 smoke compiled the shipped default and `_OFF` bindings
from `1a131612b`. The default selected 1,836 Taxi and 2,940 Istella tile4
launches with zero original-route launches; OFF selected the same counts on the
original route with zero tile4 launches. The same run matched complete model,
metadata/classes, prediction, probability, and quality outputs across all ten
retained records per dataset. Evidence is at
`mojolearn-evidence/2026-09-21_rf_gdw_h100/pulled/rf-default-smoke/`.
RunPod `9c8xn6uorujf8k` was then destroyed (DELETE 204, verification GET 404).
