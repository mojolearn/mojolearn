# RF / ExtraTrees experiment board

Own-only full fits: one excluded warmup and one scored sample. Pending comparison is not IDENTICAL qualification. No opponent ratios or default changes.

| Vendor | Dataset | Profile | Scored ms | Evidence |
|---|---|---|---:|---|
| amd | istella | rf-k4 | 44244.218 | CAPTURED_PENDING_COMPARISON |
| amd | istella | rf-k1 | 44305.355 | CAPTURED_PENDING_COMPARISON |
| amd | istella | rf-k2 | 44360.828 | CAPTURED_PENDING_COMPARISON |
| amd | istella | rf-k8 | 44600.162 | CAPTURED_PENDING_COMPARISON |
| amd | istellareg | et-float | 20698.697 | CAPTURED_PENDING_COMPARISON |
| amd | istellareg | et-u16 | 21401.863 | CAPTURED_PENDING_COMPARISON |
| amd | year | et-float | 4554.551 | CAPTURED_PENDING_COMPARISON |
| amd | year | et-u16 | 4629.698 | CAPTURED_PENDING_COMPARISON |
| nvidia | taxi | rf-k4 | 8066.457 | CAPTURED_PENDING_COMPARISON |
| nvidia | taxi | rf-k1 | 8122.243 | CAPTURED_PENDING_COMPARISON |
| nvidia | taxi | rf-k2 | 8103.067 | CAPTURED_PENDING_COMPARISON |
| nvidia | taxi | rf-k8 | 8085.612 | CAPTURED_PENDING_COMPARISON |
| nvidia | istella | rf-k4 | 24471.208 | CAPTURED_PENDING_COMPARISON |
| nvidia | istella | rf-k1 | 24548.632 | CAPTURED_PENDING_COMPARISON |
| nvidia | istella | rf-k2 | 24514.661 | CAPTURED_PENDING_COMPARISON |
| nvidia | istella | rf-k8 | 24486.532 | CAPTURED_PENDING_COMPARISON |
| nvidia | istellareg | et-float | 30634.269 | CAPTURED_PENDING_COMPARISON |
| nvidia | istellareg | et-u16 | 29712.217 | CAPTURED_PENDING_COMPARISON |
| nvidia | year | et-float | 3282.053 | CAPTURED_PENDING_COMPARISON |
| nvidia | year | et-u16 | 3831.254 | CAPTURED_PENDING_COMPARISON |
| amd | taxi | rf-k4 | 16390.002 | CAPTURED_PENDING_COMPARISON |
| amd | taxi | rf-k1 | 16398.556 | CAPTURED_PENDING_COMPARISON |
| amd | taxi | rf-k2 | 16377.838 | CAPTURED_PENDING_COMPARISON |
| amd | taxi | rf-k8 | 16370.723 | CAPTURED_PENDING_COMPARISON |

## Candidate decisions

- rf-k1: REJECTED_KEEP_DEFAULT; ratios {'nvidia/taxi': 1.0069157995883447, 'amd/taxi': 1.0005219035360704, 'nvidia/istella': 1.0031638814070807, 'amd/istella': 1.001381807674847}
- rf-k2: REJECTED_KEEP_DEFAULT; ratios {'nvidia/taxi': 1.004538547716798, 'amd/taxi': 0.9992578402369933, 'nvidia/istella': 1.0017756785852174, 'amd/istella': 1.0026355986221749}
- rf-k8: REJECTED_KEEP_DEFAULT; ratios {'nvidia/taxi': 1.0023746484980953, 'amd/taxi': 0.9988237341276712, 'nvidia/istella': 1.000626205293993, 'amd/istella': 1.008044983414556}
- et-u16: REJECTED_KEEP_DEFAULT; ratios {'nvidia/istellareg': 0.9699012893044714, 'amd/istellareg': 1.033971510380581, 'nvidia/year': 1.1673345920982994, 'amd/year': 1.0164993212283713}

## Pending and failed evidence

Pending cells: 0; current failed cells: 0. Historical attempts are retained in board.json.
