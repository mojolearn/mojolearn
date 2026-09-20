# GPT-3-small deterministic multi-GPU audit

This record qualifies the ordered two-device transport path at the production
probe shape `B=1, L=2048, d_model=768, heads=12, layers=12,
intermediate=2048, vocab=50257` (162,147,840 parameters), using two logical
shards and seed 93261 on two NVIDIA L40S devices.

The current production implementation was bitwise identical between one and
two physical devices after three optimizer steps. Losses and the complete
exported parameter, first moment, second moment, optimizer-flag, and gradient
buffers matched. The full SHA-256 values are retained in the JSON records.
The initial steady upper medians were 0.391859 s on one device and 0.341179 s
on two devices (1.14854x).

## Rejected local-owner reduction candidate

The candidate read each owner's already-local gradient slice directly instead
of staging it through the owner's incoming buffer. This removed 648,591,360
bytes of device-to-device copies per two-device optimizer step without changing
the FP32 fold order. It passed both strict comparisons: one versus two devices,
and baseline versus candidate, including all full-state hashes.

It was not a reproducible performance winner. The initial two-device upper
median moved from 0.341179 s to 0.335628 s (1.01654x), but a longer rotated run
gave these five-sample medians:

- baseline 1: 0.307101 s
- candidate 1: 0.333959 s
- candidate 2: 0.334170 s
- baseline 2: 0.338619 s

The pooled ten-sample medians were 0.321247 s baseline and 0.334089 s
candidate. The baseline drift across its two positions is larger than the
initial effect and the candidate does not improve the pooled result. The source
candidate was therefore reverted; this is evidence only, not a recommended
production change.

The guarded pod `k9y4sjop9nfa9o` was deleted at 2026-09-20 14:37:53 EDT:
DELETE returned HTTP 204 and the immediate verification GET returned HTTP 404.
No credentials or transient endpoint metadata are recorded here.
