# Complete-dependency CPU lane pairs, 2026-09-17

Source `1d7fc53fe9ee5406bd42b4641324fb864ccba6de`, Linux x86-64, RunPod
`17o53wuev9wtbr`, two vCPUs. Two single-thread workers; all nine fixtures,
two repetitions, default step/full and applicable RLPAIR properties.
Seventeen production and fifteen sabotage native families were built.

22 of 24 full lane pairs pass: all twelve BF16/INT8 weight lanes and ten
vendor-held candidates. All passing pairs have clean STABLE/N/A properties
and native training sabotage detected on all nine fixtures. Combined native
faults are used, not one-fault-at-a-time localization.

`gp-normalize-y` and `gp-sample-y-normalize` REFUSED: preprocessing was omitted
from this build list. Both failures and their complete records are retained.
Future batches should build all 32 families to avoid transitive omissions.
A CPU pass alone does not admit the twelve vendor-held candidates: their
NVIDIA/AMD evidence remains owed.

Build 253s, command 1844s, billed 2189s, $0.0365. Pod VERIFIED DELETED;
`teardown.txt` records termination verification. No active rental remains
from this workstream. This is source-built CPU evidence, not wheel or release
qualification. Installed artifact evidence lives in the separate probe tree.
