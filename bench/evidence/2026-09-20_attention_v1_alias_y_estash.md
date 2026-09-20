# Attention-v1 full-estash y lifetime alias

The opt-in `alias-y` memory profile preserves the production v1 full exponent
layout. During backward, the zdot kernel reads each exponent for the last time,
computes `y = identical_div(e, denominator)`, and writes y back to that same
cell. The unchanged dQ/dK/dV kernels then read the exponent buffer as y. This
removes the separate full-layout y scratch without changing a fold, operand,
launch, or downstream address. `packed` plus `alias-y` is compile-time refused
because packed exponent and full y addresses differ.

Apple's exhaustive trial build passed all 15 cases x 25 arms: every RAN ctx,
amax, denominator, zdot, dQ, dK, and dV buffer was bit-identical to eager.

RunPod pod `66yxws56aauz3y`, NVIDIA L40S 46,068 MiB, driver 580.159.03,
qualified the target and two adjacent shapes. Each binary used three warmups
and nine timed alternating rounds, repeated with binary order reversed (36
full-step samples/profile/case). All logs passed the eager oracle and all seven
digests matched baseline exactly.

| shape | baseline median | alias-y median | ratio |
|---|---:|---:|---:|
| B1/H12/L1024/HD64 causal | 1.427016 ms | 1.398165 ms | 1.0206x |
| B1/H12/L2048/HD64 causal | 4.414118 ms | 4.406293 ms | 1.0018x |
| B1/H12/L2048/HD64 window 512 | 1.908968 ms | 1.840833 ms | 1.0370x |

These deterministic hashed-operand timings are a no-regression screen, not a
real-corpus throughput claim. The structural memory reduction is exact: at the
GPT-3-small target the removed y allocation is
`1*12*2048*2048*4 = 201,326,592` bytes. The retained exponent allocation and
dy scratch are unchanged. Use
`MOJOLEARN_ATTENTION_MEMORY_PROFILE=alias-y`; the default remains `estash`.

The pod watchdog was armed for 45 minutes. Results were fetched, the pod was
deleted with HTTP 204, and absence was verified with GET HTTP 404.
