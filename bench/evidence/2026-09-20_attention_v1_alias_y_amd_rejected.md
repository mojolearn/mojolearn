# AMD full-estash `y` alias: exact, memory-positive, speed-neutral rejection

Date: 2026-09-20
Base: `f8b3a1967`
Device: Hot Aisle AMD Instinct MI300X VF (`gfx942`), Mojo 1.0.0
(`ed45d567`)
Mode: IDENTICAL, `B=1 L=2048 nh=12 nkv=12 hd=64`, causal

## Candidate

The shipped AMD attention arm is now
`stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`.  Qualify the
portable `MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH` lifetime reuse for that arm:
after the exponent stash cell is consumed by zdot, overwrite it with `y`
instead of allocating a second full `[B,nh,L,L]` Float32 matrix.  Kernel
order and arithmetic spelling are unchanged.

This removes `1 * 12 * 2048 * 2048 * 4 = 201,326,592` bytes of peak scratch
per attention call.  It does not remove a launch or arithmetic work.

## Current production phase profile

The serialized phase-timer run (diagnostic only) showed the repeated
backward cost is dominated by the three numerical kernels, not scratch
allocation:

| phase | warmed time |
|---|---:|
| backward scratch allocation | 0.019-0.021 ms |
| zdot / estash / dres | 0.906-0.911 ms |
| dQ tiled | 2.457-2.491 ms |
| dK+dV kvgrid | 1.778-1.792 ms |

## Exactness

Three alternating process-order pairs used the hashed fixture, eager oracle,
three warmups and nine measured rounds.  Every run passed and all seven
surfaces matched bit for bit: `ctx`, `amax`, `denom`, `zdot`, `dq`, `dk`,
and `dv`.  The common digests included:

```
ctx   13ccc93eac28789a
amax  afaf4c3ad51a2071
denom 461ef8b28777ac11
zdot  aba1ca249126f2a9
dq    d66200e4f5fecee1
dk    cec17a18979026ec
dv    1082386d9d9af404
```

## Alternating whole-call timing

The medians below are the first (baseline) row from each process, so each
number represents the same production arm compiled wholly without or wholly
with aliasing.

| pair/order | default fwd+bwd | alias fwd+bwd |
|---|---:|---:|
| 0, default then alias | 7.596813 ms | 7.610369 ms |
| 1, alias then default | 7.598761 ms | 7.611109 ms |
| 2, default then alias | 7.623825 ms | 7.603442 ms |
| median of processes | **7.598761 ms** | **7.610369 ms** |

The alias is `0.99847x` (0.15% slower), a `+0.011608 ms` per-layer delta or
about `+0.139 ms` over 12 layers.  This is noise-sized but not a repeatable
speed improvement, so AMD automatic routing remains unchanged.  The explicit
profile remains available when the 201 MB/layer memory reduction is more
valuable than the neutral timing.

## Reproduction and teardown

`tools/attention_alias_y_amd_body.sh` builds the phase profile and both A/B
binaries.  Raw logs were captured under
`bench/results/e1g/2026-09-20_amd-mi300x-attention-alias-y-route-run3/`.
The guarded rental finished with `extra_exit=0`; DELETE returned HTTP 204 and
the follow-up GET returned 404 (`destroy_confirmed=1`).
