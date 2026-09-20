# NVIDIA default promotion for attention-v1 alias-y

The production routing audit found full-layout estash defaults on NVIDIA and
AMD, and no estash default on Apple. The alias-y qualification covers NVIDIA
L40S only, so this change promotes alias-y automatically only when the compiled
column is NVIDIA and the shipped full-estash backward is active.

The guard is false for trial builds and non-estash columns. Explicit `packed`
and `recompute` profiles take precedence and cannot be combined with alias-y's
full layout. AMD remains on its existing full-estash allocation pending a
vendor A/B; Apple remains unchanged because its default carries no estash bit.

Qualification inherited from
[the alias-y receipt](2026-09-20_attention_v1_alias_y_estash.md): exact seven
buffer hashes on L40S at B1/H12/HD64 for L1024 causal, L2048 causal, and L2048
window-512; 201,326,592 bytes removed at L2048; 36 samples/profile/shape with
no timing regression. The exhaustive Apple trial gate covered 15 cases x 25
arms bit-for-bit.

The routing gate builds and runs four columns/profiles:

- Apple default: `v1-estash-default`;
- NVIDIA default: `v1-alias-y-estash`;
- NVIDIA packed: `v1-packed-estash`;
- NVIDIA recompute: `v1-recompute`.

No AMD promotion or AMD performance claim is made.
