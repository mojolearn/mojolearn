# The fourteen low-bit lanes on one Apple M4: the Metal column against the CPU column (2026-09-17)

Lane lane/identical-lowbit-inference, at commit 875f25178 (the harness lanes)
with the identical GPU builds of `_mojolearn`, linalg, transformer, mamba and
training and the six host bindings (core, linalg, transformer, mamba,
training, neural) built in the lane worktree. Contract
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`.

| column | how | files |
|---|---|---|
| metal | `tools/identity_break.py --lanes <lane> --fixtures base,ties,denormal --repeats 2 --vendor apple-m4-lowbit`, one lane per call (the harness refuses a broad Apple matrix by name) | `metal/apple-m4-lowbit-<lane>.json` |
| cpu | the same, `--require-cpu`, with the package imported from a copy that carries no `identical/` set and `MOJOLEARN_HOST_DIR` naming the six host bindings | `cpu/cpu-lowbit-<lane>.json` |

Every column read `cells=3 stable=3 moved=0 refused=0`, and
`--diff metal cpu --require-columns 2` read, per lane:

| lane | train | infer / model | batch |
|---|---|---|---|
| gemm-bf16 | IDENTICAL=3 | N/A=6 | N/A=3 |
| gemm-int8 | IDENTICAL=3 | N/A=6 | N/A=3 |
| transformer-bf16w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| transformer-int8w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| mamba1-bf16w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| mamba1-int8w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| mamba2-bf16w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| mamba2-int8w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| mamba3-bf16w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| mamba3-int8w | IDENTICAL=3 | IDENTICAL=3, N/A=3 | IDENTICAL=3 |
| mlp-bf16w | IDENTICAL=3 | IDENTICAL=6 | N/A=3 |
| mlp-int8w | IDENTICAL=3 | IDENTICAL=6 | N/A=3 |
| samba-bf16w | IDENTICAL=3 | IDENTICAL=6 | N/A=3 |
| samba-int8w | IDENTICAL=3 | IDENTICAL=6 | N/A=3 |

No DIVERGENT cell. The N/A cells are the parts the lanes declare they do not
have (a GEMM has no held-out probe; the MLP and Samba forwards declare no
batch part of their own).

What this record is: one box, two columns (Metal through the GPU builds,
the CPU through the host oracles), and the first record of these lanes. What
it is not: a three-vendor certificate. The NVIDIA and AMD columns are owed
and will replace this record as the gate's columns when taken.
