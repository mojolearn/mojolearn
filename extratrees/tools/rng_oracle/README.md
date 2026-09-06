# rng_oracle

The reference values for `extratrees/checks/pcg_rng.mojo`.

`main.cpp` is a small C++ program that **copies** RAFT's `PCGenerator`, RAFT's
`custom_next` uniform-int and uniform-float overloads, RAFT's `wmul_64bit`, and
cuML's `fnv1a32`, with the CUDA decorations (`HDI`, `DI`, `__CUDA_ARCH__`)
defined away. Every line of arithmetic in it is upstream text, not a
re-derivation. It writes `pcg_reference.txt`, which
`extratrees/checks/pcg_rng_check.mojo` compares against cell by cell.

That is the point: the Mojo port is checked against **their** arithmetic, not
against a tally we wrote twice.

## Pins

| upstream | commit | path in this box |
| --- | --- | --- |
| RAFT | `661a3b840c3300f95f053812a560c952c9d049a4` | `~/CascadeProjects/upstream/raft` |
| cuML | `00094f7e4e4b5da3a968d193a4da6085fa38f11b` | `~/CascadeProjects/upstream/cuml` |

Both pins are written into the first lines of `pcg_reference.txt`, and the Mojo
check **refuses to run** if they are not the pins it expects. Bumping an
upstream therefore means regenerating the reference and reading the diff,
because that diff is the behaviour change.

## Regenerating

```sh
extratrees/tools/rng_oracle/build.sh
```

Needs only `/usr/bin/clang++`. It compiles to a temporary file, runs it in this
directory, and deletes the binary; the only tracked output is
`pcg_reference.txt`.

## The file format

Line-oriented, one record per line, fields separated by single spaces.
Comment lines start with `#`. Integers are hex without a `0x` prefix except
where noted.

| line | meaning |
| --- | --- |
| `raft_pin <sha>` / `cuml_pin <sha>` | the pins above |
| `fnv_cases <n>` then `fnv <hash> <txt> <out>` | one `fnv1a32` step |
| `chain_cases <n>` then `chain <feature> <tree> <node> <subseq>` | the whole key chain |
| `streams <n>` then `stream <seed> <sub> <off> <k>` + `k` × `u32 <hex>` | raw `next_u32` |
| `u64_streams <n>` then `u64stream ...` + `u64 <hex>` | raw `next_u64` |
| `uint32_cases <n>` then `uint32 <seed> <sub> <off> <start> <diff> <k>` + `i32 <decimal>` | `custom_next`, 32-bit diff |
| `uint64_cases <n>` then `uint64 ...` + `i64 <decimal>` | `custom_next`, 64-bit diff (the overload cuML instantiates) |
| `float_streams <n>` then `floatstream ...` + `f <decimal>/<hexbits>` | raw `next_float` |
| `ufloat_cases <n>` then `ufloat <seed> <sub> <off> <start> <end> <k>` + `uf <decimal>/<hexbits>` | `custom_next`, `UniformDistParams<float>` |
| `end` | terminator |

`start` and `diff` on the integer lines are decimal, because they are also
range bounds a human reads.

### Why floats carry their bits

`String(Float32)` in Mojo does not round-trip — a measured 0.46% of float32
values come back one ULP wrong — so a float written only as decimal is not
evidence of anything. Every float in this file is written
`<decimal>/<hexbits>`, and the Mojo check compares **only the hex half**. The
decimal is there so a human reading a diff can see what moved.

## Coverage, and why these cases

Nine `(seed, subsequence, offset)` triples. Six of the nine subsequences come
out of the fnv1a32 key chain rather than being round numbers, because a
generator seeded only at `subsequence = 0, 1, 2` never exercises the high bits
of `inc`. Offsets are `0, 1, 5, 64, 1023, 10**6, 0xFFFFFFFF`, so `skipahead` is
exercised with a single bit, with an exact power of two, with a run of ones,
and with a 32-bit ladder — not only with zero, where the whole function is a
no-op.

Integer ranges include `diff = 1` and `diff = 2` (degenerate), primes 97 and
1000003 (non-power-of-two, so Lemire's rejection threshold is non-zero), a
power of two 4096, `0xFFFFFFFF`, and on the 64-bit side `2**63 + 1`, which puts
a bit in every position of the wide product's high word.
