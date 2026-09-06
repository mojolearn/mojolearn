# Verify an IDENTICAL build

`verify` runs one pinned k-means fixture, captures its ordered identity-trace
card, and compares it with a reference card shipped in the installation.

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify
```

Use `--json` for automation, `--all` to list every divergent stage, and
`--keep` to retain a matching generated card. `python -m mojolearn env` shows
which binaries and mode the process selected without running the GPU.

## Interpreting the result

| Exit | Meaning |
|---|---|
| 0 | `VERIFIED`: this build reproduced the selected reference card. |
| 1 | `MISMATCH`: the run completed and at least one recorded stage differs. |
| 2 | Invalid command usage. |
| 3 | Refused because the loaded FAST build makes no identity claim. |
| 4 | Could not run or judge: for example no GPU, missing binary, failed fit, or missing comparator. |
| 5 | No usable reference card is installed. |

A green local result proves only that this build and device reproduced this
reference fixture at its recorded checkpoints. It does not certify arbitrary
inputs, untraced work, another device, or the whole library. Cross-vendor
certification additionally requires completed hardware legs and provenance;
see [the support matrix](../SUPPORT_MATRIX.md).

The fixture uses exactly representable FP32 inputs and exercises the pinned
k-means accumulation path. Check its input hashes without a GPU:

```sh
python -m mojolearn check-fixture
```

The repository uses `tools/identity_trace_diff.py` as its one card comparator.
It aligns stage tags before comparing dtype, element count, and raw-bit hash,
so structural and numerical divergences remain distinguishable.

## Maintainer reference workflow

Never replace a reference from one machine alone.

On the producing device:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify \
  --emit-reference /tmp/producer.card
```

On a second vendor, from the same source state:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m mojolearn verify \
  --emit-reference /tmp/confirmer.card \
  --confirm-reference /tmp/producer.card
```

After transferring both cards to one checkout:

```sh
python -m mojolearn install-reference producer.card confirmer.card
```

The installer refuses missing provenance, source/profile disagreement,
placeholder fields, and divergent cards. Reference changes require the same
sabotage-sensitive review as any other numerical contract change.

For portable external-implementation artifacts, use
[conformance bundles](CONFORMANCE.md).
