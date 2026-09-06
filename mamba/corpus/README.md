# Mamba reference corpus

This directory contains independently generated tolerance references for
Mamba 1, Mamba 2 (SSD), and Mamba 3 (SISO). `gen_corpus.py` transcribes cited
upstream PyTorch references and emits deterministic inputs plus per-stage
float64 and float32 results.

`ref64` is the correctness reference. `ref32` is informative calibration.
Neither is a bitwise identity oracle; the Mojo host oracles and identity cards
define that stronger result.

## Generate and verify

The declared `skgpu` environment owns PyTorch, NumPy, and einops:

```sh
pixi run -e skgpu mamba-corpus-m1
pixi run -e skgpu mamba-corpus-m2
pixi run -e skgpu mamba-corpus-m3
```

Each task regenerates into a temporary directory and runs all verification
arms without overwriting committed fixtures. To intentionally refresh the
corpus, run `gen_corpus.py --family <family> --verify --out mamba/corpus` in
that environment and review every binary/hash change before committing it.

Recorded reproduction hashes:

- Mamba 1: `318a67310ad3d9edb260875a42a40ba32003ac10a7ef6b306751ca5ee07886ca`
- Mamba 2: `c928478e7a28ab7400e1cc508bc54a514bd550b84e4bb79d3c7d615017bd7f62`
- Mamba 3: the hash in `mamba3/manifest.json` is authoritative.

## File schema

Each family has a top-level `manifest.json`. Mamba 1 cases are direct children
of this directory; Mamba 2 and 3 cases live under `mamba2/` and `mamba3/`.

```text
<family>/<case>/
  manifest.json
  <input-or-parameter>.f32
  ref64/<stage>.f64
  ref32/<stage>.f32
```

Tensor files are raw little-endian IEEE-754 values in row-major order with no
header. The case manifest is authoritative for shapes, order, seeds, tensor
IDs, ranges, stages, hashes, elisions, continuation state, and runtime options
such as Mamba 2 `dt_limit`.

Inputs are at the case root. `ref*/input.x.*` is a recorded stage, not the
source input. Paths are repository-root-relative, so checks run from the root.

## Deterministic input hash

All families use `mojolearn.mamba.corpus.hash.v1` with disjoint tensor IDs and
family seed bases. Unsigned arithmetic wraps modulo 2^64.

```c
uint64 splitmix64(uint64 z) {
    z += 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

uint64 key = splitmix64(seed ^ (tensor_id << 32));
uint64 h = splitmix64(key + flat_row_major_index);
double f = (double)(h >> 40) * 0x1p-24;
float value = (float)(lo + (hi - lo) * f);
```

Seed bases are `0x4D616D6261436F72` (Mamba 1),
`0x4D6D6232436F7270` (Mamba 2), and `0x4D6D6233436F7270`
(Mamba 3). Case `k` uses `seed_base + 0x1000*k`; composition-row manifests
record when they reuse their parent's seed and input slice. Exact tensor IDs,
ranges, overrides, and signed-zero plants live in the manifests and executable
case tables in `gen_corpus.py`.

## Family contracts

### Mamba 1

- References: state-spaces/mamba `e9594ce` selective scan and Hugging Face
  transformers `d56c55b` Mamba fallback block.
- Constants: state 16, convolution 4, expansion 2, RMS epsilon `1e-5`.
- Cases cover lengths 1, 4, 16, 64, 257; batch composition; softplus
  threshold; extreme/near-zero A; signed zeros; and gate saturation.
- Short cases record all-token scan state for decode comparison.

### Mamba 2

- References: state-spaces/mamba `e9594ce` `ssd_minimal_discrete`/`segsum`;
  Hugging Face chunk scan is the second independent spelling.
- Constants: state 128, convolution 4, expansion 2, head dimension 64, group
  count 1, chunk size 256, norm epsilon `1e-5`.
- Padding the final chunk is arithmetic; outputs truncate to real length.
  Large `seg.L`/`cb.G` stages may only be elided explicitly in the manifest.
- Cases bracket chunk boundaries and cover multi-chunk state passing, batch
  composition, `dt_limit`, nonzero initial state, and decode resumption.
- Decode state is convolution window, state entering the open chunk, and
  buffered open-chunk xBC/raw-dt rows. The after-padding report state is not
  the boundary state.

### Mamba 3

- Reference: state-spaces/mamba `e9594ce` SISO forward and step references;
  Hugging Face has no Mamba 3 reference model.
- Constants: state 128, expansion 2, head dimension 64, group count 1, chunk
  size 64, 32 RoPE angles, norm epsilon `1e-5`, A floor `1e-4`.
- The case table mirrors `mamba/checks/mamba3_fixture.mojo`, which is
  authoritative together with the emitted manifests.
- Cases bracket 64 tokens and cover A-floor clamp, angle wrap, trap
  saturation, signed zeros, nonzero state, composition, and decode behavior.
- Continuation is tolerance-checked; it is not promised bitwise-equal to an
  unbroken prefill.

## Compare a device dump

A driver writes `<stage>.f32` files with names/layouts from the case manifest:

```sh
python tools/mamba_corpus_check.py mamba/corpus/<case> <dump-dir>
python tools/mamba_corpus_check.py mamba/corpus/mamba2/<case> <dump-dir>
python tools/mamba_corpus_check.py mamba/corpus/mamba3/<case> <dump-dir>
```

The checker requires a stage, rejects shape/nonfinite mismatches, and uses
`abs(error) <= atol + rtol*abs(reference)`. `--self-test` calibrates against
`ref32`. Tasks default to `rtol=1e-7, atol=1e-6` for Mamba 1 and
`rtol=1e-5, atol=1e-6` for Mamba 2/3; large-magnitude adversarial cases may
require a manifest-documented looser tolerance.

Independent whole-block gradient references come from
`tools/mamba_gradient_oracle.py`; see `mamba/README.md`.
