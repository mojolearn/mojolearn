# Fresh NVIDIA comparator campaign: audit and execution plan

Source audit only; no commands below have been executed by the audit agent.
Only the main lane runs builds, tests, timings, or rentals. Run one GPU job
at a time with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`
and `MAX_JOBS=2`; prefer prebuilt vendor wheels over extension compilation.

## Admission and scope

Follow `bench/speed/README.md`: same commit, device, session, input bytes and
weights; one excluded warm-up; at least five interleaved FAST, IDENTICAL,
external rounds; synchronization on both sides of each clock; median, IQR,
paired-ratio range, output hashes, and accuracy results. Build both Mojo
binaries before the measurement window. Record dependency versions and
strict-FP32 versus TF32/BF16 explicitly. Cross-vendor Mojo byte equality is
separate from tolerance/quality agreement with a different implementation.

## First bounded batch: reproduce the four trouble spots

The exact current identity-cost driver is `bench/lanes_price_main.mojo`:

```sh
MOJOLEARN_LANES_PRICE_LANES='knn gemv nt gram' \
MOJOLEARN_LANES_PRICE_ROUNDS=5 \
MOJOLEARN_LANES_PRICE_KNN_INDEX=100000 \
MOJOLEARN_LANES_PRICE_GEMV_DIM=2048 \
MOJOLEARN_LANES_PRICE_NT_ROWS=65536 \
MOJOLEARN_LANES_PRICE_GRAM_ROWS=4000000 \
MOJOLEARN_LANES_PRICE_KEEP_BIN=1 tools/lanes_price.sh
```

This is executable **two-arm identity-cost** coverage, not yet an admitted
three-arm external race. Its kNN fixture is 100,000 index rows, 1,000 query
rows, 32 features, k=10. Its Gram fixture needs about 1.54 GB for three
4,000,000-by-32 buffers; it is modest in GPU memory but avoid laptop runs.
NT is 65,536-by-64 times 64-by-64 transposed; GEMV is square 2,048.

An external driver must reproduce `_price_u01` (kNN salts 3 and 5) and
`_linalg_val`/`_gemm_mix` (Gram salt 11, NT 21/22, GEMV 31/32), or consume
raw fixture dumps with hashes. These generators are **not** the generators
in existing `speed_cuml_arm.py` or `speed_gemm_arm.py`.

Use cuVS `brute_force` with squared Euclidean metric for kNN; cuML brute
NearestNeighbors is an acceptable explicitly labelled alternative that
returns rooted distances. Build the index and upload queries before timing
search. Compare neighbor sets/indices, tie handling and distance tolerance
outside timing. Do not demand cross-library distance bits.

Use PyTorch CUDA/cuBLAS strict FP32 for `A @ v`, `A @ B.T`, and `X.T @ X`,
with operands already on device. Add TF32 as a separate labelled arm if
desired, never silently substitute it. Compare finite outputs and relative
norm/max errors outside timing; use float64 on the GPU for a separate
reference where practical. Output allocation policy must be explicit.

**Existing harness gaps:** `lanes_price.sh` interleaves only two arms;
`speed_gemm_arm.py` generates torch random operands and therefore cannot
be paired with the identity-cost fixture unchanged; `speed_cuml_arm.py`
has a different kNN fixture and emits no accuracy checks. Do not concatenate
these logs into a purported same-input three-arm result. Add external
invocations to a dedicated orchestrator using the retained ready binaries.

## Second batch: sequence forward

`bench/speed/seq_speed_main.mojo` and `tools/speed_torch_seq.py` already
share shape definitions, weight witnesses and output-dump agreement checks.
Start with row 2 (Llama prefill t8) and row 8 (Mamba130m prefill t8); then
row 3/9 (t128) if affordable. Rows 0/6 are only build/launch smoke.

For each prepared FAST/IDENTICAL binary, use:

```sh
MOJOLEARN_SPEED_LANE=transformer MOJOLEARN_SPEED_ROW=2 \
MOJOLEARN_SPEED_ROUNDS=1 MOJOLEARN_SPEED_DUMP_DIR=/tmp/seq-identical \
  /path/to/prebuilt-sequence-binary
python3 tools/speed_torch_seq.py --lane transformer --row 2 \
  --rounds 1 --warmups 1 --dump-dir /tmp/seq-identical
```

Repeat five interleaved rounds with separate per-mode dump directories.
For Mamba substitute lane `mamba`, row 8. For kernel attribution use
`attention`, `mlp`, `rmsnorm`, or `selective_scan` with the corresponding
family's row. Check compiled mode headers and all `FSPEED-WEIGHTS` records.
The external script names eager FP32, TF32 and forced SDPA backends, and
reports refusal for unavailable backends. Preserve refusals in results.

Install a compatible prebuilt `mamba-ssm`/`selective_scan_cuda`. If the arm
says `torch-ref-scan-gpu`, it is a Python-loop correctness reference and
does not establish competitiveness with production Mamba CUDA kernels.

**Scope gap:** these speed drivers cover Mamba **1 forward**, not Mamba2,
Mamba3, or any backward path. The existing Mamba1/2/3 backward certificates
and `tools/mamba_gradient_oracle.py` are correctness evidence, not timings.
The latter uses float64 PyTorch reference transcriptions plus finite
differences; it must not be relabelled as native vendor throughput.
Fresh Mamba2/3 full-block and backward comparisons need explicit upstream
CUDA/Triton adapters, matching weight layouts, objectives, state gradients
and shapes. Cache setup and gradient reset belong outside timing.

## Third batch: UMAP

No existing admitted full-estimator three-arm UMAP timing harness was found.
`umap/bench/optimizer_bench.mojo` runs one cold 2,048-node optimizer call
per arm without excluded warm-ups, round distributions, or external arm;
it is not a publishable end-to-end UMAP benchmark.

Start a new fit_transform benchmark at 256 and 1,024 dense float32 rows,
16 features, n_neighbors=15, two components, spectral init, Euclidean
metric, random_state=19 and explicitly fixed n_epochs=50, min_dist=0.1,
spread=1. Match cuML parameters; force brute-force graph construction where
supported and record the actual version/settings. Existing Mojo dense
quadratic graph storage makes a 100,000-row UMAP test inappropriate.
Time complete estimator fit_transform in all three arms, with the same
host/device input policy and explicit synchronization. Compare finiteness,
trustworthiness and neighborhood retention outside timing; coordinates
need not match cuML's random stream or orientation. Separately certify
Mojo IDENTICAL output bytes across devices on named fixed fixtures.

## Reporting

For every admitted row report absolute milliseconds and both
IDENTICAL/FAST and IDENTICAL/external (plus FAST/external), with precision
and scope in its label. Do not extrapolate a slow primitive into a whole
PCA/regression/UMAP slowdown. State explicitly which Mamba generations,
forward/backward surfaces, sizes and installed-API paths remain unmeasured.

## Implemented UMAP adapter (source only, not executed)

```sh
python tools/nvidia_public_compare.py --lane umap --umap-rows 256 \
  --umap-epochs 50 --rounds 7 --out /tmp/nvidia-umap256
```

Uses the same fixed 16-dimensional Swiss-roll projection in all arms,
host-input fit_transform including estimator construction and host output,
2D spectral initialization, seed 19 and explicit epochs. The cuML arm uses
`build_algo='brute_force_knn'` and `force_serial_epochs=False`; it is a
parallel-epoch production-speed comparator with no bitwise repeatability
promise. These are documented cuML parameters:
https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.manifold.umap/

Coordinates are saved and hashed; IDENTICAL must repeat its own bytes.
Separate post-timing trustworthiness and k-neighbor retention scores use
k=10. Admission requires trustworthiness >=0.85 for every arm/round and
Mojo no more than 0.05 below cuML. Both thresholds are explicit CLI knobs;
failure retains raw results and refuses a performance conclusion. Fixture
size is bounded to 32..1024 rows to cap quadratic quality-scoring CPU work.

## Additional sequence audit findings: do not quote unqualified ratios

The same-input design is present: shape constants are parsed from Mojo,
the two sides use the same tensor IDs/seeds/ranges, and the Python generator
imports the corpus hash generator. However, emitted witnesses sample only
up to 4096 cells and `FSPEED-AGREE` reports error without enforcing a bound.
An admitting runner still needs to verify every expected witness and apply
an explicit finite-output/error gate; matching sampled hashes alone is not
a full-input byte certificate.

Two concrete defects prevent blindly running the existing sequence script:

1. `undeterminize()` sets torch threads to `os.cpu_count()` after corpus
   imports, overriding the intended CPU cap. A bounded runner must replace
   that with at most two threads before invoking the script's `main()`.
2. In `run_mamba_row`, the `mamba-ssm-cuda` **full-block** arm computes
   `mamba_prefix` outside the clock and times only scan, output projection,
   and residual. Mojo times the complete block. Its fused full-block ratio
   is therefore invalid until the prefix moves inside the timed closure.
   The reference full-block arm calls the full corpus block, but is clearly
   labelled `torch-ref-scan-gpu` and is not a production-speed comparator.

The `selective_scan` lane has matching intended work (prefix outside both
clocks), but scan input tensors are generated through each implementation's
prefix and may differ numerically. To meet strict same-input admission,
export Mojo scan input buffers and feed those exact bytes to the external
scan. Until then classify it as a composed-operand comparison, not an exact
same-input primitive race.

Transformer eager-FP32 full-block comparison is the strongest existing
candidate: same generator and complete block on both sides, plus optional
small float64 corpus cross-check. The earlier row-2 commands are executable
qualification commands, **not automatically admitted measurements**. Keep
the CPU cap, validate witnesses and errors, and interleave the three arms.
No Mamba2/3 or backward production comparator is added by this adapter.

### September 5 source fixes now applied, awaiting main-lane execution

The concrete full-block prefix and CPU-thread defects described above are
fixed in `tools/speed_torch_seq.py`. It now gates finite output agreement
with configurable `--agree-rtol`/`--agree-atol` and exits 2 on admission
failure. `--mojo-log` is repeatable and required for successful admission;
it checks all input witnesses found in those Mojo logs. A further witness
bug was corrected: Python previously discarded the length-prefixed FNV
state before folding the sampled tensor, contrary to the Mojo algorithm.
It also emits separate full SHA256 hashes for tensors up to 16 MB; these
do not masquerade as corresponding full hashes from the Mojo driver.

Given prebuilt binaries and freshly captured mode-specific Mojo output:

```sh
MOJOLEARN_CPU_THREADS=2 python tools/speed_torch_seq.py \
  --lane transformer --row 2 --rounds 1 --warmups 1 --no-bf16 \
  --dump-dir /tmp/seq-identical \
  --mojo-log /tmp/transformer-fast.log --mojo-log /tmp/transformer-identical.log

MOJOLEARN_CPU_THREADS=2 python tools/speed_torch_seq.py \
  --lane mamba --row 8 --rounds 1 --warmups 1 --require-fused \
  --dump-dir /tmp/seq-identical \
  --mojo-log /tmp/mamba-fast.log --mojo-log /tmp/mamba-identical.log
```

These are one leg each; orchestrate seven rotating three-arm rounds for
timing admission. `--require-fused` ensures a missing/broken production
Mamba CUDA implementation cannot be mistaken for a valid comparison with
the Python reference. Run separately against the FAST dump as well to
qualify both outputs. Primitive selective-scan operands still need common
stage dumps for strict same-input admission; that is not fixed here.
