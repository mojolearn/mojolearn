# Transformer direct caller transfers

IDENTICAL uploads/downloads now copy directly between live NumPy buffers and
DeviceContext, synchronizing before caller lifetimes end. Other numeric
modes retain pinned staging. `MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER`
is the diagnostic fallback. No model arithmetic, comparator or tolerance
changed. Production default source: f31508bb; initial opt-in source: e944add4.

The reusable tools/transformer_transfer_check.py compares 82 full-array
SHA256 records across separate builds: fresh outputs, chunked prefill,
decode, linear and ring caches including unused bytes, and every backward
gradient, for HD64/HD128. All match on Apple and NVIDIA. Both public surface
and HD128 suites passed for the initial candidate; final default bindings
were rebuilt and matched the same 82 records. The existing generic build
gate is skipped by the IDENTICAL build script; explicit checks are recorded.

H100 80GB HBM3, driver580.126.09, Mojo1.0.0(ed45d567), Python3.11.10,
NumPy1.26.3. The original large seed-7 public harness uses one logged warmup
and seven timed rounds. Timing is host-to-host, including weight checks,
all transfers, and the complete output/cache return. No opponent was timed.

| Order | Narrow legacy/candidate ms | Wide legacy/candidate ms |
|---|---:|---:|
| Legacy then opt-in candidate |265.319614 /245.827597|261.891093 /246.565180|
| Final default then legacy |255.330876 /245.223133|260.435883 /245.318148|

Both large outputs contain 16,777,216 Float32 cells; full raw SHA256 matches
between all builds/orders. Hashes, raw samples, compiler/runtime and source
hashes are under h100/transformer-transfer/. The reversed-order pair confirms
approximately4%/6% less request time; the initial narrow baseline was noisy.
This does not qualify a Torch performance ratio: original admission still
fails, as documented in transformer_admission_2026-09-10.

Apple runs alternate candidate/legacy order at B2/L512 with HD64 and HD128,
two warmups and five rounds. HD64 pairs consistently improve by about17%;
HD128 is variable, ranging from a small regression to a win. The final
production build has no experimental enable flag and matches all 82 arrays.
These are host request times on a shared laptop, not a universal speed claim.

The first H100 wrapper stopped before compilation because git archive had
excluded the archived fixture helpers. The exact matched helper pair was
copied explicitly; successful r2 and final wrappers/logs are retained.
The original r2 script applies to e944add4; on current source reproduce its
legacy arm with -D MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER=1 and its
candidate arm with no transfer define. The final wrapper uses the previously
built legacy binary to reverse timing order. Binary and full tensor dumps
are omitted from this archive; their complete hashes and generators remain.
