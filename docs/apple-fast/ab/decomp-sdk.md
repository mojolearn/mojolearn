# SDK decomp control — WIP ownership-transfer checkpoint

UNCOMPILED, no quality helper or binding yet. Do not queue, merge or promote.
Base a78f0ab5e4f63f32029e7b212fc96cdf7784a524. Work intentionally stopped on
2026-10-04 for manager/thread ownership transfer.

Default-OFF `MOJOLEARN_DECOMP_SDK_NNNT`, FAST Apple only; audit flag
`MOJOLEARN_DECOMP_SDK_AUDIT`. SDK_INTERPOSE is either flag. Source written:

- `experiments/apple_fast/gemm/decomp_sdk.mojo`: NN/NT DeviceBuffer→TileTensor
  SDK matmul, refuses N1, empty K, transpose A, caller's split plan and direct
  output alias. No allocation, upload, transpose materialization or sync.
- `x_decomp/device.mojo::launch_gemm_buffers`: calculates incumbent split
  eligibility (64x64 tile occupancy) before SDK selection, then old raw-pointer
  launch fallback. DevExec.gemm uses wrapper only under interpose flag.
- `x_decomp/resident.mojo::pool_gemm`: validates existing pool IDs/capacities,
  takes buffer copies retaining the same allocations, uses wrapper; original
  raw-pointer route when default OFF. dev_gemm_py now calls pool_gemm.
- `x_decomp/kit_device.mojo::mm`: zero-offset DMat inputs use pool_gemm only
  under flag; row views preserve existing raw-pointer route. No pointer-based
  ownership wrapping. Raw-pointer-only Lanczos N1 calls stay unchanged.

Existing decomp AFN default d2fe2e056 was measured against per-cell kit GEMM,
not SDK: RSVD istella711→533ms, same reconstruction, taxi−1.3%; later7bb6250dd
NMF8155→6333ms. SDK screen G0 rank cannot establish SDK versus AFN. In
particular square1024³ currently splits in decomp, so this adapter properly
refuses it. Square non-split shapes need a distinct actual-AFN comparison.

Remaining work: review source and compile API/lifetimes, add actual-wrapper
quality binding using one process_ctx and exact manifests/hashes, independent
FP64 zero-regression gate, actual route counters including fallback shapes,
NN/NT alias and offsets/pool reuse controls, split-boundary639/640 tiles,
then M2 build and M3 quality only. Add downstream actual resident/Kit checks
before any speed/default claim. No local runtime or cloud operations performed.
