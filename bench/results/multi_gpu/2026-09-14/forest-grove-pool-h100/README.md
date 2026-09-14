# Resident RF/ExtraTrees grove pooling — two H100s

The production, lifecycle, public and injected-failure gates pass on RunPod
`smqlvlvt7exixd`, two H100 80GB HBM3 GPUs, driver 580.126.09, Mojo/MAX 26.5,
2026-09-14. Both separate-array and packed-node layouts were built and checked.
No local builds or tests ran.

The pool owns complete logical groves: grove g evaluates trees g,g+32,... .
Only canonical unaveraged totals return to the first owner for the original
16/8/4/2/1 fold and global-tree-count division. The root has no complete model
buffers. Production scratch is bounded to 4096 row/output cells times 32 lanes;
full host model/output staging and replicated query tiles remain. Whole groves
must fit their owners and existing integer bounds remain. Owners execute
sequentially. This is resident model ownership qualification, not a measured
beyond-one-GPU capacity run or throughput improvement. The reference contract
is single-GPU `parallel_groves`; legacy sequential prediction runs on the host
and has a different association.

## Gates

- Each layout passes 74 native fixtures: RF/ET traversal policies, tree counts
  1/3/31/32/33/65/97, output widths 1/2/3/8/9, plus 129/4097-output tiling cases.
  They compare all sampled intermediate grove totals and final prediction
  bytes, verify complete nonduplicated tree ownership and absent root model
  allocations, and exercise caller-buffer prediction.
- Separate fault builds pass those 74 fixtures with post-contribution failures
  on every active owner, unchanged output canaries and exact recovery.
- Two additional RF/ET fixtures in each layout place [2**26,1,-2**26,1] in
  grove 0's trees 0/32/64/96. Every sampled grove-0 result must have the exact
  bits of 1, independently of reference agreement. These supplement the
  original fixtures because their values did not distinguish reordering
  within a grove strongly enough. Default native-gate coverage is now 76.
- Existing resident-model layout/lifecycle gates pass in both layouts,
  including scalar/vector widths, borrowed/List/transient paths, workspace
  reuse, release, stale handles and refusal cases.
- Sixteen public fits per layout cover all four RF/ExtraTrees estimators,
  tree-count boundaries, 2/3/8/9 classes, string/int64/float labels, original
  FP32/FP64 output precision, repeated/ragged/empty queries, reversed devices,
  one-device adapters, frozen snapshots, closure and refusal behavior.
  The complete packed/separate JSON reports agree.
- MLP and clipped attention/dropout Samba replay JSONs remain unchanged after
  generic workers release incidental RPC request/response storage while idle.

## Source and receipts

`out/` retains build/gate logs, complete reports, binary/hardware/corpus hashes,
and exact job scripts/return codes. `out/comparison.log` records successful
cross-layout and neural comparisons. The production snapshot is
`out/forest-grove-final-source.tgz`, SHA256
`4a5e846d86c2cf43e4c280d55692e6ad5a4d659884fe929220e7b0613baa0ce9`.
The supplemental gate is `out/order-check.mojo`, SHA256
`38d25eb0d4518e2fb9c62299653eef9f756296aa959c678cb61fc1857eef84d8`.
It changes only test fixtures; implementation and production binaries stayed
unchanged. The separate supplemental logs record its two cases per layout.

The initial native test failed parsing a reserved `out` parameter name. Its
source archive and failed log are preserved. The corrected source also pins
upload/staging buffers through exception-path synchronization. All later
qualification passed; no failing numerical run was discarded.

The H100 pod remains leased for an eight-logical-shard replay reference while
a new RTX 5090 pod is prepared. Cross-architecture qualification and termination
are recorded separately; neither is claimed by this intermediate receipt.
