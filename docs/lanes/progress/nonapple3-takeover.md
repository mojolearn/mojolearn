# Non-Apple3 takeover review, 2026-09-28

Integration base: `243f74dc79b99b9457fe89dda557aeb843c62012`.
Merged Python consolidation: `2db8a8f36907848d8ad86f8a935a417d5f06b38a`.
This contains all 12 Python lanes, including py-shared, and the final Apple2
changes. Every branch below ending in `-apple` or `-apple2` is an ancestor
of the base or the Python consolidation; no additional final tip is missing.
Apple3 is integrated separately before qualification.

## Resolution and preservation

The prep merge retains the current mandatory-entry failure and optional-export
ImportError handling, extending it to the new optional ranges entry. Both the
historical decomp performance evidence and newer consolidated repair notes
remain. Reference tables, verifier revisions, applicability and admission code
are unchanged by this merge. Existing dirty worktrees are untouched.

The uncommitted `merged-nbmetrics-fix/core/knn_host_predict.mojo` patch is
superseded by the base: `_host_knn_block_rows` already routes each of the five
extra metrics through `host_metric_cell_ptr` and the common block selection.
The old separate row helper is therefore not imported.

## Evidence and outstanding qualification

`py-consolidated.md` and the 12 family progress notes retain precise source
commits, previous measurements and pending work. Prior successful CNN and
model-selection before/after runs cover those revisions only. The consolidated
Python verdict was pending at its final recorded tip; Apple2 timing evidence
is historical. No new native builds, GPU jobs, cloud operations, or M2/M3
operations were performed by this takeover.

This merge changes native/Python numerical paths; the earlier main 445-lane
Metal/HIP report does not qualify this new source. Some Python lanes explicitly
change previous rounding or sampling semantics. Existing references must be
assessed against the combined source without concealing mismatches. Broad
native build, cross-vendor identity, applicable batch probes and targeted
performance validation remain for the final consolidated lane. Python AST
parsing and focused portable-math/range-helper tests are only local structural
and pure-Python checks, not native qualification.

Generated historical `.patch` fixtures retain their exact context whitespace;
`git diff --check` warnings in those patch data files are not source whitespace
edits. No arbitrary worktree cleanup or evidence deletion was performed.

## Frozen source tips

```text
lane/ann-apple 9bc0b74c996b5f06a7c83d12d530ed31bb6c0a68
lane/ann-apple2 96ea1b2106058107ac362839de514bd3a3df4769
lane/cluster-apple 98e554dd5336b56cca7f9fc4bfe3869d305701a3
lane/cluster-apple2 5ba7abc91afdd34415af97d51bf360ad2514db11
lane/cnn-apple ad54ba3e2483720b2611f574b744db550737733a
lane/cnn-apple2 f0782afb06421ada0f25532c1bf0b0aabfc47690
lane/decomp-apple 81e38e02c18ffd9bb7f6e7b7bfe9eb27c04157ff
lane/decomp-apple2 4752a6a3cd8cebf6c0382ddc4c7b98659a3cbea6
lane/linear-apple 03c28abbcf7551881d93a587ce497eaee5511cce
lane/linear-apple2 08335fd90c1f25f12077e2880bfc93f3f9977562
lane/metrics-apple 81f4dc54872c5b917dff3d0f31d551127f34a625
lane/metrics-apple2 31c790ba7da6093cc1e9d0be6c679add85f60a80
lane/neighbors-apple 3ce936ef4f2d37d76dc946c3523e572c76a3b8c9
lane/neighbors-apple2 75ddcae2e90c077f76d00828cfcbd6380b3fcdc9
lane/neural-apple 975e81e2cf68c7ab80db110f420eb939414b5b40
lane/neural-apple2 9a4d06a94a9738c0f40aae54cce31bf2fc1b46bc
lane/prep-apple 780b5f9c7673cf973f946213b82329f22ad015da
lane/prep-apple2 d71c6fac656f1df0f17a825e7203ed2ad2cd48e9
lane/py-bugs 4edf69335a8f7f1073f7fd807ce4140f0f35e66c
lane/py-consolidated 2db8a8f36907848d8ad86f8a935a417d5f06b38a
lane/py-decomp-nbrs aadfff6406a869e6de464425786dbc0486b5fa2f
lane/py-dn-ann 87d6d58753fa7c18d40b40b997fabf62e709c239
lane/py-dn-kern 7203261f6b2f9d40b0d9b26028af7a64f3634459
lane/py-dn-svm 6aaa86d031c6c5d3ba8101f62751381ac9084cc8
lane/py-lm 3f2da7fc020a5f5f8033134df06f526624fe92af
lane/py-misc 893120acfabd9154d9ef7d8ce720d0aeac933dd9
lane/py-misc-metrics 5a541c18bf78c5803f29a90d3f295ef4572661ea
lane/py-misc-msel e93c05c1160f80e17879091f8ad0cd3ac50c157c
lane/py-misc-prep 2af9eef2d77b3823532b9f8cb00345aa2348ae1d
lane/py-sequence 6f0267b28c0857a0edafe1fd8aa8e0916c44ec27
lane/py-shared bad280b0a01d012e3e9607874d74d5b236c8f732
lane/sequence-apple d408f41ee26a6768b1f26bc64a8f8515ac259416
lane/sequence-apple2 9d77a29d098967d8dcd6cd7037e8c721a382e332
lane/trees-apple b8b0b04e7bd1fc3841135f480a2776a202d2a637
lane/trees-apple2 8b382db2b1b6a50a5a6294c992ccb53d7102b135
```
