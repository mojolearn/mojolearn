# Hot Aisle MI300X broad identity gate

This compact receipt records the post-promotion AMD `gfx942` identity gate on
source `38cd6ce6cc84546a27b54e4aea3f80f74590a407`. The full raw output remains
under `/Users/andrewhendel/mojolearn-evidence/2026-09-21_knn_amd_smem/hotaisle-mi300x-identity-r2/`.

`identity-summary.json` records the complete broad matrices and hashes their
raw JSON. Both default-vs-CPU and narrow default-vs-sabotage comparisons exited
zero over 22 lanes, nine fixtures, two repeats, and 198 cells. The three
`width-*.json` files retain every full-output digest for 36 wide cases.
`verdict.json` applies the corrected reach set to those original files.

The on-box judge originally required `radius-d32` to react to the shared-tile
sabotage. That was an admission error: `RadiusNeighbors` uses the independent
ball-cover count/fill implementation. `verdict-uncorrected.json` preserves that
failed judge output. The corrected reach set comprises the 14 wide
Euclidean/squared-Euclidean nearest-neighbor, classifier, and regressor cases
that traverse the promoted brute-force tile; all 14 moved under sabotage.
Radius, KDE, and the non-L2 metrics remain collateral default-vs-CPU identity
checks.

`stage.log` records hash-verified Cloudflare R2 staging. The identity body uses
fixed generated inputs, so the staged datasets were not read by this gate.
`teardown.txt` records HTTP 204 deletion followed by GET 404 and absence from
the provider listing.
