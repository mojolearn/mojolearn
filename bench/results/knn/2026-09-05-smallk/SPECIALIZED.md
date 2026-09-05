# Experimental specialized small-k selector, 2026-09-05

This is not the production dispatch and is not an installed-wheel qualification.
The explicit K=8/10/16 specialization preserves the existing composite-key order
and copies selected float bits. Other K values use the runtime candidate.

Main-lane Apple M4 and NVIDIA RTX 4090 runs passed all 72 small fixtures.
The 1,848 retained shared CELL lines match exactly across these devices.
The NVIDIA run additionally passed nine large fixtures (32 queries, 100,000
candidates; K=8/10/16; uniform, duplicate and clustered values), including
post-timing bit checks. Each timing arm has nine rotating samples.

For K=10, median selector milliseconds, legacy / specialized:

| Input profile | Legacy | Specialized |
| --- | ---: | ---: |
| Uniform | 3.525017 | 0.093570 |
| Duplicates | 8.161395 | 0.094440 |
| Clustered | 11.725492 | 0.094600 |

These are selection-component measurements, not end-to-end k-NN speedups.
No AMD qualification or production promotion is claimed.

The NVIDIA base campaign used commit
1d1f8fdd37022dad1c9e5752311296580e7762dd. These two supplemental source files
were copied afterward and matched the local files byte for byte when evidence
was reviewed. Their SHA-256 hashes identify the tested changes:

- `bench/knn_smallk_selection_main.mojo`: `ca67bc22e9f6587745e73d84949d4a3af94993775bcec8f1e8b56b90444ec917`
- `neighbors/checks/select_smallk_identical_candidate.mojo`: `51bfaa82bfb5de25ff360d50d7ee8eb79456603a23b5b37d10fabf236a9663c5`
