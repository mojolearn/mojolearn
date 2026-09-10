# Mamba3 fresh-prefill continuation — September 10

Accepted IDENTICAL defaults remove discarded host state transfers for stateless prefill and copy directly between live NumPy buffers and device buffers. Full numerical execution and reports are preserved; explicit state/decode/continuation, mutable-weight refusals and non-IDENTICAL paths retain their contracts.

Final H100 public medians are 70.828952ms narrow /124.409189ms wide, down 18.8%/25.1% from same-pod baseline. Reused admitted torch ratios are 4.32×/6.42×; the 1–1.5× target remains unmet. No new opponent measurement was taken. Root owns the opponent ledger and pod cleanup.

Evidence, source manifests, all timing samples, full-output SHA witnesses and reproduction scripts: [fresh-prefill artifact](../../bench/results/mamba3/2026-09-10-fresh/README.md). NVIDIA final default passed 39,087,232 y/report bit comparisons, mutable-weight refusals and 102 surface checks. Apple combined candidate and rebuilt final integrated default both passed 146,560 cells, mutable refusals and 102 surface checks.

Source commits: f1704718,2396fa6e,c6806c87,8a14ff97,bb98a6cd (parent cherry-picked). Final default uses pristine 5011239a GEMM, with no accepted GEMM experiment. Legacy switches allow isolated regression comparisons. The next performance work should attribute the remaining complete device block; do not bypass any state/refusal/report semantics to improve the public price.
