# Rejected GEMM fold capacity specialization

Baseline source5011239a, IDENTICAL, same dedicated H100580.126.09.
Candidate sized the per-thread fold stack to1,4,8or16slots according to
leaf count, preserving leaf boundaries and fold order. Seven Apple and
seven NVIDIA device gates passed. Six shape probe outputs passed.

The dense rows did not improve: QKV2.36066→2.36191ms,
MLPup9.20585→9.20692ms, MLPdown8.18322→8.20406ms.
These are the probe's seven-call mean timings, not medians. Each binary's
two printed arms are both its current dispatcher; compare separate
full-price/small-price files for the experiment. No cross-binary output
hash is claimed by those self-comparisons; independent device gates are
the numerical evidence. The patch was removed from production.

A subsequent existing64x64 tile probe also lost to the current128x128
plan: QKV3.135vs2.375ms, MLPup10.957vs9.217ms,
MLPdown10.995vs8.209ms. Its raw log is in the sibling
`../gemm-residual/64tile-price.log`. No dispatch change was retained.
