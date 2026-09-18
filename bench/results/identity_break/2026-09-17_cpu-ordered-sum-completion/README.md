# Ordered gradient sum native control

Fresh Mac training production and sabotage builds at `0d1d1c29b`, all nine
fixtures twice. The original Linux control was inert because pairwise add had
no native fault. The new sabotage-only fault zeros the first accumulated value.
All nine wrong results are caught by the independent Float32 left-fold oracle.
The harness records repeated measured output hashes as DIVERGENT and exits one;
it does not turn the assertion into a STABLE reference or accept an arbitrary
exception. All nine clean results pass. Other loaded bindings are reused Mac
bindings; binding hashes and sabotage readback are recorded in each arm.
No production accumulation arithmetic changed.
