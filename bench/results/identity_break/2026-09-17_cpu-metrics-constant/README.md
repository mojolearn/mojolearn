# Native H/C/V controls, including the constant-label branch

Fresh Mac production and sabotage metrics bindings, source `b9902bb46`, one
compiler/numerical worker. All nine unchanged fixtures, two repetitions per
arm. Every training control moves. The first Linux batch at `f0d53b4ed` had
moved eight fixtures but not `negative`: its all-negative inputs produce a
constant label partition, so changing one label within its range does nothing.
The sabotage define now also makes the zero-entropy homogeneity branch return
0 instead of its correct 1. This is compiled out of production.

Clean hashes match the first Linux clean column. The columns record full source,
input/protocol and native-binding digests. The core buffer helper is shared
between arms. This proves sensitivity to the listed native faults, not every
possible metric defect, and does not qualify a final release wheel.
