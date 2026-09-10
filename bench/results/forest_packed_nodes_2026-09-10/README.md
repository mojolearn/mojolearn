# Packed resident forest: initial correctness evidence

September 10, 2026. Experimental RF/ET GPU inference layout, not a speed result.
The default remains separate arrays. Enable the candidate in diagnostic builds
with `-D MOJOLEARN_FOREST_PACKED_NODES=1`.

Source: nvForest v26.08.00 `cef3a50da0f74b0015876b9d6d424c86141898dc`,
`cpp/include/nvforest/detail/node.hpp:81–175`, `evaluate_tree.hpp:44–65`,
`decision_forest_builder.hpp:135–149`. Packed fields and compact leaf vectors
follow these sources. The candidate retains MojoLearn's sibling order, local
child indices, inclusive comparison and fixed grove reduction; it is not a
literal port of nvForest's default depth-first layout or numerical topology.

Run `pixi run check-forest-resident-layouts` to compile and execute four named
small correctness arms: separate/packed × FAST/IDENTICAL. All passed on Metal.
The retained run logs cover RF/ET, outputs 1/2/3/5/8/9, a ragged 33-tree forest,
root-only leaves, five-row tails, equality/subnormal features, complete output
bits against direct-layout GPU inference, explicit cached/uncached I/O,
resize/empty/repeat/lifecycle and validation errors. The packed arm also poisons
its compact device leaf buffer and requires a changed answer.

Both candidate FAST extensions were built separately under `/tmp`, without
replacing installed bindings. `check_binding.py <extension-directory>` passed
layout readback and exact split outputs through staged, borrowed and reused I/O
for both RF and ET. The small fixtures are test oracles, not a CPU learner.
`source-sha256.json` records the tested source. Existing compiler warnings remain.

This establishes neither NVIDIA/HIP correctness nor cross-vendor identity, and
contains no large-data timing or promotion decision. Node padding can offset
leaf compaction, especially in regression. Next run NVIDIA IDENTICAL checks and
a same-process resident-layout A/B with identical I/O handling on large
HIGGS/Year/Covtype; separately measure preparation and memory. The existing
transient/resident benchmark now records layout but confounds layout with model
upload, so it cannot alone qualify this candidate. No rental was used here.
