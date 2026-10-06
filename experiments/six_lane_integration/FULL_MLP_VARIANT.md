# Full MLP estimator registration

`mlp-full-v1` adds four saved full-estimator workloads for native NVIDIA's
`I.X.complete-proposed`: MLPClassifier and MLPRegressor, each on Taxi and Istella.
The original matrix did not contain these estimator recipes. Its retained MLP
training-step cell `19744574fa8682fda571` is a candidate-scope link only. The new
IDs begin `expanded:mlp-clf` or `expanded:mlp-reg`; they do not relabel or complete
that neural fixture, any independent idea, or the complete campaign.

The saved caller is `tools/bench_board_algos.py`, lanes `mlp-clf` and `mlp-reg`.
The reviewed contract pins source, exact full archive and sidecar hashes, all
array dimensions, constructor settings, seed, output scope and timing boundary.
Hidden layers remain (256,256), Adam, batch4096, five epochs, seed7 and shuffle.
There is no new dataset preparation and no intrinsic lane cap. Classifier Taxi
uses the original credit-card population (4,110,786 training rows); regressor
Taxi uses all5,250,086 saved regression rows. Istella uses2,043,304 training rows;
every case consumes all500,000 saved query outputs. This is a new full-input
variant of the saved estimator recipe, not an unchanged historical capped race.

Run `tools/six_lane_register_full_mlp.py` with `--cls-directory`,
`--reg-directory`, `--deployments` and a fresh external `--output`. Admission
requires a clean reviewed source freeze and both retained core and x_sequence
bindings in each package. The controller still requires explicit authorization.
One excluded warmup and one scored execution per arm are required. No compilation,
product preflight or numerical verification is part of registration.

A is the complete proposed candidate, B the incumbent. Retained candidate
x_sequence enables IDN_MLP_DEVICE_EPOCH_ORDER and NI18_FUSED_EPOCH_GATHER;
this is not the separate NI52 no-op alias. Source-compatible retained binaries
must pass existing exact source/flags/compiler/target/artifact admission.
Apple FAST has no accepted retained x_sequence pair and remains blocked.

Outputs are full float64 predictions and, for classification, all positive-class
probabilities. Typed fitted-state coverage is reported separately; optimizer,
RNG and complete-model identity remain unproven where the existing capture says
so. Timing or output hashes alone cannot qualify/promote a candidate. Historical
Python product wrappers also remain noncompliant with the current no-product-
Python rule; this registration does not claim runtime migration.
