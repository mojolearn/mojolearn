# Lane: peer5-boost-iforest (Bagging/Ada/DART/RandomTreesEmbedding/register_boost lifecycle + IsolationForest)
Read COMMON.md. Worktrees: /Users/andrewhendel/CascadeProjects/mojolearn-tree-boost-lifecycle (lane/peer5-tree-boost-lifecycle-20261028
@ 6235849274, 9 dirty, 12 commits ahead) and /Users/andrewhendel/CascadeProjects/mojolearn-peer5-iforest (lane/peer5-iforest-lifecycle-20261028
@ 115b5eb8bb, 1 commit; "IForest full dispatcher compile still pending"). Read the lanes' handoffs under experiments/ (grep boost, bagging,
iforest, gbdt_parameter_native_20261028) and the root handoff Peer5 section. Evidence: ~/mojolearn-evidence/peer5-boost-iforest-20261028/.
Goal: finish the boost lifecycle draft (register_boost region currently still leaves Python lifecycle decorators; Bagging estimators_
property schema/native dispatch is additive in _native_public_attributes.py/public_attributes_native.mojo), commit, merge origin/main
(and green 77ed135648 GBDTAdapter + 0f0b336d87 GP if not yet in main), compile the tree dispatcher root(s) named in
experiments/gbdt_parameter_native_20261028/HANDOFF.txt and the tree batch receipts (rent a CPU box: the full tree dispatcher took ~430 s),
push a green head. Then the IForest full dispatcher compile; preserve the forest_release_gpu + gbdt_resident_release + iforest_resident_release
tuple and the gbdt/iforest elif registrations. Do not edit the explainer region of _expansion_trees (another lane owns it).
