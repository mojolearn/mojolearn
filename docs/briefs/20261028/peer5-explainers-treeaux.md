# Lane: peer5-explainers-treeaux (Tree/Kernel/Permutation explainers + remaining ensemble lifecycle)
Read COMMON.md. Worktrees: /Users/andrewhendel/CascadeProjects/mojolearn-explainers (lane/peer5-explainers-20261028 @ 20ab53ec02, 3 dirty,
nothing committed) and /Users/andrewhendel/CascadeProjects/mojolearn-peer5-trees (lane/peer5-tree-aux-20261027 @ 77ed135648 green, 7 dirty
files = continuation beyond the green GBDTAdapter head). Read experiments/gbdt_parameter_native_20261028/HANDOFF.txt, the tree batch
receipts and the root handoff Peer5 section. Evidence: ~/mojolearn-evidence/peer5-explainers-treeaux-20261028/.
Goal: (1) explainers: TreeExplainer/KernelExplainer/PermutationExplainer region of _expansion_trees owns validation/preparation/execution/
results natively; commit the draft in steps, merge origin/main, compile the affected root, push green. (2) tree-aux: finish the 7 dirty
files' remaining ensemble lifecycle, commit, merge origin/main, compile, push green. Do not edit the register_boost/Bagging region
(peer5-boost lane owns it) beyond conflict resolution. Preserve the CPU review's exact GBDT host implementation files.
