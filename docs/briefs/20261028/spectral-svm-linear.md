# Lane: spectral-svm-linear (spectral public lifecycle, then whole SVM lifecycle, then expansion-linear)
Read COMMON.md. Worktree ~/mojolearn-wt/spectral-public-lifecycle-20261028, branch lane/spectral-public-lifecycle-20261028 (@ 780cd217d6,
4 dirty, nothing committed). Read experiments/cluster_density_linalg_20261027/ (SVM_STATE_COEF.txt, ROOT_COMPACTION_HANDOFF under
~/mojolearn-evidence/cluster-density-linalg-20261027/) and the root handoff "Local /root/cluster_density_linalg" section: SVC/SVR,
expansion-linear and spectral classes still retain _mode validation/output wrappers and lack native public slots; the targeted SVM batch
7cacec39ba (green, being integrated) covers state/coef/probability-availability and SGD predict_proba, not the whole lifecycle; new native
family_svc_public state route precedes binding selection; public coef properties remain Python; classical parent owns shared
classical_labels.mojo label-kind/sorted/flatten/list protocol repair and SVM-local numbers.Real/Integral ABC callback removal.
Evidence: ~/mojolearn-evidence/spectral-svm-linear-20261028/.
Goal: (1) commit and finish the spectral public lifecycle draft (predict availability metadata, native slots, validation/output ownership),
merge origin/main + 7cacec39ba if not yet in main, compile the spectral/classical root, push green. (2) Whole SVM lifecycle: coef
properties and remaining wrappers native, classical_labels protocol repair, ABC callback removal; compile; push. (3) Start
expansion-linear as a separate commit series; deliver what compiles. Preserve Peer4 mode/slot work and Peer5 gp_label helpers.
