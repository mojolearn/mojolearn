# peer5-boost-iforest HANDOFF (2026-10-28)

Integrate: lane/peer5-iforest-lifecycle-20261028 @ e667e684ee (superset: IForest + boost + main + 77ed135648 + 0f0b336d87).
Boost alone: lane/peer5-tree-boost-lifecycle-20261028 @ 4bc490b6f3.

Changes
- Boost/Bagging/Ada/DART/RTE public doors -> trees_public_boost_* (python/mojolearn/_expansion_trees.py), native owner providers/python/tree_boost_public_lifecycle_pyapi.mojo, admission bindings/trees_boost_admission.mojo, dispatch bindings/trees_public_native.mojo.
- Bagging estimators_ get/set/slice native: bindings/trees_bagging_members_native.mojo; schema 'bagging_estimators' in _native_public_attributes.py:161 and public_attributes_native.mojo _property.
- Python lifecycle decorators: the 7 boost classes declare `_mojolearn_native_public = True` in class bodies (no Python wrappers installed); the draft's register_boost unwrap loop removed, register_boost == main.
- IForest branch: merges resolved in _native_trees_bridge.py:151 (release tuple forest_release_gpu+gbdt_resident_release+iforest_resident_release) and :202/:206 (gbdt and iforest elifs kept); trees_public_native.mojo:15-16 both imports kept.

Compile receipts (CPU column, identical, O0, x86-64-v3, -j 32, RunPod CPU; compile only)
- boost: experiments/tree_boost_lifecycle_20261028/compile.json, exit 0, 54 s, entry checks/tree_boost_public_compile.mojo (full trees_public_operation + attribute lookup), pod 2gwq69lclpm0f6 reaped/verified 404.
- combined: experiments/iforest_public_lifecycle_20261028/full_dispatcher_compile.json, iforest_public_boundary_compile exit 0 88 s, tree_boost_public_compile exit 0 89 s, pod wrcr0qphvj32vs reaped/verified 404.
- Logs/binaries: ~/mojolearn-evidence/peer5-boost-iforest-20261028/pod-boost, pod-iforest.

Not covered / pending
- No runtime, import, identity, GPU (NVIDIA/AMD/Apple targets) or FAST-define compile. Python edits are source-only (ast parse).
- Not merged to main (orchestrator). Explainer region of _expansion_trees untouched.
- IForest still unwraps decorators at register time in _connect_public_iforest (peer design, left as is).
