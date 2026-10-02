# prep (prep + prep2)
Two worktrees: ~/mojolearn-wt/prep (lane/apple-fast-prep: minmax-scaler, onehot, ordinal, multilabel-binarizer; conflicts in bindings/_mojolearn_preprocessing.mojo, preprocessing/estimator.mojo, python/mojolearn/_expansion_prep.py, python/mojolearn/preprocessing.py) and ~/mojolearn-wt/prep2 (lane/apple-fast-prep2: target-encoder, simple-imputer, robust-scaler, iterative-imputer, eigh on a 32-thread block; conflicts in python/mojolearn/_expansion_prep.py).
1. In prep: `git merge origin/main`, resolve, commit, push.
2. In prep2: `git merge origin/main`, resolve, commit, push. Its eigh switch is an env switch (MOJOLEARN_X_PREP_FAST_EIGH_BLOCK): convert to a -D define and switch its ab line to afc_ab_def.sh.
3. Settle the overlap: `git merge-tree --write-tree lane/apple-fast-prep lane/apple-fast-prep2`; if it conflicts, in prep2 `git merge lane/apple-fast-prep` and resolve. Push.
4. Both ab files in the light form; Python-side changes must not add NumPy/sklearn math on a FAST path.
