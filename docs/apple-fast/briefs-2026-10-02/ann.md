# ann
Worktree ~/mojolearn-wt/ann, branch lane/apple-fast-ann (tsne, cagra, ivf-*; conflicts in x_ann/knn_device.mojo, x_ann/tsne_device.mojo).
1. `git merge origin/main`, resolve (main wins; FAST switches re-expressed on top). Commit, push.
2. Check docs/apple-fast/ab/ann.txt against the light A/B form (1 2, no -ident lines, one tag per lane x dataset, one dataset first); fix the lines if they loop or use -ident. Any env switch on a hot path becomes a -D define (then the line uses afc_ab_def.sh).
