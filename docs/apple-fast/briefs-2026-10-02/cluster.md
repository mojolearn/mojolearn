# cluster (+ cluster2 overlap)
Two worktrees: ~/mojolearn-wt/cluster (lane/apple-fast-cluster: meanshift, minibatch-kmeans; conflicts in x_cluster/device_ops.mojo, host/host_ops.mojo, ops.mojo) and ~/mojolearn-wt/cluster2 (lane/apple-fast-cluster2: affinity-prop, bayesian-gmm, bisecting-kmeans, optics; conflicts in those plus bisect.mojo, optics.mojo).
1. In cluster: `git merge origin/main`, resolve, commit, push.
2. In cluster2: `git merge origin/main`, resolve, commit, push.
3. Settle the overlap: run `git merge-tree --write-tree lane/apple-fast-cluster lane/apple-fast-cluster2` (read-only). If it conflicts, in cluster2 do `git merge lane/apple-fast-cluster` and resolve so both families' switches coexist (cluster2 lands second). Push.
4. Check both ab/*.txt files follow the light form (1 2, no -ident, one tag per lane x dataset). Env switches on hot paths become -D defines.
