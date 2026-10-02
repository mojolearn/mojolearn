# neighbors (isotonic-knn + neighbors2)
Two worktrees: ~/mojolearn-wt/isotonic-knn (lane/apple-fast-isotonic-knn: isotonic, tiled kNN MMA route for lof/label-prop, lle; conflicts in python/mojolearn/_expansion_decomp.py, x_neighbors/iter_device.mojo) and ~/mojolearn-wt/neighbors2 (lane/apple-fast-neighbors2: kNN via MMA, pagerank reductions, svgp device solve; conflicts in bindings/_mojolearn_x_neighbors.mojo, bindings/_mojolearn_x_neighbors_host.mojo, python/mojolearn/_surface_neighbors.py, x_neighbors/iter_device.mojo).
1. In isotonic-knn: `git merge origin/main`, resolve, commit, push.
2. In neighbors2: `git merge origin/main`, resolve, commit, push. Its svgp switch is an env switch (MOJOLEARN_SVGP_FAST_GPU=1): convert it to a -D define and change its ab line to the afc_ab_def.sh form.
3. Settle the overlap: `git merge-tree --write-tree lane/apple-fast-isotonic-knn lane/apple-fast-neighbors2`; if it conflicts, in neighbors2 `git merge lane/apple-fast-isotonic-knn` and resolve. Push.
4. Check both ab files follow the light form.
