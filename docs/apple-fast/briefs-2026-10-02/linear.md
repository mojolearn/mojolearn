# linear
Worktree ~/mojolearn-wt/linear, branch lane/apple-fast-linear (lasso, elasticnet, logreg, linearsvc, linearsvr, gmm; conflicts in bindings/_mojolearn_solver.mojo, glm/impl/qn/glm_base.mojo, python/mojolearn/_solver_impl.py).
1. `git merge origin/main`, resolve (main's IDENTICAL code and host-route removals win; note main already has LassoCV/ElasticNetCV grid work on the device: do not reintroduce anything main removed). Commit, push.
2. Check docs/apple-fast/ab/linear.txt follows the light form. Env switches on hot paths become -D defines.
