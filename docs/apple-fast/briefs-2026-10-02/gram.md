# gram (+ kernel overlap)
Two worktrees: ~/mojolearn-wt/gram (lane/apple-fast-gram: lars, lasso-lars, ridge-clf, ridge-cv, lda-clf, qda; conflicts in x_linear/device.mojo, lars.mojo, ridge.mojo) and ~/mojolearn-wt/kernel (lane/apple-fast-kernel: bayesian-ridge, ard, gpr, nystroem; conflicts in kernel_methods/estimator.mojo, x_linear/bayes.mojo, x_linear/device.mojo).
1. In gram: `git merge origin/main`, resolve, commit, push.
2. In kernel: `git merge origin/main`, resolve, commit, push. Note main rebuilt BayesianRidge on grid kernels (bayes_yy_parts_kernel, bayes_step_gram_kernel in x_linear/device.mojo): main's grid path wins; re-express kernel's bayesian-ridge/ard FAST switch on top only if it still adds something; otherwise drop it and say so. (A separate lane, lane/apple-fast-bayes, ports the NaN guard to the grid kernels: do not do that here.)
3. Settle the overlap: `git merge-tree --write-tree lane/apple-fast-gram lane/apple-fast-kernel`; if it conflicts, in kernel `git merge lane/apple-fast-gram` and resolve. Push.
4. Check both ab/*.txt files follow the light form (1 2, no -ident, one tag per lane x dataset).
