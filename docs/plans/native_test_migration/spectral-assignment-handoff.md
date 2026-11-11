# Spectral alternative assignment

CPU 2 and GPU 2 are authored and unrun, using the frozen assignment API
`4084fbbe82` and actual fit backends `764e2ad158`. Import
`spectral_assignment_{cpu,gpu}_suite.run_checks`; standard identical and CPU
column flags apply. The original 180 by 3 input, solver settings and the
discretize/cluster-QR settings remain unchanged.

A separate NumPy input-only certificate checks that the neighbor graph has
three connected components corresponding to the planted 60-row groups. Both
self-inclusive and self-exclusive ten-neighbor conventions have a minimum
cross-edge squared-distance margin above 25.85. The native oracle checks that
partition independently of class-label permutation. It does not compare
eigenvector signs, fitted reference hashes or historical outputs. Six reference
preparation jobs used the host's 10 workers; no product import or execution ran.

The result moves out of its descriptor output room and is validated after its
input/fit buffers are destroyed. CPU graphs and row assertions are parallel;
GPU graphs are ordered on one physical GPU while numerical kernels stay
parallel. The separate shared-context concurrency failure remains open.

Nothing removed: the Python subprocess/second-invocation regression test stays
because it tests process and loader lifetime. These are supplementary native
mathematical and ownership checks over its original workloads, not deleted
Python definitions. No product/kernel, algorithm, default, tolerance or timing
changes. All native compilation/execution remains owed to the integrator.
