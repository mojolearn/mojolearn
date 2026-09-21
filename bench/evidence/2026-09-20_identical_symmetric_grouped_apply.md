# IDENTICAL symmetric-tree grouped resident apply

Base: `origin/main` `f8b3a1967`. Hardware: Apple M4. Numeric mode:
IDENTICAL. The candidate groups at most four consecutive oblivious trees in
one resident-inference kernel. Each row retains tree order, level order,
split predicates, and one Float32 addition per tree. FAST and DETERMINISTIC
keep the previous one-launch-per-tree path.

The analytic device gate compares four separate launches with one grouped
launch over 1,000,003 rows, two model dimensions, mixed depths 1/2/3/1,
ordered and equality predicates, and hashed leaves. All 2,000,006 cells were
bitwise equal. Repeating those four trees 25 times also produced exact final
cells. Warm kernel-only milliseconds for 100 tree applications were:

```
separate 63.359 63.824 63.703 63.367
grouped  32.972 54.347 31.679 31.299
```

Public `GradientBoosting.predict` A/B used explicit learning rate 0.03,
No bootstrap, zero random strength, and independently reconstructed fits and
queries. The first call includes resident preparation; medians below exclude
it. Model, complete prediction, and complete probability SHA-256 values were
identical between builds.

| seed/features/rows/classes/trees/depth | baseline ms | grouped ms | speedup |
|---|---:|---:|---:|
| 7/8/1M/2/100/6 | 42.619 | 27.858 | 1.53x |
| 19/16/1M/2/100/6 | 45.588 | 31.846 | 1.43x |
| 41/32/250k/2/100/6 | 19.472 | 14.095 | 1.38x |
| 3/8/1M/3/17/3 | 27.584 | 23.598 | 1.17x |
| 29/16/250k/8/7/8 | 18.832 | 16.577 | 1.14x |

Representative complete hashes:

| case | model | prediction | probability |
|---|---|---|---|
| 7/8/1M/2/100/6 | `9be9d6927618f8b3deda39d387fcaec897d654ab3d265009bf246dda1b6de862` | `e9c5fb17bafd686fdfba755e4a4728f9d96691352a39ee2c1d0768cac89842bb` | `030a2f71c1e2377c8a9984e1d1884eb21de5d5a73d0c09c4e6dda272b3df017a` |
| 3/8/1M/3/17/3 | `8dcf0929e1d30e55a12d7048663f41f5b0ffbfd7ead6425a65c0a937bb3e8d91` | `51449ad747ffba8779490e8b611abb7a6c2cc05083638946c8d565bc5279c7e5` | `a9403dc53dafaf9186f3a198ef30eb7e9bbf5c0ae0a96a1bf2cd32800b932b8d` |
| 29/16/250k/8/7/8 | `771f6fcff9a2cbbe747897aaca3deb95705ef2917bd684a6b96c0dd7ba213d24` | `941d4c001b5a100763c0c2ef0c76fdec32bda61e7ab5ff3bd5c48ddb7151a952` | `6fefbc93697d2fc17cf9165c2b646cb2e353641aeabbc497e6e21b659e626bf3` |

Focused Python gates: 49 passed plus 13 subtests. Three failures were solely
the fresh worktree lacking `python/mojolearn/.dylibs/libMojolearnMath.dylib`;
all failed before fitting, and are unrelated to the candidate. The public
A/B avoids that packaging omission with an explicit learning rate.
