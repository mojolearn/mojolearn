# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""What the Newton walker needs from a leaves-estimation oracle.

PORT OF `catboost/cuda/methods/leaves_estimation/oracle_interface.h` at
CatBoost `54a8143a`, cut to the surface the DIAGONAL pointwise path uses.
Transliterated where there is code to transliterate; the interface itself
is a shape.

WHAT IS DELIBERATELY NOT HERE, so nobody goes looking:

* `AddLangevinNoiseToDerivatives` -- a no-op unless
  `boostingOptions.Langevin` (posterior sampling) is on
  (`pointwise_oracle.cpp:199-204`), which no configuration this
  repository runs sets. The hook is omitted rather than stubbed; wiring
  Langevin means porting `AddLangevinNoise` and putting the hook back in
  the walker AT THEIR CALL SITES, one of which noises the GRADIENT twice
  and the Hessian never (`descent_helpers.cpp:190-196`) -- their code,
  their order, not a thing to fix silently.

`regularize` carries `RegularizeImpl`'s contract (`oracle_interface.h:43-52`):
zero every bin whose weight sum is under `MinLeafWeight` (their hardcoded
`1e-20`, `leaves_estimation_config.h:62`).
"""


trait LeavesEstimationOracle:
    def point_dim(self) -> Int:
        ...

    def hessian_block_size(self) -> Int:
        """`HessianBlockSize()` (`oracle_interface.h`).

        ONE for every single-dimensional loss, which sends the walker down
        `UpdateMoveDirectionDiagonal`; `numClasses - 1` for MultiClass,
        which sends it down the per-leaf Cholesky solve. Their
        `TBinOptimizedOracle` derives it from the target
        (`pointwise_oracle.h`), and so does ours.
        """
        ...

    def move_to(mut self, point: List[Float32]) raises:
        ...

    def write_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ) raises:
        ...

    def write_second_derivatives(mut self, mut second_der: List[Float64]) raises:
        ...

    def make_estimation_result(
        self, point: List[Float32]
    ) -> List[Float32]:
        """`MakeEstimationResult` (`pointwise_oracle.cpp:18-33`).

        Identity for every single-dimensional loss. For MultiClass it
        projects the walker's `numClasses`-wide point down to the cursor's
        `numClasses - 1` by subtracting the pinned component.

        THE WALKER APPLIES IT ON THE WAY OUT, at BOTH of its exits
        (`descent_helpers.cpp:153` and `:204`), so what a caller receives
        is always in the CURSOR's gauge and never in the walker's. This
        port returned the raw point at first, which for MultiClass is a
        vector one component too wide in a gauge the cursor cannot read.
        """
        ...

    def regularize(self, mut point: List[Float32]):
        ...
