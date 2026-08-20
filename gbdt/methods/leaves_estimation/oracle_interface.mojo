"""What the Newton walker needs from a leaves-estimation oracle.

PORT OF `catboost/cuda/methods/leaves_estimation/oracle_interface.h` at
CatBoost `54a8143a`, cut to the surface the DIAGONAL pointwise path uses.
Transliterated where there is code to transliterate; the interface itself
is a shape.

WHAT IS DELIBERATELY NOT HERE, so nobody goes looking:

* `HessianBlockSize()` -- the blocked-Hessian arm of the walker is
  MultiClass-only (`descent_helpers.cpp:92-118` solves per-block Cholesky
  systems); pointwise losses are `HessianBlockSize == 1` and this port
  raises on anything else by not offering the hook.
* `MakeEstimationResult` -- identity for every single-dim loss
  (`pointwise_oracle.cpp:18-33`; the non-identity arm is MultiClass's
  last-dimension elimination).
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

    def move_to(mut self, point: List[Float32]) raises:
        ...

    def write_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ) raises:
        ...

    def write_second_derivatives(mut self, mut second_der: List[Float64]) raises:
        ...

    def regularize(self, mut point: List[Float32]):
        ...
