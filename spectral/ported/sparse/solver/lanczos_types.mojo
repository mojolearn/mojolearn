"""`raft/sparse/solver/lanczos_types.hpp`: `LANCZOS_WHICH` and
`lanczos_solver_config<ValueTypeT>`, field for field.

`seed` is `std::optional<uint64_t>` in theirs; here `has_seed` + `seed`.
An absent seed means `std::random_device{}()` at
`detail/lanczos.cuh:781` -- a non-reproducible start vector -- and is
REFUSED BY NAME by `lanczos_compute_eigenpairs` (DEVIATION 772 in
`detail/lanczos.mojo`), so `has_seed` exists to carry the refusal, not a
second behavior.
"""

#: `LANCZOS_WHICH`, `lanczos_types.hpp:20-29`.
comptime LANCZOS_LA = 0  # largest algebraic -- the only arm cuVS reaches
comptime LANCZOS_LM = 1  # largest magnitude
comptime LANCZOS_SA = 2  # smallest algebraic
comptime LANCZOS_SM = 3  # smallest magnitude


def lanczos_which_name(which: Int) -> String:
    if which == LANCZOS_LA:
        return String("LA")
    if which == LANCZOS_LM:
        return String("LM")
    if which == LANCZOS_SA:
        return String("SA")
    if which == LANCZOS_SM:
        return String("SM")
    return String("WHICH?")


@fieldwise_init
struct LanczosSolverConfig(Copyable, Movable):
    """`lanczos_solver_config<float>`, `lanczos_types.hpp:40-70`."""

    var n_components: Int
    var max_iterations: Int
    var ncv: Int
    var tolerance: Float32
    var which: Int
    var has_seed: Bool
    var seed: UInt64
