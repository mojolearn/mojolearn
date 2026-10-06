"""Untimed host full block-solver regression. No convergence relaxation."""
from std.memory import bitcast
from x_decomp.rr_solve import host_eigh_rb_sorted

def main() raises:
    var sizes: List[Int] = [64, 65, 96]
    for n in sizes:
        var a = List[Float32](length=n * n, fill=Float32(0.0))
        for i in range(n):
            for j in range(n):
                # Exact power-of-two rank-one fixture, plus repeated diagonal.
                var ui = Float32((i * 17) % 31 - 15) / Float32(128.0)
                var uj = Float32((j * 17) % 31 - 15) / Float32(128.0)
                a[i * n + j] = ui * uj + (Float32(2.0) if i == j else Float32(0.0))
        var w = List[Float32]()
        var v = List[Float32]()
        var sweeps = host_eigh_rb_sorted(a, n, w, v)
        print("FULL_BLOCK_CONVERGED", n, sweeps)
        for i in range(n):
            print("FULL_EIGEN_BITS", n, i, bitcast[DType.uint32](w[i]))
        for i in range(n * n):
            print("FULL_VECTOR_BITS", n, i, bitcast[DType.uint32](v[i]))
