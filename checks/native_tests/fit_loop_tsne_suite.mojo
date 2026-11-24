# SPDX-License-Identifier: Apache-2.0
"""The fit loop clock of TSNE's gradient loop (lane ab-loop-mode / alm-iter,
2026-11-24): `tsne_fit_device` and `tsne_fit_host` on an 8-row fixture with
max_iter K. The loop always runs max_iter steps (TSNE's n_iter_), so the check
is that each column writes a non-negative loop time through `loop_word`
(the fit's `fit_loop_` milliseconds).

Suite module (no `main`): checks/native_tests/fit_loop_tsne.mojo runs
`run_checks`."""
from checks.native_tests.testsuite_cases import require
from x_ann.tsne_device import tsne_fit_device
from x_ann.host.tsne_host import tsne_fit_host

comptime K = 6
comptime N = 8


def _fixture() -> List[Float32]:
    var values: List[Float32] = [0.0, 0.1, 1.0, 1.2, 0.2, 0.0, 1.1, 0.9, 3.0, 3.1, 4.0, 4.2, 3.2, 3.0, 4.1, 3.9]
    return values^


def _start() -> List[Float32]:
    var values = List[Float32](length=2 * N, fill=Float32(0.0))
    for i in range(2 * N):  # small-loop(16 start coordinates)
        values[i] = Float32(i % 5) * Float32(1e-4)
    return values^


def _fit(device: Bool) raises:
    var word = List[Int](length=1, fill=-1)
    var y = List[Float32]()
    var kl = Float32(0.0)
    if device:
        tsne_fit_device(_fixture(), N, 2, _start(), Float32(2.0), Float32(12.0), Float32(50.0), Float32(1.0), K, 2,
                        y, kl, 0, Int(word.unsafe_ptr()))
    else:
        tsne_fit_host(_fixture(), N, 2, _start(), Float32(2.0), Float32(12.0), Float32(50.0), Float32(1.0), K, 2,
                      y, kl, Int(word.unsafe_ptr()))
    var column = String(" device") if device else String(" host")
    require(word[0] >= 0, "TSNE" + column + ": the loop time was not written")
    require(len(y) == 2 * N, "TSNE" + column + ": embedding size")


def run_checks() raises:
    _fit(False)
    _fit(True)
