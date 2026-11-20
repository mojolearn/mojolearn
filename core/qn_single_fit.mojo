# SPDX-License-Identifier: Apache-2.0
"""Single-target QN metadata owner; the existing provider owns the solver."""
from core.logistic_native_api import QnFitRequest, QnNativeDriver


struct QnSingleFitPlan(Movable):
    var info: List[Float32]
    var iterations: Int

    def __init__(out self):
        self.info = List[Float32](length=2, fill=0)
        self.iterations = 0

    def run(mut self, driver: QnNativeDriver, var request: QnFitRequest) raises:
        request.info = Int(self.info.unsafe_ptr())
        try:
            self.iterations = driver.fit(request)
        finally:
            _ = self.info

    def objective(self) -> Float64:
        return Float64(self.info[0])

    def retcode(self) -> Int:
        return Int(self.info[1])
