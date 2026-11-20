# SPDX-License-Identifier: Apache-2.0
"""One native resident handle; an Arc lease keeps active queries alive."""
from core.kmeans_variants_api import KMeansVariantsDriver


@fieldwise_init
struct VariantsResidentState(Movable):
    var driver: KMeansVariantsDriver
    var handle: Int

    def __del__(deinit self):
        if self.handle >= 0:
            try:
                self.driver.release(self.handle)
            except:
                pass  # Existing resident teardown is quiet.
