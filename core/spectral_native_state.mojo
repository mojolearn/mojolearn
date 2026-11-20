# SPDX-License-Identifier: Apache-2.0
"""A resident Spectral handle owned by native query/cache Arc references."""
from core.spectral_native_api import SpectralDriver


@fieldwise_init
struct SpectralResidentState(Movable):
    var driver: SpectralDriver
    var handle: Int

    def __del__(deinit self):
        if self.handle >= 0:
            try:
                self.driver.release(self.handle)
            except:
                pass  # Preserve quiet resident cleanup.
