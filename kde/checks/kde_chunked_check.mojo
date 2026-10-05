# SPDX-License-Identifier: Apache-2.0
"""Current KDE dispatch versus its host fold; legacy schedules versus staged.

The full historical trace-card suite should additionally run with
MOJOLEARN_IDN_KDE_CHUNK_LSE_OFF because its trace stages use the serial fold.
"""
from kde.checks.kde_check import check_kde_tiled_equals_staged


def main() raises:
    check_kde_tiled_equals_staged()
