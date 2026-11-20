# SPDX-License-Identifier: Apache-2.0
"""Typed membership test over connected type identity words (RUN)."""


def _address_in(address: Int, members: List[Int]) -> Bool:
    for i in range(len(members)):  # driver-loop: over the few connected type identities
        if members[i] == address:
            return True
    return False
