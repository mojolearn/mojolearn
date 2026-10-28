# SPDX-License-Identifier: Apache-2.0
# Compile only (-emit shared-lib); do not execute this exported probe.
# Instantiates the real auxiliary owner, including PCS capability and draw.
from x_neighbors.native_descriptor_aux_host import xn_aux_execute


@export
def peer1_pcs_draw_compile(seed: Int, components: Int, total: Int,
                          index_address: Int, sign_address: Int) abi("C") -> Int:
    try:
        return xn_aux_execute(4, [index_address, sign_address], [seed, components, total], [])
    except error:
        return -1
