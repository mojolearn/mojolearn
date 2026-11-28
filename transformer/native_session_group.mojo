# SPDX-License-Identifier: Apache-2.0
# Operation-local native reservation descriptor. The provider retains every
# corresponding typed owner and image until reserve/execute/release completes.
# Algorithms interpret their own operation and scalar/address vectors; this
# shared record contains no Python objects, dispatch policy or numerical work.
struct NativeSessionGroup(Movable):
    var owner: Int
    var operation: Int
    var layers: List[Int]
    var addresses: List[Int]
    var parameters: List[Int]

    def __init__(out self, owner: Int, operation: Int, var layers: List[Int],
                 var addresses: List[Int], var parameters: List[Int]):
        self.owner = owner
        self.operation = operation
        self.layers = layers^
        self.addresses = addresses^
        self.parameters = parameters^
