# SPDX-License-Identifier: Apache-2.0
"""GP resident lease identity and take-once state, independent of Python."""


struct GPLeaseState(Movable):
    var handle: Int
    var provider: Int
    var generation: Int
    var owners: List[Int]

    def __init__(out self, handle: Int, provider: Int, generation: Int, var owners: List[Int]):
        self.handle = handle
        self.provider = provider
        self.generation = generation
        self.owners = owners^

    def matches_header(self, provider: Int, generation: Int, count: Int) -> Bool:
        return self.handle > 0 and self.provider == provider and self.generation == generation and len(self.owners) == count

    def matches_owner(self, index: Int, owner: Int) -> Bool:
        return index >= 0 and index < len(self.owners) and self.owners[index] == owner

    def matches(self, provider: Int, generation: Int, owners: List[Int]) -> Bool:
        if not self.matches_header(provider, generation, len(owners)):
            return False
        for i in range(len(owners)):  # driver-loop: identity of each fitted buffer, never buffer contents
            if not self.matches_owner(i, owners[i]):
                return False
        return True

    def take(mut self) -> Int:
        var handle = self.handle
        self.handle = 0
        return handle
