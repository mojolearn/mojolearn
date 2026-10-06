# SPDX-License-Identifier: Apache-2.0
"""M06 ownership-key building block; caller integration remains pending.

NOT TESTED — NOT COMPILED — NOT MEASURED. This is control-plane source,
not an enabled cache or a claim that mutable Python arrays have generations.
The existing sessions continue exact-byte comparison until an actual owned
input/weight API supplies and invalidates these keys around every mutation.
"""


struct MambaOwnedGeneration(Copyable, Movable):
    """A cache lease is tied to an owner, an epoch and its shape/profile.

    An owner ID must come from the resource owner, never a recycled address.
    Its epoch increases after optimizer updates, explicit replacement,
    checkpoint restore and any external write. The caller must invalidate
    before queuing a mutation; successful completion may publish a new key.
    """

    var owner: UInt64
    var epoch: UInt64
    var elements: Int
    var numerical_profile: UInt64
    var valid: Bool

    def __init__(out self):
        self.owner = 0
        self.epoch = 0
        self.elements = 0
        self.numerical_profile = 0
        self.valid = False

    def invalidate(mut self):
        self.valid = False

    def publish(mut self, owner: UInt64, epoch: UInt64,
                elements: Int, numerical_profile: UInt64):
        self.owner = owner
        self.epoch = epoch
        self.elements = elements
        self.numerical_profile = numerical_profile
        self.valid = owner != 0 and elements >= 0

    def matches(self, owner: UInt64, epoch: UInt64,
                elements: Int, numerical_profile: UInt64) -> Bool:
        return (self.valid and self.owner == owner and self.epoch == epoch
                and self.elements == elements
                and self.numerical_profile == numerical_profile)
