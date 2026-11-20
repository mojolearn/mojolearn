# SPDX-License-Identifier: Apache-2.0
"""PCA query settings and admission identities contain only native values."""


@fieldwise_init
struct PCAQueryState(Copyable, Movable):
    var fit_rows: Int
    var features: Int
    var count: Int
    var whiten: Bool
    var connection: Int


struct PCAQueryIdentity(Copyable, Movable):
    var words: List[Int]
    var valid: Bool

    def __init__(out self, valid: Bool = True):
        self.words = List[Int]()
        self.valid = valid

    def same(self, other: Self) -> Bool:
        if not self.valid or not other.valid or len(self.words) != len(other.words):
            return False
        for index in range(len(self.words)):  # driver-loop: fixed admitted metadata identities
            if self.words[index] != other.words[index]:
                return False
        return True
