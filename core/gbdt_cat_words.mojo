# SPDX-License-Identifier: Apache-2.0
"""The Gbdt fit door's cat_features range refusal over a member's parameter words (lane mf-cal5).

`gbdt_cat_outside_words(params, d)`: NEW's feature-list tail (`[..., cat indices, one-hot indices, C, H]`,
core/gbdt_member.mojo `_feature_tail`) read for its first cat_features index, in the given order, outside the
fit's `d` columns, in the words of the fit door's refusal at its X (bindings/gbdt_category_unpack.mojo
`_cat_slots`). The member boundary (providers/python/member_gbdt_boundary_door.mojo `gbdt_member_fit_params`)
raises them where the caller used to fall back to that door to raise them (THE FALLBACK RULE). Typed values
only, no Python: callable from a native test. "" when every index is inside X (or the tail is malformed).
"""


def gbdt_cat_range_words(index: Int, n_features: Int) -> String:
    """`_cat_slots`'s words for one cat_features index outside `n_features` columns."""
    return ("mojolearn: cat_features index " + String(index) + " out of range for " + String(n_features)
            + " features")


def gbdt_cat_outside_words(params: List[Float64], d: Int) -> String:
    """The refusal for the first cat_features index of the tail outside `d` columns, in the given order; ""
    when there is none."""
    var count = len(params)
    if count < 4:
        return String("")
    var h = Int(params[count - 1])
    var c = Int(params[count - 2])
    var start = count - 2 - c - h
    if c <= 0 or h < 0 or start < 0:
        return String("")
    for k in range(c):  # small-loop(cat_features): the first index outside X
        var index = Int(params[start + k])
        if index < 0 or index >= d:
            return gbdt_cat_range_words(index, d)
    return String("")
