"""Do the options refuse what is not honored?

The point of the file under test is that an option present and ignored is
worse than one absent. This checks the refusals actually fire, because a
`check()` nobody exercises is exactly the kind of machinery this repository
keeps finding unwired.
"""

from gbdt.options.catboost_options import (
    CatBoostOptions,
    DETERMINISM_CROSS_DEVICE,
    DETERMINISM_DEVICE,
    DETERMINISM_OFF,
    GROW_DEPTHWISE,
    GROW_LOSSGUIDE,
    GROW_SYMMETRIC,
    determinism_name,
    grow_policy_name,
)


def expect_refusal(mut o: CatBoostOptions, what: String) raises:
    var refused = False
    try:
        o.check()
    except:
        refused = True
    if not refused:
        raise Error(what + " should have been refused and was not")
    print("    refused:", what)


def check_options() raises:
    var d = CatBoostOptions.default()
    d.check()
    print("  defaults pass:")
    print(
        "    depth",
        d.depth,
        " grow_policy",
        grow_policy_name(d.grow_policy),
        " l2_leaf_reg",
        d.l2_leaf_reg,
        " border_count",
        d.border_count,
    )
    print("    determinism", determinism_name(d.determinism))

    var a = CatBoostOptions.default()
    a.grow_policy = GROW_LOSSGUIDE
    expect_refusal(a, String("grow_policy=Lossguide"))

    var b = CatBoostOptions.default()
    b.grow_policy = GROW_DEPTHWISE
    expect_refusal(b, String("grow_policy=Depthwise"))

    var c = CatBoostOptions.default()
    c.leaf_estimation_iterations = 10
    expect_refusal(c, String("leaf_estimation_iterations=10"))

    var e = CatBoostOptions.default()
    e.random_strength = 1.0
    expect_refusal(e, String("random_strength=1.0"))

    var f = CatBoostOptions.default()
    f.rsm = 0.8
    expect_refusal(f, String("rsm=0.8"))

    var g = CatBoostOptions.default()
    g.min_data_in_leaf = 20
    expect_refusal(g, String("min_data_in_leaf=20"))

    var h = CatBoostOptions.default()
    h.depth = 0
    expect_refusal(h, String("depth=0"))

    # The three determinism levels must all be accepted.
    var levels = List[Int]()
    levels.append(DETERMINISM_OFF)
    levels.append(DETERMINISM_DEVICE)
    levels.append(DETERMINISM_CROSS_DEVICE)
    for i in range(len(levels)):
        var k = CatBoostOptions.default()
        k.determinism = levels[i]
        k.check()
    print("    all three determinism levels accepted")
    print("  every unported option refuses by name")
