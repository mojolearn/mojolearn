# SPDX-License-Identifier: Apache-2.0
"""Prescribed original Agglomerative workloads and bounded additional routes."""

@fieldwise_init
struct AggloCase(Copyable, Movable):
    var name: String
    var rows: Int
    var features: Int
    var clusters: Int
    var linkage: Int
    var tree: Bool


def profile(unit: Int) -> AggloCase:
    if unit == 0:
        return AggloCase("grid18", 18, 2, 2, 3, False)
    if unit == 1:
        return AggloCase("duplicate36", 36, 2, 2, 3, False)
    if unit == 2:
        return AggloCase("singletons17", 18, 2, 17, 3, False)
    if unit == 3:
        return AggloCase("parallel200", 200, 4, 3, 3, False)
    if unit == 4:
        return AggloCase("ward180", 180, 3, 3, 0, True)
    if unit == 5:
        return AggloCase("grid18_tree", 18, 2, 2, 3, True)
    if unit == 6:
        return AggloCase("average6", 6, 2, 2, 2, True)
    else:
        return AggloCase("complete6", 6, 2, 2, 1, True)


def query_count(unit: Int) -> Int:
    if unit == 0 or unit == 5:
        return 3
    return 1 if unit == 1 or unit == 2 else 0


def query_name(batch: Int) -> String:
    return "train" if batch == 0 else ("tie" if batch == 1 else "near")


def query_rows(unit: Int, batch: Int) -> Int:
    if batch == 1:
        return 1
    if batch == 2:
        return 2
    return 36 if unit == 1 else 18
