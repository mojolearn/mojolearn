# SPDX-License-Identifier: Apache-2.0
"""Original DBSCAN workload descriptors; independent refs recorded beside inputs."""

@fieldwise_init
struct DBSCANCase(Copyable, Movable):
    var name: String
    var rows: Int
    var features: Int
    var eps: Float64
    var min_samples: Int
    var method: Int
    var metric: Int
    var weighted: Bool
    var clusters: Int
    var core_count: Int


def profile(unit: Int) -> DBSCANCase:
    if unit == 0:
        return DBSCANCase("grid_brute", 20, 2, 1.0, 4, 0, 0, False, 2, 10)
    if unit == 1:
        return DBSCANCase("grid_rbc", 20, 2, 1.0, 4, 1, 0, False, 2, 10)
    if unit == 2:
        return DBSCANCase("grid_wide", 20, 2, 3.0, 4, 0, 0, False, 2, 18)
    if unit == 3:
        return DBSCANCase("grid_duplicates", 40, 2, 1.0, 4, 0, 0, False, 2, 36)
    if unit == 4:
        return DBSCANCase("grid_manhattan", 20, 2, 1.0, 4, 0, 1, False, 2, 10)
    if unit == 5:
        return DBSCANCase("grid_ones", 20, 2, 1.0, 4, 0, 0, True, 2, 10)
    if unit == 6:
        return DBSCANCase("grid_empty", 20, 2, 0.1, 50, 1, 0, False, 0, 0)
    if unit == 7:
        return DBSCANCase("host512", 512, 5, 0.9, 5, 1, 0, False, 5, 130)
    if unit == 8:
        return DBSCANCase("parallel400", 400, 4, 0.9, 5, 1, 0, False, 1, 257)
    if unit == 9:
        return DBSCANCase("weighted2_brute", 8, 2, 1.0, 3, 0, 0, True, 2, 8)
    if unit == 10:
        return DBSCANCase("weighted15_brute", 8, 2, 1.0, 3, 0, 0, True, 1, 6)
    if unit == 11:
        return DBSCANCase("duplicated_brute", 9, 2, 1.0, 3, 0, 0, False, 2, 9)
    if unit == 12:
        return DBSCANCase("weighted2_rbc", 8, 2, 1.0, 3, 1, 0, True, 2, 8)
    if unit == 13:
        return DBSCANCase("weighted15_rbc", 8, 2, 1.0, 3, 1, 0, True, 1, 6)
    if unit == 14:
        return DBSCANCase("duplicated_rbc", 9, 2, 1.0, 3, 1, 0, False, 2, 9)
    if unit == 15:
        return DBSCANCase("negative_inhibit", 6, 2, 0.75, 3, 0, 0, True, 1, 3)
    else:
        return DBSCANCase("precomputed_grid", 20, 20, 1.0, 4, 0, 3, False, 2, 10)


def query_count(unit: Int) -> Int:
    if unit == 0:
        return 5
    return 1 if unit <= 6 else 0


def query_name(unit: Int, batch: Int) -> String:
    if unit == 0:
        var names: List[String] = ["train", "at_eps", "past_eps", "far", "reverse"]
        return names[batch].copy()
    if unit == 2:
        return "tie"
    if unit == 4:
        return "boundary"
    if unit == 6:
        return "first_three"
    return "train"


def query_rows(unit: Int, batch: Int) -> Int:
    if unit == 0:
        var rows: List[Int] = [20, 2, 1, 2, 20]
        return rows[batch]
    if unit == 2:
        return 1
    if unit == 3:
        return 40
    if unit == 4 or unit == 6:
        return 3
    return 20
