"""Device-independent options of the grouped LightGBM ranking objectives."""

comptime OBJECTIVE_LAMBDARANK = 18
comptime OBJECTIVE_RANK_XENDCG = 19


@always_inline
def is_lightgbm_ranking(objective: Int) -> Bool:
    return objective == OBJECTIVE_LAMBDARANK or objective == OBJECTIVE_RANK_XENDCG


def is_lightgbm_ranking_name(name: String) -> Bool:
    return name == "lambdarank" or name == "rank_xendcg"


def ranking_seed(tree_seed: UInt64) -> Int:
    return Int(UInt32(tree_seed) + UInt32(5))


def check_rank_estimator(method: Int, newton: Int, iterations: Int) raises:
    if method != newton or iterations != 1:
        raise Error("lambdarank and rank_xendcg take one Newton leaf update per tree")


def check_rank_grade(valid: Bool) raises:
    if not valid:
        raise Error("lambdarank and rank_xendcg require integer relevance grades in [0, 30]")


def check_rank_inputs(categorical: Bool, validation: Bool) raises:
    if categorical:
        raise Error("lambdarank and rank_xendcg accept numeric features only")
    if validation:
        raise Error("lambdarank and rank_xendcg do not support eval_set")
