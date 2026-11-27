"""Device-independent options of the grouped LightGBM ranking objectives."""

comptime OBJECTIVE_LAMBDARANK = 18
comptime OBJECTIVE_RANK_XENDCG = 19
comptime OBJECTIVE_QUERY_SOFTMAX = 20


@always_inline
def is_lightgbm_ranking(objective: Int) -> Bool:
    return objective == OBJECTIVE_LAMBDARANK or objective == OBJECTIVE_RANK_XENDCG


def is_lightgbm_ranking_name(name: String) -> Bool:
    return name == "lambdarank" or name == "rank_xendcg"


@always_inline
def uses_grouped_rank_workspace(objective: Int) -> Bool:
    return is_lightgbm_ranking(objective) or objective == OBJECTIVE_QUERY_SOFTMAX


def uses_grouped_rank_workspace_name(name: String) -> Bool:
    return is_lightgbm_ranking_name(name) or name == "QuerySoftMax"


def ranking_loss(objective: Int, metric: Float64) -> Float64:
    return metric if objective == OBJECTIVE_QUERY_SOFTMAX else 1.0 - metric


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
