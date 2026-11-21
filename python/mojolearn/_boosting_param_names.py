# SPDX-License-Identifier: Apache-2.0
"""The one table of CatBoost, XGBoost and LightGBM parameter names that the
gradient-boosting classes accept beside their own (plan 2026-11-21, section 3,
items 3 and 5).

Data only: this module imports nothing from the package, so
`tools/gen_boosting_parameter_names.py` reads it without loading the compiled
library, and `docs/api/BOOSTING_PARAMETER_NAMES.md` is generated from it.

A row is `(name, canonical, libraries, same, note)`:
- `same=True`: `name` means exactly what `canonical` means here; the
  constructor and `set_params` store the value under `canonical`, and
  `get_params` reports `canonical`. A name is accepted only when it means the
  same in every listed library that uses it.
- `same=False`: the name exists in those libraries with a different meaning;
  it is refused with `note`, which names the parameter to use instead. It is
  never mapped.

Resolution happens once per constructor or `set_params` call, natively
(bindings/trees_gbdt_parameter_names_unpack.mojo), and only for a keyword the
class's own signature does not hold: a call that uses the canonical names does
no table lookup at all. Nothing here is read by `fit` or `predict`.
"""

_C, _X, _L = "CatBoost", "XGBoost", "LightGBM"

BOOSTING_PARAMETER_NAMES = (
    # -- number of trees
    ("iterations", "n_estimators", (_C,), True, ""),
    ("num_boost_round", "n_estimators", (_C, _X, _L), True, ""),
    ("num_trees", "n_estimators", (_C, _L), True, ""),
    ("num_tree", "n_estimators", (_L,), True, ""),
    ("num_iterations", "n_estimators", (_L,), True, ""),
    ("num_iteration", "n_estimators", (_L,), True, ""),
    ("n_iter", "n_estimators", (_L,), True, ""),
    ("num_round", "n_estimators", (_L,), True, ""),
    ("num_rounds", "n_estimators", (_L,), True, ""),
    ("nrounds", "n_estimators", (_L,), True, ""),
    ("max_iter", "n_estimators", (_L,), True, ""),
    # -- step size
    ("eta", "learning_rate", (_C, _X), True, ""),
    ("shrinkage_rate", "learning_rate", (_L,), True, ""),
    # -- tree shape
    ("depth", "max_depth", (_C,), True, ""),
    ("num_leaves", "max_leaves", (_C, _L), True, ""),
    ("max_leaf", "max_leaves", (_L,), True, ""),
    ("max_leaf_nodes", "max_leaves", (_L,), True, ""),
    # -- L2 regularization of leaf values
    ("reg_lambda", "l2_leaf_reg", (_C, _X, _L), True, ""),
    ("lambda", "l2_leaf_reg", (_X, _L), True, ""),
    ("lambda_l2", "l2_leaf_reg", (_L,), True, ""),
    ("l2_regularization", "l2_leaf_reg", (_L,), True, ""),
    # -- seed
    ("random_seed", "random_state", (_C, _L), True, ""),
    ("seed", "random_state", (_X, _L), True, ""),
    # -- row subsampling (read with bootstrap_type='Bernoulli' or 'Poisson')
    ("bagging_fraction", "subsample", (_L,), True, ""),
    ("sub_row", "subsample", (_L,), True, ""),
    # -- per-tree feature subsampling
    ("colsample_bytree", "feature_fraction", (_X, _L), True, ""),
    ("sub_feature", "feature_fraction", (_L,), True, ""),
    # -- minimum Hessian sum in a child
    ("min_child_weight", "min_child_hessian", (_X, _L), True, ""),
    ("min_sum_hessian_in_leaf", "min_child_hessian", (_L,), True, ""),
    ("min_sum_hessian_per_leaf", "min_child_hessian", (_L,), True, ""),
    ("min_sum_hessian", "min_child_hessian", (_L,), True, ""),
    ("min_hessian", "min_child_hessian", (_L,), True, ""),
    # -- rows sampled for the border search
    ("bin_construct_sample_cnt", "border_build_max_samples", (_L,), True, ""),
    ("subsample_for_bin", "border_build_max_samples", (_L,), True, ""),
    # -- overfitting detector
    ("od_pval", "od_pvalue", (_C,), True, ""),
    # -- loss (CatBoost's loss names; other libraries' objective strings are refused by value)
    ("loss_function", "loss", (_C,), True, ""),
    ("objective", "loss", (_C, _X, _L), True, ""),
    # -- categorical features (column indices)
    ("categorical_feature", "cat_features", (_L,), True, ""),
    ("categorical_features", "cat_features", (_L,), True, ""),
    ("cat_feature", "cat_features", (_L,), True, ""),
    ("categorical_column", "cat_features", (_L,), True, ""),
    ("cat_column", "cat_features", (_L,), True, ""),
    # -- where it runs
    ("device_type", "device", (_L,), True, ""),

    # -- names whose meaning differs: refused, never mapped
    ("max_bin", "border_count", (_C, _X, _L), False,
     "max_bin counts bins in XGBoost and LightGBM but split borders in CatBoost; "
     "use border_count, the number of split borders per numeric feature"),
    ("min_child_samples", "min_data_in_leaf", (_C, _L), False,
     "min_child_samples bounds each child's rows in LightGBM; "
     "use min_data_in_leaf, CatBoost's meaning: a leaf with at most that many rows is not split"),
    ("min_data_per_leaf", "min_data_in_leaf", (_L,), False,
     "LightGBM's min_data_per_leaf bounds each child's rows; "
     "use min_data_in_leaf, CatBoost's meaning: a leaf with at most that many rows is not split"),
    ("min_data", "min_data_in_leaf", (_L,), False,
     "LightGBM's min_data bounds each child's rows; "
     "use min_data_in_leaf, CatBoost's meaning: a leaf with at most that many rows is not split"),
    ("min_samples_leaf", "min_data_in_leaf", (_L,), False,
     "LightGBM's min_samples_leaf bounds each child's rows; "
     "use min_data_in_leaf, CatBoost's meaning: a leaf with at most that many rows is not split"),
    ("gamma", "min_split_gain", (_X,), False,
     "XGBoost's gamma is a loss reduction; use min_split_gain, whose units follow score_function"),
    ("min_split_loss", "min_split_gain", (_X,), False,
     "XGBoost's min_split_loss is a loss reduction; use min_split_gain, whose units follow score_function"),
    ("min_gain_to_split", "min_split_gain", (_L,), False,
     "LightGBM's min_gain_to_split is a loss reduction; use min_split_gain, whose units follow score_function"),
    ("min_gain_split", "min_split_gain", (_L,), False,
     "LightGBM's min_gain_split is a loss reduction; use min_split_gain, whose units follow score_function"),
    ("rsm", "feature_fraction", (_C,), False,
     "CatBoost's rsm samples features at each split; use feature_fraction, which samples features once per tree"),
    ("colsample_bylevel", "feature_fraction", (_C, _X), False,
     "colsample_bylevel samples features per level; use feature_fraction, which samples features once per tree"),
    ("colsample_bynode", "feature_fraction", (_X,), False,
     "colsample_bynode samples features per node; use feature_fraction, which samples features once per tree"),
    ("feature_fraction_bynode", "feature_fraction", (_L,), False,
     "feature_fraction_bynode samples features per node; use feature_fraction, which samples features once per tree"),
    ("early_stopping_rounds", "od_wait", (_C, _X, _L), False,
     "early stopping here is the overfitting detector: use od_type='Iter' and od_wait=<rounds>, "
     "with eval_set in fit"),
    ("early_stopping_round", "od_wait", (_L,), False,
     "early stopping here is the overfitting detector: use od_type='Iter' and od_wait=<rounds>, "
     "with eval_set in fit"),
    ("early_stopping", "od_wait", (_L,), False,
     "early stopping here is the overfitting detector: use od_type='Iter' and od_wait=<rounds>, "
     "with eval_set in fit"),
    ("n_iter_no_change", "od_wait", (_L,), False,
     "early stopping here is the overfitting detector: use od_type='Iter' and od_wait=<rounds>, "
     "with eval_set in fit"),
    ("one_hot_max_size", "one_hot_features", (_C,), False,
     "one_hot_max_size is a cardinality threshold; use one_hot_features, the column indices to one-hot encode"),
    ("max_cat_to_onehot", "one_hot_features", (_X, _L), False,
     "max_cat_to_onehot is a cardinality threshold; use one_hot_features, the column indices to one-hot encode"),
    ("class_weight", "class_weights", (_L,), False,
     "LightGBM's class_weight takes a dict or 'balanced'; use class_weights, one weight per class in label order"),
    ("auto_class_weights", "class_weights", (_C,), False,
     "auto_class_weights computes weights; use class_weights, one weight per class in label order"),
    ("scale_pos_weight", "class_weights", (_X, _L), False,
     "scale_pos_weight weights the positive class only; use class_weights=[1.0, <weight>] for two classes"),
    ("boosting", "boosting_type", (_L,), False,
     "LightGBM's boosting picks gbdt, dart, goss or rf; boosting_type here is CatBoost's "
     "'Plain' or 'Ordered' (LightGBM's gbdt is 'Plain')"),
    ("boost", "boosting_type", (_L,), False,
     "LightGBM's boost picks gbdt, dart, goss or rf; boosting_type here is CatBoost's "
     "'Plain' or 'Ordered' (LightGBM's gbdt is 'Plain')"),
    ("task_type", "device", (_C,), False,
     "CatBoost's task_type takes 'CPU' or 'GPU'; use device='cpu' or device='gpu'"),
)

# Names whose spelling is already a parameter here but whose values follow
# another library's vocabulary. They are canonical names, so they are not in
# the table above; the value checks refuse the other vocabulary. Listed for
# the generated page only.
SAME_NAME_OTHER_VALUES = (
    ("boosting_type", (_L,), "LightGBM's 'gbdt', 'dart', 'goss' or 'rf'; here CatBoost's 'Plain' or 'Ordered'"),
    ("grow_policy", (_X,), "XGBoost's 'depthwise' or 'lossguide'; here CatBoost's 'SymmetricTree', 'Depthwise' or 'Lossguide'"),
    ("min_data_in_leaf", (_L,), "LightGBM bounds each child's rows; here a leaf with at most that many rows is not split (CatBoost)"),
    ("device", (_X, _L), "'cuda' is refused; here None, 'gpu' or 'cpu'"),
)


def parameter_name_table(parameters, defaults=None):
    """`{name: (canonical or None, refusal text, index, default)}` for one
    class, built once at import from its constructor parameter names (in
    signature order) and, for a positional constructor, their defaults.

    A row is kept only when its canonical name is a parameter of the class
    and its own name is not; `index` is the canonical name's position in
    `parameters` and `default` its default (None and -1 when `defaults` is
    not given)."""
    order = {name: i for i, name in enumerate(parameters)}  # glue: import-time table build
    table = {}
    for name, canonical, libraries, same, note in BOOSTING_PARAMETER_NAMES:  # glue: import-time table build
        if canonical not in order or name in order:
            continue
        text = None if same else (
            "ValueError: " + name + " (" + ", ".join(libraries) + ") is not accepted: " + note)
        index = order[canonical] if defaults is not None else -1
        default = defaults[canonical] if defaults is not None else None
        table[name] = (canonical if same else None, text, index, default)
    return table
