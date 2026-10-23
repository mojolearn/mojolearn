# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Shared data owner for trained GBDT models. Serialization imports this module
# without importing the GPU trainer. train re-exports this same type; do not
# define a second TrainedModel or change the field order at this boundary.
from gbdt.models.oblivious_model import TAdditiveModel
from gbdt.models.ctr_value_table import TCtrValueTable
from gbdt.models.tensor_ctr_value_table import TTensorCtrRegistry


@fieldwise_init
struct TrainedModel(Movable):
    """The ensemble plus the quantization grid it was trained on, which
    is what applying it to NEW raw floats requires -- their model carries
    its borders for the same reason."""

    var model: TAdditiveModel
    var fold_counts: List[Int]
    var one_hot: List[Bool]
    var borders: List[List[Float32]]
    #: their `TFloatFeature::NanValueTreatment`, one per column
    #: (`libs/model/features.h:51-61`). `AsIs` for a column the learn pool
    #: had no NaN in, and a NaN arriving on such a column at apply time is
    #: their `CB_ENSURE` rather than a guess. It travels WITH the borders
    #: because it is part of the grid: the sentinel border is meaningless
    #: without the substitution that reaches it.
    var nan_treatment: List[Int]
    var losses: List[Float64]
    #: the HELD-OUT loss per iteration, empty when no eval set was given.
    #: Their `TestCursor`'s error curve, which is what the overfitting
    #: detector reads and what a caller should plot -- the learn curve
    #: falls almost by construction and says nothing about stopping.
    var test_losses: List[Float64]
    #: the index of the lowest TEST loss, or of the lowest learn loss with
    #: no eval set. Their `TLearnProgress` best-iteration bookkeeping.
    #: **-1 means NOT RECORDED**, which is what a model loaded from text
    #: carries: the text holds no held-out curve and
    #: `load_model_text` does not invent one.
    var best_iteration: Int
    #: True when the detector fired before `n_estimators` was reached
    var stopped_early: Bool
    var ctr_column_count: Int
    """How many of the columns above are CTR values rather than raw input
    features. It is a SAFETY count: a model that declares CTR columns and
    carries no tables for them cannot be applied to raw rows, and
    `predict_floats` refuses it. It survives serialization for exactly that
    reason -- a round trip that dropped it would turn a model that refuses
    into one that silently scores."""
    var ctr_tables: List[TCtrValueTable]
    """The apply-time CTR tables, one per CTR column, their
    `ctr_data.hash_map` (`libs/model/ctr_value_table.h`). With these
    present and covering every declared CTR column, `predict_floats` maps a
    raw category to the statistic the LEARN pool produced and scores the
    row; without them it refuses. See
    `gbdt/models/ctr_value_table.mojo`."""
    var tensor_ctr_registry: TTensorCtrRegistry
    """Winning combination/tree CTR tables. Serialized model apply rebuilds
    these columns and their split-history bins in level order. Public
    combination training remains refused until this registry is populated by
    the production structure-search driver rather than focused checks."""
