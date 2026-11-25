# SPDX-License-Identifier: Apache-2.0
"""What `model.digest_` and `model.training_digest_` hash, per class, and their one native connection.

`digest_` means "this model gives these answers": the hash of exactly the state
that decides what predict, transform, score and the other answer methods
return (the architecture, the hyperparameters those methods read, the fitted
parameters and buffers, class labels, category maps), and of nothing that only
records how the model was trained (optimizer moments, step and epoch counters,
loss and score histories, convergence records, timing, device, thread counts).
That training record has its own value, `training_digest_`, never mixed into
`digest_`. bindings/fitted_digest_door.mojo has the walk (one walk for both),
core/fitted_digest.mojo the format. The compiled walk reads the tables below;
`tools/gen_digest_fields.py` writes docs/api/DIGEST_FIELDS.md from the same
tables and the save schemas, so the table and the code cannot disagree.

The field source of a class, first match in its ancestry:

0. The class name hashed is the public estimator's: a CPU twin (`CANONICAL`)
   is hashed under the name of the class it loads, and takes that class's
   field source through its ancestry.
1. `REFUSED`: no digest yet; the read raises AttributeError with the reason.
2. A digest schema (`SCHEMAS`), else the answer members of the connected save
   schema (`_native_archive_bridge.register`: the members `save` writes,
   less the training record and the run settings, `answer_steps`). A fitted
   model the save schema refuses (a format the file does not carry) is hashed
   by rule 3.
3. Otherwise: every public fitted attribute (an instance field whose name ends
   in "_" and does not start with "_", value not None, name not in
   `NOT_STATE` or `TRAINING`), plus the private fields and the constructor
   parameters in `EXTRA` (`EXTRA_BY_NAME`, `PARAMS_BY_NAME`).

`training_digest_` hashes, by rule 3's form, the public `TRAINING` fields the
model holds and the private fields in `TRAINING_EXTRA`.

This module holds declarations and import-time glue only: nothing here runs
when a model is fitted, and the read itself is one native call.
"""
#: class name -> the private instance fields that are fitted state (they
#: decide the model's answers and no public attribute carries them).
#: A name the model does not hold is passed over, so a base class may list
#: the union of its subclasses' fields. Listed here: state `fit` writes that
#: `predict` / `transform` reads. Not listed: device handles, pins, lifecycle
#: flags and caches written by a later call (they are not fitted state, and a
#: handle differs per device).
EXTRA_BY_NAME = {
    # preprocessing: encoders, imputers, discretizers (bindings/prep_encode_native_door.mojo)
    "SimpleImputer": ("_fill", "_keep", "_indicator"),
    "KBinsDiscretizer": ("_edges", "_ne", "_stride"),
    "OrdinalEncoder": ("_missing", "_infrequent", "_grouping"),
    "OneHotEncoder": ("_infrequent", "_grouping", "_drop_after"),
    "SplineTransformer": ("_knots", "_lo", "_hi", "_nk", "_k"),
    "TargetEncoder": ("_T", "_cmax", "_enc", "_meta"),
    "_SelectorMixin": ("_mask",),
    # the imputer's generator word and posterior step advance with each sampled transform:
    # they decide the next transform's answers, so they are state; so do the kept columns marked not missing
    # (`_empty_kept`) and a classifier estimator's per-step labels (`_step_classes`).
    "IterativeImputer": ("_keep", "_indicator", "_bounds_k", "_posterior", "_rng", "_post_step", "_empty_kept",
                         "_step_classes"),
    # stateless-fit transformers (bindings/prep_stateless_native_door.mojo)
    "PolynomialFeatures": ("_poly_index", "_poly_start"),
    "QuantileTransformer": ("_q",),
    "PowerTransformer": ("_pt_anchor", "_pt_anchor_kind", "_mean", "_scale", "_method"),
    # label encoders (bindings/prep_labels_native_door.mojo)
    "LabelEncoder": ("_classes", "_cats"),
    "LabelBinarizer": ("_classes", "_cats"),
    "MultiLabelBinarizer": ("_classes", "_cats", "_given"),
    # naive Bayes and discriminant analysis (bindings/prep_classify_native_door.mojo)
    "GaussianNB": ("_const", "_raw_var"),
    "_DiscreteNB": ("_w", "_bias", "_cc", "_flp", "_cmax"),
    "LinearDiscriminantAnalysis": ("_coef", "_inter", "_rank", "_scal_full", "_max_components"),
    "QuadraticDiscriminantAnalysis": ("_rot", "_logc"),
    # linear models: `coef_` and `intercept_` decide every answer. The centering of the fit (`_x_mean`,
    # `_y_mean`) is not listed: predict never reads it, a saved file does not hold it, and a loaded copy
    # (which has the same answers) must have the same digest.
    # the stacked weights predict reads (bindings/qn_boundary_pack.mojo, bindings/svc_linear_boundary_pack.mojo)
    "QNRegressor": ("_w",),
    "_LinearSVMBase": ("_w",),
    # the link the GLM predict applies (bindings/x_linear_regressors_pack.mojo)
    "_GLMBase": ("_link",),
    # kernel methods: the kernel's parameters and seed as fitted (bindings/kernel_methods_boundary_pack.mojo)
    "KernelRidge": ("_kernel_params", "_squeeze"),
    "Nystroem": ("_kernel_params",),
    "RBFSampler": ("_params",),
    # decomposition and manifold (bindings/ica_native_unpack.mojo, rp_native_door.mojo, pls_boundary_pack.mojo,
    # lda_native_door.mojo, mani_decomp_native_door.mojo, mcd_native_door.mojo)
    "FastICA": ("_whiten",),
    "_RandomProjection": ("_rp_components",),
    "_PLS": ("_x_mean", "_x_std", "_y_mean", "_y_std", "_x_scores", "_y_scores", "_predict_1d"),
    # the draw counter advances with each fit / partial_fit and seeds the next transform
    "LatentDirichletAllocation": ("_seed", "_draw"),
    "Isomap": ("_kind", "_pw", "_k_col", "_k_all", "_fit_X", "_knn", "_iso_v_m_"),
    "LocallyLinearEmbedding": ("_fit_X", "_knn"),
    "MDS": ("_norm",),
    "ClassicalMDS": ("_norm",),
    "MinCovDet": ("_precision", "_seed", "_draws"),
    # clustering (bindings/cluster_fit_public_native_pack.mojo, providers/python/member_xcluster_boundary_door.mojo)
    "BisectingKMeans": ("_tree_centers",),
    "BayesianGaussianMixture": ("_full_pchol", "_log_consts", "_score_words"),
    # neighbors, outliers, kernel maps, semi-supervised, sparse GP (bindings/neighbors_*_native_door.mojo,
    # kpca_native_door.mojo, svc_ocsvm_boundary_pack.mojo).
    "LocalOutlierFactor": ("_fit_X", "_fit_dist", "_lrd"),
    "NearestCentroid": ("_codes_classes",),
    # k-NN outside its save format (string class labels; bindings/neighbors_*_native_door.mojo)
    "NearestNeighbors": ("_index",),
    "KNeighborsClassifier": ("_index", "_y_cols", "_label_names", "_classes_list"),
    "KNeighborsRegressor": ("_index", "_y_cols"),
    "KNNImputer": ("_fit_X", "_valid", "_miss_cols"),
    "OneClassSVM": ("_gamma",),
    "KernelPCA": ("_gamma", "_fit_X", "_fit_cols", "_fit_all"),
    "SkewedChi2Sampler": ("_random_weights",),
    "_LabelPropagationBase": ("_classical_raw_graph",),
    "SVGP": ("_alpha", "_C"),
    # tree ensembles and meta-estimators (bindings/trees_public_boost_door.mojo, trees_public_meta_door.mojo,
    # trees_gbdt_adapter_lifecycle_door.mojo). A member estimator contributes its own digest.
    "_DARTBase": ("_bag_at", "_KIND"),
    "RandomTreesEmbedding": ("_tree_base", "_node_col"),
    "_StackingBase": ("_widths_", "_binary"),
    "MultiOutputRegressor": ("_multi_ridge",),
    "_GBDTAdapter": ("_learner_",),
    # forecasting (bindings/xseq_forecast_native_door.mojo, xseq_prophet_native_door.mojo,
    # tsa_autoarima_native_door.mojo): the training series a forecast extends. `_fitted_h` and `_last` are
    # per-horizon caches a later predict fills: not fitted state.
    "AutoTheta": ("_one", "_y"),
    "_Croston": ("_one", "_y"),
    "ETS": ("_one", "_y", "_season"),
    "GARCH": ("_one", "_y", "_fc"),
    "ProphetForecaster": ("_one", "_nh", "_K"),
    "AutoARIMA": ("_fitted",),
    "_BaseMLP": ("_y_1d",),
    # bagging: the members as one forest (`_batched` = (forest, the unfitted base)) or as a member list
    # (`_estimators`), and each member's out-of-bag rows (bindings/trees_public_boost_door.mojo)
    "_BaggingBase": ("_batched", "_estimators", "_oob_rows"),
    # recurrent models: the weights are the public flat block `params_` (bindings/xseq_neural_native_door.mojo)
    "_RecurrentBase": ("_y_1d",),
    # Sequential: the weights are the public flat block `params_`; the layer list decides what they mean
    # (bindings/sequential_description_unpack.mojo: one tuple of codes and sizes per layer); predict and
    # predict_proba also read the loss the model was fitted with (`_fit_loss`: logits -> labels or
    # probabilities) and whether y was 1-D (`_y_1d`: the answer's shape), bindings/sequential_admission_door.mojo
    # `_read_fields_full`. Not listed: the optimizer and row-order state (`_fit_state`, `_curve`, `_fit_batch`,
    # `_fit_shuffle`, `_target`), which only `fit(..., resume=True)` reads (TRAINING_EXTRA_BY_NAME).
    "Sequential": ("_layers", "_sample_shape", "_fit_loss", "_y_1d"),
    # CNNClassifier: `layers_` and `head_` are layer objects, entered by the walk (LAYERS)
    "CNNClassifier": ("_flat",),
}

#: class name -> the constructor parameters an answer method reads at the call
#: (they decide the answers and no fitted array carries them): a stateless
#: transformer's settings, a coder's transform settings, a network's
#: activation and pooling. Hashed by name and value like the private fields of
#: `EXTRA_BY_NAME` (rule 3). Not listed: parameters that only steer fit (a
#: learning rate, a tolerance, an iteration cap, a seed): their effect is in the
#: fitted arrays, and two fits that end in the same arrays give the same answers.
PARAMS_BY_NAME = {
    # stateless transformers: fit records the input width only
    "Binarizer": ("threshold",),
    "Normalizer": ("norm",),
    "AdditiveChi2Sampler": ("sample_steps", "sample_interval"),
    # kernel maps whose transform scales the input by these (the fitted hashes / weights do not carry them)
    "PolynomialCountSketch": ("gamma", "degree", "coef0"),
    "SkewedChi2Sampler": ("skewedness",),
    # label propagation: predict builds the kernel between new points and the fitted ones
    "_LabelPropagationBase": ("kernel", "gamma", "n_neighbors"),
    # the coders: transform solves a sparse code with these settings for the fitted dictionary
    "_BaseSparseCoding": ("transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha", "split_sign",
                          "positive_code", "transform_max_iter"),
    "DictionaryLearning": ("transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha", "split_sign",
                           "positive_code", "transform_max_iter"),
    "MiniBatchDictionaryLearning": ("transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha",
                                    "split_sign", "positive_code", "transform_max_iter"),
    "SparseCoder": ("transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha", "split_sign",
                    "positive_code", "transform_max_iter"),
    # the ridge of the transform's least squares
    "SparsePCA": ("ridge_alpha",),
    "MiniBatchSparsePCA": ("ridge_alpha",),
    "IncrementalPCA": ("whiten",),
    # predict / decision_function / score_samples exist only with novelty=True
    "LocalOutlierFactor": ("novelty",),
    # the hidden layers' activation (the fitted weights do not say which one predict applies)
    "_BaseMLP": ("activation",),
    # the recurrent network the flat `params_` block belongs to (its length alone does not say), and the
    # plain RNN's activation (bindings/xseq_neural_native_door.mojo); `predict_chunk` only tiles predict
    # and (lane rnn-parity) the head's reading of it: `output` (final state or every step) and `bidirectional`
    "_RecurrentBase": ("hidden_size", "num_layers", "output", "bidirectional"),
    "RNNClassifier": ("hidden_size", "num_layers", "nonlinearity", "output", "bidirectional"),
    "RNNRegressor": ("hidden_size", "num_layers", "nonlinearity", "output", "bidirectional"),
    # the network the fitted weights belong to: pooling and input layout are not in the weights' shapes
    "CNNClassifier": ("input_shape", "conv_channels", "kernel_size", "pool_size"),
    # k-NN: hashed by its save schema; these are for a fitted model the file does not carry (string class
    # labels), which is hashed by rule 3 with the private index and labels of EXTRA_BY_NAME
    "NearestNeighbors": ("n_neighbors", "radius", "metric", "p", "algorithm"),
    "KNeighborsClassifier": ("n_neighbors", "weights", "metric", "p", "algorithm"),
    "KNeighborsRegressor": ("n_neighbors", "weights", "metric", "p", "algorithm"),
}

#: Public names (ending in "_") that record how a call ran, not what was
#: fitted: whether the input was copied on the way in, which shards or which
#: query tile the last call used. They differ with the input's container or
#: the machine while the fitted model is the same, so no class hashes them
#: (neither digest).
NOT_STATE = ("input_copied_", "fortran_copied_", "last_shards_", "used_query_tile_", "n_candidate_distances_",
             # a layer's gradient buffers of the last training step
             "grad_weight_", "grad_bias_", "grad_weight_r_",
             # a search's resolved scorer object: a setting (`scoring`), not a fitted value
             "scorer_",
             # the fit loop's executed count and its wall time (ab-loop-mode, 2026-11-24): timing, never state
             "fit_loop_")

#: Public names that record how the model was trained, not what it answers:
#: iteration, epoch and step counters, loss, score and objective histories,
#: convergence records, the optimizer's moments, a search's per-candidate
#: results (with their fit times). `digest_` never hashes them;
#: `training_digest_` hashes exactly these (and `TRAINING_EXTRA`). A model
#: loaded from an inference export, which drops them, keeps its `digest_`.
TRAINING = (
    # counters
    "n_iter_", "t_", "n_steps_", "n_batch_iter_", "epochs_run_", "best_epoch_", "best_iteration_",
    "num_batches_tracked_",
    # loss, score and objective records
    "loss_", "loss_curve_", "losses_", "validation_loss_curve_", "validation_scores_", "best_loss_",
    "best_validation_score_", "training_loss_", "error_", "reconstruction_err_", "objective_", "dual_gap_", "kl_divergence_", "stress_",
    "lower_bound_", "converged_", "stopped_early_", "retcode_", "info_", "sse_", "criterion_",
    "log_marginal_likelihood_value_", "n_boruvka_rounds_",
    # the optimizer's state, which only a resumed fit reads
    "optimizer_m_", "optimizer_v_", "optimizer_runs_",
    # a search's or a cross-validated model's record of the candidates (scores, fit / score times, paths):
    # the chosen candidate is in the fitted state (`best_estimator_`, `best_params_`, `alpha_`, `C_`)
    "cv_results_", "best_score_", "scores_", "coefs_paths_", "mse_path_",
)

#: class name -> private training fields: state that only a resumed fit reads.
TRAINING_EXTRA_BY_NAME = {
    # bindings/sequential_execution_pack.mojo: the optimizer and row-order counters, the curve buffer, the
    # batch size, shuffle flag and target code of the fit (sequential_checkpoint_door.mojo writes them)
    "Sequential": ("_fit_state", "_curve", "_fit_batch", "_fit_shuffle", "_target",
                   # nn-blocks-pgseq: compile's parameter groups as the fit resolved them (per-tensor table)
                   "_param_groups"),
}

#: Save-schema members that are training records or run settings: `answer_steps`
#: leaves them out of a save schema's digest. The public `TRAINING` and
#: `NOT_STATE` names, plus the settings that only steer a fit (an iteration
#: cap, a tolerance, an initialization, restarts, a covariance floor) and the
#: performance knobs (a query tile, a kernel cache size), none of which any
#: answer method reads.
SCHEMA_NOT_ANSWERS = NOT_STATE + TRAINING + (
    "max_iter", "maxiter", "tol", "query_tile", "cache_size", "nochange_steps", "reg_covar", "init_params",
    "_n_init",
    # the SVMs' solver settings as stored (python/mojolearn/_svm_impl.py `hyper`/`meta`)
    "_hp_tol", "_hp_cache_size", "_hp_max_iter", "_hp_nochange_steps",
)

#: Base classes of layer objects a layered model keeps in its fitted fields
#: (`CNNClassifier.layers_`, `head_`). The walk enters such an object: its
#: class name, then its own public learned fields (`weight_`, `bias_`, ...).
#: Filled by `layer` when the layer module is imported.
LAYERS = []

#: class name -> why it has no digest yet, in words a user reads.
REFUSED_BY_NAME = {}

#: class name -> what is hashed, for a class whose digest is its own compiled
#: walk and not the shared one (documentation for the generated page; the
#: walk itself is named in the row).
OWN_WALK_BY_NAME = {
    # bindings/causal_lm_public_door.mojo `_causal_digest_members`: the walk the checkpoint fingerprint shares
    "CausalLM": "`kind`, `numeric_profile`, `norm_eps`, `max_positions`, `embed`, `norm` (when present), `head`, "
                "then per block its options and every weight (and the int15 planes of a packed model), in model order",
}

#: The tables the compiled walk reads, keyed by class (filled by `declare`).
#: `EXTRA[cls]`: the private fields and the constructor parameters rule 3 adds
#: (`EXTRA_BY_NAME` and `PARAMS_BY_NAME`, each the first entry in the class's
#: ancestry). `TRAINING_EXTRA[cls]`: the private training fields.
EXTRA = {}
TRAINING_EXTRA = {}
REFUSED = {}

#: class -> the schema steps its digest is built from, in place of its save
#: schema (bindings/archive_schema_native_door.mojo runs either). Every class
#: with a save schema has one: by default the save schema's answer members
#: (`answer_steps`, entered by `answers` when the save schema is registered);
#: a class may declare its own (`schema`), for example a tree's, built without
#: the save refusal of a splitter='random' tree. Filled by `answers` and
#: `schema`.
SCHEMAS = {}

#: CPU twin class -> the name of the public class it is hashed under. A `Host*`
#: class is the same estimator bound to the CPU binding (a saved file loaded by
#: `host_model`), so it has the digest of the class that wrote the file: the
#: name in the hash and in the save schema's `estimator` member is the public
#: one. Filled by `twin` where the twins are defined (_classical_host.py).
CANONICAL = {}

#: The one connection of the `digest_` and `training_digest_` properties: the
#: two property objects, the compiled entry and its context. Filled in place by
#: `connect`; `_mode` puts this dictionary in its metadata context, where
#: bindings/fitted_digest_property_pack.mojo reads it.
CONNECTION = {"property": None, "training_property": None, "entry": None, "context": None}

#: The argument tuple of the compiled entry that selects `training_digest_`.
TRAINING_ARGUMENTS = ("training",)


def _first(table, cls):
    """The entry of the first class in `cls`'s ancestry that `table` names, or ()."""
    for klass in cls.__mro__:  # glue: once per class definition, over its ancestry
        if klass.__name__ in table:
            return tuple(table[klass.__name__])
    return ()


def declare(cls):
    """Enter `cls` in the class-keyed tables when a table above names it
    (called once per class, at its definition, by the shared estimator
    base)."""
    name = cls.__name__
    if name in EXTRA_BY_NAME or name in PARAMS_BY_NAME:
        EXTRA[cls] = _first(EXTRA_BY_NAME, cls) + _first(PARAMS_BY_NAME, cls)
    if name in TRAINING_EXTRA_BY_NAME:
        TRAINING_EXTRA[cls] = TRAINING_EXTRA_BY_NAME[name]
    if name in REFUSED_BY_NAME:
        REFUSED[cls] = REFUSED_BY_NAME[name]


def schema(cls, steps):
    """Declare the schema steps `cls` is hashed from, in place of its save
    schema (called once, where the class's save schema is registered)."""
    SCHEMAS[cls] = steps


#: Step kinds that write one member from one field (the field is `step[2]`).
_MEMBER_STEPS = ("str", "lower", "array", "flat", "utf8", "init_name", "classes", "head", "reshape", "device_array")
#: Step kinds that pack several fields into one vector member (entries `step[2]`, each `(kind, field, ...)`).
_PACKED_STEPS = ("i8", "i4", "f8", "f4")
#: Step kinds that run nested steps, and the position of the nested steps.
_NESTED_STEPS = {"when": 3, "unless": 3, "when_in": 3, "when_set": 2, "when_none": 2}


def _not_answer(field):
    return type(field) is str and field in SCHEMA_NOT_ANSWERS


def answer_steps(steps):
    """The save steps `steps` without the members and packed entries read from
    a field in `SCHEMA_NOT_ANSWERS` (a training record or a run setting); every
    other step (the refusals, the answer members) is kept in order."""
    kept = []
    for step in steps:  # glue: once per class definition, over its fixed save steps
        kind = step[0] if type(step) is tuple and step and type(step[0]) is str else None
        if kind in _MEMBER_STEPS and len(step) > 2 and _not_answer(step[2]):
            continue
        if kind in _PACKED_STEPS:
            entries = tuple(entry for entry in step[2]
                            if not (type(entry) is tuple and len(entry) > 1 and _not_answer(entry[1])))
            if not entries:
                continue
            step = (step[0], step[1], entries) + tuple(step[3:])
        elif kind in _NESTED_STEPS:
            at = _NESTED_STEPS[kind]
            step = step[:at] + (answer_steps(step[at]),) + tuple(step[at + 1:])
        kept.append(step)
    return tuple(kept)


def answers(cls, save):
    """Enter the digest schema of a class whose save schema `save` was just
    registered: its answer members (`answer_steps`). A schema the class
    declared with `schema` is kept."""
    SCHEMAS.setdefault(cls, answer_steps(save))


def twin(cls, name):
    """Declare `cls` a CPU twin of the public class called `name` (called
    once per twin, where the twins are defined)."""
    if cls.__name__ != name:
        CANONICAL[cls] = name


def twin_field(cls, field):
    """Declare `cls` a CPU twin of several public classes: each instance is
    hashed under the public class name its field `field` holds (the
    `estimator` a saved file names; HostForest, HostGBDT)."""
    CANONICAL[cls] = (field,)


#: The groves suffix of a forest format (`<format>-parallel-groves-1`).
GROVES_SUFFIX = "-parallel-groves-1"


def forest_steps(fmt, engine):
    """The digest steps of a forest saved in the format `fmt` (a random
    forest, extra trees, or the CPU twin HostForest): the members `save`
    writes that decide the answers. `engine` is the step that adds the
    groves format and its numeric mode for a model predicting with the
    'parallel_groves' engine (its fold order changes the answers' bits).
    Not hashed: the fit settings an extra-trees file also records in `meta`
    (`depth_cap_bound_`, `max_depth_resolved_`, `max_features_`), which no
    prediction reads and a HostForest does not hold."""
    return (
        ("need", "_offsets", "RuntimeError", "this estimator is not fitted yet"),
        ("format", "format", fmt),
        ("estimator", "estimator"),
        ("format", "device", "gpu"),
        ("array", "offsets", "_offsets"),
        ("array", "colid", "_colid"),
        ("array", "quesval", "_quesval"),
        ("array", "left_child", "_left_child"),
        ("array", "leaves", "_leaves"),
        ("i8", "meta", (("int", "n_features_in_"), ("int", "_n_trees"), ("int", "_num_outputs"))),
        ("classes", "classes", "classes_"),
        engine,
    )


def host_forest_engine(fmt):
    """The `engine` step of `forest_steps` for a HostForest: it holds the
    engine its file named (`_inference_engine`); a groves file is refused
    at load unless it was saved under numeric_mode='identical'."""
    return ("when", "_inference_engine", "parallel_groves",
            (("format", "format", fmt + GROVES_SUFFIX), ("format", "numeric_mode", "identical")))


def gbdt_steps(fmt):
    """The digest steps of a gradient-boosting model saved in the format
    `fmt` (GradientBoosting and the CPU twin HostGBDT): the model text, the
    loss, the output shape, the bias and the category map. Not hashed: the
    fit records `best_iteration_` and `stopped_early_`, which the file keeps
    in `meta` but no prediction reads (the model text is already the model
    that predicts)."""
    return (
        ("need", "model_", "AttributeError", "'{cls}' object has no attribute 'model_'"),
        ("need_set", "model_", "RuntimeError", "mojolearn: digest_ before fit()"),
        ("format", "format", fmt),
        ("mode", "numeric_mode", "mojolearn: "),
        ("estimator", "estimator"),
        ("array", "loss", "loss"),
        ("utf8", "model", "model_"),
        ("i8", "meta", (("int", "n_features_in_"), ("int", "approx_dim_"), ("none", "n_classes_", -1))),
        ("f8", "bias", (("float", "bias_"),)),
        ("when_set", "category_map_", (("array", "category_map", "category_map_"),)),
    )


def layer(cls):
    """Declare `cls` a base class of layer objects (called once, where the
    layer classes are defined)."""
    if cls not in LAYERS:
        LAYERS.append(cls)


def connect(base):
    """Connect the compiled digest entry once: `(entry, context)`. `base` is
    the shared estimator base whose `digest_` property this serves. On an
    installation without the saved-model owner the entry raises
    ModuleNotFoundError in words, as `save` does there."""
    if CONNECTION["context"] is not None:
        return CONNECTION["entry"], CONNECTION["context"]
    from ._native_archive_bridge import connect as connect_archive
    entry, context = connect_archive()
    if "archive_unavailable" not in context:
        # `digest_skip`: the public names `digest_` never hashes; `digest_training`: the public names
        # `training_digest_` hashes (with `digest_training_extra`).
        context.update(digest_base=base, digest_extra=EXTRA, digest_refused=REFUSED,
                       digest_skip=NOT_STATE + TRAINING, digest_not_state=NOT_STATE, digest_training=TRAINING,
                       digest_training_extra=TRAINING_EXTRA, digest_layers=LAYERS, digest_canonical=CANONICAL,
                       digest_schemas=SCHEMAS)
        CONNECTION["entry"] = entry
    CONNECTION["context"] = context
    return entry, context


def read(model, arguments=()):
    """`model.digest_` (`arguments` empty) or `model.training_digest_`
    (`TRAINING_ARGUMENTS`): one native call."""
    entry = CONNECTION["entry"]
    if entry is None:
        from ._mode import CheckMixin
        entry = connect(CheckMixin)[0]
    return entry(model, "fitted_digest", arguments, CONNECTION["context"])

