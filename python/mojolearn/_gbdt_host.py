# SPDX-License-Identifier: Apache-2.0
"""CPU inference for saved GradientBoosting models (the GBDT host lane,
2026-09-13, the forest host lane's sibling).

`HostGBDT.from_file(path)` loads a model written by `GradientBoosting.save`
and predicts on the CPU through `_mojolearn_forest_host`, the binding
`core/gbdt_host_predict.mojo` is compiled into with no accelerator target.
`predict` returns what `GradientBoosting.predict` returns for the same file,
the raw float32 approxes, `(n_rows,)` for a one-dimensional loss and
`(n_rows, approx_dim)` otherwise. `predict_proba` is defined for `Logloss`
and `CrossEntropy` and returns the float64 `[1 - p, p]` columns exactly as
`GradientBoosting.predict_proba` builds them, the sigmoid through the host
binding's copy of `gbdt_sigmoid`. The multi-output probability transforms
(`MultiClass` softmax, `MultiClassOneVsAll` sigmoids) run on the device in
`gbdt_predict_multi` and are NOT carried here; `predict_proba` refuses those
losses by name rather than restating an unmeasured transform.

THE MODEL TEXT IS PARSED HERE, in Python, because the Mojo parser
(`gbdt/models/model_text.mojo:706`, `load_model_text`) returns a
`TrainedModel` from `gbdt/train.mojo`, which imports the device side. This
parser reads the same records with the same strictness on what it accepts
(header order, gap-free indices, counts that add up, an unknown keyword is
an error). Every float
comes from the hex bits half of its token, as the Mojo loader reads it; the
decimal half is not consulted (the Mojo loader checks it agrees within one
ULP, which this parser does not, so a hand-edited decimal is not caught
here; the bits are what both sides predict with).

CTR TABLES AND TENSOR CTRS (lane/inference-gbdt-ctr-tables, 2026-09-15).
A model whose categorical input was above `one_hot_max_size` carries
`ctr_columns`, one `ctr_table` per CTR column and its `ctr_entry` counts; an
`ExperimentalTwoLevelFeatureFreq` model whose tree split on a combination
carries `tensor_ctr_registry` and its `feature_freq_tensor` records. They are
parsed here with `load_model_text`'s checks and handed to the forest host
binding's `forest_host_gbdt_expand_ctr`, which rebuilds the tables and calls
the same `expand_raw_columns` and `expand_tensor_ctr_columns` the GPU
`predict_floats` calls, then the ordinary quantize and walk run on the model
columns. What `predict_floats` refuses is refused here, at load: a model
with both kinds, a model declaring more CTR columns than it carries tables,
and a table of a type with no apply-time arithmetic (Buckets,
BinarizedTargetMeanValue, FloatTargetMeanValue), by name. The tensor record's
hash is checked for shape only (two unsigned 32-bit halves); the Mojo reader
recomputes it from the sources and splits, which this parser does not.

This module holds no arithmetic beyond the `1 - p` column
`GradientBoosting.predict_proba` also computes in Python (DEVIATION 2333),
and with a binding that carries `forest_host_gbdt_sigmoid_pair` (DEVIATION
2902) not even that: both columns come from the binding in one pass.
What it promises is what the gate measured: tools/forest_host_gate.py
compares the host predictions of a recorded model and fixture against the
SHA-256 a GPU run recorded.
"""
import hashlib
import struct

from . import _serialize
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty, frombytes
from ._forest_host import _load
from ._labels import flat_view

#: `_MODEL_FORMAT` in ensemble.py.
GBDT_FORMAT = 'mojolearn-gbdt-1'
#: `MODEL_TEXT_NAME` and `MODEL_TEXT_VERSION`, `gbdt/models/model_text.mojo:225-229`.
_TEXT_NAME = 'mojolearn-model'
_TEXT_VERSION = 2
#: `NAN_TREATMENT_*`, `gbdt/data/quantization.mojo:73-76`, keyed by the
#: token `nan_treatment_token` writes (`model_text.mojo:232-245`).
_NAN_CODES = {'as_is': 0, 'as_false': 1, 'as_true': 2}
_PROBA_LOSSES = ('Logloss', 'CrossEntropy')
_MULTI_LOSSES = ('MultiClass', 'MultiClassOneVsAll')
_CLASSIFICATION_LOSSES = _PROBA_LOSSES + _MULTI_LOSSES
#: `ctr_type_name`, `gbdt/ctrs/ctr.mojo`: the names a `ctr_table` may carry.
_CTR_TYPE_CODES = {'Borders': 0, 'Buckets': 1, 'BinarizedTargetMeanValue': 2,
                   'FloatTargetMeanValue': 3, 'Counter': 4, 'FeatureFreq': 5}
#: The types `TCtrValueTable.value_for` has apply-time arithmetic for; the
#: others raise there and are refused here by name.
_CTR_APPLIED = ('Borders', 'Counter', 'FeatureFreq')
#: Records no reader of this version understands, refused by name.
_REFUSED_RECORDS = ()


def _bits32(token, what):
    """The float32 whose IEEE bits are the hex half of `<decimal>/<hex>`."""
    parts = token.split('/')
    if len(parts) != 2 or len(parts[1]) != 8:
        raise ValueError(f"mojolearn: {what} token {token!r} is not <decimal>/<8 hex digits>")
    return int(parts[1], 16)


def _bits64(token, what):
    parts = token.split('/')
    if len(parts) != 2 or len(parts[1]) != 16:
        raise ValueError(f"mojolearn: {what} token {token!r} is not <decimal>/<16 hex digits>")
    return int(parts[1], 16)


def _f32_from_bits(bits):
    return frombytes(struct.pack(f'<{len(bits)}I', *bits), '<f4', (len(bits),))


def _i32(values):
    return Array.from_list([int(v) for v in values], '<i4')


def _padded_i32(values):
    """An int32 Array of at least one element, so the binding never
    receives a null address for an empty split table (a constant-only
    ensemble has no splits); the true count travels in `params`."""
    return _i32(values if values else [0])


def _u32_word(token, what):
    v = int(token)
    if v < 0 or v > 0xffffffff:
        raise ValueError(f'{what} {token!r} does not fit an unsigned 32-bit word')
    return v


def _parse_tensor_table(t):
    """One `feature_freq_tensor 2` record, with
    `parse_feature_freq_tensor_table`'s checks (`gbdt/models/
    tensor_ctr_value_table.mojo`) except the recomputed tensor hash and the
    canonical split order, which need the tensor builder."""
    p = [1]

    def word(expect=None):
        if p[0] >= len(t):
            raise ValueError('a truncated feature_freq_tensor record')
        tok = t[p[0]]
        p[0] += 1
        if expect is not None and tok != expect:
            raise ValueError(f'expected tensor {expect}, got {tok!r}')
        return tok

    if int(word()) != 2:
        raise ValueError('unsupported feature_freq_tensor version')
    word('hash_hi'); _u32_word(word(), 'tensor hash_hi')
    word('hash_lo'); _u32_word(word(), 'tensor hash_lo')
    word('sources')
    n_sources = int(word())
    sources = [int(word()) for _ in range(n_sources)]
    word('cardinalities')
    if int(word()) != n_sources:
        raise ValueError('tensor source/cardinality count mismatch')
    cards, product = [], 1
    for _ in range(n_sources):
        card = int(word())
        if card < 1 or product > 10000000 // card:
            raise ValueError('invalid tensor cardinality product')
        product *= card
        cards.append(card)
    word('splits')
    n_splits = int(word())
    if n_splits < 0 or n_splits > 30:
        raise ValueError('invalid tensor split count')
    splits = []
    for _ in range(n_splits):
        feature, bin_idx, kind = int(word()), int(word()), int(word())
        if feature < 0 or feature > 2147483647 or bin_idx < 0 or bin_idx > 2147483647 or kind not in (0, 1):
            raise ValueError('invalid tensor split record')
        if product > 10000000 // 2:
            raise ValueError('tensor split table exceeds 10,000,000 entries')
        product *= 2
        splits.append((feature, bin_idx, kind))
    word('classes')
    classes = int(word())
    if classes != 0 and (classes < 2 or classes > 256):
        raise ValueError('tensor target classes must be zero or at least two')
    word('target_border')
    target_border = int(word())
    if (classes == 0 and target_border != -1) or (classes > 0 and not 0 <= target_border < classes - 1):
        raise ValueError('invalid tensor target border')
    word('prior_bits')
    prior_num, prior_denom = _u32_word(word(), 'tensor prior'), _u32_word(word(), 'tensor prior')
    if classes > 0 and prior_denom & 0x7fffffff == 0:
        raise ValueError('Borders tensor prior denominator must be non-zero')
    word('denominator')
    denominator = int(word())
    if denominator < 0:
        raise ValueError('tensor denominator must be non-negative')
    word('counts')
    n_counts = int(word())
    width = classes if classes > 0 else 1
    if product > 10000000 // width or n_counts != product * width:
        raise ValueError('tensor count length does not match cardinalities')
    counts = [int(word()) for _ in range(n_counts)]
    if any(c < 0 for c in counts):
        raise ValueError('tensor counts must be non-negative')
    if classes == 0 and sum(counts) != denominator:
        raise ValueError('tensor counts do not sum to denominator')
    if classes > 0 and denominator != 0:
        raise ValueError('Borders tensor denominator must be zero')
    if p[0] != len(t):
        raise ValueError('trailing fields in feature_freq_tensor record')
    if any(s < 0 for s in sources) or any(a >= b for a, b in zip(sources, sources[1:])):
        raise ValueError('tensor sources are not canonical and unique')
    return dict(sources=sources, cardinalities=cards, splits=splits, classes=classes,
                target_border=target_border, prior_num=prior_num, prior_denom=prior_denom,
                denominator=denominator, counts=counts)


def _ctr_plan(n_features, kinds, ctr_column_count, ctr_tables, tensor_first, tensor_declared, tensor_tables):
    """`load_model_text`'s cross-checks of the CTR half, `predict_floats`'s
    refusals, and `model_input_features`: the RAW input column count."""
    for tab in ctr_tables:
        width = tab['classes'] if tab['classes'] > 0 else 1
        if tab['seen'] != tab['entries'] or len(tab['counts']) != tab['entries'] * width:
            raise ValueError(f"mojolearn: ctr_table for column {tab['column']} declares {tab['entries']} "
                             f"entries and carries {tab['seen']}")
        if kinds[tab['column']] != 'ctr':
            raise ValueError(f"mojolearn: column {tab['column']} carries a CTR table and its `feature` "
                             f"record says type '{kinds[tab['column']]}'")
    named = {tab['column'] for tab in ctr_tables}
    for f, k in enumerate(kinds):
        if k == 'ctr' and f not in named:
            raise ValueError(f'mojolearn: column {f} says type ctr and no `ctr_table` record names it')
    if len(ctr_tables) > ctr_column_count:
        raise ValueError(f'mojolearn: the header declares {ctr_column_count} CTR columns and the file '
                         f'carries {len(ctr_tables)} CTR tables')
    if len(tensor_tables) != tensor_declared:
        raise ValueError(f'mojolearn: tensor CTR registry declares {tensor_declared} tables and carries '
                         f'{len(tensor_tables)}')
    if tensor_first >= 0 and tensor_first + tensor_declared > n_features:
        raise ValueError('mojolearn: tensor CTR registry exceeds model feature columns')
    for f, k in enumerate(kinds):
        in_registry = tensor_first >= 0 and tensor_first <= f < tensor_first + tensor_declared
        if (k == 'tensor_ctr') != in_registry:
            raise ValueError(f'mojolearn: feature {f} of type {k} and the tensor CTR registry disagree')
    if tensor_tables and ctr_tables:
        raise ValueError('mojolearn: combined simple-CTR and tensor-CTR model apply needs a composed '
                         'column plan and is not wired yet (predict_floats refuses this model)')
    if ctr_column_count != len(ctr_tables):
        raise ValueError(
            f'mojolearn: a model with {ctr_column_count} CTR columns and {len(ctr_tables)} CTR tables '
            'cannot score a new row: a CTR value is a statistic of the learn pool (predict_floats '
            'refuses it)')
    if tensor_tables:
        if tensor_first + tensor_declared != n_features:
            raise ValueError('mojolearn: tensor CTR apply plan does not end at the last model column')
        return tensor_first
    if not ctr_tables:
        return n_features
    # `column_plan` (gbdt/models/ctr_value_table.mojo)
    table_of = {tab['column']: tab for tab in ctr_tables}
    c = f = 0
    while c < n_features:
        tab = table_of.get(c)
        if tab is None:
            c += 1
            f += 1
            continue
        if tab['source'] != f:
            raise ValueError(f"mojolearn: CTR table for column {c} names input feature {tab['source']}, "
                             f"but the columns before it account for {f} inputs")
        src = tab['source']
        while c < n_features and c in table_of and table_of[c]['source'] == src:
            c += 1
        f += 1
    return f


def parse_model_text(text):
    """The flat arrays of a `mojolearn-model 2` text. Returns a dict; see
    `HostGBDT.__init__` for the members. Mirrors `load_model_text`'s
    refusals for everything the host walk reads."""
    header = 0
    n_features = n_flags = n_trees = n_losses = None
    bias = 0.0
    bias_seen = False
    fold_counts, one_hot, nan_codes, border_lists = [], [], [], []
    trees = []          # per tree: dict(shape, size, dim, splits, leaves)
    losses_seen = 0
    kinds = []          # the `type` token of each feature
    ctr_column_count = 0
    ctr_tables = []     # dict(column, source, type, ints..., counts)
    tensor_first, tensor_declared, tensor_tables = -1, 0, []
    for line_no, raw in enumerate(text.split('\n'), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        t = line.split()
        kind = t[0]
        try:
            if kind == 'format':
                if header != 0:
                    raise ValueError('a second `format` record')
                if len(t) != 3 or t[1] != _TEXT_NAME:
                    raise ValueError(f'not a {_TEXT_NAME} file')
                if int(t[2]) != _TEXT_VERSION:
                    raise ValueError(f'format version {t[2]}, this reader is version {_TEXT_VERSION}')
                header = 1
            elif kind == 'features':
                if header != 1:
                    raise ValueError('`features` must follow `format`')
                n_features, n_flags = int(t[1]), int(t[2])
                if n_features < 0 or n_flags not in (0, n_features):
                    raise ValueError(f'one_hot flag count {n_flags} is neither 0 nor {n_features}')
                header = 2
            elif kind == 'trees':
                if header != 2:
                    raise ValueError('`trees` must follow `features`')
                n_trees = int(t[1])
                header = 3
            elif kind == 'losses':
                if header != 3:
                    raise ValueError('`losses` must follow `trees`')
                n_losses = int(t[1])
                header = 4
            elif kind == 'bias':
                if header != 4 or bias_seen:
                    raise ValueError('`bias` must follow `losses`, once')
                if len(t) != 2:
                    raise ValueError('`bias` takes one float token')
                bias = struct.unpack('<d', struct.pack('<Q', _bits64(t[1], 'bias')))[0]
                bias_seen = True
            elif kind == 'ctr_columns':
                if header != 4 or len(t) != 2:
                    raise ValueError('`ctr_columns` must follow `losses`')
                ctr_column_count = int(t[1])
            elif kind == 'tensor_ctr_registry':
                if len(t) != 3 or tensor_first != -1:
                    raise ValueError('malformed or duplicate tensor_ctr_registry')
                tensor_first, tensor_declared = int(t[1]), int(t[2])
                if tensor_first < 0 or tensor_declared < 0:
                    raise ValueError('negative tensor CTR registry field')
            elif kind == 'feature_freq_tensor':
                if tensor_first < 0:
                    raise ValueError('tensor table appears before its registry')
                tensor_tables.append(_parse_tensor_table(t))
            elif kind == 'ctr_table':
                if len(fold_counts) != n_features:
                    raise ValueError('a `ctr_table` before the feature block ended')
                if trees:
                    raise ValueError('a `ctr_table` after the first `tree`')
                if len(t) != 22 or [t[i] for i in range(2, 22, 2)] != [
                        'source', 'type', 'prior_num', 'prior_denom', 'shift', 'scale', 'denom',
                        'classes', 'target_border', 'entries']:
                    raise ValueError('a `ctr_table` record has 22 fields')
                col = int(t[1])
                if col < 0 or col >= n_features:
                    raise ValueError(f'ctr_table names column {col} of {n_features}')
                if ctr_tables and col <= ctr_tables[-1]['column']:
                    raise ValueError('ctr_table records must arrive in ascending column order')
                if t[5] not in _CTR_TYPE_CODES:
                    raise ValueError(f"unknown ctr type name '{t[5]}'")
                if t[5] not in _CTR_APPLIED:
                    raise ValueError(
                        f'a `ctr_table` of type {t[5]}; no apply-time table arithmetic is implemented '
                        'for it (TCtrValueTable.value_for applies Borders, Counter and FeatureFreq), '
                        'so the model is refused')
                classes, border_idx, entries = int(t[17]), int(t[19]), int(t[21])
                if classes < 0 or classes == 1:
                    raise ValueError(f'ctr_table declares {classes} target classes')
                if border_idx < 0 or entries < 0:
                    raise ValueError('ctr_table declares a negative target_border or entry count')
                ctr_tables.append(dict(
                    column=col, source=int(t[3]), type=_CTR_TYPE_CODES[t[5]], type_name=t[5],
                    prior_num=_bits32(t[7], 'prior_num'), prior_denom=_bits32(t[9], 'prior_denom'),
                    shift=_bits32(t[11], 'shift'), scale=_bits32(t[13], 'scale'), denom=int(t[15]),
                    classes=classes, target_border=border_idx, entries=entries, seen=0, counts=[]))
            elif kind == 'ctr_entry':
                if not ctr_tables:
                    raise ValueError('a `ctr_entry` before any `ctr_table`')
                tab = ctr_tables[-1]
                if int(t[1]) != tab['column']:
                    raise ValueError(f"a `ctr_entry` for column {t[1]} under the `ctr_table` for column {tab['column']}")
                width = tab['classes'] if tab['classes'] > 0 else 1
                if len(t) != 3 + width:
                    raise ValueError(f'a `ctr_entry` record has {3 + width} fields, this one has {len(t)}')
                if int(t[2]) != tab['seen']:
                    raise ValueError(f"column {tab['column']} categories must arrive in order")
                tab['counts'].extend(int(v) for v in t[3:])
                tab['seen'] += 1
            elif kind == 'feature':
                if header != 4:
                    raise ValueError('a `feature` record before the header ended')
                f = int(t[1])
                if f != len(fold_counts) or f >= n_features:
                    raise ValueError(f'feature {f} out of order')
                if (len(t) < 12 or t[2] != 'folds' or t[4] != 'one_hot' or t[6] != 'type'
                        or t[8] != 'nan' or t[10] != 'borders'):
                    raise ValueError('malformed `feature` record')
                folds, flag, ftype, nan_tok, n_b = int(t[3]), int(t[5]), t[7], t[9], int(t[11])
                if ftype not in ('float', 'cat', 'ctr', 'tensor_ctr'):
                    raise ValueError(f'feature {f} has unknown type {ftype!r}')
                if flag not in (0, 1) or (ftype == 'cat') != (flag == 1 and n_flags != 0):
                    raise ValueError(f'feature {f}: type {ftype} and one_hot {flag} disagree')
                if nan_tok not in _NAN_CODES:
                    raise ValueError(f'feature {f} has unknown nan treatment {nan_tok!r}')
                if folds < 0 or n_b < 0 or len(t) != 12 + n_b:
                    raise ValueError(f'feature {f} declares {n_b} borders and carries {len(t) - 12}')
                fold_counts.append(folds)
                kinds.append(ftype)
                one_hot.append(flag if n_flags else 0)
                nan_codes.append(_NAN_CODES[nan_tok])
                border_lists.append([_bits32(tok, f'feature {f} border') for tok in t[12:]])
            elif kind in ('tree', 'ntree'):
                if len(fold_counts) != n_features:
                    raise ValueError('a tree before every feature was read')
                ti = int(t[1])
                if ti != len(trees) or ti >= n_trees:
                    raise ValueError(f'tree {ti} out of order')
                if kind == 'tree':
                    if len(t) != 8 or t[2] != 'depth' or t[4] != 'dim' or t[6] != 'weights':
                        raise ValueError('a `tree` record has 8 fields')
                    size = int(t[3])
                    if size < 0 or size > 31:
                        raise ValueError(f'tree depth {size} is not sane')
                    n_leaves = 1 << size
                else:
                    if len(t) != 8 or t[2] != 'nodes' or t[4] != 'dim' or t[6] != 'weights':
                        raise ValueError('an `ntree` record has 8 fields')
                    size = int(t[3])
                    if size < 0:
                        raise ValueError('negative node count')
                    n_leaves = size + 1
                dim = int(t[5])
                if dim < 1:
                    raise ValueError(f'tree {ti} has dim {dim}')
                if trees and trees[0]['shape'] != kind:
                    raise ValueError('a file mixing oblivious and non-symmetric trees')
                trees.append(dict(shape=kind, size=size, dim=dim, n_leaves=n_leaves,
                                  splits=[], leaves=[], weights=int(t[7])))
            elif kind in ('split', 'node'):
                ti = int(t[1])
                if ti != len(trees) - 1 or ti < 0:
                    raise ValueError(f'a `{kind}` for tree {ti} outside that tree')
                tree = trees[ti]
                if tree['shape'] != ('tree' if kind == 'split' else 'ntree'):
                    raise ValueError(f'a `{kind}` record in a tree of the other shape')
                i = int(t[2])
                if i != len(tree['splits']) or i >= tree['size']:
                    raise ValueError(f'`{kind}` {i} of tree {ti} out of order')
                base = 5 if kind == 'split' else 7
                take_bin = 0
                if len(t) == base + 2:
                    if t[base] != 'split_type' or t[base + 1] != 'take_bin':
                        raise ValueError(f'a `{kind}` record\'s split_type is `take_bin` or absent')
                    take_bin = 1
                elif len(t) != base:
                    raise ValueError(f'a `{kind}` record has {base} fields or `split_type take_bin`')
                feature, bin_idx = int(t[3]), int(t[4])
                if feature < 0 or feature >= n_features or bin_idx < 0:
                    raise ValueError(f'`{kind}` {i} of tree {ti} names feature {feature} bin {bin_idx}')
                if take_bin != one_hot[feature]:
                    raise ValueError(
                        f'`{kind}` {i} of tree {ti} is a {"TakeBin" if take_bin else "TakeGreater"} '
                        f'split on feature {feature}, which is {"one-hot" if one_hot[feature] else "ordered"}')
                if kind == 'split':
                    tree['splits'].append((feature, bin_idx, take_bin, 0, 0))
                else:
                    left, right = int(t[5]), int(t[6])
                    if left < 1 or right < 1 or left > 65535 or right > 65535:
                        raise ValueError(f'node {i} of tree {ti} has subtrees {left} and {right}')
                    tree['splits'].append((feature, bin_idx, take_bin, left, right))
            elif kind == 'leaf':
                ti = int(t[1])
                if ti != len(trees) - 1 or ti < 0:
                    raise ValueError(f'a `leaf` for tree {ti} outside that tree')
                tree = trees[ti]
                if len(tree['splits']) != tree['size']:
                    raise ValueError(f'a `leaf` before every split of tree {ti}')
                i = int(t[2])
                if i != len(tree['leaves']) or i >= tree['n_leaves'] * tree['dim'] or len(t) != 4:
                    raise ValueError(f'leaf {i} of tree {ti} out of order')
                tree['leaves'].append(_bits32(t[3], f'leaf {i} of tree {ti}'))
            elif kind == 'weight':
                ti = int(t[1])
                if ti != len(trees) - 1 or ti < 0 or not trees[ti]['weights'] or len(t) != 4:
                    raise ValueError(f'a `weight` for tree {ti} that declares none')
                # leaf weights are not read by any apply path; the token is
                # validated for shape only
                (_bits64 if trees[ti]['shape'] == 'ntree' else _bits32)(t[3], 'weight')
            elif kind == 'loss':
                if header != 4 or int(t[1]) != losses_seen or len(t) != 3:
                    raise ValueError('a `loss` record out of order')
                _bits64(t[2], 'loss')
                losses_seen += 1
            else:
                raise ValueError(f'unknown record keyword {kind!r}')
        except (ValueError, IndexError) as exc:
            raise ValueError(f'mojolearn: model text line {line_no}: {exc}') from None
    if header != 4:
        raise ValueError('mojolearn: model text is missing its header')
    if len(fold_counts) != n_features or len(trees) != n_trees or losses_seen != n_losses:
        raise ValueError('mojolearn: model text counts do not add up')
    for ti, tree in enumerate(trees):
        if len(tree['splits']) != tree['size'] or len(tree['leaves']) != tree['n_leaves'] * tree['dim']:
            raise ValueError(f'mojolearn: tree {ti} is incomplete')
    n_input = _ctr_plan(n_features, kinds, ctr_column_count, ctr_tables, tensor_first,
                        tensor_declared, tensor_tables)
    dims = {tree['dim'] for tree in trees}
    if len(dims) > 1:
        raise ValueError('mojolearn: trees disagree on dim')
    dim = dims.pop() if dims else 1
    shape = trees[0]['shape'] if trees else 'tree'
    border_offsets = [0]
    for borders in border_lists:
        border_offsets.append(border_offsets[-1] + len(borders))
    tree_offsets, leaf_offsets = [0], [0]
    for tree in trees:
        tree_offsets.append(tree_offsets[-1] + tree['size'])
        leaf_offsets.append(leaf_offsets[-1] + len(tree['leaves']))
    splits = [s for tree in trees for s in tree['splits']]
    ctr_ints, ctr_floats, tensor_ints, tensor_floats, counts = [], [], [], [], []
    for tab in ctr_tables:
        ctr_ints += [tab['column'], tab['source'], tab['type'], tab['denom'], tab['classes'],
                     tab['target_border'], len(counts), len(tab['counts'])]
        ctr_floats += [tab['prior_num'], tab['prior_denom'], tab['shift'], tab['scale']]
        counts += tab['counts']
    for tab in tensor_tables:
        tensor_ints += [len(tab['sources'])] + tab['sources'] + tab['cardinalities'] + [len(tab['splits'])]
        for split in tab['splits']:
            tensor_ints += list(split)
        tensor_ints += [tab['classes'], tab['target_border'], tab['denominator'], len(counts), len(tab['counts'])]
        tensor_floats += [tab['prior_num'], tab['prior_denom']]
        counts += tab['counts']
    if any(c > 2147483647 for c in counts + tensor_ints + ctr_ints):
        raise ValueError('mojolearn: a CTR count or field does not fit int32')
    return dict(
        n_features=n_features, n_trees=n_trees, dim=dim, bias=bias,
        n_input_features=n_input,
        n_ctr_tables=len(ctr_tables), n_tensor_tables=len(tensor_tables),
        ctr_ints=_padded_i32(ctr_ints), n_ctr_ints=len(ctr_ints),
        ctr_floats=_f32_from_bits(ctr_floats or [0]), n_ctr_floats=len(ctr_floats),
        tensor_ints=_padded_i32(tensor_ints), n_tensor_ints=len(tensor_ints),
        tensor_floats=_f32_from_bits(tensor_floats or [0]), n_tensor_floats=len(tensor_floats),
        ctr_counts=_padded_i32(counts), n_ctr_counts=len(counts),
        non_symmetric=(shape == 'ntree'),
        border_offsets=_i32(border_offsets),
        borders=_f32_from_bits([b for borders in border_lists for b in borders] or [0]),
        n_borders=border_offsets[-1],
        fold_counts=_padded_i32(fold_counts), one_hot=_padded_i32(one_hot),
        nan_treatment=_padded_i32(nan_codes),
        tree_offsets=_i32(tree_offsets),
        split_feature=_padded_i32([s[0] for s in splits]),
        split_bin=_padded_i32([s[1] for s in splits]),
        split_take_bin=_padded_i32([s[2] for s in splits]),
        node_left=_padded_i32([s[3] for s in splits]),
        node_right=_padded_i32([s[4] for s in splits]),
        n_splits=len(splits),
        leaf_offsets=_i32(leaf_offsets),
        leaves=_f32_from_bits([v for tree in trees for v in tree['leaves']] or [0]),
        n_leaf_values=leaf_offsets[-1],
    )


#: The `estimator` members a `mojolearn-gbdt-1` archive may carry, with the
#: losses each class can save. `OrderedRMSE` and
#: `ExperimentalTwoLevelFeatureFreq` subclass `GradientBoosting` and inherit
#: its `save`, so their archives are the same format holding the same model
#: text records (symmetric trees, float and one-hot `cat` features); the name
#: is kept so a report says which class trained the file. A class whose model
#: text carries CTR or tensor CTR records is still refused by the parser, by
#: record name, whatever the class.
GBDT_ESTIMATORS = {
    'GradientBoosting': None,
    'OrderedRMSE': ('RMSE',),
    'ExperimentalTwoLevelFeatureFreq': ('RMSE',),
}


class HostGBDT:
    """A saved GradientBoosting, OrderedRMSE or ExperimentalTwoLevelFeatureFreq
    model that predicts on the CPU."""

    estimator = 'GradientBoosting'

    def __init__(self, *, loss, text, n_features_in, approx_dim, n_classes=None,
                 numeric_mode=None, bias=None, estimator='GradientBoosting'):
        if estimator not in GBDT_ESTIMATORS:
            raise ValueError(f"mojolearn: {estimator!r} is not a gradient boosting class this loader reads")
        losses = GBDT_ESTIMATORS[estimator]
        if losses is not None and str(loss) not in losses:
            raise ValueError(f"mojolearn: a {estimator} archive with loss {loss!r}; that class saves "
                             f"{', '.join(losses)} only, the file is corrupt")
        self.estimator = estimator
        self.loss = str(loss)
        self.numeric_mode = numeric_mode
        self.model_ = str(text)
        arrays = parse_model_text(self.model_)
        if arrays['n_input_features'] != int(n_features_in):
            raise ValueError(
                f"mojolearn: the archive says {int(n_features_in)} features, its model "
                f"text reads {arrays['n_input_features']}; the file is corrupt")
        if arrays['dim'] != int(approx_dim):
            raise ValueError(
                f"mojolearn: the archive stores approx_dim {int(approx_dim)} but its "
                f"model text holds {arrays['dim']}; the file is corrupt")
        if bias is not None and float(bias) != arrays['bias']:
            raise ValueError(
                f"mojolearn: the archive stores bias {float(bias)!r} but its model text "
                f"holds {arrays['bias']!r}; the file is corrupt")
        self._arrays = arrays
        self.n_features_in_ = arrays['n_input_features']
        #: model columns: the input columns, or more for a CTR model
        self.n_model_columns_ = arrays['n_features']
        self.approx_dim_ = arrays['dim']
        self.bias_ = arrays['bias']
        self.n_classes_ = None if n_classes is None else int(n_classes)
        self.is_classifier = self.loss in _CLASSIFICATION_LOSSES
        self.has_proba = self.loss in _PROBA_LOSSES
        self._binding = _load()

    @classmethod
    def from_file(cls, path):
        """A model from a file written by `GradientBoosting.save` (or the
        `OrderedRMSE` and `ExperimentalTwoLevelFeatureFreq` save it inherits)."""
        arrays = _serialize.read_npz(path, GBDT_FORMAT)
        saved_as = _serialize.scalar_str(arrays, 'estimator')
        if saved_as not in GBDT_ESTIMATORS:
            raise ValueError(f"mojolearn: {path!r} was saved by {saved_as}, not "
                             f"{' or '.join(GBDT_ESTIMATORS)}")
        mode = None
        if 'numeric_mode' in arrays:
            mode = _serialize.scalar_str(arrays, 'numeric_mode')
            if mode not in ('fast', 'deterministic', 'identical'):
                raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
        text = _serialize.exact(arrays, 'model', '<u1').tobytes().decode('utf-8')
        meta = _serialize.exact(arrays, 'meta', '<i8')
        if meta.size < 3:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, at least 3 are needed")
        bias = _serialize.exact(arrays, 'bias', '<f8').tolist()
        while isinstance(bias, list):
            bias = bias[0]
        return cls(loss=_serialize.scalar_str(arrays, 'loss'), text=text,
                   n_features_in=int(meta[0]), approx_dim=int(meta[1]),
                   n_classes=None if int(meta[2]) < 0 else int(meta[2]),
                   numeric_mode=mode, bias=float(bias), estimator=saved_as)

    @property
    def n_trees(self):
        return self._arrays['n_trees']

    @property
    def inference_engine(self):
        return 'sequential'

    def model_sha256(self):
        """SHA-256 over the model text's UTF-8, which is the archive's
        `model` member byte for byte."""
        return hashlib.sha256(self.model_.encode('utf-8')).hexdigest()

    def predict(self, X):
        """The raw approxes, float32, `(n_rows,)` or `(n_rows, approx_dim)`,
        as `GradientBoosting.predict` returns them."""
        # `GradientBoosting._check_fitted` stages X COLUMN-major through
        # `as_f32_colmajor`, whose 2-D flip is the base binding's
        # `transpose_f32`, a stub on a CPU-only install. The float32 cast is
        # the same `cast_f64_to_f32` helper here (the host binding carries
        # it), and the flip is the package's own `_reorder`, which
        # `transpose_f32` documents itself as ("a pure move, no arithmetic:
        # every float32 bit pattern, NaN payloads included, arrives
        # unchanged", `bindings/_mojolearn.mojo:765-768`); the bytes the
        # binding reads are therefore the GPU path's by that contract.
        Xc, _ = as_f32_c(X, ndim=2, name='X')
        n_rows, n_features = Xc.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"mojolearn: model was fitted on {self.n_features_in_} features, got {n_features}")
        a = self._arrays
        dim = self.approx_dim_
        n_cols = self.n_model_columns_
        row_major = not (a['n_ctr_tables'] or a['n_tensor_tables'])
        Xa = Xc if row_major else Xc._as_order('F')
        if a['n_ctr_tables'] or a['n_tensor_tables']:
            expanded = empty((n_rows * n_cols,), '<f4')
            names = ('ctr_ints', 'ctr_floats', 'tensor_ints', 'tensor_floats', 'ctr_counts',
                     'border_offsets', 'borders', 'one_hot')
            wrote = self._binding.forest_host_gbdt_expand_ctr(
                [addr_ro(Xa, name='X')] + [addr_ro(a[name], name=name) for name in names]
                + [addr(expanded, name='expanded')],
                [int(n_rows), int(n_features), n_cols, a['n_ctr_tables'], a['n_ctr_ints'],
                 a['n_ctr_floats'], a['n_tensor_tables'], a['n_tensor_ints'], a['n_tensor_floats'],
                 a['n_ctr_counts'], a['n_borders']])
            if int(wrote) != n_cols:
                raise RuntimeError(f"forest_host_gbdt_expand_ctr wrote {wrote} of {n_cols} columns")
            Xa, n_features = expanded, n_cols
        out = empty((n_rows * dim,), '<f4')
        names = ('border_offsets', 'borders', 'fold_counts', 'one_hot', 'nan_treatment',
                 'tree_offsets', 'split_feature', 'split_bin', 'split_take_bin',
                 'node_left', 'node_right', 'leaf_offsets', 'leaves')
        addresses = [addr_ro(a[name], name=name) for name in names]
        addresses += [addr_ro(Xa, name='X'), addr(out, name='out')]
        wrote = self._binding.forest_host_gbdt_predict(
            addresses,
            [int(n_rows), int(n_features), a['n_trees'], dim, 1 if a['non_symmetric'] else 0,
             a['n_splits'], a['n_leaf_values'], a['n_borders'], 1 if row_major else 0],
            float(self.bias_))
        if int(wrote) != n_rows:
            raise RuntimeError(f"forest_host_gbdt_predict wrote {wrote} of {n_rows} rows")
        return out if dim == 1 else out.reshape((n_rows, dim))

    def predict_proba(self, X):
        """`GradientBoosting.predict_proba` for Logloss and CrossEntropy:
        the raw score widened to float64, the sigmoid through the binding,
        and the `[1 - p, p]` columns built the way ensemble.py builds them."""
        if self.loss in _MULTI_LOSSES:
            raise NotImplementedError(
                f"mojolearn: HostGBDT.predict_proba does not carry the {self.loss} "
                "transform; the GPU binding applies it in gbdt_predict_multi and no host "
                "restatement has been measured")
        if not self.has_proba:
            raise ValueError(
                f"mojolearn: predict_proba is defined for Logloss and CrossEntropy; this "
                f"model was fitted with {self.loss!r}")
        raw = self.predict(X).astype('<f8')
        n_rows = raw.shape[0]
        pair = getattr(self._binding, 'forest_host_gbdt_sigmoid_pair', None)
        if pair is not None:
            # DEVIATION 2902 (lane/infer-speed-trees, 2026-09-17): both
            # columns from the binding in one pass, the same `p` and the
            # same one double subtraction per row as the comprehension
            # below; the O(rows) Python loop is gone, the bits are not.
            out = empty((n_rows, 2), '<f8')
            wrote = pair(addr_ro(raw, name='raw'), addr(out, name='proba'), n_rows)
            if int(wrote) != n_rows:
                raise RuntimeError(f"forest_host_gbdt_sigmoid_pair wrote {wrote} of {n_rows} rows")
            return out
        p1 = empty((n_rows,), '<f8')
        self._binding.forest_host_gbdt_sigmoid(addr_ro(raw, name='raw'), addr(p1, name='p1'), n_rows)
        pv = flat_view(p1, 'd')
        return Array.from_list([[1.0 - p, p] for p in pv], '<f8')
