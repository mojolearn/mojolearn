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
an error) and REFUSES BY NAME what the host walk does not carry: CTR columns
and tables (`ctr_columns`, `ctr_table`, `ctr_entry`, `tensor_ctr_registry`,
`feature_freq_tensor`, a feature of type `ctr` or `tensor_ctr`). Every float
comes from the hex bits half of its token, as the Mojo loader reads it; the
decimal half is not consulted (the Mojo loader checks it agrees within one
ULP, which this parser does not, so a hand-edited decimal is not caught
here; the bits are what both sides predict with).

This module holds no arithmetic beyond the `1 - p` column
`GradientBoosting.predict_proba` also computes in Python (DEVIATION 2333).
What it promises is what the gate measured: tools/forest_host_gate.py
compares the host predictions of a recorded model and fixture against the
SHA-256 a GPU run recorded, and the brief in
docs/lanes/BRIEF_forest_host_inference_2026-09-13.md records on which CPUs
that has passed.
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
#: Records the host walk does not carry, refused by name.
_REFUSED_RECORDS = ('ctr_columns', 'ctr_table', 'ctr_entry', 'tensor_ctr_registry',
                    'feature_freq_tensor')


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
            elif kind in _REFUSED_RECORDS:
                raise ValueError(
                    f'a `{kind}` record; the host walk applies float and one-hot '
                    'features only, CTR models are refused')
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
                if ftype in ('ctr', 'tensor_ctr'):
                    raise ValueError(f'feature {f} is of type {ftype}; the host walk refuses CTR models')
                if ftype not in ('float', 'cat'):
                    raise ValueError(f'feature {f} has unknown type {ftype!r}')
                if flag not in (0, 1) or (ftype == 'cat') != (flag == 1 and n_flags != 0):
                    raise ValueError(f'feature {f}: type {ftype} and one_hot {flag} disagree')
                if nan_tok not in _NAN_CODES:
                    raise ValueError(f'feature {f} has unknown nan treatment {nan_tok!r}')
                if folds < 0 or n_b < 0 or len(t) != 12 + n_b:
                    raise ValueError(f'feature {f} declares {n_b} borders and carries {len(t) - 12}')
                fold_counts.append(folds)
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
    return dict(
        n_features=n_features, n_trees=n_trees, dim=dim, bias=bias,
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


class HostGBDT:
    """A saved GradientBoosting model that predicts on the CPU."""

    estimator = 'GradientBoosting'

    def __init__(self, *, loss, text, n_features_in, approx_dim, n_classes=None,
                 numeric_mode=None, bias=None):
        self.loss = str(loss)
        self.numeric_mode = numeric_mode
        self.model_ = str(text)
        arrays = parse_model_text(self.model_)
        if arrays['n_features'] != int(n_features_in):
            raise ValueError(
                f"mojolearn: the archive says {int(n_features_in)} features, its model "
                f"text holds {arrays['n_features']}; the file is corrupt")
        if arrays['dim'] != int(approx_dim):
            raise ValueError(
                f"mojolearn: the archive stores approx_dim {int(approx_dim)} but its "
                f"model text holds {arrays['dim']}; the file is corrupt")
        if bias is not None and float(bias) != arrays['bias']:
            raise ValueError(
                f"mojolearn: the archive stores bias {float(bias)!r} but its model text "
                f"holds {arrays['bias']!r}; the file is corrupt")
        self._arrays = arrays
        self.n_features_in_ = arrays['n_features']
        self.approx_dim_ = arrays['dim']
        self.bias_ = arrays['bias']
        self.n_classes_ = None if n_classes is None else int(n_classes)
        self.is_classifier = self.loss in _CLASSIFICATION_LOSSES
        self.has_proba = self.loss in _PROBA_LOSSES
        self._binding = _load()

    @classmethod
    def from_file(cls, path):
        """A model from a file written by `GradientBoosting.save`."""
        arrays = _serialize.read_npz(path, GBDT_FORMAT)
        saved_as = _serialize.scalar_str(arrays, 'estimator')
        if saved_as != cls.estimator:
            raise ValueError(f"mojolearn: {path!r} was saved by {saved_as}, not {cls.estimator}")
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
                   numeric_mode=mode, bias=float(bias))

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
        Xa = Xc._as_order('F')
        a = self._arrays
        dim = self.approx_dim_
        out = empty((n_rows * dim,), '<f4')
        names = ('border_offsets', 'borders', 'fold_counts', 'one_hot', 'nan_treatment',
                 'tree_offsets', 'split_feature', 'split_bin', 'split_take_bin',
                 'node_left', 'node_right', 'leaf_offsets', 'leaves')
        addresses = [addr_ro(a[name], name=name) for name in names]
        addresses += [addr_ro(Xa, name='X'), addr(out, name='out')]
        wrote = self._binding.forest_host_gbdt_predict(
            addresses,
            [int(n_rows), int(n_features), a['n_trees'], dim, 1 if a['non_symmetric'] else 0,
             a['n_splits'], a['n_leaf_values'], a['n_borders']],
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
        p1 = empty((n_rows,), '<f8')
        self._binding.forest_host_gbdt_sigmoid(addr_ro(raw, name='raw'), addr(p1, name='p1'), n_rows)
        pv = flat_view(p1, 'd')
        return Array.from_list([[1.0 - p, p] for p in pv], '<f8')
