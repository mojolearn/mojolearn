#!/usr/bin/env python3
"""One shape description, shared by the capture harness and every verifier.

DEVIATION 2682. The byte LM's training identity is claimed PER SHAPE, because
nine of the weight gradients contract over the token count, so the gradient at
128 tokens is not the bits of the gradient at 64 tokens accumulated twice. A
second shape is therefore a new certificate, not an extension of the first, and
it needs its own capture, its own schedule and its own comparison.

Before this module the shape lived as literals in five places: the capture
harness, the corpus manifest, the independent gradient oracle, the state
comparator and the CPU training gate. Adding a shape by editing five sets of
literals is how one of them ends up describing a different model than the
others, silently, because every check compares a literal against a literal.

So the shape is derived here from nine integers, and the risk that derivation
introduces is met head on: `check_default()` asserts that everything derived
for the default shape equals the literal constants that the certified b2-l32
run was produced and admitted with, character for character where it is a
string. Any drift in the derivation fails at import, on the default path, in
CI, rather than in a comparison nobody re-ran.
"""
from __future__ import annotations
import math

#: The nine dimensions, in the order the native binding takes them.
FIELDS = ('batch', 'length', 'd_model', 'n_heads', 'n_kv', 'head_dim',
          'intermediate', 'n_layers', 'vocab_size')
DEFAULT = (2, 32, 32, 4, 2, 8, 64, 2, 256)

#: Literals of the certified default run, kept verbatim as the derivation's test.
DEFAULT_PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
DEFAULT_SCHEDULE = ('step s zero-based: row b reads bytes[(s*64+b*32) % 65504 : '
                    'start+33]; targets shifted one byte')
DEFAULT_VALIDATION_STARTS = list(range(65536, 66048, 64))
DEFAULT_N_TOTAL = 34944
DEFAULT_IDS = 66
DEFAULT_FLAGS = 20

#: The corpus is pinned independently of the shape, and both are the same file.
TRAIN_RANGE = (0, 65536)
VALIDATION_RANGE = (65536, 73728)
PLANNED_STEPS = 128
#: Every validation schedule reads this many target bytes, whatever the shape,
#: so the held-out loss of two shapes is a mean over the same amount of text.
VALIDATION_TARGETS = 512


class Shape:
    """Nine dimensions plus everything a capture or a verifier derives from them."""

    def __init__(self, fields=None):
        values = DEFAULT if fields is None else tuple(fields)
        if len(values) == 7:
            values += (2, 256)
        if len(values) != 9 or any(type(x) is not int or isinstance(x, bool)
                                  or not 0 < x <= 1 << 20 for x in values):
            raise ValueError('expected nine positive bounded integer dimensions')
        b, l, dm, h, kv, hd, ff, layers, vocab = values
        if l > 8192 or dm != h * hd or h % kv or hd % 2:
            raise ValueError('requires L <= 8192, DM = H*HD, H divisible by KV, and even HD')
        if max(b * l * vocab, b * l * dm, b * l * ff, b * h * l * l) >= 2 ** 31:
            raise ValueError('model extent exceeds int32 indexing')
        self.fields = values
        for name, value in zip(FIELDS, values):
            setattr(self, name, value)

    # -- identity ---------------------------------------------------------
    @property
    def profile(self):
        """The native profile string. The default keeps its v1 spelling, which
        is what every retained b2-l32 capture.json records."""
        if self.fields == DEFAULT:
            return DEFAULT_PROFILE
        b, l, dm, h, kv, hd, ff, layers, vocab = self.fields
        suffix = ('-v256-blocks2.fp32.v2' if (layers, vocab) == (2, 256)
                  else f'-v{vocab}-blocks{layers}.fp32.v3')
        return (f'mojolearn.byte-lm.b{b}-l{l}-d{dm}-h{h}-kv{kv}-hd{hd}'
                f'-ff{ff}{suffix}')

    @property
    def slug(self):
        """A short name for a directory or a manifest file."""
        return f'b{self.batch}-l{self.length}'

    # -- registry ---------------------------------------------------------
    @property
    def parameter_shapes(self):
        dm, kd, ff = self.d_model, self.n_kv * self.head_dim, self.intermediate
        block = ((dm,), (dm, dm), (kd, dm), (kd, dm), (dm, dm),
                 (dm,), (ff, dm), (ff, dm), (dm, ff))
        return ((self.vocab_size, dm), *(block * self.n_layers), (self.vocab_size, dm))

    @property
    def parameter_names(self):
        names = ('norm1_w', 'w_q', 'w_k', 'w_v', 'w_o', 'norm2_w',
                 'w_gate', 'w_up', 'w_down')
        return ('embed',
                *(f'block{layer}.{name}' for layer in range(self.n_layers)
                  for name in names),
                'lm_head')

    def registry(self):
        """The parameter tensors in flat order, each with its offset. An offset
        is what turns a flat element index into a tensor name and an index
        inside it, which is how a mismatch gets localized."""
        entries, offset = [], 0
        for name, shape in zip(self.parameter_names, self.parameter_shapes):
            count = math.prod(shape)
            entries.append(dict(name=name, shape=list(shape), offset=offset,
                                count=count))
            offset += count
        return entries

    @property
    def n_tensors(self):
        return 2 + 9 * self.n_layers

    @property
    def n_total(self):
        return sum(math.prod(shape) for shape in self.parameter_shapes)

    # -- the recorded arrays ----------------------------------------------
    @property
    def n_ids(self):
        """int32 ids per training step, the `[batch, length + 1]` layout."""
        return self.batch * (self.length + 1)

    @property
    def n_flags(self):
        return DEFAULT_FLAGS

    def counts(self):
        """Element count of every array a step capture records."""
        counts = {key: self.n_total for key in
                  ('initial_p', 'initial_m', 'initial_v',
                   'post_p', 'post_m', 'post_v', 'grad')}
        counts.update(initial_flags=self.n_flags, post_flags=self.n_flags,
                      loss=1, ids=self.n_ids)
        return counts

    # -- the data schedule ------------------------------------------------
    @property
    def tokens_per_step(self):
        return self.batch * self.length

    @property
    def schedule(self):
        """The training schedule, as prose, because it is the field the corpus
        manifest pins and the capture compares literally."""
        stride = self.tokens_per_step
        return (f'step s zero-based: row b reads bytes[(s*{stride}+b*{self.length})'
                f' % {TRAIN_RANGE[1] - self.length} : start+{self.length + 1}];'
                ' targets shifted one byte')

    def train_start(self, step, row):
        return ((step * self.tokens_per_step + row * self.length)
                % (TRAIN_RANGE[1] - self.length))

    @property
    def validation_batches(self):
        """Held-out batches, chosen so every shape reads VALIDATION_TARGETS
        target bytes. The default's eight batches of two rows and a four-row
        shape's four batches are the same 512 targets over the same bytes."""
        batches, remainder = divmod(VALIDATION_TARGETS, self.tokens_per_step)
        if remainder or batches < 1:
            raise ValueError('held-out targets must divide evenly into batches')
        return batches

    @property
    def validation_starts(self):
        base = VALIDATION_RANGE[0]
        starts = [base + index * self.tokens_per_step
                  for index in range(self.validation_batches)]
        last = starts[-1] + (self.batch - 1) * self.length + self.length + 1
        if last > VALIDATION_RANGE[1]:
            raise ValueError('held-out schedule leaves the validation range')
        return starts

    def validation_ids_bytes(self):
        """Byte size of one held-out batch's recorded int32 ids."""
        return self.n_ids * 4

    # -- manifest ---------------------------------------------------------
    def manifest_fields(self, *, corpus_sha, corpus_bytes):
        """The shape-dependent fields of a corpus manifest, which is the file
        the capture harness refuses to run against if anything differs."""
        return dict(schema='mojolearn.byte-lm.corpus.v1', sha256=corpus_sha,
                    bytes=corpus_bytes, train_range=list(TRAIN_RANGE),
                    validation_range=list(VALIDATION_RANGE),
                    vocabulary=self.vocab_size, batch=self.batch,
                    context=self.length, planned_steps=PLANNED_STEPS,
                    train_batch_schedule=self.schedule,
                    validation_batch_starts=self.validation_starts)

    @property
    def manifest_name(self):
        """The default keeps `manifest.json`; a second shape gets its own file,
        so neither run can read the other's schedule."""
        return 'manifest.json' if self.fields == DEFAULT else f'manifest-{self.slug}.json'

    def to_json(self):
        return dict(zip(FIELDS, self.fields))

    def __eq__(self, other):
        # DUCK TYPED ON PURPOSE, and this is not a style preference. Every tool
        # here loads this module BY PATH, under its own private module name,
        # because these scripts run on a leased host where the tools directory
        # is not on the import path. That means two loaded copies define two
        # different Shape classes, and an isinstance test between them is False
        # even when the nine dimensions are identical. A shape passed from one
        # tool to another would then silently take the wrong branch, which is
        # how a second shape's tree gets admitted against the first shape's
        # counts. Compare what a shape IS, which is its nine integers.
        fields = getattr(other, 'fields', None)
        if not isinstance(fields, tuple) or len(fields) != 9:
            return NotImplemented
        return self.fields == fields

    def __hash__(self):
        return hash(self.fields)

    def __repr__(self):
        return f'Shape{self.fields}'


def parse(text):
    """`--shape 4,32` or the nine comma-separated dimensions, or None."""
    if text is None or text == 'default':
        return Shape()
    parts = [p.strip() for p in str(text).split(',')]
    if any(not p or not p.isdigit() for p in parts):
        raise ValueError('shape must be comma-separated positive integers')
    values = [int(p) for p in parts]
    if len(values) == 2:  # batch and length, the rest of the default profile
        values = list(values) + list(DEFAULT[2:])
    return Shape(values)


def from_capture(config):
    """The shape a retained capture.json describes. Captures written before
    DEVIATION 2682 carry no `model_shape` and are the default by construction,
    since that is the only shape that existed when they were written."""
    if not isinstance(config, dict):
        raise ValueError('capture config must be an object')
    fields = config.get('model_shape')
    shape = Shape() if fields is None else Shape([fields[name] for name in FIELDS]
                                                 if isinstance(fields, dict) else fields)
    if config.get('profile') not in (None, shape.profile):
        raise ValueError('capture profile disagrees with its recorded shape')
    return shape


def check_default():
    """Derivation must reproduce the certified run's literals exactly."""
    shape = Shape()
    if shape.profile != DEFAULT_PROFILE:
        raise ValueError('derived default profile differs from the certified one')
    if shape.schedule != DEFAULT_SCHEDULE:
        raise ValueError('derived default schedule differs from the pinned manifest')
    if shape.validation_starts != DEFAULT_VALIDATION_STARTS:
        raise ValueError('derived default held-out starts differ from the pinned ones')
    if (shape.n_total, shape.n_ids, shape.n_tensors) != (DEFAULT_N_TOTAL, DEFAULT_IDS, 20):
        raise ValueError('derived default registry differs from the certified one')
    if shape.validation_batches != 8:
        raise ValueError('derived default held-out batch count differs')
    if [e['name'] for e in shape.registry()][:2] != ['embed', 'block0.norm1_w']:
        raise ValueError('derived registry order differs')
    return shape


#: Fails at import on the default path if any derivation above drifts.
check_default()


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser(description='Print a shape description.')
    ap.add_argument('--shape', default=None)
    args = ap.parse_args()
    s = parse(args.shape)
    print(json.dumps(dict(
        fields=list(s.fields), profile=s.profile, slug=s.slug,
        manifest=s.manifest_name, n_total=s.n_total, n_ids=s.n_ids,
        n_tensors=s.n_tensors, tokens_per_step=s.tokens_per_step,
        schedule=s.schedule, validation_batches=s.validation_batches,
        validation_starts=s.validation_starts, counts=s.counts()), indent=1))
