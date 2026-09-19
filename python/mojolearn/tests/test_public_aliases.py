# SPDX-License-Identifier: Apache-2.0
"""The public names that are ALIASES of another class, held to being that
class (lane/laneless-public-classes, 2026-09-19).

WHY THESE ARE TESTS AND NOT LANES. A census of `mojolearn.__all__` against
`tools/identity_break.py` on 2026-09-19 returned three public names the lane
registry never mentions. One, `DistributedIVFIndex`, is a class of its own
and got a lane (`par-ivf`). The other two are aliases:

    language_model.py   LanguageModelConfig = ByteLanguageModelConfig
    tokenizer.py        _DEPRECATED_ALIASES = {"GPT2Tokenizer": "BpeTokenizer"}

`tools/verification_matrix.py::package_index` already reads BOTH spellings
and resolves them, so the matrix counts each alias as covered by the lanes of
the class it names -- which is right, and is exactly why a LANE for either
would be worthless: it would hash the same bytes the `tokenizer` and
`byte-lm*` lanes already hash, under a second name, and would cost a cell on
every column for no claim the record does not already carry.

What is NOT covered by those lanes is the ALIAS ITSELF. If
`GPT2Tokenizer` stopped resolving, or resolved to a different class, or
stopped warning, every lane would still read IDENTICAL and the break would
ship: a lane can only measure a class it can reach, and the alias is how a
0.8.x caller reaches it. That is a behaviour, it has no arithmetic, and a
test is the honest place for it. The `language-model-config` lane hashes the
`LanguageModelConfig is ByteLanguageModelConfig` identity as one of its flags
for the same reason, and this file is where the rest of that class's
host-only helpers are held.

`DistributedIVFIndex` is not in this file. It has a lane.
"""
import warnings

import pytest

import mojolearn as ml
from mojolearn import _byte_lm_config, language_model, tokenizer


# ------------------------------------------------------- GPT2Tokenizer

def test_gpt2_tokenizer_is_the_bpe_tokenizer_itself():
    """Not a subclass and not a wrapper: the SAME object. A subclass would
    pass an isinstance check and still pickle, repr and compare differently
    from what 0.8.x shipped."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert tokenizer.GPT2Tokenizer is tokenizer.BpeTokenizer
        assert ml.GPT2Tokenizer is tokenizer.BpeTokenizer


def test_both_doors_to_the_old_name_warn():
    """`mojolearn.GPT2Tokenizer` delegates to `mojolearn.tokenizer`'s module
    __getattr__, so the warning must come out of either door. It names the
    new spelling, because a deprecation that does not say what to write
    instead is a dead end."""
    for get in (lambda: tokenizer.GPT2Tokenizer, lambda: ml.GPT2Tokenizer):
        with pytest.warns(DeprecationWarning) as caught:
            got = get()
        assert got is tokenizer.BpeTokenizer
        assert "BpeTokenizer" in str(caught[0].message)


def test_the_old_name_is_still_importable_and_still_exported():
    """It is in `mojolearn.__all__`, so `from mojolearn import *` binds it;
    dropping it from there would break that import silently for anyone who
    used it, which is the whole reason the alias exists."""
    assert "GPT2Tokenizer" in ml.__all__
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from mojolearn import GPT2Tokenizer", namespace)
    assert namespace["GPT2Tokenizer"] is tokenizer.BpeTokenizer


def test_a_name_that_is_not_an_alias_is_still_refused():
    """The alias door must not become a door to everything: a module
    __getattr__ that returns `globals()[name]` for any name would make every
    private helper public by accident."""
    with pytest.raises(AttributeError):
        tokenizer.NotATokenizer
    with pytest.raises(AttributeError):
        ml.NotAnEstimator


def test_the_alias_encodes_what_the_class_encodes():
    """The one arithmetic statement worth making here, and the reason it is
    one line rather than a lane: the alias is the class, so its ids are the
    `tokenizer` lane's ids by construction. A lane would restate that at the
    price of a cell on every column."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = ml.GPT2Tokenizer._synthetic()
    new = tokenizer.BpeTokenizer._synthetic()
    raw = bytes(range(256)) * 4
    assert old.encode_bytes(raw) == new.encode_bytes(raw)
    assert old.identity == new.identity


# --------------------------------------------------- LanguageModelConfig

def test_language_model_config_is_the_byte_lm_config_itself():
    assert ml.LanguageModelConfig is ml.ByteLanguageModelConfig
    assert language_model.LanguageModelConfig is _byte_lm_config.ByteLanguageModelConfig
    assert "LanguageModelConfig" in ml.__all__ and "ByteLanguageModelConfig" in ml.__all__
    assert "LanguageModelConfig" in language_model.__all__


def test_the_config_round_trips_through_its_own_dict():
    """`to_dict` is what a checkpoint writes and `ByteLanguageModelConfig(**d)`
    is what reads it back, so the pair has to be the identity or a saved
    model comes back a different model."""
    cfg = ml.LanguageModelConfig(batch=3, length=7, d_model=24, n_heads=3, n_kv=3,
                                 head_dim=8, intermediate=40, n_layers=3, vocab_size=37)
    assert ml.LanguageModelConfig(**cfg.to_dict()) == cfg
    assert set(cfg.to_dict()) == set(ml.LanguageModelConfig().to_dict())
    assert cfg.offsets[-1] == cfg.n_total
    assert len(cfg.parameter_shapes) == cfg.n_tensors == len(cfg.parameter_names)


@pytest.mark.parametrize("kwargs,message", [
    (dict(batch=0), "Byte-LM dimensions must be in [1, 2**20]"),
    (dict(batch=True), "requires integer dimensions"),
    (dict(length=1.5), "requires integer dimensions"),
    (dict(length=8193), "L <= 8192"),
    (dict(d_model=33), "DM = H*HD"),
    (dict(n_kv=3), "divisible by KV"),
])
def test_the_config_refuses_a_shape_it_cannot_describe(kwargs, message):
    with pytest.raises(ValueError) as caught:
        ml.LanguageModelConfig(**kwargs)
    assert message in str(caught.value)


def test_require_shape_and_state_shape():
    """The two private helpers every byte LM door goes through. They are not
    on the public surface, so the `language-model-config` lane leaves them
    here rather than reaching into a private module from a cell."""
    default = ml.LanguageModelConfig()
    assert _byte_lm_config.require_shape(None) == default
    assert _byte_lm_config.require_shape(default) is default
    with pytest.raises(TypeError, match="must be a ByteLanguageModelConfig"):
        _byte_lm_config.require_shape(default.to_dict())
    assert _byte_lm_config.state_shape({}) == default
    shape = dict(batch=1, length=5, d_model=16, n_heads=2, n_kv=1, head_dim=8,
                 intermediate=24, n_layers=1, vocab_size=61)
    assert _byte_lm_config.state_shape({"model_shape": shape}) == ml.LanguageModelConfig(**shape)
    # The pre-n_layers/vocab_size spelling is still read; anything else is not.
    legacy = {k: v for k, v in default.to_dict().items() if k not in ("n_layers", "vocab_size")}
    assert _byte_lm_config.state_shape({"model_shape": legacy}) == default
    for bad in ({"batch": 2}, {**default.to_dict(), "extra": 1}, [1, 2]):
        with pytest.raises(ValueError, match="missing or unknown dimensions"):
            _byte_lm_config.state_shape({"model_shape": bad})


def test_the_profile_names_the_shape_and_not_only_the_default():
    """`profile` is the string `LanguageModelInference` holds the compiled
    binding to. Two different shapes must not share one, or a checkpoint
    would load into the wrong model and pass the check."""
    shapes = (dict(), dict(vocab_size=257), dict(n_layers=3), dict(batch=1),
              dict(batch=1, length=5, d_model=16, n_heads=2, n_kv=1, head_dim=8,
                   intermediate=24, n_layers=1, vocab_size=61))
    seen = set()
    for kwargs in shapes:
        cfg = ml.LanguageModelConfig(**kwargs)
        assert cfg.profile.startswith("mojolearn.byte-lm.")
        seen.add(cfg.profile)
    assert len(seen) == len(shapes)
    # And the reverse: the SHIPPED shape spelled out in full is the shipped
    # profile, `.v1`, not the `.v3` spelling a different shape would get.
    assert ml.LanguageModelConfig(n_layers=2, vocab_size=256).profile == ml.LanguageModelConfig().profile
    assert ml.LanguageModelConfig().profile.endswith("-v256-blocks2.fp32.v1")
