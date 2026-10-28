"""Focused installed-runtime controls; source lane does not execute these."""
from array import array
import hashlib
import json
import mmap

import pytest

from mojolearn import lm_corpus


def _tokens(tmp_path, batch=2, length=3):
    directory = tmp_path / "tokens"
    directory.mkdir()
    payload = array("i", range(48)).tobytes()
    (directory / "tokens.i32").write_bytes(payload)
    manifest = dict(schema=lm_corpus.TOKENS_SCHEMA, source=dict(sha256="c" * 64),
                    vocabulary=dict(schema=lm_corpus.VOCABULARY_SCHEMA, sha256="v" * 64, n_vocab=257),
                    sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), tokens=48,
                    train_range=[4, 40])
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return lm_corpus.TokenBatches(directory, batch, length), directory


def test_batches_keep_readonly_mapping_and_original_schedule(tmp_path):
    batches, _ = _tokens(tmp_path)
    assert isinstance(batches.ids_all._store, mmap.mmap)
    assert batches.ids_all._readonly
    rows = batches.ids(2)
    assert rows.shape == (2, 4)
    assert list(memoryview(rows._store)) == list(range(16, 20)) + list(range(19, 23))
    assert batches.data_schedule(extra="retained")["extra"] == "retained"
    assert batches.describe()["train_range"] == [4, 40]


def test_prefetch_is_lazy_and_closes_after_native_field_failure(tmp_path):
    batches, _ = _tokens(tmp_path)
    iterator = batches.prefetch(2, 3)
    assert iter(iterator) is iterator
    assert next(iterator)[0] == 2
    touched = []

    class Conversion:
        def __int__(self):
            touched.append("int")
            return 3

    batches.length = Conversion()
    with pytest.raises(TypeError):
        next(iterator)
    assert touched == []
    with pytest.raises(StopIteration):
        next(iterator)


def test_prefetch_refuses_replaced_ids_without_calling_it(tmp_path):
    batches, _ = _tokens(tmp_path)
    touched = []
    batches.ids = lambda step: touched.append(step)
    with pytest.raises(TypeError, match="overridden Python ids"):
        batches.prefetch(0, 2)
    assert touched == []


def test_constructor_refuses_replaced_dictionary_descriptor(tmp_path):
    _, directory = _tokens(tmp_path)
    touched = []

    class Replaced(lm_corpus.TokenBatches):
        @property
        def __dict__(self):
            touched.append("dict")
            return {}

    with pytest.raises(TypeError):
        Replaced(directory, 1, 3)
    assert touched == []


def test_native_path_and_progress_admission_precedes_hooks(tmp_path):
    touched = []

    class PathHook:
        def __fspath__(self):
            touched.append("path")
            return str(tmp_path / "missing.txt")

    with pytest.raises(NotImplementedError, match="progress callbacks"):
        lm_corpus.prepare(PathHook(), progress=lambda message: touched.append(message))
    with pytest.raises(TypeError):
        lm_corpus.prepare(PathHook())
    assert touched == []


def test_corpus_property_refuses_scalar_conversion_hook(tmp_path):
    _, directory = _tokens(tmp_path)
    corpus = lm_corpus.TokenizedCorpus(directory, tmp_path / "ranks.tsv")
    touched = []

    class Conversion:
        def __int__(self):
            touched.append("int")
            return 257

    corpus.manifest["vocabulary"]["n_vocab"] = Conversion()
    with pytest.raises(TypeError):
        _ = corpus.n_vocab
    assert touched == []


def test_checkpoint_identity_does_not_traverse_model_weights():
    vocabulary = dict(sha256="v" * 64, n_vocab=257)
    checkpoint = dict(model_weights=object(), data_schedule=dict(vocabulary=vocabulary))
    assert lm_corpus._expected_vocabulary(checkpoint) == vocabulary


def test_schedule_descriptor_is_refused_without_execution():
    touched = []

    class Trainer:
        @property
        def data_schedule(self):
            touched.append("schedule")
            return {}

    with pytest.raises(TypeError):
        lm_corpus._expected_vocabulary(Trainer())
    assert touched == []


def test_payload_corruption_is_not_rebuilt(tmp_path):
    _, directory = _tokens(tmp_path)
    (directory / "tokens.i32").write_bytes(array("i", [7] * 48).tobytes())
    with pytest.raises(ValueError, match="length/SHA mismatch"):
        lm_corpus.TokenBatches(directory, 1, 3)


@pytest.mark.parametrize("step", [0.5, -0.1, True])
def test_batch_offset_multiplies_before_integer_conversion(tmp_path, step):
    batches, _ = _tokens(tmp_path)
    result = batches.ids(step)
    base = int(step * batches.batch * batches.length)
    expected = []
    for row in range(batches.batch):
        start = batches.lo + (base + row * batches.length) % batches.modulus
        expected.extend(range(start, start + batches.length + 1))
    assert list(memoryview(result._store)) == expected


def test_constructor_retains_native_numpy_scalar_conversion(tmp_path):
    np = pytest.importorskip("numpy")
    _, directory = _tokens(tmp_path)
    batches = lm_corpus.TokenBatches(directory, np.int64(2), np.float32(3.9))
    assert batches.batch == 2 and batches.length == 3


@pytest.mark.parametrize("kind,value,batch", [("float16", 0.1, 10), ("float32", 0.7, 10),
                                               ("float64", -0.01, 2), ("int8", 2, 2),
                                               ("uint64", 2 ** 53 + 1, 2), ("bool_", True, 2)])
@pytest.mark.parametrize("promotion", ["legacy", "weak"])
def test_numpy_step_uses_installed_scalar_promotion(tmp_path, kind, value, batch, promotion):
    np = pytest.importorskip("numpy")
    setter = getattr(np, "_set_promotion_state", None)
    if setter is None and promotion != "weak":
        pytest.skip("NumPy 2.2 and later have only weak scalar promotion")
    prior = np._get_promotion_state() if setter is not None else None
    batches, _ = _tokens(tmp_path, batch=batch, length=1)
    step = getattr(np, kind)(value)
    try:
        if setter is not None:
            setter(promotion)
        base = int(step * batches.batch * batches.length)  # independent legacy expression oracle
        result = batches.ids(step)
    finally:
        if setter is not None:
            setter(prior)
    expected = []
    for row in range(batches.batch):
        start = batches.lo + (base + row * batches.length) % batches.modulus
        expected.extend(range(start, start + batches.length + 1))
    assert list(memoryview(result._store)) == expected


def test_numpy_step_overflow_refuses_before_seterrcall(tmp_path):
    np = pytest.importorskip("numpy")
    setter = getattr(np, "_set_promotion_state", None)
    prior = np._get_promotion_state() if setter is not None else None
    prior_callback = np.geterrcall()
    batches, _ = _tokens(tmp_path)
    touched = []
    try:
        if setter is not None:
            setter("weak")
        np.seterrcall(lambda *arguments: touched.append(arguments))
        with np.errstate(over="call"):
            with pytest.raises(ValueError, match="overflows"):
                batches.ids(np.int8(100))
    finally:
        np.seterrcall(prior_callback)
        if setter is not None:
            setter(prior)
    assert touched == []
