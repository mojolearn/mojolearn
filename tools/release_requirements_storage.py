"""Release ledger storage: a small current index plus immutable content-addressed objects.

Fingerprints always describe the expanded logical document. References are a
storage detail, never substitutes for missing evidence. Large immutable leaf
maps/lists share memory after decoding.

Layout v2 (lane ledger-storage, 2026-11-04). ``requirements.json`` is the
index: the logical ledger with every attempt, run and interruption replaced by
an ordered ``{"$release_record": sha}`` pointer, and every value at a SHARED key
(inputs, receipts, proofs, evidence, ...) by ``{"$release_input": sha}`` or a
small inline node. ``requirements.objects/<aa>/<sha>.json`` are canonical JSON
files named by the SHA-256 of their bytes, published once with link(2) (never
replaced, never rewritten) and verified on every read. Record lists are
append-only: an edit that changes or drops a historical record is refused. An
edit writes its new objects first and then the index (atomic rename), so a
reader without the lock always sees one complete index.

Why: the 0.8.65 ledger was 1.10 GB and 72% of it was one 157 KB producer
provenance map repeated in 6,331 attempt receipts (253 distinct values), and
again in 2,334 status rows (~/mojolearn-evidence/ledger-storage-notes/FIELD_CENSUS.md).

Readers accept v2, the v1 document with an inline ``_shared_inputs`` table, and
plain JSON. ``migrate`` converts a v1 or plain ledger once, streaming record by
record; it verifies every record against the original before the index is
switched and keeps the original bytes as ``requirements.objects/legacy/<sha>.json``.
"""
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

SCHEMA = "mojolearn.release-requirements-storage.v1"
INDEX_SCHEMA = "mojolearn.release-requirements-storage.v2"
TABLE = "_shared_inputs"
STORAGE = "_storage"
REF = "$release_input"
RECORD = "$release_record"
FIELDS = frozenset({"inputs", "provenance_sources", "dependency_limitations"})
# Values under these keys are stored as shared content-addressed nodes in v2 (the census's repeated fields).
SHARED = FIELDS | frozenset({"receipt", "equivalence_proof", "failure_equivalence_proof", "evidence",
                             "invalidation_evidence", "reference"})
RECORDS = ("attempts", "runs", "interruptions")
# A node whose canonical JSON is shorter stays inline in its parent. 0.8.65 measurement: 1 KB gave 31,443
# files (18,931 of them 1-3 KB source-map buckets) and a 16.6 s read dominated by opens; 8 KB keeps each
# distinct source closure and producer provenance map one file (FIELD_CENSUS.md, MEASUREMENTS.md).
INLINE = 8192
# A value at a SHARED key of a shell (index row or record) is a reference from this size on, so the index
# stays one small row per requirement and a receipt is one object shared by its attempt and status row.
ROOT_INLINE = 512
SHA = re.compile(r"[0-9a-f]{64}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bucket_name(key):
    return format(int(hashlib.sha256(key.encode()).hexdigest()[:2], 16) // 4, "02x")


class FrozenMap(dict):
    def _refuse(self, *args, **kwargs):
        raise TypeError("shared release inputs are immutable; replace the map to change inputs")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refuse

    def __deepcopy__(self, memo):
        return self


class FrozenList(list):
    def _refuse(self, *args, **kwargs):
        raise TypeError("shared release inputs are immutable; replace the list to change inputs")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = reverse = sort = __iadd__ = __imul__ = _refuse

    def __deepcopy__(self, memo):
        return self


class RecordMap(dict):
    """An immutable decoded record or shared subtree. deepcopy gives a mutable copy (shared leaves stay
    shared), so ``copy.deepcopy(attempt["manifest"])`` then editing the copy works; editing in place raises."""

    def _refuse(self, *args, **kwargs):
        raise TypeError("historical ledger records are immutable; copy.deepcopy the value to change a copy")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refuse

    def __deepcopy__(self, memo):
        return _thaw(self)


class RecordList(list):
    def _refuse(self, *args, **kwargs):
        raise TypeError("historical ledger records are immutable; copy.deepcopy the value to change a copy")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = reverse = sort = __iadd__ = __imul__ = _refuse

    def __deepcopy__(self, memo):
        return _thaw(self)


IMMUTABLE = (FrozenMap, FrozenList, RecordMap, RecordList)


def _leaf(value):
    return isinstance(value, (dict, list)) and len(value) >= 16 and all(
        not isinstance(item, (dict, list)) for item in (value.values() if isinstance(value, dict) else value))


# ---------------------------------------------------------------- v1: inline table (read; kept for encode tests)

def encode(document):
    """v1: intern selected input subtrees into an inline table; leave history/status records inspectable."""
    if TABLE in document:
        raise ValueError("encode requires a decoded release requirements document")
    objects, visited = {}, {}

    def store(value, partition=True):
        if not isinstance(value, (dict, list)):
            return value
        previous = visited.get(id(value))
        if previous is not None:
            return {REF: previous[1]}
        if isinstance(value, dict):
            if partition and len(value) >= 64 and _leaf(value):
                # Stable hash buckets share the common parts of overlapping
                # source closures; sorted-size chunks would shift on additions.
                buckets = {}
                for key, item in value.items():
                    bucket = bucket_name(key)
                    buckets.setdefault(bucket, {})[key] = item
                node = dict(kind="map", value={key: store(item, False) for key, item in sorted(buckets.items())})
            else:
                node = dict(kind="dict", value={key: store(item) for key, item in value.items()})
        elif len(value) >= 128 and _leaf(value):
            node = dict(kind="sequence", value=[store(value[start:start + 64]) for start in range(0, len(value), 64)])
        else:
            node = dict(kind="list", value=[store(item) for item in value])
        key = identity(node)
        objects.setdefault(key, node)
        # Retain the object too: temporary hash-bucket dictionaries otherwise
        # release their IDs while encoding, permitting accidental ID reuse.
        visited[id(value)] = (value, key)
        return {REF: key}

    def walk(value):
        if isinstance(value, dict):
            return {key: (store(item) if key in FIELDS and isinstance(item, (dict, list)) else walk(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    result = walk(document)
    result[TABLE] = dict(schema=SCHEMA, objects=objects)
    return result


def _legacy_walk(table):
    """The v1 resolver over one validated inline ``_shared_inputs`` table: a function decoding any subtree."""
    if not isinstance(table, dict) or table.get("schema") != SCHEMA or not isinstance(table.get("objects"), dict):
        raise ValueError("unsupported shared release input storage")
    objects, decoded, active = table["objects"], {}, set()
    for key, node in objects.items():
        if not isinstance(node, dict) or identity(node) != key:
            raise ValueError("shared release input object digest mismatch: " + str(key))

    def resolve(value):
        if not isinstance(value, dict):
            if isinstance(value, list):
                raise ValueError("malformed shared input node")
            return value
        if set(value) != {REF} or not isinstance(value[REF], str):
            raise ValueError("malformed shared input reference")
        key = value[REF]
        if key in decoded:
            return decoded[key]
        if key in active or key not in objects:
            raise ValueError("cyclic or missing shared release input object: " + key)
        active.add(key)
        node = objects[key]
        kind, payload = node.get("kind"), node.get("value")
        if kind in ("dict", "map") and isinstance(payload, dict):
            result = {name: resolve(item) for name, item in payload.items()}
            if kind == "map":
                merged = {}
                for bucket, part in result.items():
                    if not isinstance(part, dict) or any(bucket_name(name) != bucket for name in part):
                        raise ValueError("invalid shared source-map bucket")
                    merged.update(part)
                result = merged
        elif kind in ("list", "sequence") and isinstance(payload, list):
            result = [resolve(item) for item in payload]
            if kind == "sequence":
                if any(not isinstance(part, list) for part in result):
                    raise ValueError("invalid shared list segment")
                result = [item for part in result for item in part]
        else:
            raise ValueError("unknown shared release input object kind")
        if _leaf(result):
            result = FrozenMap(result) if isinstance(result, dict) else FrozenList(result)
        decoded[key] = result
        active.remove(key)
        return result

    def walk(value):
        if isinstance(value, dict):
            return {key: (copy.deepcopy(resolve(item)) if key in FIELDS and isinstance(item, dict) and set(item) == {REF}
                          else walk(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk


def decode(document):
    """Expand a v1 (inline table) or plain ledger document."""
    if TABLE not in document:
        return document
    walk = _legacy_walk(document[TABLE])
    return walk({key: value for key, value in document.items() if key != TABLE})


# ---------------------------------------------------------------- v2: index + immutable object files

def objects_dir(path):
    """The object store beside a ledger index: requirements.json -> requirements.objects."""
    return Path(path).with_suffix(".objects")


class ObjectStore:
    """Write-once content-addressed JSON files; every read verifies the SHA-256 of the bytes."""

    def __init__(self, root, *, durable=True):
        self.root = Path(root)
        self.durable = durable
        self.known = set()
        self.written = 0

    def path(self, sha):
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError("malformed ledger object reference: " + repr(sha)[:80])
        return self.root / sha[:2] / (sha + ".json")

    def put(self, data):
        sha = hashlib.sha256(data).hexdigest()
        if sha in self.known:
            return sha
        target = self.path(sha)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".write-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(data)
                    out.flush()
                    if self.durable:
                        os.fsync(out.fileno())
                try:
                    # link(2) publishes a complete file and never replaces one: write once.
                    os.link(temporary, target)
                    self.written += 1
                except FileExistsError:
                    pass
            finally:
                os.unlink(temporary)
        self.known.add(sha)
        return sha

    def get(self, sha):
        try:
            data = self.path(sha).read_bytes()
        except FileNotFoundError:
            raise ValueError("ledger object is missing: " + sha) from None
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("ledger object digest mismatch: " + sha)
        return data


class Encoder:
    """Logical values -> index/record JSON, storing shared nodes in ``store``.

    ``memo`` ({id: (frozen object, sha)}) comes from the read: an unchanged frozen leaf is referenced
    without re-serializing it. Encoding is deterministic, so an unchanged record gets its old SHA.
    Canonical text is assembled bottom-up (each value serialized once, whatever its depth)."""

    def __init__(self, store, memo=None):
        self.store = store
        self.memo = dict(memo or {})

    def node(self, value, inline=INLINE):
        """(stored form, its canonical JSON text): a scalar, an inline node or a reference."""
        if not isinstance(value, (dict, list)):
            text = canonical(value)
            return value, text
        hit = self.memo.get(id(value))
        if hit is not None and hit[0] is value:
            return {REF: hit[1]}, '{"' + REF + '":"' + hit[1] + '"}'
        if _scalars(value):
            # One C serialization for an all-scalar container (most of the ledger's structure).
            body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            kind = "dict" if isinstance(value, dict) else "list"
            node = dict(kind=kind, value=value)
            data = '{"kind":"' + kind + '","value":' + body + "}"
        elif isinstance(value, dict):
            _string_keys(value)
            parts, payload = [], {}
            for key in sorted(value):
                stored, text = self.node(value[key])
                payload[key] = stored
                parts.append(canonical(key) + ":" + text)
            node = dict(kind="dict", value=payload)
            data = '{"kind":"dict","value":{' + ",".join(parts) + "}}"
        else:
            parts, payload = [], []
            for item in value:
                stored, text = self.node(item)
                payload.append(stored)
                parts.append(text)
            node = dict(kind="list", value=payload)
            data = '{"kind":"list","value":[' + ",".join(parts) + "]}"
        if len(data) < inline:
            return node, data
        sha = self.store.put(data.encode())
        if isinstance(value, IMMUTABLE):
            # Only immutable values may be recognized by identity later; holding mutable ones would
            # also keep every encoded value alive (a migration would hold the whole document).
            self.memo[id(value)] = (value, sha)
        return {REF: sha}, '{"' + REF + '":"' + sha + '"}'

    def text(self, value):
        """Canonical JSON text of a shell (index member or record), storing its SHARED values."""
        if isinstance(value, dict):
            _string_keys(value)
            return "{" + ",".join(canonical(key) + ":" + (self.node(value[key], ROOT_INLINE)[1] if key in SHARED and isinstance(value[key], (dict, list))
                                                          else self.text(value[key])) for key in sorted(value)) + "}"
        if isinstance(value, list):
            return "[" + ",".join(self.text(item) for item in value) + "]"
        return canonical(value)

    def walk(self, value):
        return json.loads(self.text(value))

    def record(self, value):
        if not isinstance(value, dict):
            raise ValueError("ledger records are JSON objects")
        hit = self.memo.get(id(value))
        if hit is not None and hit[0] is value:
            return hit[1]   # an unchanged (immutable) record read from this store
        return self.store.put(self.text(value).encode())


_SCALARS = (str, int, float, bool, type(None))


def _scalars(value):
    items = value.values() if isinstance(value, dict) else value
    if isinstance(value, dict) and not all(type(key) is str for key in value):
        return False
    return all(type(item) in _SCALARS for item in items)


def _string_keys(value):
    if not all(isinstance(key, str) for key in value):
        raise TypeError("ledger JSON object keys must be strings")


def _thaw(value):
    """Mutable copy of decoded containers; frozen shared leaves stay shared (the v1 deepcopy contract)."""
    kind = type(value)
    if kind is dict or kind is RecordMap:
        return {key: _thaw(item) for key, item in value.items()}
    if kind is list or kind is RecordList:
        return [_thaw(item) for item in value]
    return value


def _freeze(value):
    kind = type(value)
    if kind is dict:
        return RecordMap((key, _freeze(item)) for key, item in value.items())
    if kind is list:
        return RecordList(_freeze(item) for item in value)
    return value


class Decoder:
    """Index/record JSON -> logical values; ``loaded`` names every object read, ``memo`` the frozen leaves."""

    def __init__(self, store):
        self.store = store
        self.cache, self.memo, self.loaded, self.active = {}, {}, set(), set()

    def resolve(self, sha):
        if sha in self.cache:
            return self.cache[sha]
        if sha in self.active:
            raise ValueError("cyclic ledger object: " + sha)
        self.active.add(sha)
        node = json.loads(self.store.get(sha))
        self.loaded.add(sha)
        result = self.build(node)
        if _leaf(result):
            result = FrozenMap(result) if isinstance(result, dict) else FrozenList(result)
        else:
            result = RecordMap(result) if isinstance(result, dict) else RecordList(result)
        # Decoded objects are immutable and shared; an unchanged one is re-referenced by identity.
        self.memo[id(result)] = (result, sha)
        self.cache[sha] = result
        self.active.discard(sha)
        return result

    def child(self, value):
        if not isinstance(value, dict):
            if isinstance(value, list):
                raise ValueError("malformed ledger node child")
            return value
        if set(value) == {REF} and isinstance(value[REF], str):
            return self.resolve(value[REF])
        if set(value) == {"kind", "value"}:
            result = self.build(value)
            if _leaf(result):
                return FrozenMap(result) if isinstance(result, dict) else FrozenList(result)
            return RecordMap(result) if isinstance(result, dict) else RecordList(result)
        raise ValueError("malformed ledger node reference")

    def build(self, node):
        if not isinstance(node, dict):
            raise ValueError("malformed ledger node")
        kind, payload = node.get("kind"), node.get("value")
        if kind in ("dict", "map") and isinstance(payload, dict):
            result = {name: self.child(item) for name, item in payload.items()}
            if kind == "map":
                merged = {}
                for part in result.values():
                    if not isinstance(part, dict):
                        raise ValueError("invalid shared source-map bucket")
                    merged.update(part)
                if len(merged) != sum(len(part) for part in result.values()):
                    raise ValueError("invalid shared source-map bucket")
                result = merged
        elif kind in ("list", "sequence") and isinstance(payload, list):
            result = [self.child(item) for item in payload]
            if kind == "sequence":
                if any(not isinstance(part, list) for part in result):
                    raise ValueError("invalid shared list segment")
                result = [item for part in result for item in part]
        else:
            raise ValueError("unknown ledger node kind")
        return result

    def shell(self, value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key in SHARED and isinstance(item, (dict, list)):
                    # Mutable containers, shared immutable leaves (the v1 decode contract).
                    out[key] = _thaw(self.child(item))
                else:
                    out[key] = self.shell(item)
            return out
        if isinstance(value, list):
            return [self.shell(item) for item in value]
        return value

    def frozen(self, value):
        """A record's own fields: immutable, SHARED values the shared decoded nodes themselves."""
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                out[key] = self.child(item) if key in SHARED and isinstance(item, (dict, list)) else self.frozen(item)
            return RecordMap(out)
        if isinstance(value, list):
            return RecordList(self.frozen(item) for item in value)
        return value

    def record(self, sha):
        payload = json.loads(self.store.get(sha))
        self.loaded.add(sha)
        if not isinstance(payload, dict):
            raise ValueError("ledger record is not a JSON object: " + sha)
        result = self.frozen(payload)
        self.memo[id(result)] = (result, sha)
        return result


class State:
    """What a read learned that its edit needs: record SHAs (append-only check), frozen memo, legacy origin."""

    def __init__(self):
        self.records, self.memo, self.legacy, self.loaded = {}, {}, None, set()


def is_index(document):
    return isinstance(document, dict) and STORAGE in document


def _record_refs(key, value):
    if not isinstance(value, list):
        raise ValueError("ledger index " + key + " must be a list of record references")
    shas = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {RECORD} or not isinstance(item[RECORD], str):
            raise ValueError("malformed ledger record reference in " + key)
        shas.append(item[RECORD])
    return shas


def read_index(path, document, state=None):
    state = state or State()
    meta = document[STORAGE]
    if not isinstance(meta, dict) or meta.get("schema") != INDEX_SCHEMA:
        raise ValueError("unsupported release requirements ledger storage")
    decoder = Decoder(ObjectStore(objects_dir(path)))
    doc = {}
    for key, value in document.items():
        if key == STORAGE:
            continue
        if key in RECORDS:
            state.records[key] = _record_refs(key, value)
            doc[key] = [decoder.record(sha) for sha in state.records[key]]
        else:
            doc[key] = decoder.shell(value)
    state.memo, state.loaded = decoder.memo, decoder.loaded
    return doc, state


def load(path):
    """(logical document, State) of a v2 index, v1 document or plain JSON ledger at ``path``."""
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("release requirements ledger must be a JSON object")
    if is_index(document):
        return read_index(path, document)
    state = State()
    state.legacy = hashlib.sha256(raw).hexdigest()
    return decode(document), state


def index_of(doc, store, state=None):
    """The v2 index of ``doc``; writes every object it references. Refuses rewritten or dropped records."""
    state = state or State()
    if TABLE in doc or STORAGE in doc:
        raise ValueError("encode requires a decoded release requirements document")
    encoder = Encoder(store, state.memo)
    index = {}
    for key, value in doc.items():
        if key in RECORDS and isinstance(value, list):
            shas = [encoder.record(item) for item in value]
            prior = state.records.get(key, [])
            if shas[:len(prior)] != prior:
                changed = next((i for i, (a, b) in enumerate(zip(shas, prior)) if a != b), len(shas))
                raise ValueError(f"historical ledger {key} are immutable and append-only: record {changed} "
                                 f"of {len(prior)} was rewritten or dropped")
            index[key] = [{RECORD: sha} for sha in shas]
        else:
            index[key] = encoder.walk(value)
    for key in state.records:
        if state.records[key] and key not in index:
            raise ValueError(f"historical ledger {key} are immutable and append-only: the list was dropped")
    index[STORAGE] = dict(schema=INDEX_SCHEMA)
    return index


def _write_atomic(path, data):
    fd, name = tempfile.mkstemp(prefix=Path(path).name + ".", dir=Path(path).parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def keep_legacy(path, sha):
    """Keep the original v1/plain bytes as an immutable object (a hard link: no copy, no extra space)."""
    target = objects_dir(path) / "legacy" / (sha + ".json")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(path, target)
    except OSError:
        import shutil
        fd, temporary = tempfile.mkstemp(prefix=".write-", dir=target.parent)
        os.close(fd)
        shutil.copyfile(path, temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        finally:
            os.unlink(temporary)
    if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
        raise ValueError("retained legacy ledger bytes differ from the ledger that was read")
    return target


def save(path, doc, state=None):
    """Write the new objects of ``doc``, then its index (atomic rename). Returns the number of new objects."""
    path = Path(path)
    state = state or State()
    store = ObjectStore(objects_dir(path))
    data = (json.dumps(index_of(doc, store, state), separators=(",", ":"), sort_keys=True) + "\n").encode()
    if state.legacy is not None and path.exists():
        keep_legacy(path, state.legacy)
    _write_atomic(path, data)
    return store.written


def object_files(path):
    """[(sha, file)] of every object the index at ``path`` reaches (a v1/plain ledger has none)."""
    path = Path(path)
    document = json.loads(path.read_bytes())
    if not is_index(document):
        return []
    _, state = read_index(path, document)
    store = ObjectStore(objects_dir(path))
    return [(sha, store.path(sha)) for sha in sorted(state.loaded)]


# ---------------------------------------------------------------- migrate (streaming, memory-bounded)

_DECODER = json.JSONDecoder()


def _skip_ws(text, i):
    while i < len(text) and text[i] in " \t\n\r":
        i += 1
    return i


def _members(text, i, *, lazy=False, done=None):
    """Yield (key|index, start, end, value) of the object/array at text[i]; values decoded one at a time.

    With ``lazy`` container members are skipped (decoding one level further down at a time) instead of
    decoded, so memory stays one member. ``done[0]`` is set to the index after the container."""
    opener = text[i]
    if opener not in "{[":
        raise ValueError("expected a JSON object or array")
    closer = "}" if opener == "{" else "]"
    i = _skip_ws(text, i + 1)
    n = 0
    while text[i] != closer:
        if opener == "{":
            key, i = _DECODER.raw_decode(text, i)
            i = _skip_ws(text, i)
            if text[i] != ":":
                raise ValueError("malformed ledger JSON")
            i = _skip_ws(text, i + 1)
        else:
            key = n
        start = i
        if lazy and text[i] in "{[":
            holder = [None]
            for _ in _members(text, i, done=holder):
                pass
            i = holder[0]
            yield key, start, i, None
        else:
            value, i = _DECODER.raw_decode(text, i)
            yield key, start, i, value
        n += 1
        i = _skip_ws(text, i)
        if text[i] == ",":
            i = _skip_ws(text, i + 1)
        elif text[i] != closer:
            raise ValueError("malformed ledger JSON")
    if done is not None:
        done[0] = i + 1


def migrate(path, *, say=print):
    """Convert the v1/plain ledger at ``path`` into a v2 index and objects, without losing any record.

    Streams one record at a time; verifies every record, status row, requirement and top-level value
    of the new index against the original before switching; keeps the original bytes. Run under the
    ledger lock (RequirementsLedger.migrate)."""
    import time
    path = Path(path)
    started = time.time()
    raw = path.read_bytes()
    legacy_sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    del raw
    i = _skip_ws(text, 0)
    spans = {key: (start, end) for key, start, end, _ in _members(text, i, lazy=True)}
    if STORAGE in spans:
        return dict(status="already-v2", ledger=str(path))
    walk = (_legacy_walk(json.loads(text[slice(*spans[TABLE])])) if TABLE in spans
            else (lambda value: value))
    store = ObjectStore(objects_dir(path), durable=False)
    encoder = Encoder(store)
    index, counts = {}, {}
    for key, (start, end) in spans.items():
        if key == TABLE:
            continue
        if key in RECORDS and text[start] == "[":
            index[key] = [{RECORD: encoder.record(walk(value))} for _, _, _, value in _members(text, start)]
        elif key in ("status", "requirements") and text[start] == "{":
            index[key] = {name: encoder.walk(walk(value)) for name, _, _, value in _members(text, start)}
        else:
            index[key] = encoder.walk(walk(json.loads(text[start:end])))
        counts[key] = len(index[key]) if isinstance(index[key], (list, dict)) else 1
    index[STORAGE] = dict(schema=INDEX_SCHEMA)
    encoded = time.time()
    # Verify the whole logical document, member by member, from the files just written.
    decoder = Decoder(ObjectStore(objects_dir(path)))
    for key, (start, end) in spans.items():
        if key == TABLE:
            continue
        if key in RECORDS and text[start] == "[":
            seen = 0
            for n, _, _, value in _members(text, start):
                seen += 1
                if n >= len(index[key]) or decoder.record(index[key][n][RECORD]) != walk(value):
                    raise ValueError(f"migration verification failed: {key}[{n}]")
            if seen != len(index[key]):
                raise ValueError("migration verification failed: " + key + " count")
        elif key in ("status", "requirements") and text[start] == "{":
            seen = 0
            for name, _, _, value in _members(text, start):
                seen += 1
                if decoder.shell(index[key][name]) != walk(value):
                    raise ValueError(f"migration verification failed: {key}[{name}]")
            if seen != len(index[key]):
                raise ValueError("migration verification failed: " + key + " count")
        elif decoder.shell(index[key]) != walk(json.loads(text[start:end])):
            raise ValueError("migration verification failed: " + key)
    verified = time.time()
    del text
    os.sync()
    keep_legacy(path, legacy_sha)
    data = (json.dumps(index, separators=(",", ":"), sort_keys=True) + "\n").encode()
    _write_atomic(path, data)
    objects = sum(1 for _ in objects_dir(path).glob("[0-9a-f][0-9a-f]/*.json"))
    result = dict(status="migrated", ledger=str(path), legacy_sha256=legacy_sha,
                  legacy=str(objects_dir(path) / "legacy" / (legacy_sha + ".json")),
                  index_bytes=len(data), objects=objects, objects_written=store.written, counts=counts,
                  encode_seconds=round(encoded - started, 1), verify_seconds=round(verified - encoded, 1),
                  seconds=round(time.time() - started, 1))
    say(json.dumps(result, sort_keys=True))
    return result
