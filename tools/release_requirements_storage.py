"""Readable, content-addressed storage for repeated release input closures.

Fingerprints always describe the expanded logical document. References are a
storage detail, never substitutes for missing evidence. Old plain JSON remains
readable. Large immutable leaf maps/lists share memory after decoding.
"""
import copy
import hashlib
import json

SCHEMA = "mojolearn.release-requirements-storage.v1"
TABLE = "_shared_inputs"
REF = "$release_input"
FIELDS = frozenset({"inputs", "provenance_sources", "dependency_limitations"})


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


def _leaf(value):
    return isinstance(value, (dict, list)) and len(value) >= 16 and all(
        not isinstance(item, (dict, list)) for item in (value.values() if isinstance(value, dict) else value))


def encode(document):
    """Intern selected input subtrees; leave history/status records inspectable."""
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


def decode(document):
    if TABLE not in document:
        return document
    table = document[TABLE]
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

    return walk({key: value for key, value in document.items() if key != TABLE})
