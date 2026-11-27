"""ONNX protobuf fields for the small, offline verification models."""


def varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def field(num, wire, payload):
    """One protobuf field: wire 0 an unsigned varint, wire 2 length-delimited bytes."""
    key = varint((num << 3) | wire)
    if wire == 0:
        return key + varint(int(payload))
    payload = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    return key + varint(len(payload)) + payload


def value_info(name, dims):
    F = field
    shape = b"".join(F(1, 2, F(2, 2, v) if isinstance(v, str) else F(1, 0, v)) for v in dims)  # glue: dims
    return F(1, 2, name) + F(2, 2, F(1, 2, F(1, 0, 1) + F(2, 2, shape)))
