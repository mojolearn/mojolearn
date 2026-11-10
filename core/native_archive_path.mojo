# SPDX-License-Identifier: Apache-2.0
"""Native filename decoding and lexical POSIX path spelling.

One UTF-8 stream has dependent byte boundaries. Path components are ordered:
an absolute component replaces preceding components. Independent archive
operations/tests run separately; these short dependent scans allocate once.
Surrogateescaped bytes remain scalar values until the fixed Unicode PACK.
"""
from core.native_archive_span import ArchiveSpan


def archive_filesystem_decode(raw: ArchiveSpan) raises -> List[UInt32]:
    var result = List[UInt32](capacity=raw.size)
    var i = 0
    while i < raw.size:
        var first = raw.byte(i)
        if first < 128:
            result.append(UInt32(first))
            i += 1
            continue
        var width = 2 if first >= 194 and first <= 223 else (3 if first >= 224 and first <= 239 else (4 if first >= 240 and first <= 244 else 0))
        var valid = width != 0 and width <= raw.size - i
        var value = first & (31 if width == 2 else (15 if width == 3 else 7))
        if valid:
            for offset in range(1, width):
                var byte = raw.byte(i + offset)
                var low = 128
                var high = 191
                if offset == 1:
                    if first == 224:
                        low = 160
                    elif first == 237:
                        high = 159
                    elif first == 240:
                        low = 144
                    elif first == 244:
                        high = 143
                valid = valid and byte >= low and byte <= high
                value = (value << 6) | (byte & 63)
        if valid:
            result.append(UInt32(value))
            i += width
        else:
            # Every offending non-ASCII byte maps independently. Reconsider
            # following bytes so a later valid UTF-8 sequence stays intact.
            result.append(UInt32(0xDC00 + first))
            i += 1
    return result^


def archive_path_join(parts: List[List[UInt32]]) -> List[UInt32]:
    var result = List[UInt32]()
    for part in parts:
        if len(result) == 0 or (len(part) != 0 and part[0] == 47):
            result = part.copy()
        elif len(part) != 0:
            # Older pathlib stores the distinct '//' root as one component.
            if not (len(result) == 2 and result[0] == 47 and result[1] == 47):
                while len(result) != 0 and result[len(result) - 1] == 47:
                    _ = result.pop()
                result.append(47)
            for point in part:
                result.append(point)
    if len(result) == 0:
        result.append(46)
    return result^


def archive_path_normalize(points: List[UInt32]) -> List[UInt32]:
    var result = List[UInt32](capacity=len(points))
    var prefix = 0
    if len(points) != 0 and points[0] == 47:
        prefix = 1
        if len(points) >= 2 and points[1] == 47 and (len(points) == 2 or points[2] != 47):
            prefix = 2
        for i in range(prefix):
            result.append(47)
    var first = 0
    while first < len(points):
        var end = first
        while end < len(points) and points[end] != 47:
            end += 1
        if end > first and not (end == first + 1 and points[first] == 46):
            if len(result) > prefix:
                result.append(47)
            for i in range(first, end):
                result.append(points[i])
        first = end + 1
    if len(result) == 0:
        result.append(46)
    return result^
