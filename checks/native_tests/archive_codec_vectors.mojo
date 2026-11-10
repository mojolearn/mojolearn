# SPDX-License-Identifier: Apache-2.0
"""Fixed external codec fixtures, generated at development time only.
Python stdlib zlib/bz2/lzma produced the compressed bytes; Zstd uses the
published raw/RLE frame format. Expected plaintext is independently planted.
No Python code executes in the native reader or test executable.
"""

def archive_codec_fixture(method: Int, vector: Int) -> List[UInt8]:
    if method == 8:
        if vector == 0:
            var data: List[UInt8] = [51, 52, 50, 54, 49, 53, 51, 183, 176, 4, 0]
            return data^
        if vector == 1:
            var data: List[UInt8] = [3, 0]
            return data^
        if vector == 2:
            var data: List[UInt8] = [237, 193, 1, 1, 0, 0, 0, 128, 144, 254, 175, 238, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106]
            return data^
    if method == 12:
        if vector == 0:
            var data: List[UInt8] = [66, 90, 104, 57, 49, 65, 89, 38, 83, 89, 252, 137, 25, 24, 0, 0, 0, 8, 0, 63, 224, 32, 0, 34, 13, 12, 8, 50, 98, 30, 13, 239, 41, 193, 119, 36, 83, 133, 9, 15, 200, 145, 145, 128]
            return data^
        if vector == 1:
            var data: List[UInt8] = [66, 90, 104, 57, 23, 114, 69, 56, 80, 144, 0, 0, 0, 0]
            return data^
        if vector == 2:
            var data: List[UInt8] = [66, 90, 104, 57, 49, 65, 89, 38, 83, 89, 215, 113, 233, 235, 0, 0, 128, 192, 0, 192, 0, 0, 8, 32, 0, 48, 128, 41, 26, 1, 164, 3, 139, 185, 34, 156, 40, 72, 107, 184, 244, 245, 128]
            return data^
    if method == 14:
        if vector == 0:
            var data: List[UInt8] = [9, 4, 5, 0, 93, 0, 0, 16, 0, 0, 24, 140, 130, 182, 196, 17, 52, 92, 78, 231, 16, 125, 224, 255, 253, 20, 128, 0]
            return data^
        if vector == 1:
            var data: List[UInt8] = [9, 4, 5, 0, 93, 0, 0, 16, 0, 0, 131, 255, 251, 255, 255, 192, 0, 0, 0]
            return data^
        if vector == 2:
            var data: List[UInt8] = [9, 4, 5, 0, 93, 0, 0, 16, 0, 0, 0, 111, 253, 255, 255, 163, 183, 255, 71, 62, 72, 21, 114, 57, 97, 81, 184, 146, 40, 230, 163, 134, 7, 249, 238, 228, 30, 130, 211, 47, 197, 58, 60, 1, 75, 177, 126, 201, 138, 138, 77, 47, 163, 13, 217, 127, 166, 227, 140, 35, 17, 83, 224, 89, 24, 197, 117, 138, 226, 119, 248, 182, 148, 127, 12, 106, 192, 222, 116, 73, 100, 92, 163, 205, 231, 163, 255, 255, 12, 28, 0, 0]
            return data^
    if method == 93:
        if vector == 0:
            var data: List[UInt8] = [40, 181, 47, 253, 32, 9, 73, 0, 0, 49, 50, 51, 52, 53, 54, 55, 56, 57]
            return data^
        if vector == 1:
            var data: List[UInt8] = [40, 181, 47, 253, 32, 0, 1, 0, 0]
            return data^
        if vector == 2:
            var data: List[UInt8] = [40, 181, 47, 253, 96, 0, 255, 3, 0, 8, 0]
            return data^
    return List[UInt8]()
