# SPDX-License-Identifier: Apache-2.0
# Shared streamed file digest; the original ByteLM chunk size/order is retained.
# Serial by nature: one SHA-256 chain over the file's bytes in order.
from core.sha256 import Sha256


def sha256_file_sized(path: String, mut size: Int) raises -> String:
    """The digest of the file and, in `size`, the byte count that was hashed
    (one pass: a caller that needs both does not stat the file)."""
    var hasher = Sha256()
    var total = 0
    with open(path, "r") as fh:
        while True:  # cpu-route: integrity hashing over host files, one 1 MiB chunk
            var chunk = fh.read_bytes(1048576)
            if len(chunk) == 0:
                break
            total += len(chunk)
            hasher.update_list(chunk)
    size = total
    return hasher.hexdigest()


def sha256_file(path: String) raises -> String:
    var size = 0
    return sha256_file_sized(path, size)
