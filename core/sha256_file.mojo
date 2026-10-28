# SPDX-License-Identifier: Apache-2.0
# Shared streamed file digest; the original ByteLM chunk size/order is retained.
from core.sha256 import Sha256


def sha256_file(path: String) raises -> String:
    var hasher = Sha256()
    with open(path, "r") as fh:
        while True:  # cpu-route: integrity hashing over host files, one 1 MiB chunk
            var chunk = fh.read_bytes(1048576)
            if len(chunk) == 0:
                break
            hasher.update_list(chunk)
    return hasher.hexdigest()
