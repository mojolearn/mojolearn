# SPDX-License-Identifier: Apache-2.0
"""Host-only registry, profile, bounds and copy-isolation checks. No GPU imports."""
from training.byte_lm_config import ByteConfig, BYTE_DEFAULT_PROFILE


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused(config: ByteConfig) raises:
    var rejected = False
    try:
        config.validate()
    except:
        rejected = True
    require(rejected, "invalid configuration accepted")


def main() raises:
    var default = ByteConfig()
    default.validate()
    require(default.profile() == String(BYTE_DEFAULT_PROFILE), "default profile changed")
    var counts: List[Int] = [8192, 32, 1024, 512, 512, 1024, 32, 2048,
        2048, 2048, 32, 1024, 512, 512, 1024, 32, 2048, 2048, 2048, 8192]
    var offsets = default.offsets()
    require(len(offsets) == 21 and offsets[0] == 0, "offset shape")
    var total = 0
    for j in range(20):
        require(default.param_count(j) == counts[j], "default tensor count changed")
        total += counts[j]
        require(offsets[j + 1] == total, "default offset changed")
    require(total == 34944 and default.n_total() == total, "default total changed")
    var alternate = ByteConfig(3, 7, 24, 3, 1, 8, 40)
    require(alternate.n_total() == 21216, "alternate GQA registry")
    require(alternate.param_count(2) == 576 and alternate.param_count(3) == 192,
            "alternate query/KV counts")
    require(alternate.profile() == "mojolearn.byte-lm.b3-l7-d24-h3-kv1-hd8-ff40-v256-blocks2.fp32.v2", "alternate profile")
    var larger = ByteConfig(1, 65, 48, 6, 3, 8, 96)
    require(larger.n_total() == 66240, "larger GQA registry")
    var copied = alternate.copy()
    copied.length = 11
    require(alternate.length == 7 and copied.profile() != alternate.profile(), "copy/profile isolation")
    require(copied.n_total() == alternate.n_total(), "batch/length changed registry")
    ByteConfig(1, 8192, 2, 1, 1, 2, 2).validate()
    refused(ByteConfig(0))
    refused(ByteConfig(-1))
    refused(ByteConfig(1048577))
    refused(ByteConfig(2, 8193))
    refused(ByteConfig(2, 32, 33))
    refused(ByteConfig(2, 32, 32, 4, 3))
    refused(ByteConfig(2, 32, 12, 4, 2, 3))
    refused(ByteConfig(2, 32, 32, 4, 2, 8, 0))
    refused(ByteConfig(1048576, 8192, 2, 1, 1, 2, 2))
    refused(ByteConfig(1, 1, 1048576, 1, 1, 1048576, 2))
    refused(ByteConfig(1, 8192, 64, 32, 1, 2, 2))
    var rejected = False
    try:
        _ = default.param_count(20)
    except:
        rejected = True
    require(rejected, "parameter20 accepted")
    rejected = False
    try:
        _ = default.param_count(-1)
    except:
        rejected = True
    require(rejected, "negative parameter accepted")
    print("PASS byte LM host-only runtime config: default registry/profile, GQA counts, copies, bounds")
