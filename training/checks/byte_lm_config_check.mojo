# SPDX-License-Identifier: Apache-2.0
"""Host-only registry, profile, bounds and copy-isolation checks.

Independent configurations run on the existing host parallel pool. Registry
loops within each fixture compare its dependent prefix offsets. No GPU or
Python imports; large shapes allocate only shape metadata, never weights.
"""
from training.byte_lm_config import ByteConfig, BYTE_DEFAULT_PROFILE
from core.host_parallel import host_parallelize

comptime CONFIG_FIXTURES = 25


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


def refused_naming(config: ByteConfig, needle: String) raises:
    """Refused BY NAME: `refused` above only proves that SOME guard fired, so a
    guard whose message drifts, or a case that trips a different guard than the
    author intended, both still pass it. This one pins WHICH refusal happened by
    requiring the message to contain `needle`."""
    var rejected = False
    var message = String("")
    try:
        config.validate()
    except e:
        rejected = True
        message = String(e)
    require(rejected, "invalid configuration accepted")
    require(
        message.find(needle) >= 0,
        String("refusal did not name '") + needle + "': " + message,
    )


def check_fixture(fixture_id: Int) raises:
    if fixture_id == 0:
        var config = ByteConfig()
        config.validate()
        require(config.profile() == String(BYTE_DEFAULT_PROFILE), "default profile changed")
        var counts: List[Int] = [8192, 32, 1024, 512, 512, 1024, 32, 2048,
            2048, 2048, 32, 1024, 512, 512, 1024, 32, 2048, 2048, 2048, 8192]
        var offsets = config.offsets()
        require(len(offsets) == 21 and offsets[0] == 0, "offset shape")
        var total = 0
        for j in range(20):
            require(config.param_count(j) == counts[j], "default tensor count changed")
            total += counts[j]
            require(offsets[j + 1] == total, "default offset changed")
        require(total == 34944 and config.n_total() == total, "default total changed")
    elif fixture_id == 1:
        var config = ByteConfig(3, 7, 24, 3, 1, 8, 40)
        require(config.n_total() == 21216, "alternate GQA registry")
        require(config.param_count(2) == 576 and config.param_count(3) == 192,
                "alternate query/KV counts")
        require(config.profile() == "mojolearn.byte-lm.b3-l7-d24-h3-kv1-hd8-ff40-v256-blocks2.fp32.v2", "alternate profile")
    elif fixture_id == 2:
        var config = ByteConfig(1, 65, 48, 6, 3, 8, 96)
        require(config.n_total() == 66240, "larger GQA registry")
    elif fixture_id == 3:
        var generalized = ByteConfig(2, 7, 24, 3, 1, 8, 40, 3, 513)
        var chunked = ByteConfig(2, 7, 24, 3, 1, 8, 40, 3, 513, True)
        require(chunked.profile() != generalized.profile(), "chunked LM-head selector missing from profile")
        require(generalized.n_tensors() == 29, "three-layer tensor count")
        require(generalized.param_count(0) == 12312 and generalized.param_count(28) == 12312,
                "configured embedding/head vocabulary")
        require(generalized.n_total() == 38016, "three-layer registry total")
        require(len(generalized.offsets()) == 30, "three-layer offsets")
    elif fixture_id == 4:
        # Migrated from test_byte_lm_runtime_shape's pure host-only large
        # registry check. This creates metadata only, not model weights.
        var large = ByteConfig(1, 2048, 768, 12, 12, 64, 2048, 12, 50257)
        require(large.n_tensors() == 110 and large.n_total() == 162147840,
                "large configured registry")
        var offsets = large.offsets()
        require(len(offsets) == 111 and offsets[0] == 0, "large registry offset shape")
        var total = 0
        for j in range(large.n_tensors()):
            total += large.param_count(j)
            require(offsets[j + 1] == total, "large registry prefix offset")
        require(total == 162147840, "large registry final offset")
    elif fixture_id == 5:
        var alternate = ByteConfig(3, 7, 24, 3, 1, 8, 40)
        var copied = alternate.copy()
        copied.length = 11
        require(alternate.length == 7 and copied.profile() != alternate.profile(), "copy/profile isolation")
        require(copied.n_total() == alternate.n_total(), "batch/length changed registry")
    elif fixture_id == 6:
        ByteConfig(1, 8192, 2, 1, 1, 2, 2).validate()
    elif fixture_id < 20:
        var invalid: List[ByteConfig] = [
            ByteConfig(1, 7, 24, 3, 1, 8, 40, 0, 513),
            ByteConfig(1, 7, 24, 3, 1, 8, 40, 3, 0),
            ByteConfig(0), ByteConfig(-1), ByteConfig(1048577),
            ByteConfig(2, 8193), ByteConfig(2, 32, 33),
            ByteConfig(2, 32, 32, 4, 3), ByteConfig(2, 32, 12, 4, 2, 3),
            ByteConfig(2, 32, 32, 4, 2, 8, 0),
            ByteConfig(1048576, 8192, 2, 1, 1, 2, 2),
            ByteConfig(1, 1, 1048576, 1, 1, 1048576, 2),
            ByteConfig(1, 8192, 64, 32, 1, 2, 2),
        ]
        # DEVIATION 812: pin WHICH bound fired. This is the absolute-position
        # ceiling, not a RoPE table limit; no table has 8192 entries.
        if fixture_id == 12:
            refused_naming(invalid[fixture_id - 7], "absolute-position ceiling")
        else:
            refused(invalid[fixture_id - 7])
    elif fixture_id < 22:
        var config = ByteConfig()
        var rejected = False
        try:
            _ = config.param_count(20 if fixture_id == 20 else -1)
        except:
            rejected = True
        require(rejected, "out-of-range parameter accepted")
    else:
        # Same generalized shape cases as the retained Python checkpoint/API
        # tests; test the typed registry without constructing a Python trainer.
        var layers: List[Int] = [1, 3, 12]
        var vocabs: List[Int] = [257, 513, 50257]
        var totals: List[Int] = [4576, 9600, 809680]
        var idx = fixture_id - 22
        var config = ByteConfig(1, 3, 8, 1, 1, 8, 8, layers[idx], vocabs[idx])
        var block_counts: List[Int] = [8, 64, 64, 64, 64, 8, 64, 64, 64]
        var offsets = config.offsets()
        require(config.n_tensors() == 2 + 9 * layers[idx], "generalized tensor count")
        require(config.n_total() == totals[idx], "generalized parameter total")
        require(len(offsets) == 3 + 9 * layers[idx] and offsets[0] == 0, "generalized offset shape")
        var total = 0
        for j in range(config.n_tensors()):
            var count = 8 * vocabs[idx] if j == 0 or j == config.n_tensors() - 1 else block_counts[(j - 1) % 9]
            require(config.param_count(j) == count, "generalized parameter count")
            total += count
            require(offsets[j + 1] == total, "generalized parameter offset")


def main() raises:
    var failures = List[Int](length=CONFIG_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            check_fixture(fixture_id)
        except error:
            print("FAIL byte_lm_config fixture", fixture_id, String(error))
            fp.unsafe_store(fixture_id, 1)
    host_parallelize(task, CONFIG_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "byte_lm_config failed " + String(failed) + " of " + String(CONFIG_FIXTURES))
    print("PASS byte LM host-only runtime config:", CONFIG_FIXTURES, "native fixtures, parallel")
