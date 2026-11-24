# SPDX-License-Identifier: Apache-2.0
"""RandomForestClassifier and RandomForestRegressor as MEMBERS (core/member_abi.mojo), family 14 (`MEMBER_FAMILY_FOREST`).

One entry, `forest_member_v2`, published by each image that serves the
family's forest entry (`forest_member_publish_image`, beside the image's
forest entry publication). The RUN is the shared forest member RUN
(core/forest_member_run.mojo: parameter words, views, kinds, blocks and the
route slot it reads, `route.entry` = the image's forest entry); this file
holds only the family's store and its publication. Kind 0 is the
classifier, kind 1 the regressor.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.forest_member_run import fm_member, fm_store_address
from core.gbdt_entry_table import gbdt_vendor_word
from core.member_abi import member_publish, member_call_from_spans
from core.member_families import MEMBER_FAMILY_FOREST
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report

comptime _STORE_NAME: StaticString = "MojolearnForestMemberStoreV2"


def forest_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of RandomForestClassifier and RandomForestRegressor (schema `MEMBER_ABI`)."""
    try:
        var word = fm_member[_STORE_NAME, True, True](operation, member_call_from_spans(operation, aa, na, np, nf), "RandomForest")
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def forest_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Forest member entry."""
    member_publish[MEMBER_FAMILY_FOREST](mode, vendor, publisher, forest_member_v2)


def forest_member_publish_image(vendor: StaticString) raises:
    """The one publication line of an image's module initialization, after
    its forest entry is published: the image's tier and the vendor word of
    the string it exports (a build with no vendor word publishes nothing,
    as its forest entry)."""
    var word = gbdt_vendor_word(vendor)
    if word < 0:
        return
    forest_member_publish(Int(GLOBAL_NUMERIC_MODE), word, fm_store_address[_STORE_NAME]())
