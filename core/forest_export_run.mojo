# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/forest_export_binding.mojo` (wave 8 run split): the typed
entry one fitted-forest export handle owns. Imports no Python."""


@fieldwise_init
struct ForestExportEntry[Model: Movable & Deinitable](Movable):
    var model: Self.Model
    var trees: Int
    var nodes: Int
    var outputs: Int
    var meta: List[Int64]
