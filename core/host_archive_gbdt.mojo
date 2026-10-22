# SPDX-License-Identifier: Apache-2.0
# Native saved-model parsing and host-layout publication share the product
# text reader, including simple and tensor CTR schemas and both tree shapes.
from std.collections import Dict
from gbdt.models.model_text import load_model_text
from gbdt.models.ctr_value_table import column_plan


@fieldwise_init
struct HostArchiveGBDT(Movable):
    var integers: Dict[String, List[Int32]]
    var floats: Dict[String, List[Float32]]
    var metadata: Dict[String, Int]
    var bias: Float64
    var non_symmetric: Bool


def _archive_i32(values: List[Int]) raises -> List[Int32]:
    var out = List[Int32]()
    for value in values:
        if value < -2147483648 or value > 2147483647:
            raise Error("saved GBDT host layout integer does not fit int32")
        out.append(Int32(value))
    if len(out) == 0:
        out.append(0)
    return out^


def _archive_f32(values: List[Float32]) -> List[Float32]:
    var out = values.copy()
    if len(out) == 0:
        out.append(0.0)
    return out^


def host_archive_gbdt_text(text: String) raises -> HostArchiveGBDT:
    var saved = load_model_text(text)
    var integers = Dict[String, List[Int32]]()
    var floats = Dict[String, List[Float32]]()
    var metadata = Dict[String, Int]()
    var features = len(saved.fold_counts)
    var trees = saved.model.size()
    var non_symmetric = not saved.model.is_oblivious()
    var dim = 1
    if trees:
        dim = saved.model.non_symmetric_models[0].dim if non_symmetric else saved.model.weak_models[0].dim
    metadata["n_features"] = features
    metadata["n_trees"] = trees
    metadata["dim"] = dim
    var n_tables = len(saved.ctr_tables)
    var n_tensors = len(saved.tensor_ctr_registry.features)
    if n_tables and n_tensors:
        raise Error("combined simple-CTR and tensor-CTR model apply needs a composed column plan")
    if saved.ctr_column_count != n_tables:
        raise Error("saved GBDT CTR columns and apply-time tables disagree")
    var n_input = features
    if n_tensors:
        n_input = saved.tensor_ctr_registry.first_model_column
        if n_input + n_tensors != features:
            raise Error("saved GBDT tensor CTR apply plan does not end at the last model column")
    elif n_tables:
        n_input = column_plan(saved.ctr_tables, features).n_input_features
    metadata["n_input_features"] = n_input
    metadata["n_ctr_tables"] = n_tables
    metadata["n_tensor_tables"] = n_tensors
    var ctr_ints = List[Int]()
    var ctr_floats = List[Float32]()
    var tensor_ints = List[Int]()
    var tensor_floats = List[Float32]()
    var counts = List[Int]()
    for i in range(n_tables):
        ref table = saved.ctr_tables[i]
        if table.ctr_type != 0 and table.ctr_type != 4 and table.ctr_type != 5:
            raise Error("saved GBDT CTR table has no prediction-time formula")
        ctr_ints.extend(List[Int](table.column, table.source_feature, table.ctr_type,
                                 table.counter_denominator, table.target_classes_count,
                                 table.target_border_idx, len(counts), len(table.counts)))
        ctr_floats.extend(List[Float32](table.prior_num, table.prior_denom, table.shift, table.scale))
        counts.extend(table.counts)
    for i in range(n_tensors):
        ref table = saved.tensor_ctr_registry.features[i].table
        tensor_ints.append(len(table.source_features))
        tensor_ints.extend(table.source_features)
        tensor_ints.extend(table.cardinalities)
        tensor_ints.append(len(table.splits))
        for split in table.splits:
            tensor_ints.extend(List[Int](Int(split.feature_id), Int(split.bin_idx), Int(split.split_type)))
        tensor_ints.extend(List[Int](table.target_classes_count, table.target_border_idx,
                                    table.denominator, len(counts), len(table.counts)))
        tensor_floats.extend(List[Float32](table.prior_num, table.prior_denom))
        counts.extend(table.counts)
    metadata["n_ctr_ints"] = len(ctr_ints)
    metadata["n_ctr_floats"] = len(ctr_floats)
    metadata["n_tensor_ints"] = len(tensor_ints)
    metadata["n_tensor_floats"] = len(tensor_floats)
    metadata["n_ctr_counts"] = len(counts)
    integers["ctr_ints"] = _archive_i32(ctr_ints)
    integers["tensor_ints"] = _archive_i32(tensor_ints)
    integers["ctr_counts"] = _archive_i32(counts)
    floats["ctr_floats"] = _archive_f32(ctr_floats)
    floats["tensor_floats"] = _archive_f32(tensor_floats)
    var flags = List[Int]()
    var borders = List[Float32]()
    var border_offsets = List[Int](0)
    for i in range(features):
        flags.append(Int(saved.one_hot[i]) if len(saved.one_hot) else 0)
        borders.extend(saved.borders[i])
        border_offsets.append(len(borders))
    metadata["n_borders"] = len(borders)
    integers["fold_counts"] = _archive_i32(saved.fold_counts)
    integers["one_hot"] = _archive_i32(flags)
    integers["nan_treatment"] = _archive_i32(saved.nan_treatment)
    integers["border_offsets"] = _archive_i32(border_offsets)
    floats["borders"] = _archive_f32(borders)
    var tree_offsets = List[Int](0)
    var leaf_offsets = List[Int](0)
    var split_feature = List[Int]()
    var split_bin = List[Int]()
    var split_take_bin = List[Int]()
    var node_left = List[Int]()
    var node_right = List[Int]()
    var leaves = List[Float32]()
    for i in range(trees):
        if non_symmetric:
            ref tree = saved.model.non_symmetric_models[i]
            if tree.dim != dim:
                raise Error("saved GBDT trees disagree on output dimension")
            for j in range(len(tree.model_structure.nodes)):
                ref node = tree.model_structure.nodes[j]
                split_feature.append(Int(node.feature_id))
                split_bin.append(Int(node.bin))
                split_take_bin.append(Int(tree.model_structure.split_types[j]))
                node_left.append(Int(node.left_subtree))
                node_right.append(Int(node.right_subtree))
            leaves.extend(tree.leaf_values)
        else:
            ref tree = saved.model.weak_models[i]
            if tree.dim != dim:
                raise Error("saved GBDT trees disagree on output dimension")
            for split in tree.structure.splits:
                split_feature.append(Int(split.feature_id))
                split_bin.append(Int(split.bin_idx))
                split_take_bin.append(Int(split.split_type))
                node_left.append(0)
                node_right.append(0)
            leaves.extend(tree.leaf_values)
        tree_offsets.append(len(split_feature))
        leaf_offsets.append(len(leaves))
    metadata["n_splits"] = len(split_feature)
    metadata["n_leaf_values"] = len(leaves)
    integers["tree_offsets"] = _archive_i32(tree_offsets)
    integers["leaf_offsets"] = _archive_i32(leaf_offsets)
    integers["split_feature"] = _archive_i32(split_feature)
    integers["split_bin"] = _archive_i32(split_bin)
    integers["split_take_bin"] = _archive_i32(split_take_bin)
    integers["node_left"] = _archive_i32(node_left)
    integers["node_right"] = _archive_i32(node_right)
    floats["leaves"] = _archive_f32(leaves)
    return HostArchiveGBDT(integers^, floats^, metadata^, saved.model.bias, non_symmetric)
