# SPDX-License-Identifier: Apache-2.0
"""Complete boosting consumers for all grow policies and categorical CTRs."""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.gbdt_partition_cache_check import tree_hash,fold
from gbdt.train import train,predict_floats
from gbdt.options.catboost_options import TCatFeatureParams

def check_grow_policy_callers(ctx: DeviceContext) raises:
    var rows=521
    var cols=5
    var x=List[Float32]()
    var y=List[Float32]()
    for feature in range(cols):
        for row in range(rows):
            x.append(Float32((row*17+row//7)%65) if feature==2 else Float32((row*37+feature*43)%127-63)*Float32(0.03125))
    for row in range(rows):
        y.append(x[row]*x[rows+row]+Float32(row%11-5)*Float32(0.03125))
    for policy in [String("SymmetricTree"),String("Depthwise"),String("Lossguide")]:
        for categorical in [False,True]:
            var categories=List[Bool]()
            for feature in range(cols):
                categories.append(categorical and feature==2)
            var reference=UInt64(0)
            for repeat in range(2):
                var fit_params=List[TCatFeatureParams]()
                if categorical:
                    fit_params.append(TCatFeatureParams.feature_freq_only())
                var model=train(ctx,x,y,rows,cols,border_count=16,n_estimators=3,max_depth=4,
                    grow_policy=policy,max_leaves=11 if policy==String("Lossguide") else -1,
                    min_data_in_leaf=1 if policy==String("SymmetricTree") else 3,
                    cat_features=categories,cat_feature_params=fit_params^,
                    random_seed=UInt64(7921),leaf_estimation_iterations=2)
                if categorical and model.ctr_column_count!=1:
                    raise Error("I17 categorical grow-policy caller did not reach feature-frequency CTR")
                var prediction=predict_floats(ctx,model,x,rows)
                var h=UInt64(14695981039346656037)
                fold(h,bitcast[DType.uint64](model.model.bias))
                fold(h,UInt64(model.ctr_column_count))
                for tree in model.model.weak_models:
                    for split in tree.structure.splits:
                        fold(h,UInt64(split.feature_id)); fold(h,UInt64(split.bin_idx)); fold(h,UInt64(split.split_type))
                    for value in tree.leaf_values:
                        fold(h,UInt64(bitcast[DType.uint32](value)))
                for tree in model.model.non_symmetric_models:
                    tree_hash(h,tree)
                for borders in model.borders:
                    for value in borders:
                        fold(h,UInt64(bitcast[DType.uint32](value)))
                for value in model.losses:
                    fold(h,bitcast[DType.uint64](value))
                for value in prediction:
                    fold(h,UInt64(bitcast[DType.uint32](value)))
                if repeat==0:
                    reference=h
                elif h!=reference:
                    raise Error("I17 grow-policy/CTR model state leaked across fits")
                print("I17_CALLER",policy,"categorical",categorical,"repeat",repeat,"fingerprint",h)
