# SPDX-License-Identifier: Apache-2.0
"""Independent packed-bin projection oracle and prepared/training agreement."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.feature_sampling import sample_tree_folds, project_tree_columns, FeatureProjectionWorkspace
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.prepared import prepare_numeric_dataset
from gbdt.train import train, predict_floats
from gbdt.models.model_text import model_text
from checks.numerics import numeric_mode_name


def projection(ctx: DeviceContext) raises:
    var n = 35
    var folds = List[Int]()
    var selected = List[Int]()
    for f in range(55):
        var width = 1 if f % 3 == 0 else (15 if f % 3 == 1 else 128)
        if f == 4:
            width = 0
        folds.append(width)
        selected.append(width if f % 4 != 0 and f != 1 else 0)
    var original = build_layout(folds)
    var target = build_layout(selected)
    var h = ctx.enqueue_create_host_buffer[DType.uint32](n*original.columns)
    for i in range(n*original.columns):
        h[i] = UInt32(0)
    for f in range(len(folds)):
        if folds[f] == 0:
            continue
        ref cf = original.features[f]
        for r in range(n):
            var value = UInt32((r*17+f*7)%(folds[f]+1))
            var i = Int(cf.offset)*n+r
            h[i] = h[i] | (value << cf.shift)
    var source = ctx.enqueue_create_buffer[DType.uint32](n*original.columns)
    ctx.enqueue_copy(dst_buf=source,src_buf=h)
    var result = project_tree_columns(ctx,source,n,original,target)
    with result.map_to_host() as out:
        for f in range(len(folds)):
            if selected[f] == 0:
                continue
            ref cf = target.features[f]
            for r in range(n):
                var expected = UInt32((r*17+f*7)%(folds[f]+1))
                var actual = (out[Int(cf.offset)*n+r] >> cf.shift)&cf.mask
                if actual != expected:
                    raise Error("projection changed original feature's bins")
    var workspace = FeatureProjectionWorkspace(ctx,n,original)
    for rep in range(3):
        var current_folds = selected.copy() if rep != 1 else folds.copy()
        var current = build_layout(current_folds)
        var reused = workspace.project(ctx,source,original,current)
        with reused.map_to_host() as out:
            for f in range(len(folds)):
                if current_folds[f] == 0:
                    continue
                ref cf = current.features[f]
                for r in range(n):
                    var expected = UInt32((r*17+f*7)%(folds[f]+1))
                    var actual = (out[Int(cf.offset)*n+r] >> cf.shift)&cf.mask
                    if actual != expected:
                        raise Error("reused projection changed original bins")
        _ = reused^
    _ = workspace^
    _ = result^
    _ = source^
    _ = h^
    var random = TRandom(UInt64(42))
    var untouched = TRandom(UInt64(42))
    var all = sample_tree_folds(folds,1,random)
    if random.next_uniform_l() != untouched.next_uniform_l():
        raise Error("default fraction consumed RNG")
    for f in range(len(folds)):
        if all[f] != folds[f]:
            raise Error("default fraction changed folds")
    var fractions: List[Float64] = [0.000001,0.5,0.99]
    for fraction in fractions:
        var chosen = sample_tree_folds(folds,fraction,random)
        var count = 0
        for f in range(len(folds)):
            if chosen[f] > 0:
                count += 1
                if chosen[f] != folds[f]:
                    raise Error("sample changed fold count")
        if count != max(1,Int(Float64(54)*fraction+0.5)):
            raise Error("sample count mismatch")


def fit_paths(ctx: DeviceContext) raises:
    var n = 65
    var f = 7
    var x = List[Float32]()
    var y = List[Float32]()
    for c in range(f):
        for r in range(n):
            x.append(Float32((r*17+c*23)%47)/16 if c != 2 else Float32(1))
    for r in range(n):
        y.append(x[r]+x[4*n+r]*Float32(0.5))
    var pool = prepare_numeric_dataset(ctx,x,y,n,f,border_count=15,random_seed=UInt64(42))
    var policies: List[String] = ["SymmetricTree","Depthwise","Lossguide"]
    for policy in policies:
        var ordinary = train(ctx,x,y,n,f,border_count=15,n_estimators=3,max_depth=3,grow_policy=policy,random_seed=UInt64(42),feature_fraction=0.5)
        var prepared = pool.fit(n_estimators=3,max_depth=3,grow_policy=policy,random_seed=UInt64(42),feature_fraction=0.5)
        if model_text(ordinary) != model_text(prepared):
            raise Error("prepared sampled model mismatch "+policy)
        var a = predict_floats(ctx,ordinary,x,n)
        var b = predict_floats(ctx,prepared,x,n)
        for i in range(n):
            if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
                raise Error("prepared sampled prediction mismatch")
        var implicit = train(ctx,x,y,n,f,border_count=15,n_estimators=2,max_depth=2,grow_policy=policy)
        var explicit = train(ctx,x,y,n,f,border_count=15,n_estimators=2,max_depth=2,grow_policy=policy,feature_fraction=1)
        if model_text(implicit) != model_text(explicit):
            raise Error("default-one model mismatch")


def main() raises:
    print("numeric_mode",numeric_mode_name())
    var ctx = DeviceContext()
    projection(ctx)
    fit_paths(ctx)
    print("PASS feature sampling: mixed-width projection, sampler counts/default RNG, prepared/default models and predictions")
