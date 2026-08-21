"""Our ExtraTrees against scikit-learn's, INTERLEAVED in one process.

    pixi run -e bench mojo run -I . extratrees/bench/sklearn_interleaved.mojo \
        <data_dir> <name> <n_rows> <n_features> <n_classes> <trees> <depth> [host]

WHY ONE PROCESS. This machine drifts: the repository has 1.7x measured on
the same binary and the same fixture twenty minutes apart. Two numbers taken
in two processes are not comparable, so the arms alternate INSIDE one
process and the ratio is what carries. Mojo's Python interop is what makes
that possible with scikit-learn on the other side.

WHAT IS BEING COMPARED. Our GPU fit against scikit-learn's CPU fit, on this
Mac, on the same bytes, at the same parameters. It is not a claim about
cuML: cuML cannot run here at all, and neither can any other GPU
implementation of this estimator. It is a claim about what a user on this
machine can reach.

THREE ARMS, NOT TWO. scikit-learn's own default is `n_jobs=None`, one
thread, and that is the arm whose parameters match ours exactly. A user who
types `n_jobs=-1` gets all ten cores of this box. Reporting only the first
would be choosing the flattering comparison, so both run, alternating with
ours, in the same window. `host` adds our own CPU arm, which is off by
default because it is a serial reference implementation and it is slow.

WHAT IS TIMED. `fit`, on both sides, and nothing else. Loading, the
transpose, the label shift and every accuracy computation are outside the
timed region on both arms.

THE ACCURACY COLUMN IS THE GATE. The two RNGs are different designs -- a
sequential xorshift per split there, counter-based and keyed here -- so the
forests are not the same forest and no bit-identity is available or claimed.
Train accuracy at matched parameters is what says the speed was not bought
with a worse model, and it is printed on every line rather than summarised.
"""

from max.gpu.host import DeviceContext
from std.python import Python
from std.sys import argv
from std.time import perf_counter_ns

from extratrees.bench.bench_data import (
    all_digits,
    dense_class_ids,
    max_features_code,
    read_column_prefix,
    read_f32,
    row_major,
)
from extratrees.estimator import (
    ExtraTreesConfig,
    resolve_max_features,
    fit_extra_trees_classifier,
    fit_extra_trees_classifier_device,
)
from extratrees.ported.randomforest.randomforest import predict_class_forest

comptime DEFAULT_REPS = 3


def main() raises:
    var args = argv()
    if len(args) < 8:
        raise Error(
            "usage: mojo run -I ."
            " extratrees/bench/sklearn_interleaved.mojo <data_dir> <name>"
            " <n_rows> <n_features> <n_classes> <trees> <depth> [host]"
        )
    var data_dir = String(args[1])
    var name = String(args[2])
    var n_rows = Int(String(args[3]))
    var n_features = Int(String(args[4]))
    var n_classes = Int(String(args[5]))
    var trees = Int(String(args[6]))
    # A COMMA LIST, so a whole depth sweep is ONE process. The arms alternate
    # per rep within a depth, which is what makes each ratio sound; putting
    # the depths in one process additionally makes the TREND sound, because a
    # window that drifts drifts across all of them together.
    var depths = List[Int]()
    for piece in String(args[7]).split(","):
        depths.append(Int(String(piece)))
    # Trailing options, order-independent: `host` adds our own CPU arm, a bare
    # integer sets the rep count. Three reps is the default because one rep of
    # anything on this box is a sample of the window, not of the code.
    var want_host = False
    var reps = DEFAULT_REPS
    var mf_specs = List[String]()
    for i in range(8, len(args)):
        var a = String(args[i])
        if a == String("host"):
            want_host = True
        elif all_digits(a):
            reps = Int(a)
        else:
            # A COMMA LIST here too. The sampled-column count is the quantity
            # both the speed and the quality story turn on, so sweeping it
            # inside ONE process is what makes the trend a measurement rather
            # than three windows laid side by side.
            for piece in a.split(","):
                mf_specs.append(String(piece))
    if len(mf_specs) == 0:
        mf_specs.append(String("sqrt"))

    var ctx = DeviceContext()
    var sys = Python.import_module("sys")
    _ = sys.path.append("extratrees/bench")
    var arm = Python.import_module("sklearn_arm")

    # ---- the data, once, outside every timed region ----------------------
    var yfull = read_f32(data_dir + "/" + name + "_y.f32")
    var total_rows = len(yfull)
    if n_rows > total_rows:
        raise Error(
            "asked for "
            + String(n_rows)
            + " rows; the label file has "
            + String(total_rows)
        )
    var labels = dense_class_ids(yfull, n_rows, n_classes)
    var x = read_column_prefix(
        data_dir + "/" + name + "_Xcol.f32", total_rows, n_rows, n_features
    )
    var xr = row_major(x, n_rows, n_features)

    print(
        "[bench]",
        ctx.name(),
        "-- scikit-learn",
        String(arm.sklearn_version()),
        "on",
        String(arm.threads_available()),
        "cores",
    )
    print(
        "[bench]",
        name,
        n_rows,
        "rows x",
        n_features,
        "features,",
        n_classes,
        "classes;",
        trees,
        "trees, max_depth",
        String(args[7]),
        ", bootstrap False, gini",
    )
    print(
        "[bench] speedup > 1 means OURS is faster. accuracy is TRAIN"
        " accuracy on the same rows, computed outside the timed region on"
        " both arms."
    )

    var cfg = ExtraTreesConfig()
    cfg.n_estimators = Int32(trees)

    for mi in range(len(mf_specs)):
     var mf_spec = mf_specs[mi]
     cfg.max_features_spec = max_features_code(mf_spec, n_features)
     for di in range(len(depths)):
      var depth = depths[di]
      cfg.max_depth = Int32(depth)
      print(
          "[max_features",
          mf_spec,
          "->",
          resolve_max_features(cfg.max_features_spec, 0.0, n_features),
          "of",
          n_features,
          "  depth",
          depth,
          "]",
      )
      for rep in range(reps):
          cfg.random_state = UInt64(rep)

          # ---- their arm, one thread: their own default ---------------------
          var t1 = arm.fit_seconds_and_accuracy(
              data_dir, name, n_rows, n_features, trees, depth, rep, 1, mf_spec
          )
          var their1_ms = t1[0].__float__() * 1000.0 / Float64(trees)
          var their1_acc = t1[1].__float__()
          var their1_nodes = t1[2].__float__()

          # ---- our device arm ----------------------------------------------
          var t0 = perf_counter_ns()
          var res = fit_extra_trees_classifier_device(
              ctx,
              x,
              labels,
              Int32(n_rows),
              Int32(n_features),
              Int32(n_classes),
              cfg,
          )
          var ours_ms = Float64(perf_counter_ns() - t0) / 1e6 / Float64(trees)
          var our_nodes = 0
          for t in range(len(res.forest.trees)):
              our_nodes += res.forest.trees[t].num_nodes()
          var right = 0
          for r in range(n_rows):
              if (
                  predict_class_forest(res.forest, xr, r * n_features)
                  == Int(labels[r])
              ):
                  right += 1
          var our_acc = Float64(right) / Float64(n_rows)

          # ---- their arm, all cores: what a user actually types --------------
          var tn = arm.fit_seconds_and_accuracy(
              data_dir, name, n_rows, n_features, trees, depth, rep, -1, mf_spec
          )
          var theirn_ms = tn[0].__float__() * 1000.0 / Float64(trees)
          var theirn_acc = tn[1].__float__()

          print(
              "  rep",
              rep,
              " sklearn-1core",
              their1_ms,
              "ms/tree  acc",
              their1_acc,
              " sklearn-10core",
              theirn_ms,
              "ms/tree  acc",
              theirn_acc,
              " ours-gpu",
              ours_ms,
              "ms/tree  acc",
              our_acc,
              " speedup vs 1core",
              their1_ms / ours_ms,
              "x  vs 10core",
              theirn_ms / ours_ms,
              "x  nodes theirs",
              their1_nodes,
              "ours",
              our_nodes,
          )

          if want_host:
              var th = perf_counter_ns()
              var hres = fit_extra_trees_classifier(
                  x,
                  labels,
                  Int32(n_rows),
                  Int32(n_features),
                  Int32(n_classes),
                  cfg,
              )
              var host_ms = Float64(perf_counter_ns() - th) / 1e6 / Float64(
                  trees
              )
              var hnodes = 0
              for t in range(len(hres.forest.trees)):
                  hnodes += hres.forest.trees[t].num_nodes()
              print(
                  "        ours-cpu",
                  host_ms,
                  "ms/tree  nodes",
                  hnodes,
                  " gpu speedup over our own cpu",
                  host_ms / ours_ms,
                  "x",
              )

    _ = x.unsafe_ptr()
    _ = xr.unsafe_ptr()
    _ = labels.unsafe_ptr()
