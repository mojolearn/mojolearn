# SPDX-License-Identifier: Apache-2.0
"""The loaded-unit boundary of per-family import on first use.

`import mojolearn` loads the core only; the first use of a family loads that
family (and the model-selection hub every estimator connects through); the
enumeration that `python -m mojolearn` runs first (`_families.load_all`, the
step before `verify --coverage`) loads every family and every unit, the set the
package loaded at import before lane lazy-family-import (2026-11-12).

Each loaded set is read in a fresh interpreter, since this suite's own process
loads everything (conftest.py). The static tests read source only.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

# The core: the tokenizer host (it carries the compiled descriptors) and the
# runtime owners its import aliases publish into it (host_surface.
# HOST_IMPORT_ALIASES), the identical base binding (Array helpers and the
# vendor read-back), the optional fast tokenizer, and at most one member of
# the fast tier (the process metadata's vendor read-back of that tier).
_CORE_LEAVES = {"_mojolearn", "_mojolearn_tokenizer_host", "_mojolearn_tokenizer_fast"}

# Families a Ridge fit never needs: neural, sequence, language-model,
# forecasting, boosting/forest, Gaussian-process, density-clustering, ANN and
# resampling units, and their Python modules.
_NOT_FOR_RIDGE_UNITS = {
    "_mojolearn_training", "_mojolearn_mamba", "_mojolearn_transformer", "_mojolearn_byte_lm",
    "_mojolearn_embedding", "_mojolearn_x_sequence", "_mojolearn_x_cnn", "_mojolearn_tsa",
    "_mojolearn_arima", "_mojolearn_gbdt", "_mojolearn_rf", "_mojolearn_trees", "_mojolearn_gp",
    "_mojolearn_hdbscan", "_mojolearn_ivf", "_mojolearn_resample", "_mojolearn_neural_host",
    "_mojolearn_training_host", "_mojolearn_mamba_host", "_mojolearn_transformer_host",
    "_mojolearn_forest_host", "_mojolearn_gbdt_host", "_mojolearn_tsa_host", "_mojolearn_arima_host",
}
_NOT_FOR_RIDGE_MODULES = {
    "mamba", "_mamba_impl", "transformer", "_transformer_impl", "training", "_samba_impl", "models",
    "embedding", "ensemble", "randomforest", "extratrees", "_expansion_trees", "_expansion_sequence",
    "_expansion_cnn", "_expansion_ann", "hdbscan", "_gp_impl", "_gpc_impl", "_ivf_impl", "umap",
    "parallel_training", "parallel_classical", "parallel_ensemble", "lm_corpus",
}

_PROBE = r"""
import json, sys
import mojolearn
from mojolearn import _families
{action}
prefix = "mojolearn."
modules = sorted(n[len(prefix):] for n, m in sys.modules.items() if n.startswith(prefix) and m is not None)
print(json.dumps(dict(units=_families.loaded_units(), modules=modules,
                      canonical=sorted(vars(sys.modules["mojolearn._backend"])["_CANONICAL_PATHS"]))))
"""

_RIDGE_FIT = """
X = mojolearn.Array.from_list([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0], [2.0, 1.0]], "<f4")
y = mojolearn.Array.from_list([1.0, 2.0, 3.0, 4.0], "<f4")
mojolearn.Ridge(alpha=1.0).fit(X, y)
"""


def _run(action):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PKG.parent), env.get("PYTHONPATH"))))
    done = subprocess.run([sys.executable, "-c", _PROBE.replace("{action}", action)], env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _leaf(member):
    return member.rpartition(".")[2]


def _alias_owners():
    tree = ast.parse((PKG / "host_surface.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "HOST_IMPORT_ALIASES":
            table = ast.literal_eval(node.value)
            return {spec["binding"] for spec in table.get("_mojolearn_tokenizer_host", ())}
    raise AssertionError("host_surface.HOST_IMPORT_ALIASES not found")


def _lazy_names():
    tree = ast.parse((PKG / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "_LAZY_NAMES":
            keys = [ast.literal_eval(key) for key in node.value.keys]
            assert len(keys) == len(set(keys)), sorted({k for k in keys if keys.count(k) > 1})
            return dict(zip(keys, (ast.literal_eval(v) for v in node.value.values)))
    raise AssertionError("_LAZY_NAMES not found in mojolearn/__init__.py")


def _literal(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


def test_import_loads_only_the_core_units_and_no_family():
    seen = _run("")
    core = _CORE_LEAVES | _alias_owners()
    fast = [member for member in seen["units"] if member.startswith("mojolearn._sets.fast.")
            and _leaf(member) not in core]
    other = sorted(member for member in seen["units"] if _leaf(member) not in core
                   and not member.startswith("mojolearn._sets.fast."))
    assert not other, f"units loaded by `import mojolearn` outside the core: {other}"
    assert len(fast) <= 1, f"more than the fast tier's read-back member loaded at import: {fast}"
    assert any(_leaf(member) == "_mojolearn_tokenizer_host" for member in seen["units"]), seen["units"]
    families = {where.lstrip(".").split(".")[0] if where else name for name, where in _lazy_names().items()}
    loaded = sorted(families & set(seen["modules"]))
    assert not loaded, f"family modules imported by `import mojolearn`: {loaded}"
    files = sorted(set(seen["units"].values()))
    print(f"import mojolearn: {len(seen['units'])} members in {len(files)} units: {files}")


def test_one_ridge_fit_loads_only_its_family_and_the_hub():
    seen = _run(_RIDGE_FIT)
    units = sorted(member for member in seen["units"] if _leaf(member) in _NOT_FOR_RIDGE_UNITS)
    assert not units, f"a Ridge fit loaded units of unrelated families: {units}"
    modules = sorted(_NOT_FOR_RIDGE_MODULES & set(seen["modules"]))
    assert not modules, f"a Ridge fit imported unrelated families: {modules}"
    assert "linear_model" in seen["modules"] and "model_selection" in seen["modules"], seen["modules"]
    files = sorted(set(seen["units"].values()))
    print(f"after one Ridge fit: {len(seen['units'])} members in {len(files)} units: {files}")


def test_verify_enumeration_loads_every_family_and_unit():
    # `python -m mojolearn` (verify --coverage included) imports __main__,
    # whose first statement is `_families.load_all()`; importing it here runs
    # exactly that enumeration without running a command.
    seen = _run("import mojolearn.__main__")
    eager = _literal(PKG / "_families.py", "EAGER_BEFORE")
    missing = sorted(set(eager) - set(seen["modules"]))
    assert not missing, f"the enumeration left families unimported: {missing}"
    loaded = {member for member in seen["units"] if member.count(".") == 1}
    absent = sorted(f"mojolearn.{name}" for name in seen["canonical"] if f"mojolearn.{name}" not in loaded)
    assert not absent, f"the enumeration left selected bindings unloaded: {absent}"
    print(f"after the enumeration: {len(seen['units'])} members in {len(set(seen['units'].values()))} units")


def test_every_public_name_is_routed_once_and_doors_match_their_all():
    table = _lazy_names()
    for door in _literal(PKG / "_families.py", "EXPANSION_DOORS"):
        exported = _literal(PKG / f"{door}.py", "__all__")
        routed = [name for name, where in table.items() if where == "." + door]
        assert sorted(exported) == sorted(routed), (door, sorted(set(exported) ^ set(routed)))
    public = _literal(PKG / "__init__.py", "__all__")
    eager = {"Array", "BpeTokenizer", "tokenizer", "GPT2Tokenizer", "__version__", "numeric_mode",
             "set_numeric_mode", "numeric_profile", "set_numeric_profile", "numeric_profiles",
             "numeric_profile_measured", "vendor", "gpu_arch", "gpu_arch_how"}
    unrouted = sorted(name for name in public if name not in table and name not in eager)
    assert not unrouted, f"public names neither core nor routed: {unrouted}"
    for name, where in table.items():
        target = name if where is None else where.lstrip(".")
        path = PKG / (target.replace(".", "/") + ".py")
        assert path.exists() or (PKG / target.replace(".", "/") / "__init__.py").exists(), (name, where)


# One family alone in a fresh interpreter (lane lazy-import-fix, 2026-11-12):
# its first import must bring in everything its native boundary reads, as
# `load_all()` does, without `load_all()`. The probe replaces `load_all` with a
# refusal, so a family that only works after the enumeration fails here, and
# it checks that the package did not import every family on the way.
_FAMILY_PROBE = r"""
import json, sys
import numpy as np
import mojolearn
from mojolearn import _families


def _refuse():
    raise AssertionError("load_all() was called by a single-family operation")


_families.load_all = _refuse
rs = np.random.RandomState(0)
X = rs.rand(400, 8).astype(np.float32)
y = (X[:, 0] > 0.5).astype(np.int64)
yr = X[:, 1].astype(np.float32)
{action}
missing = [name for name in _families.EAGER_BEFORE if _families.loaded(name) is None]
print(json.dumps(dict(missing=missing)))
"""


def _family(action):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PKG.parent), env.get("PYTHONPATH"))))
    done = subprocess.run([sys.executable, "-c", _FAMILY_PROBE.replace("{action}", action)], env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen["missing"], "a single-family operation imported every family"
    return seen


def test_neural_inference_imports_alone():
    # neural_inference -> _transformer_impl must not reach models.causal_lm
    # while neural_inference is half-initialized; the causal connection still
    # runs at this family's import (DEPENDS) and fills the neural slots.
    _family("""
import mojolearn.neural_inference as ni
from mojolearn import _native_neural_bridge as bridge
assert ni.TransformerBlockInference.__mro__[1].__name__ == "TransformerBlock"
assert bridge.RUNTIME is not None and type(bridge.CONTEXT) is dict
assert "mojolearn.models.causal_lm" in sys.modules
""")


def test_transformer_block_imports_alone_and_connects():
    _family("""
from mojolearn import TransformerBlock
from mojolearn import _native_neural_bridge as bridge
assert bridge.RUNTIME is not None and type(bridge.CONTEXT) is dict
""")


def test_nearest_centroid_alone():
    _family("mojolearn.NearestCentroid().fit(X, y).predict(X)")


def test_label_propagation_alone():
    _family("""
from mojolearn.semi_supervised import LabelPropagation
yy = y.copy(); yy[::2] = -1
LabelPropagation(max_iter=5).fit(X, yy).predict(X)
""")


def test_label_spreading_alone():
    _family("""
from mojolearn.semi_supervised import LabelSpreading
yy = y.copy(); yy[::2] = -1
LabelSpreading(max_iter=5).fit(X, yy).predict(X)
""")


def test_radius_neighbors_alone():
    _family("""
rn = mojolearn.RadiusNeighbors(radius=0.5).fit(X)
distances, indices = rn.radius_neighbors(X[:20], sort_results=True)
assert len(indices) == 20
""")


def test_bayesian_gaussian_mixture_alone():
    _family("""
for ct in ("full", "tied", "diag", "spherical"):
    mojolearn.BayesianGaussianMixture(n_components=2, covariance_type=ct, max_iter=5, random_state=0).fit(X)
""")


def test_voting_classifier_with_package_level_members_alone():
    # The members come from the package attributes, as the race builds them:
    # `mojolearn.DecisionTreeClassifier` routes to `_expansion_trees`, never
    # importing `tree`, so its CV registration must not depend on `tree`.
    _family("""
vc = mojolearn.VotingClassifier([("lr", mojolearn.LogisticRegression(max_iter=200)), ("nb", mojolearn.GaussianNB()),
                                 ("dt", mojolearn.DecisionTreeClassifier(max_depth=3, random_state=0))], voting="soft")
vc.fit(X, y).predict(X[:10])
""")


def test_stacking_regressor_with_package_level_members_alone():
    _family("""
sr = mojolearn.StackingRegressor([("lasso", mojolearn.Lasso(alpha=0.01, tol=1e-3, max_iter=1000)),
                                  ("dt", mojolearn.DecisionTreeRegressor(max_depth=3, random_state=0))],
                                 final_estimator=mojolearn.Ridge(alpha=1.0), cv=5)
sr.fit(X, yr).predict(X[:10])
""")


def test_cross_validation_alone():
    # The CV owner reads its fast-define masks (bindings/msel_cv_door.mojo) and
    # the scorer's metrics program without the resample family loaded.
    _family("""
scores = mojolearn.cross_val_score(mojolearn.Ridge(alpha=1.0), X, yr, cv=3)
assert len(scores) == 3
""")
