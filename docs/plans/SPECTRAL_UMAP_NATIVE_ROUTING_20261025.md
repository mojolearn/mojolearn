# Spectral and UMAP native routing (2026-10-25)

The public spectral fit/predict/embedding and UMAP fit/transform methods now
hand their input arguments directly to one compiled owner. Binding selection
previously ran in Python at each call, and UMAP random/PCA initialization called
the Python backend resolver again from Mojo.

`_native_spectral_umap_bridge.py` resolves the offered tiers at package
connection and copies only compiled functions into exact `ModuleType` objects.
Missing optional entries remain absent; an unavailable tier remains a named
error. The native route reads the backend's live default from its retained
dictionary, so changing the process default does not freeze callers to the
import-time value. Model-specific modes and UMAP's fitted mode select their own
retained table. UMAP rejects changed parameters/mode before looking up the fitted
route, preserving the refit error even when the changed live mode is unavailable.

The numerical routines and operands are unchanged. The native MKit helper now
reads exact-module namespaces directly when checking optional entries, avoiding
Python module error formatting and proxy lookup. Its existing non-module caller
path is unchanged. No source build or GPU qualification ran in this lane.

Source validation covers proxy detachment, Python callable rejection, retained
missing-tier errors, one-time connections, direct public argument handoff, and
the transitive decomp symbols needed by UMAP random/PCA initialization; all 16
tests passed. Complete
logs are in `~/mojolearn-evidence/spectral-umap-entry-final-20261025/`.

Integrated qualification owed:

- Build the consolidated native source, including the tokenizer/classical owner
  and runtime manifold owner. Numerical qualification remains in identical mode;
  this lane does not authorize an Apple fast campaign.
- Run the existing spectral embedding/predict and UMAP options/transform suites
  from the installed integrated wheel. Include dense, precomputed/nearest-neighbor
  spectral affinities, RBF, supported alternate label assignment, and UMAP
  spectral/random/PCA/array initializers.
- Trace the actual public operation for Python frames after package connection.
  Trap the Python backend binding resolver and proxy lookup to prove they are
  unreachable during fit, predict, transform and UMAP initialization.
- Check UMAP unfitted refusal, a changed parameter, a changed live default, and
  a changed explicit mode after fitting; ensure refit refusal precedes an
  unavailable-tier error. Preserve existing input ownership and fitted-array
  behavior. Source tests do not replace these native runtime checks.
