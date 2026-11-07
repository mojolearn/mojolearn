# scikit-learn's module layout in Mojolearn

`from sklearn.<module> import <Name>` becomes `from mojolearn.<module> import <Name>`.
The modules scikit-learn has and Mojolearn's package did not are their own files in
`python/mojolearn/`, each re-exporting classes defined in the implementation modules
(`_sklearn_modules.py` covers the modules that existed before). Nothing in the package
imports them: users reach them by their own path, as below. `packaging/wheel_ci.py
inventory` counts each `from mojolearn.<module> import` line here as that module's
importer, so a module removed from this list is reported as shipping unimported.

```python
from mojolearn.calibration import CalibratedClassifierCV
from mojolearn.covariance import EllipticEnvelope, MinCovDet
from mojolearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from mojolearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from mojolearn.feature_selection import RFE, SelectKBest, VarianceThreshold
from mojolearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from mojolearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from mojolearn.isotonic import IsotonicRegression
from mojolearn.kernel_approximation import Nystroem, RBFSampler
from mojolearn.kernel_ridge import KernelRidge
from mojolearn.multiclass import OneVsRestClassifier
from mojolearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from mojolearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from mojolearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from mojolearn.semi_supervised import LabelPropagation, LabelSpreading
```
